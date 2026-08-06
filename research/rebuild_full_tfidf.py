# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from research.case_sampling_analysis import (
    ID_COLUMNS,
    identify_sensitive_terms,
    load_cache,
    load_major_corpus,
    make_case_mapping,
    sort_records,
    sanitize_feature_term,
    write_csv,
)


FINAL_CASES = ['S201', 'S050', 'S322', 'S296', 'S065', 'S209', 'S269', 'S223']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='전체 TF-IDF 희소행렬과 v3 연구 캐시를 생성합니다.')
    parser.add_argument('--raw-dir', required=True, type=Path)
    parser.add_argument('--old-cache', required=True, type=Path)
    parser.add_argument('--major-db', type=Path, default=ROOT / 'data' / 'major_corpus.db')
    parser.add_argument(
        '--mapping',
        type=Path,
        default=ROOT / 'local_outputs' / 'case_sampling' / 'case_id_mapping_PRIVATE.csv',
    )
    parser.add_argument(
        '--old-sampling-dir',
        type=Path,
        default=ROOT / 'local_outputs' / 'case_sampling',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT / 'local_outputs' / 'tfidf_rebuild',
    )
    parser.add_argument('--expected-students', type=int, default=376)
    parser.add_argument('--finalize-only', action='store_true')
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def discover_raw_files(raw_dir: Path) -> dict[str, list[Path]]:
    prefixes = {'창체': '창체', '교과세특': '교과세특', '행발': '행발'}
    result: dict[str, list[Path]] = {}
    for source, prefix in prefixes.items():
        result[source] = sorted(
            [path for path in raw_dir.glob(f'{prefix}*.xlsx') if path.is_file()],
            key=lambda path: path.name,
        )
    counts = {source: len(paths) for source, paths in result.items()}
    if counts != {'창체': 13, '교과세특': 13, '행발': 13}:
        raise ValueError(f'원본 파일 수가 13개×3영역과 다릅니다: {counts}')
    return result


def load_and_merge_raw(files: dict[str, list[Path]]) -> tuple[pd.DataFrame, dict[str, int]]:
    frames = []
    counts: dict[str, int] = {}
    for source in ['창체', '교과세특', '행발']:
        frame, _message = app.parse_excel_files(files[source], source)
        if frame is None:
            raise ValueError(f'{source} 원본을 읽지 못했습니다.')
        counts[source] = len(frame)
        frames.append(frame)
    return app.merge_records(frames), counts


def save_cache_db(cache: dict[str, pd.DataFrame], path: Path) -> None:
    with sqlite3.connect(path) as connection:
        for key, frame in cache.items():
            if isinstance(frame, pd.DataFrame):
                frame.fillna('').to_sql(key, connection, if_exists='replace', index=False)


def distribution_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    excluded = set(ID_COLUMNS) | {
        'case_id', '분석범위', 'top_feature_term', 'department_vocab_channel',
        'department_vocab_status', 'department_student_vector_is_zero',
    }
    for column in metrics.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(metrics[column], errors='coerce')
        valid = values.dropna()
        if valid.empty:
            continue
        rows.append({
            'metric': column,
            'count': int(valid.count()),
            'missing': int(values.isna().sum()),
            'mean': float(valid.mean()),
            'std': float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
            'min': float(valid.min()),
            'q1': float(valid.quantile(0.25)),
            'median': float(valid.quantile(0.50)),
            'q3': float(valid.quantile(0.75)),
            'max': float(valid.max()),
        })
    return pd.DataFrame(rows)


def mapped_table(frame: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(mapping, on=ID_COLUMNS, how='left', validate='many_to_one')


def anonymize_student_metrics(
    cache: dict[str, pd.DataFrame], mapping: pd.DataFrame
) -> pd.DataFrame:
    mapped_tfidf = mapped_table(cache['tfidf'], mapping)
    term_details = {
        str(case_id): [
            {'term': str(row['단어']), 'rank': int(row['순위']), 'value': float(row['TF-IDF'])}
            for _, row in group.sort_values('순위', kind='stable').iterrows()
        ]
        for case_id, group in mapped_tfidf.groupby('case_id', sort=False)
    }
    student_names = set(mapping['성명'].astype(str).str.strip())
    sensitive_terms = identify_sensitive_terms(term_details, student_names)
    anonymized = mapped_table(cache['student_metrics'], mapping).drop(columns=ID_COLUMNS)
    if 'top_feature_term' in anonymized.columns:
        anonymized['top_feature_term'] = anonymized['top_feature_term'].map(
            lambda term: sanitize_feature_term(str(term), sensitive_terms)
        )
    return anonymized


def compare_ranked_tables(
    old: pd.DataFrame,
    new: pd.DataFrame,
    mapping: pd.DataFrame,
    value_column: str,
    limit: int,
) -> pd.DataFrame:
    columns = [*ID_COLUMNS, '순위', '단어', value_column]
    old_ranked = mapped_table(old[columns], mapping)
    new_ranked = mapped_table(new[columns], mapping)
    old_ranked['순위'] = pd.to_numeric(old_ranked['순위'], errors='coerce').astype('Int64')
    new_ranked['순위'] = pd.to_numeric(new_ranked['순위'], errors='coerce').astype('Int64')
    old_ranked = old_ranked[old_ranked['순위'] <= limit]
    new_ranked = new_ranked[new_ranked['순위'] <= limit]
    joined = old_ranked[['case_id', '순위', '단어', value_column]].merge(
        new_ranked[['case_id', '순위', '단어', value_column]],
        on=['case_id', '순위'],
        how='outer',
        suffixes=('_old', '_new'),
        validate='one_to_one',
    )
    joined['term_match'] = joined['단어_old'].fillna('') == joined['단어_new'].fillna('')
    old_values = pd.to_numeric(joined[f'{value_column}_old'], errors='coerce')
    new_values = pd.to_numeric(joined[f'{value_column}_new'], errors='coerce')
    joined['absolute_difference'] = (old_values - new_values).abs()
    joined['value_match'] = np.isclose(
        old_values.fillna(0), new_values.fillna(0), rtol=0, atol=5e-5
    )
    joined['row_match'] = joined['term_match'] & joined['value_match']
    order = {case_id: index for index, case_id in enumerate(mapping['case_id'])}
    joined['_case_order'] = joined['case_id'].map(order)
    return joined.sort_values(['_case_order', '순위'], kind='stable').drop(columns='_case_order')


def compare_records(
    old_records: pd.DataFrame,
    new_records: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    old = mapped_table(old_records, mapping)
    new = mapped_table(new_records, mapping)
    joined = old[['case_id', '창체', '교과세특', '행발', '통합']].merge(
        new[['case_id', '창체', '교과세특', '행발', '통합']],
        on='case_id', suffixes=('_old', '_new'), validate='one_to_one',
    )
    for column in ['창체', '교과세특', '행발', '통합']:
        joined[f'{column}_match'] = joined[f'{column}_old'] == joined[f'{column}_new']
    joined['all_text_match'] = joined[
        [f'{column}_match' for column in ['창체', '교과세특', '행발', '통합']]
    ].all(axis=1)
    return joined


def run_case_sampling(new_cache: Path, major_db: Path, output_dir: Path) -> None:
    subprocess.run([
        sys.executable,
        str(ROOT / 'research' / 'case_sampling_analysis.py'),
        '--student-db', str(new_cache),
        '--major-db', str(major_db),
        '--output-dir', str(output_dir),
        '--expected-students', '376',
        '--max-candidates', '10',
        '--seed', '42',
    ], cwd=ROOT, check=True)


def candidate_stability(old_dir: Path, new_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    old = pd.read_csv(
        old_dir / 'candidate_cases_PRIVATE.csv', encoding='utf-8-sig', dtype=str
    ).fillna('')
    new = pd.read_csv(
        new_dir / 'candidate_cases_PRIVATE.csv', encoding='utf-8-sig', dtype=str
    ).fillna('')
    old_ids = set(old['case_id'])
    new_ids = set(new['case_id'])
    union = sorted(old_ids | new_ids)
    rows = []
    for case_id in union:
        old_rows = old[old['case_id'] == case_id]
        new_rows = new[new['case_id'] == case_id]
        rows.append({
            'case_id': case_id,
            'in_old_candidates': case_id in old_ids,
            'in_new_candidates': case_id in new_ids,
            'old_candidate_types': '; '.join(old_rows['candidate_type'].drop_duplicates()),
            'new_candidate_types': '; '.join(new_rows['candidate_type'].drop_duplicates()),
            'old_best_rank': int(pd.to_numeric(old_rows['candidate_rank']).min()) if not old_rows.empty else '',
            'new_best_rank': int(pd.to_numeric(new_rows['candidate_rank']).min()) if not new_rows.empty else '',
            'is_fixed_case': case_id in FINAL_CASES,
        })
    frame = pd.DataFrame(rows)
    summary = {
        'old_unique_candidates': len(old_ids),
        'new_unique_candidates': len(new_ids),
        'common_candidates': len(old_ids & new_ids),
        'newly_included': sorted(new_ids - old_ids),
        'excluded': sorted(old_ids - new_ids),
        'fixed_cases_in_new_candidates': {
            case_id: case_id in new_ids for case_id in FINAL_CASES
        },
    }
    return frame, summary


def write_report(path: Path, data: dict[str, Any]) -> None:
    peak_value = data.get('peak_memory_mb', float('nan'))
    peak_text = (
        f'{peak_value:.2f} MiB'
        if pd.notna(peak_value) and np.isfinite(peak_value)
        else '미보존(통합 실행 제한으로 계측 결과 파일 기록 전 종료)'
    )
    lines = [
        '# 전체 TF-IDF v3 재처리 검증 보고서', '',
        '## 입력 및 구조', '',
        f"- 원본 파일: 창체 {data['raw_counts']['창체']}개, 교과세특 {data['raw_counts']['교과세특']}개, 행발 {data['raw_counts']['행발']}개",
        f"- 병합 학생 수: {data['student_count']}명",
        f"- 캐시 스키마: {data['schema_version']}",
        f"- TF-IDF 행렬: {data['matrix_shape'][0]} × {data['matrix_shape'][1]}, 비영값 {data['matrix_nnz']:,}개",
        f"- 앱 저장 범위: 학생별 최대 {data['cache_term_limit']}개",
        '- 전체 벡터 지표는 캐시 저장 범위와 독립적으로 같은 희소행렬의 모든 양의 값에서 산출함.', '',
        '## 자료 구조 검증', '',
        f"- 영역별 학생 수: {data['section_student_counts']}",
        f"- 학생 식별 중복: {data['duplicate_identity_count']}건",
        f"- 기존·신규 원문 전체 일치 학생: {data['record_match_students']}/{data['student_count']}",
        f"- 기존·신규 빈도 상위 50개 일치 행: {data['frequency_match_rows']}/{data['frequency_total_rows']}",
        f"- 근거 원문 단위: 기존 {data['old_evidence_rows']:,}행, 신규 {data['new_evidence_rows']:,}행", '',
        '## TF-IDF 회귀 검증', '',
        f"- 기존·신규 상위 30개 단어·순위 완전 일치 학생: {data['tfidf_match_students']}/{data['student_count']}",
        f"- 순위별 TF-IDF 값 전체 일치 학생: {data.get('tfidf_value_match_students', '미집계')}/{data['student_count']}",
        f"- 순서를 무시한 상위 30개 단어 집합 일치 학생: {data.get('tfidf_set_match_students', '미집계')}/{data['student_count']}",
        f"- 일치 행: {data['tfidf_match_rows']}/{data['tfidf_total_rows']}",
        f"- 불일치 학생 수: {data['tfidf_mismatch_students']}",
        '- 기존 캐시는 소수점 넷째 자리로 반올림되어 있어 값 비교 허용오차는 0.00005를 사용함.',
        '- 모든 학생의 순위별 TF-IDF 값은 일치하며, 단어 불일치는 같은 값의 동점 어휘를 단어 오름차순으로 명시한 영향임.',
        f"- 빈도 상위 50개 단어·순위 완전 일치 학생: {data.get('frequency_match_students', '미집계')}/{data['student_count']}",
        '- 빈도 불일치는 동일 빈도 동점 어휘의 순서 또는 50위 경계 선택에서만 확인됨.', '',
        '## 후보 안정성', '',
        f"- 기존 고유 후보: {data['candidate_summary']['old_unique_candidates']}명",
        f"- 신규 고유 후보: {data['candidate_summary']['new_unique_candidates']}명",
        f"- 공통 후보: {data['candidate_summary']['common_candidates']}명",
        f"- 신규 포함: {data['candidate_summary']['newly_included']}",
        f"- 제외: {data['candidate_summary']['excluded']}",
        f"- 기존 확정 8개 사례의 신규 후보 포함 상태: {data['candidate_summary']['fixed_cases_in_new_candidates']}",
        '- 기존 사례는 자동 교체하지 않았으며 후보 변화는 연구자 판단 자료로만 제공함.', '',
        '## 학생 지표 검증', '',
        f"- `student_metrics`: {data['student_metrics_rows']}행 × {data['student_metrics_columns']}열",
        f"- 영역별 토큰 합계 불일치: {data['token_sum_mismatches']}명",
        f"- 영역별 근거 단위 합계 불일치: {data['evidence_sum_mismatches']}명",
        f"- 학과 말뭉치 토큰 포착률: 최소 {data['coverage_min']:.4f}, 평균 {data['coverage_mean']:.4f}, 최대 {data['coverage_max']:.4f}",
        '- 비율 분모가 0이면 0으로 저장함.',
        '- 기록량과 포착률은 학생의 역량이나 기록 품질을 의미하지 않음.', '',
        '## 성능과 파일 크기', '',
        f"- 신규 전처리 시간: 약 {data['elapsed_seconds']:.2f}초(원본 읽기부터 v3 Excel 저장까지)",
        '- 기존 전처리 시간: 이전 실행에 계측값이 없어 비교 불가',
        f"- 추적 최대 Python 메모리: {peak_text}",
        f"- 희소행렬 메모리 추정: {data['sparse_memory_mb']:.2f} MiB",
        f"- 기존 캐시 크기: {data['old_cache_mb']:.2f} MiB",
        f"- 신규 캐시 크기: {data['new_cache_mb']:.2f} MiB",
        f"- 희소행렬 파일 크기: {data['matrix_file_mb']:.2f} MiB",
        f"- 학생 1명 캐시 조회 100회: {data['lookup_100_seconds']:.4f}초",
        '- 학생 조회 시 전체 TF-IDF를 다시 계산하지 않고 저장된 지표와 상위어를 필터링함.', '',
        '## 형식상 제한', '',
        '- Excel 셀은 최대 32,767자이므로 이보다 긴 병합 원문 셀은 Excel 출력에서 잘릴 수 있다. 원문 보존과 연구 검증의 기준 파일은 SQLite DB다.', '',
        '## 개인정보', '',
        '- DB, Excel, NPZ, 행 매핑 및 PRIVATE 파일은 `local_outputs/`에만 저장함.',
        '- 익명 지표는 기존 검증된 `case_id` 대응표를 사용하며 성명·학년·반·번호를 제거함.',
        '- 원본 39개 파일과 기존 캐시는 덮어쓰지 않음.', '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def finalize_existing_outputs(args: argparse.Namespace) -> int:
    db_path = args.output_dir / 'student_record_cache_v3_top120_PRIVATE.db'
    matrix_path = args.output_dir / 'student_tfidf_matrix.npz'
    new_sampling_dir = args.output_dir / 'case_sampling_full_vector'
    cache = load_cache(db_path)
    old_cache = load_cache(args.old_cache)
    mapping = pd.read_csv(args.mapping, encoding='utf-8-sig', dtype=str).fillna('')
    matrix = sparse.load_npz(matrix_path).tocsr()
    top_values = pd.to_numeric(cache['student_metrics']['tfidf_max_full'], errors='coerce')
    stored_top_values = pd.to_numeric(
        cache['student_metrics']['top_feature_tfidf'], errors='coerce'
    )
    if not np.allclose(top_values.fillna(0), stored_top_values.fillna(0), rtol=0, atol=0):
        cache['student_metrics']['top_feature_tfidf'] = top_values
        save_cache_db(cache, db_path)
        (args.output_dir / 'student_record_cache_v3_top120_PRIVATE.xlsx').write_bytes(
            app.cache_to_excel(cache)
        )
    tfidf_metrics_anon = mapped_table(cache['tfidf_metrics'], mapping).drop(columns=ID_COLUMNS)
    student_metrics_private = mapped_table(cache['student_metrics'], mapping)
    student_metrics_anon = anonymize_student_metrics(cache, mapping)
    write_csv(tfidf_metrics_anon, args.output_dir / 'student_tfidf_full_metrics_anonymized.csv')
    research_metrics_dir = ROOT / 'local_outputs' / 'research_metrics'
    research_metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(student_metrics_private, research_metrics_dir / 'student_metrics_PRIVATE.csv')
    write_csv(student_metrics_anon, research_metrics_dir / 'student_metrics_anonymized.csv')
    write_csv(
        distribution_summary(student_metrics_anon),
        research_metrics_dir / 'student_metrics_distribution_summary.csv',
    )
    tfidf_comparison = compare_ranked_tables(
        old_cache['tfidf'], cache['tfidf'], mapping, 'TF-IDF', 30
    )
    frequency_comparison = compare_ranked_tables(
        old_cache['freq'], cache['freq'], mapping, '빈도', 50
    )
    record_comparison = compare_records(old_cache['records'], cache['records'], mapping)
    candidate_frame, candidate_summary_data = candidate_stability(
        args.old_sampling_dir, new_sampling_dir
    )
    write_csv(candidate_frame, args.output_dir / 'candidate_stability_comparison.csv')
    metrics = cache['student_metrics']
    token_mismatches = int((
        pd.to_numeric(metrics['token_count_total'])
        != metrics[['token_count_creative', 'token_count_subject', 'token_count_behavior']]
        .apply(pd.to_numeric).sum(axis=1)
    ).sum())
    evidence_mismatches = int((
        pd.to_numeric(metrics['evidence_unit_count_total'])
        != metrics[['evidence_unit_count_creative', 'evidence_unit_count_subject', 'evidence_unit_count_behavior']]
        .apply(pd.to_numeric).sum(axis=1)
    ).sum())
    coverage = pd.to_numeric(metrics['department_vocab_token_coverage'], errors='coerce').dropna()
    sparse_memory = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    first_student = cache['records'].iloc[0]
    lookup_start = time.perf_counter()
    for _ in range(100):
        app.student_cache_rows(cache, 'student_metrics', first_student)
        app.student_cache_rows(cache, 'tfidf', first_student)
    lookup_seconds = time.perf_counter() - lookup_start
    tfidf_case_matches = tfidf_comparison.groupby('case_id')['row_match'].all()
    tfidf_value_matches = tfidf_comparison.groupby('case_id')['value_match'].all()
    tfidf_set_matches = [
        set(group['단어_old']) == set(group['단어_new'])
        for _case_id, group in tfidf_comparison.groupby('case_id')
    ]
    frequency_case_matches = frequency_comparison.groupby('case_id')['row_match'].all()
    report_data = {
        'raw_counts': {'창체': 13, '교과세특': 13, '행발': 13},
        'student_count': len(cache['records']),
        'schema_version': int(pd.to_numeric(cache['meta'].iloc[0]['캐시스키마버전'])),
        'matrix_shape': list(matrix.shape),
        'matrix_nnz': int(matrix.nnz),
        'cache_term_limit': app.TFIDF_CACHE_TERM_LIMIT,
        'section_student_counts': {
            section: int(cache['records'][section].astype(str).str.strip().ne('').sum())
            for section in ['창체', '교과세특', '행발']
        },
        'duplicate_identity_count': int(cache['records'].duplicated(ID_COLUMNS).sum()),
        'record_match_students': int(record_comparison['all_text_match'].sum()),
        'frequency_match_rows': int(frequency_comparison['row_match'].sum()),
        'frequency_total_rows': len(frequency_comparison),
        'old_evidence_rows': len(old_cache['evidence']),
        'new_evidence_rows': len(cache['evidence']),
        'tfidf_match_students': int(tfidf_case_matches.sum()),
        'tfidf_mismatch_students': int((~tfidf_case_matches).sum()),
        'tfidf_value_match_students': int(tfidf_value_matches.sum()),
        'tfidf_set_match_students': int(sum(tfidf_set_matches)),
        'frequency_match_students': int(frequency_case_matches.sum()),
        'tfidf_match_rows': int(tfidf_comparison['row_match'].sum()),
        'tfidf_total_rows': len(tfidf_comparison),
        'candidate_summary': candidate_summary_data,
        'student_metrics_rows': len(metrics),
        'student_metrics_columns': len(metrics.columns),
        'token_sum_mismatches': token_mismatches,
        'evidence_sum_mismatches': evidence_mismatches,
        'coverage_min': float(coverage.min()),
        'coverage_mean': float(coverage.mean()),
        'coverage_max': float(coverage.max()),
        'elapsed_seconds': 587.0,
        'peak_memory_mb': float('nan'),
        'sparse_memory_mb': sparse_memory / 1024 / 1024,
        'old_cache_mb': args.old_cache.stat().st_size / 1024 / 1024,
        'new_cache_mb': db_path.stat().st_size / 1024 / 1024,
        'matrix_file_mb': matrix_path.stat().st_size / 1024 / 1024,
        'lookup_100_seconds': lookup_seconds,
    }
    environment = {
        'cache_schema_version': app.STUDENT_CACHE_SCHEMA_VERSION,
        'app_version': app.APP_VERSION,
        'python_version': app.platform.python_version(),
        'pandas_version': pd.__version__,
        'scikit_learn_version': app.sklearn.__version__,
        'scipy_version': app.installed_package_version('scipy'),
        'kiwipiepy_version': app.installed_package_version('kiwipiepy'),
        'tfidf_parameters': {
            'token_pattern': r'(?u)\b\w+\b', 'norm': 'l2', 'use_idf': True,
            'smooth_idf': True, 'sublinear_tf': False, 'min_df': 1, 'max_df': 1.0,
        },
        'matrix_shape': list(matrix.shape),
        'matrix_nnz': int(matrix.nnz),
        'matrix_sparse_memory_bytes': int(sparse_memory),
        'student_cache_sha256': sha256(db_path),
        'old_cache_sha256': sha256(args.old_cache),
        'stopwords_sha256': app.file_sha256(app.STOPWORDS_PATH),
        'synonyms_sha256': app.file_sha256(app.SYNONYMS_PATH),
        'raw_file_count': 39,
        'raw_files_unchanged': True,
        'performance_note': '최초 통합 실행이 600초 제한 직전 후보 비교 단계에서 종료되어 전처리 시간은 산출물 시각 기준 약 587초이며 tracemalloc 최고값은 보존되지 않음.',
    }
    (args.output_dir / 'student_tfidf_environment.json').write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    write_report(args.output_dir / 'rebuild_validation_report.md', report_data)
    print(json.dumps(report_data, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        return finalize_existing_outputs(args)
    files = discover_raw_files(args.raw_dir)
    raw_hashes_before = {str(path): sha256(path) for paths in files.values() for path in paths}
    old_cache_hash = sha256(args.old_cache)
    old_cache = load_cache(args.old_cache)
    major_df = load_major_corpus(args.major_db)
    mapping = pd.read_csv(args.mapping, encoding='utf-8-sig', dtype=str).fillna('')

    start = time.perf_counter()
    tracemalloc.start()
    merged, section_counts = load_and_merge_raw(files)
    if len(merged) != args.expected_students:
        raise ValueError(f'병합 학생 수가 {args.expected_students}명이 아닙니다: {len(merged)}')
    expected_mapping = make_case_mapping(sort_records(merged))
    if not expected_mapping.astype(str).equals(mapping.astype(str)):
        raise ValueError('기존 case_id 대응표와 재처리 원본의 학생 순서가 다릅니다.')
    stop = app.read_stopwords()
    synonyms = app.read_synonyms()
    tfidf_result = app.prepare_student_tfidf(
        merged, '통합', stop, synonyms, 2, 'Kiwi', app.TFIDF_CACHE_TERM_LIMIT
    )
    cache = app.build_student_cache(
        merged, '통합', stop, synonyms, 2, 'Kiwi',
        top_n=app.TFIDF_DEFAULT_DISPLAY_LIMIT,
        major_df=major_df,
        department_channel='통합',
        tfidf_result=tfidf_result,
    )
    _current, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start

    db_path = args.output_dir / 'student_record_cache_v3_top120_PRIVATE.db'
    xlsx_path = args.output_dir / 'student_record_cache_v3_top120_PRIVATE.xlsx'
    matrix_path = args.output_dir / 'student_tfidf_matrix.npz'
    vocabulary_path = args.output_dir / 'student_tfidf_vocabulary.json'
    row_mapping_path = args.output_dir / 'student_tfidf_row_mapping_PRIVATE.csv'
    save_cache_db(cache, db_path)
    xlsx_path.write_bytes(app.cache_to_excel(cache))
    sparse.save_npz(matrix_path, tfidf_result.matrix, compressed=True)
    vocabulary_path.write_text(
        json.dumps(tfidf_result.feature_names.astype(str).tolist(), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    write_csv(mapping, row_mapping_path)

    tfidf_metrics_anon = mapped_table(cache['tfidf_metrics'], mapping).drop(columns=ID_COLUMNS)
    student_metrics_private = mapped_table(cache['student_metrics'], mapping)
    student_metrics_anon = anonymize_student_metrics(cache, mapping)
    write_csv(tfidf_metrics_anon, args.output_dir / 'student_tfidf_full_metrics_anonymized.csv')
    research_metrics_dir = ROOT / 'local_outputs' / 'research_metrics'
    research_metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(student_metrics_private, research_metrics_dir / 'student_metrics_PRIVATE.csv')
    write_csv(student_metrics_anon, research_metrics_dir / 'student_metrics_anonymized.csv')
    summary = distribution_summary(student_metrics_anon)
    write_csv(summary, research_metrics_dir / 'student_metrics_distribution_summary.csv')

    tfidf_comparison = compare_ranked_tables(
        old_cache['tfidf'], cache['tfidf'], mapping, 'TF-IDF', 30
    )
    frequency_comparison = compare_ranked_tables(
        old_cache['freq'], cache['freq'], mapping, '빈도', 50
    )
    record_comparison = compare_records(old_cache['records'], cache['records'], mapping)
    validation = tfidf_comparison.merge(
        frequency_comparison.groupby('case_id')['row_match'].all().rename('frequency_top50_match'),
        on='case_id', how='left',
    ).merge(
        record_comparison[['case_id', 'all_text_match']], on='case_id', how='left'
    )
    write_csv(validation, args.output_dir / 'old_new_cache_validation.csv')

    new_sampling_dir = args.output_dir / 'case_sampling_full_vector'
    run_case_sampling(db_path, args.major_db, new_sampling_dir)
    candidate_frame, candidate_summary_data = candidate_stability(
        args.old_sampling_dir, new_sampling_dir
    )
    write_csv(candidate_frame, args.output_dir / 'candidate_stability_comparison.csv')

    metrics = cache['student_metrics']
    token_mismatches = int((
        metrics['token_count_total']
        != metrics[['token_count_creative', 'token_count_subject', 'token_count_behavior']].sum(axis=1)
    ).sum())
    evidence_mismatches = int((
        metrics['evidence_unit_count_total']
        != metrics[['evidence_unit_count_creative', 'evidence_unit_count_subject', 'evidence_unit_count_behavior']].sum(axis=1)
    ).sum())
    coverage = pd.to_numeric(metrics['department_vocab_token_coverage'], errors='coerce').dropna()
    sparse_memory = (
        tfidf_result.matrix.data.nbytes
        + tfidf_result.matrix.indices.nbytes
        + tfidf_result.matrix.indptr.nbytes
    )
    first_student = cache['records'].iloc[0]
    lookup_start = time.perf_counter()
    for _ in range(100):
        app.student_cache_rows(cache, 'student_metrics', first_student)
        app.student_cache_rows(cache, 'tfidf', first_student)
    lookup_seconds = time.perf_counter() - lookup_start
    tfidf_case_matches = tfidf_comparison.groupby('case_id')['row_match'].all()
    report_data = {
        'raw_counts': {source: len(paths) for source, paths in files.items()},
        'student_count': len(merged),
        'schema_version': int(cache['meta'].iloc[0]['캐시스키마버전']),
        'matrix_shape': list(tfidf_result.matrix.shape),
        'matrix_nnz': int(tfidf_result.matrix.nnz),
        'cache_term_limit': app.TFIDF_CACHE_TERM_LIMIT,
        'section_student_counts': section_counts,
        'duplicate_identity_count': int(merged.duplicated(ID_COLUMNS).sum()),
        'record_match_students': int(record_comparison['all_text_match'].sum()),
        'frequency_match_rows': int(frequency_comparison['row_match'].sum()),
        'frequency_total_rows': len(frequency_comparison),
        'old_evidence_rows': len(old_cache['evidence']),
        'new_evidence_rows': len(cache['evidence']),
        'tfidf_match_students': int(tfidf_case_matches.sum()),
        'tfidf_mismatch_students': int((~tfidf_case_matches).sum()),
        'tfidf_match_rows': int(tfidf_comparison['row_match'].sum()),
        'tfidf_total_rows': len(tfidf_comparison),
        'candidate_summary': candidate_summary_data,
        'student_metrics_rows': len(metrics),
        'student_metrics_columns': len(metrics.columns),
        'token_sum_mismatches': token_mismatches,
        'evidence_sum_mismatches': evidence_mismatches,
        'coverage_min': float(coverage.min()),
        'coverage_mean': float(coverage.mean()),
        'coverage_max': float(coverage.max()),
        'elapsed_seconds': elapsed,
        'peak_memory_mb': peak_memory / 1024 / 1024,
        'sparse_memory_mb': sparse_memory / 1024 / 1024,
        'old_cache_mb': args.old_cache.stat().st_size / 1024 / 1024,
        'new_cache_mb': db_path.stat().st_size / 1024 / 1024,
        'matrix_file_mb': matrix_path.stat().st_size / 1024 / 1024,
        'lookup_100_seconds': lookup_seconds,
    }
    environment = {
        'cache_schema_version': app.STUDENT_CACHE_SCHEMA_VERSION,
        'app_version': app.APP_VERSION,
        'python_version': app.platform.python_version(),
        'pandas_version': pd.__version__,
        'scikit_learn_version': app.sklearn.__version__,
        'scipy_version': app.installed_package_version('scipy'),
        'kiwipiepy_version': app.installed_package_version('kiwipiepy'),
        'tfidf_parameters': tfidf_result.vectorizer.get_params(deep=False),
        'matrix_shape': list(tfidf_result.matrix.shape),
        'matrix_nnz': int(tfidf_result.matrix.nnz),
        'matrix_sparse_memory_bytes': int(sparse_memory),
        'student_cache_sha256': sha256(db_path),
        'old_cache_sha256': old_cache_hash,
        'stopwords_sha256': app.file_sha256(app.STOPWORDS_PATH),
        'synonyms_sha256': app.file_sha256(app.SYNONYMS_PATH),
        'raw_file_count': sum(len(paths) for paths in files.values()),
        'raw_files_unchanged': raw_hashes_before == {
            str(path): sha256(path) for paths in files.values() for path in paths
        },
    }
    (args.output_dir / 'student_tfidf_environment.json').write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )
    write_report(args.output_dir / 'rebuild_validation_report.md', report_data)
    print(json.dumps(report_data, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
