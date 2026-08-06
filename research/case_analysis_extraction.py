# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
    read_analysis_settings,
    sort_records,
    student_key,
    write_csv,
)


CASES = [
    ('S201', '주후보', '기록과 학과 관련 어휘가 비교적 수렴하는 사례', ''),
    ('S050', '주후보', '하나의 관심 주제가 여러 유사 학과로 제시되는 사례', ''),
    ('S322', '주후보', '기록량과 근거가 제한되어 해석에 주의가 필요한 사례', ''),
    ('S296', '주후보', '특징어는 뚜렷하지만 학과 말뭉치의 포착 범위가 제한되는 사례', ''),
    ('S065', '예비후보', 'S201의 예비후보', 'S201'),
    ('S209', '예비후보', 'S050의 예비후보', 'S050'),
    ('S269', '예비후보', 'S322의 예비후보', 'S322'),
    ('S223', '예비후보', 'S296의 예비후보', 'S296'),
]
CASE_ORDER = {case_id: index for index, (case_id, *_rest) in enumerate(CASES)}
FOCUS_TERMS = {
    'S201': ['응급', '응급구조사', '의료', '구조', '지혈', '감염병', '소생술'],
    'S050': ['디자인', '패키지', '작품', '시각', '디자이너'],
    'S322': ['화재', '진압', '소방관', '소방', '보컬', '암호'],
    'S296': ['변리사', '특허', '기술', '촉감', '공학'],
}
PRIVATE_ID_COLUMNS = ['학년', '반', '번호', '성명']
SECTION_NAMES = ['창체', '교과세특', '행발']
OVERVIEW_METRICS = [
    'total_token_count',
    'creative_activity_token_count',
    'subject_note_token_count',
    'behavior_comment_token_count',
    'evidence_sentence_count',
    'cross_section_feature_count',
    'department_similarity_rank1',
    'department_similarity_rank1_rank5_gap',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='선정 사례 8명의 논문 사례 분석 자료를 추출합니다.')
    parser.add_argument('--student-db', required=True, type=Path)
    parser.add_argument('--major-db', type=Path, default=ROOT / 'data' / 'major_corpus.db')
    parser.add_argument(
        '--sampling-dir', type=Path, default=ROOT / 'local_outputs' / 'case_sampling'
    )
    parser.add_argument(
        '--output-dir', type=Path, default=ROOT / 'local_outputs' / 'case_analysis'
    )
    parser.add_argument('--expected-students', type=int, default=376)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_keywords(value: Any) -> list[str]:
    return [item.strip() for item in str(value or '').split(',') if item.strip()]


def ordered(frame: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result['_case_order'] = result['case_id'].map(CASE_ORDER)
    result = result.sort_values(['_case_order', *extra], kind='stable')
    return result.drop(columns='_case_order').reset_index(drop=True)


def private_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in PRIVATE_ID_COLUMNS if column in frame.columns]


def anonymize_term(term: Any, sensitive_terms: set[str]) -> str:
    value = str(term or '').strip()
    return '[희귀 고유명사 제거]' if value in sensitive_terms else value


INSTITUTION_RE = re.compile(
    r'([가-힣A-Za-z0-9·]{2,20}?(?:초등학교|중학교|고등학교|대학교|학교|병원|센터|연구소|복지관|수련관|협회|재단|주식회사|회사))'
)
PERSON_ROLE_RE = re.compile(r'(?<![가-힣])([가-힣]{2,4})(?=(?:\s?(?:선생님|교사|강사|교수|님)))')


def anonymize_text(
    text: Any,
    student_names: set[str],
    rare_entities: set[str] | None = None,
) -> str:
    value = str(text or '')
    for name in sorted((name for name in student_names if name), key=len, reverse=True):
        value = value.replace(name, '[인명 제거]')
    value = PERSON_ROLE_RE.sub('[인명 제거]', value)
    value = INSTITUTION_RE.sub('[기관명 제거]', value)
    for entity in sorted((rare_entities or set()), key=len, reverse=True):
        if entity:
            value = value.replace(entity, '[고유명사 제거]')
    return value


def read_sampling_inputs(sampling_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        'mapping': sampling_dir / 'case_id_mapping_PRIVATE.csv',
        'metrics': sampling_dir / 'case_sampling_metrics_PRIVATE.csv',
        'candidates': sampling_dir / 'candidate_cases_PRIVATE.csv',
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('필요한 기존 사례 선정 파일이 없습니다: ' + ', '.join(missing))
    return tuple(
        pd.read_csv(path, encoding='utf-8-sig', dtype=str).fillna('')
        for path in paths.values()
    )


def validate_mapping(
    cache: dict[str, pd.DataFrame], mapping: pd.DataFrame, expected_students: int
) -> dict[str, Any]:
    expected = make_case_mapping(sort_records(cache['records']))
    exact_columns = mapping.columns.tolist() == ['case_id', *ID_COLUMNS]
    normalized = mapping[['case_id', *ID_COLUMNS]].astype(str).reset_index(drop=True)
    expected = expected[['case_id', *ID_COLUMNS]].astype(str).reset_index(drop=True)
    return {
        'row_count': len(mapping),
        'expected_row_count': expected_students,
        'row_count_matches': len(mapping) == expected_students,
        'exact_required_columns': exact_columns,
        'no_blank_values': not normalized.eq('').any().any(),
        'no_duplicate_case_id': not normalized['case_id'].duplicated().any(),
        'no_duplicate_identity': not normalized[ID_COLUMNS].duplicated().any(),
        'exact_cache_order_and_identity_match': normalized.equals(expected),
    }


def build_overview(
    mapping: pd.DataFrame, metrics: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    selected = pd.DataFrame(
        CASES, columns=['case_id', '후보 구분', '분석 조건', '대응 주후보']
    )
    metric_columns = [
        'case_id', *PRIVATE_ID_COLUMNS,
        *OVERVIEW_METRICS,
        'total_token_count_percentile', 'total_token_count_quartile',
        'evidence_sentence_count_percentile', 'evidence_sentence_count_quartile',
        'cross_section_feature_count_percentile', 'cross_section_feature_count_quartile',
        'department_similarity_rank1_percentile', 'department_similarity_rank1_quartile',
        'department_similarity_rank1_rank5_gap_percentile',
        'department_similarity_rank1_rank5_gap_quartile',
        'data_quality_note',
    ]
    available = [column for column in metric_columns if column in metrics.columns]
    result = selected.merge(metrics[available], on='case_id', how='left', validate='one_to_one')
    grouped = candidates[candidates['case_id'].isin(CASE_ORDER)].groupby('case_id', sort=False)
    candidate_details = grouped.agg({
        'candidate_type': lambda values: '; '.join(dict.fromkeys(map(str, values))),
        'candidate_type_name': lambda values: '; '.join(dict.fromkeys(map(str, values))),
        'matched_conditions': lambda values: '; '.join(dict.fromkeys(map(str, values))),
    }).reset_index()
    result = result.merge(candidate_details, on='case_id', how='left')
    return ordered(result, [])


def selected_cache_table(
    frame: pd.DataFrame, mapping: pd.DataFrame, case_ids: Iterable[str]
) -> pd.DataFrame:
    result = frame.merge(mapping, on=ID_COLUMNS, how='inner', validate='many_to_one')
    return result[result['case_id'].isin(case_ids)].copy()


def build_features_and_evidence(
    cache: dict[str, pd.DataFrame], mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    case_ids = list(CASE_ORDER)
    tfidf = selected_cache_table(cache['tfidf'], mapping, case_ids)
    tfidf['순위'] = pd.to_numeric(tfidf['순위'], errors='raise').astype(int)
    tfidf['TF-IDF'] = pd.to_numeric(tfidf['TF-IDF'], errors='raise').astype(float)
    tfidf = ordered(tfidf[tfidf['순위'] <= 30], ['순위'])

    evidence = selected_cache_table(cache['evidence'], mapping, case_ids)
    evidence['문장번호'] = pd.to_numeric(evidence['문장번호'], errors='raise').astype(int)
    feature_lookup = {
        case_id: {
            str(row['단어']): int(row['순위'])
            for _, row in group.iterrows()
        }
        for case_id, group in tfidf.groupby('case_id', sort=False)
    }
    rows: list[dict[str, Any]] = []
    sentence_payload: dict[str, dict[str, dict[str, Any]]] = {}
    for _, row in evidence.iterrows():
        case_id = str(row['case_id'])
        keywords = split_keywords(row['키워드목록'])
        matches = sorted(
            (term for term in keywords if term in feature_lookup.get(case_id, {})),
            key=lambda term: feature_lookup[case_id][term],
        )
        if not matches:
            continue
        raw = str(row['원문'])
        source = str(row['출처'])
        sentence_number = int(row['문장번호'])
        identity = {column: str(row[column]) for column in PRIVATE_ID_COLUMNS}
        fingerprint = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]
        sentence_id = f'{case_id}-{source}-{sentence_number:04d}-{fingerprint[:6]}'
        duplicate_group_id = f'DUP-{fingerprint}'
        sentence_payload.setdefault(case_id, {})[sentence_id] = {
            'sentence_id': sentence_id,
            'duplicate_group_id': duplicate_group_id,
            '출처 영역': source,
            '문장번호': sentence_number,
            '원문': raw,
            '해당 문장에 연결된 전체 키워드목록': keywords,
        }
        for term in matches:
            rows.append({
                'case_id': case_id,
                **identity,
                'feature_term': term,
                'feature_rank': feature_lookup[case_id][term],
                'sentence_id': sentence_id,
                'duplicate_group_id': duplicate_group_id,
                '출처 영역': source,
                '문장번호': sentence_number,
                '원문': raw,
                '해당 문장에 연결된 전체 키워드목록': ', '.join(keywords),
            })
    evidence_rows = ordered(pd.DataFrame(rows), ['feature_rank', '출처 영역', '문장번호'])

    count_rows: list[dict[str, Any]] = []
    for _, row in tfidf.iterrows():
        matching = evidence_rows[
            (evidence_rows['case_id'] == row['case_id'])
            & (evidence_rows['feature_term'] == row['단어'])
        ]
        counts = matching['출처 영역'].value_counts().to_dict()
        count_rows.append({
            'case_id': row['case_id'],
            **{column: row[column] for column in PRIVATE_ID_COLUMNS},
            'feature_rank': int(row['순위']),
            'feature_term': str(row['단어']),
            'tfidf_value': float(row['TF-IDF']),
            'total_evidence_sentence_count': len(matching),
            'creative_activity_evidence_count': int(counts.get('창체', 0)),
            'subject_note_evidence_count': int(counts.get('교과세특', 0)),
            'behavior_comment_evidence_count': int(counts.get('행발', 0)),
            'appearing_section_count': sum(int(counts.get(section, 0) > 0) for section in SECTION_NAMES),
        })
    features = ordered(pd.DataFrame(count_rows), ['feature_rank'])
    payload = {
        'description': '상위 30개 TF-IDF 특징어에 연결된 학생 캐시 evidence 문장',
        'cases': [],
    }
    for case_id, role, condition, reserve_for in CASES:
        identity = mapping.loc[mapping['case_id'] == case_id].iloc[0]
        payload['cases'].append({
            'case_id': case_id,
            **{column: str(identity[column]) for column in PRIVATE_ID_COLUMNS},
            '후보 구분': role,
            '분석 조건': condition,
            '대응 주후보': reserve_for,
            'sentences': list(sentence_payload.get(case_id, {}).values()),
            'feature_links': evidence_rows.loc[
                evidence_rows['case_id'] == case_id,
                ['feature_term', 'feature_rank', 'sentence_id'],
            ].to_dict('records'),
        })
    feature_details = {
        case_id: group[['feature_rank', 'feature_term', 'tfidf_value']]
        .rename(columns={'feature_rank': 'rank', 'feature_term': 'term', 'tfidf_value': 'value'})
        .to_dict('records')
        for case_id, group in features.groupby('case_id', sort=False)
    }
    return features, evidence_rows, payload, feature_details


def build_frequency(cache: dict[str, pd.DataFrame], mapping: pd.DataFrame) -> pd.DataFrame:
    frequency = selected_cache_table(cache['freq'], mapping, CASE_ORDER)
    frequency['순위'] = pd.to_numeric(frequency['순위'], errors='raise').astype(int)
    frequency['빈도'] = pd.to_numeric(frequency['빈도'], errors='raise').astype(int)
    frequency = frequency[frequency['순위'] <= 50].copy()
    frequency = frequency.rename(columns={
        '분석범위': 'analysis_scope', '순위': 'frequency_rank',
        '단어': 'term', '빈도': 'frequency',
    })
    columns = ['case_id', *PRIVATE_ID_COLUMNS, 'analysis_scope', 'frequency_rank', 'term', 'frequency']
    return ordered(frequency[columns], ['frequency_rank'])


def build_departments(
    records: pd.DataFrame,
    major_df: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    documents = tuple(app.corpus_texts(major_df, '통합'))
    vectorizer, major_matrix, terms = app.prepare_major_index(
        documents,
        tuple(sorted(settings['stopwords'])),
        tuple(sorted(settings['synonyms'].items())),
        settings['min_len'],
        settings['analyzer'],
    )
    if vectorizer is None:
        raise RuntimeError('학과 말뭉치 TF-IDF 인덱스를 만들 수 없습니다.')
    name_column = '학과명' if '학과명' in major_df.columns else '학과'
    department_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    app_checks: dict[str, Any] = {}
    for _, record in ordered(records, []).iterrows():
        case_id = str(record['case_id'])
        student_text = str(record.get(settings['scope'], record.get('통합', '')))
        student_doc = app.tokenized(
            student_text, settings['stopwords'], settings['synonyms'],
            settings['min_len'], settings['analyzer']
        )
        student_vector = vectorizer.transform([student_doc])
        scores = cosine_similarity(student_vector, major_matrix).ravel()
        indices = scores.argsort()[::-1][:10]
        app_result = app.similarity(
            student_text, major_df, settings['stopwords'], settings['synonyms'],
            settings['min_len'], settings['analyzer'], top_k=10, channel='통합'
        )
        direct_pairs: list[tuple[str, float]] = []
        for rank, major_index in enumerate(indices, 1):
            target = major_df.iloc[int(major_index)]
            similarity_raw = round(float(scores[int(major_index)]), 4)
            direct_pairs.append((str(target.get(name_column, '')), similarity_raw))
            gap, common, _gap_dict, metrics = app.gap_analysis(
                student_text, target, major_df, settings['stopwords'], settings['synonyms'],
                settings['min_len'], settings['analyzer'], top_n=100000, channel='통합'
            )
            display_common = ''
            if not app_result.empty and rank <= len(app_result):
                display_common = str(app_result.iloc[rank - 1].get('공통핵심어', ''))
            department_rows.append({
                'case_id': case_id,
                **{column: str(record[column]) for column in PRIVATE_ID_COLUMNS},
                'department_rank': rank,
                '학과명': str(target.get(name_column, '')),
                '행정 계열': str(target.get('계열', '')),
                'cosine_similarity_raw': similarity_raw,
                'cosine_similarity_percent': similarity_raw * 100,
                '앱 표시 공통핵심어': display_common,
                'keyword_intersection_ratio': float(metrics.get('jaccard', 0)),
                'student_top_term_count': int(metrics.get('student_count', 0)),
                'department_top_term_count': int(metrics.get('target_count', 0)),
                'common_top_term_count': int(metrics.get('common_count', 0)),
                'gap_analysis_cosine_raw': float(metrics.get('cosine', 0)),
            })
            if rank <= 5 and not common.empty:
                for common_rank, (_, item) in enumerate(common.iterrows(), 1):
                    common_rows.append({
                        'case_id': case_id,
                        **{column: str(record[column]) for column in PRIVATE_ID_COLUMNS},
                        'department_rank': rank,
                        '학과명': str(target.get(name_column, '')),
                        'common_term_rank': common_rank,
                        'common_term': str(item['키워드']),
                        'common_score': float(item['공통점수']),
                        'student_weight_or_frequency': '지원하지 않음',
                        'department_weight_or_frequency': '지원하지 않음',
                    })
            for common_rank, (_, item) in enumerate(common.head(30).iterrows(), 1):
                comparison_rows.append({
                    'case_id': case_id,
                    **{column: str(record[column]) for column in PRIVATE_ID_COLUMNS},
                    'department_rank': rank,
                    '학과명': str(target.get(name_column, '')),
                    'comparison_type': '공통 어휘',
                    'comparison_rank': common_rank,
                    'term': str(item['키워드']),
                    'common_score': float(item['공통점수']),
                    'student_tfidf': '지원하지 않음',
                    'department_tfidf': '지원하지 않음',
                    'gap_score': '',
                })
            for gap_rank, (_, item) in enumerate(gap.head(30).iterrows(), 1):
                comparison_rows.append({
                    'case_id': case_id,
                    **{column: str(record[column]) for column in PRIVATE_ID_COLUMNS},
                    'department_rank': rank,
                    '학과명': str(target.get(name_column, '')),
                    'comparison_type': '보완 키워드',
                    'comparison_rank': gap_rank,
                    'term': str(item['키워드']),
                    'common_score': '',
                    'student_tfidf': float(item['학생부_TFIDF']),
                    'department_tfidf': float(item['목표학과_TFIDF']),
                    'gap_score': float(item['부족도']),
                })
        app_pairs = [
            (str(row['학과명']), float(row['유사도'])) for _, row in app_result.iterrows()
        ]
        app_checks[case_id] = {
            'direct_top10': direct_pairs,
            'app_top10': app_pairs,
            'matches': direct_pairs == app_pairs,
        }
    return (
        ordered(pd.DataFrame(department_rows), ['department_rank']),
        ordered(pd.DataFrame(common_rows), ['department_rank', 'common_term_rank']),
        ordered(pd.DataFrame(comparison_rows), ['department_rank', 'comparison_type', 'comparison_rank']),
        app_checks,
    )


def collect_rare_entities(
    evidence: pd.DataFrame, protected_terms: set[str]
) -> set[str]:
    kiwi = app.get_kiwi()
    if kiwi is None:
        return set()
    counts: Counter[str] = Counter()
    for text in evidence['원문'].drop_duplicates().astype(str):
        counts.update(
            token.form.strip()
            for token in kiwi.tokenize(text)
            if token.tag == 'NNP' and len(token.form.strip()) >= 2
        )
    return {
        term for term, count in counts.items()
        if count <= 2 and term not in protected_terms
    }


def anonymized_copy(
    frame: pd.DataFrame,
    sensitive_terms: set[str],
    student_names: set[str],
    rare_entities: set[str],
) -> pd.DataFrame:
    result = frame.drop(columns=private_columns(frame)).copy()
    term_columns = [
        column for column in ['feature_term', 'term', 'common_term'] if column in result.columns
    ]
    for column in term_columns:
        result[column] = result[column].map(lambda value: anonymize_term(value, sensitive_terms))
    if '원문' in result.columns:
        result['원문'] = result['원문'].map(
            lambda value: anonymize_text(value, student_names, rare_entities)
        )
    if '해당 문장에 연결된 전체 키워드목록' in result.columns:
        result['해당 문장에 연결된 전체 키워드목록'] = result[
            '해당 문장에 연결된 전체 키워드목록'
        ].map(lambda value: ', '.join(
            anonymize_term(term, sensitive_terms) for term in split_keywords(value)
        ))
    if '앱 표시 공통핵심어' in result.columns:
        result['앱 표시 공통핵심어'] = result['앱 표시 공통핵심어'].map(
            lambda value: ', '.join(
                anonymize_term(term, sensitive_terms) for term in split_keywords(value)
            )
        )
    return result


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame[columns] if columns else frame
    if selected.empty:
        return '_자료 없음_'
    safe = selected.copy().fillna('')
    for column in safe.columns:
        safe[column] = safe[column].astype(str).str.replace('|', '\\|', regex=False).str.replace('\n', ' ', regex=False)
    header = '| ' + ' | '.join(map(str, safe.columns)) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(safe.columns)) + ' |'
    rows = [
        '| ' + ' | '.join(str(value) for value in row) + ' |'
        for row in safe.itertuples(index=False, name=None)
    ]
    return '\n'.join([header, divider, *rows])


def build_sourcebook(
    overview: pd.DataFrame,
    features: pd.DataFrame,
    frequency: pd.DataFrame,
    evidence: pd.DataFrame,
    departments: pd.DataFrame,
    common: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    parts = [
        '# 학생부 탐색기 사례 분석 자료집 (PRIVATE)',
        '',
        '> 이 문서는 기명 로컬 전용 자료입니다. TF-IDF 표는 학생부 전체 어휘 분포가 아니라 캐시에 저장된 상위 30개 특징어 가중치 분포입니다.',
        '> 유사도 백분율은 코사인 유사도 원값에 100을 곱한 표시값입니다. 진로 판정이나 추천 확률이 아닙니다.',
        '',
    ]
    for case_id, role, condition, reserve_for in CASES:
        base = overview[overview['case_id'] == case_id].iloc[0]
        case_features = features[features['case_id'] == case_id]
        case_frequency = frequency[frequency['case_id'] == case_id]
        case_evidence = evidence[evidence['case_id'] == case_id]
        case_departments = departments[departments['case_id'] == case_id]
        case_common = common[common['case_id'] == case_id]
        case_comparison = comparison[comparison['case_id'] == case_id]
        focus = FOCUS_TERMS.get(case_id, case_features.head(10)['feature_term'].tolist())
        focus_rows = []
        for term in focus:
            matched = case_features[case_features['feature_term'] == term]
            if matched.empty:
                focus_rows.append({'확인 어휘': term, 'TF-IDF 상위 30 포함': '아니오', '근거 문장 수': 0, '출현 영역 수': 0})
            else:
                item = matched.iloc[0]
                focus_rows.append({'확인 어휘': term, 'TF-IDF 상위 30 포함': '예', '근거 문장 수': int(item['total_evidence_sentence_count']), '출현 영역 수': int(item['appearing_section_count'])})
        facts = []
        for item in focus_rows:
            if item['근거 문장 수'] == 1:
                facts.append(f"{item['확인 어휘']}: 근거가 한 문장에 연결됨")
            elif item['출현 영역 수'] >= 2:
                facts.append(f"{item['확인 어휘']}: {item['출현 영역 수']}개 영역에서 확인됨")
            elif item['근거 문장 수'] == 0:
                facts.append(f"{item['확인 어휘']}: 상위 30개 특징어 근거에서 확인되지 않음")
        parts.extend([
            f"# {case_id} · {base['성명']}", '',
            '## 1. 기본 정보와 선정 근거', '',
            markdown_table(pd.DataFrame([base])[['case_id', *PRIVATE_ID_COLUMNS, '후보 구분', '분석 조건', '대응 주후보', 'candidate_type', 'matched_conditions']]), '',
            '## 2. 기록량과 전체 분포상 위치', '',
            markdown_table(pd.DataFrame([base])[[
                'total_token_count', 'total_token_count_percentile', 'total_token_count_quartile',
                'evidence_sentence_count', 'evidence_sentence_count_percentile', 'evidence_sentence_count_quartile',
                'cross_section_feature_count', 'cross_section_feature_count_percentile', 'cross_section_feature_count_quartile',
                'department_similarity_rank1', 'department_similarity_rank1_percentile', 'department_similarity_rank1_quartile',
                'department_similarity_rank1_rank5_gap', 'department_similarity_rank1_rank5_gap_percentile', 'department_similarity_rank1_rank5_gap_quartile',
            ]]), '',
            '## 3. 상위 30개 TF-IDF 특징어', '',
            markdown_table(case_features, ['feature_rank', 'feature_term', 'tfidf_value']), '',
            '## 4. 빈도 상위 50개 단어', '',
            markdown_table(case_frequency, ['analysis_scope', 'frequency_rank', 'term', 'frequency']), '',
            '## 5. 특징어별 영역 분포', '',
            markdown_table(case_features, [
                'feature_rank', 'feature_term', 'total_evidence_sentence_count',
                'creative_activity_evidence_count', 'subject_note_evidence_count',
                'behavior_comment_evidence_count', 'appearing_section_count',
            ]), '',
            '## 6. 특징어별 근거 문장', '',
            markdown_table(case_evidence, [
                'feature_rank', 'feature_term', 'sentence_id', '출처 영역', '문장번호',
                '원문', '해당 문장에 연결된 전체 키워드목록',
            ]), '',
            '## 7. 학과 유사도 상위 10개', '',
            markdown_table(case_departments, [
                'department_rank', '학과명', '행정 계열', 'cosine_similarity_raw',
                'cosine_similarity_percent', 'keyword_intersection_ratio', '앱 표시 공통핵심어',
            ]), '',
            '## 8. 상위 학과별 공통 어휘', '',
            markdown_table(case_common, [
                'department_rank', '학과명', 'common_term_rank', 'common_term',
                'common_score', 'student_weight_or_frequency', 'department_weight_or_frequency',
            ]), '',
            '## 9. 보완 키워드 또는 비교 키워드', '',
            markdown_table(case_comparison, [
                'department_rank', '학과명', 'comparison_type', 'comparison_rank', 'term',
                'common_score', 'student_tfidf', 'department_tfidf', 'gap_score',
            ]), '',
            '## 10. 데이터 품질 및 해석상 확인 사항', '',
            markdown_table(pd.DataFrame(focus_rows)), '',
            *([f'- {fact}' for fact in facts] or ['- 별도 기계적 주의 사항 없음']),
            '- 공통 어휘의 학생 측·학과 측 개별 가중치는 현재 앱이 제공하지 않아 `지원하지 않음`으로 기록함.',
            '- 앱 화면의 공통핵심어는 집합에서 최대 12개를 가져오므로 표시 순서가 고정되지 않을 수 있음.',
            '',
        ])
    return '\n'.join(parts)


def build_handoff(
    overview: pd.DataFrame,
    features: pd.DataFrame,
    frequency: pd.DataFrame,
    evidence: pd.DataFrame,
    departments: pd.DataFrame,
    common: pd.DataFrame,
    comparison: pd.DataFrame,
    generated_files: list[str],
) -> str:
    parts = [
        '# ChatGPT 사례 분석 인계 자료', '',
        '이 문서는 익명 사례 ID만 사용한다. TF-IDF 표는 전체 어휘 분포가 아니라 캐시에 저장된 상위 30개 특징어 가중치 분포다.',
        '학과 유사도 백분율은 코사인 유사도 원값에 100을 곱한 표시값이며 진로 판정 또는 추천 확률이 아니다.',
        '', '## 사례 구성', '',
        markdown_table(overview, ['case_id', '후보 구분', '분석 조건', '대응 주후보']), '',
        '## 주후보-예비후보 비교', '',
        markdown_table(overview, [
            'case_id', '후보 구분', '대응 주후보', 'total_token_count',
            'evidence_sentence_count', 'cross_section_feature_count',
            'department_similarity_rank1', 'department_similarity_rank1_rank5_gap',
        ]), '',
    ]
    for case_id, role, condition, reserve_for in CASES:
        parts.extend([
            f'## {case_id} · {role}', '',
            f'- 분석 조건: {condition}',
            f'- 대응 주후보: {reserve_for or "해당 없음"}', '',
            '### 정량 선정 근거', '',
            markdown_table(overview[overview['case_id'] == case_id], [
                'total_token_count', 'total_token_count_percentile', 'total_token_count_quartile',
                'evidence_sentence_count', 'evidence_sentence_count_percentile', 'evidence_sentence_count_quartile',
                'cross_section_feature_count', 'cross_section_feature_count_percentile', 'cross_section_feature_count_quartile',
                'department_similarity_rank1', 'department_similarity_rank1_percentile', 'department_similarity_rank1_quartile',
                'department_similarity_rank1_rank5_gap', 'department_similarity_rank1_rank5_gap_percentile', 'department_similarity_rank1_rank5_gap_quartile',
            ]), '',
            '### 상위 30개 TF-IDF 특징어와 영역 출현', '',
            markdown_table(features[features['case_id'] == case_id], [
                'feature_rank', 'feature_term', 'tfidf_value', 'total_evidence_sentence_count',
                'creative_activity_evidence_count', 'subject_note_evidence_count',
                'behavior_comment_evidence_count', 'appearing_section_count',
            ]), '',
            '### 빈도 상위 50개 단어', '',
            markdown_table(frequency[frequency['case_id'] == case_id], ['frequency_rank', 'term', 'frequency', 'analysis_scope']), '',
            '### 상위 10개 특징어의 익명화 근거 문장', '',
            markdown_table(evidence[(evidence['case_id'] == case_id) & (evidence['feature_rank'] <= 10)], [
                'feature_rank', 'feature_term', 'sentence_id', '출처 영역', '문장번호', '원문',
            ]), '',
            '### 학과 유사도 상위 10개', '',
            markdown_table(departments[departments['case_id'] == case_id], [
                'department_rank', '학과명', '행정 계열', 'cosine_similarity_raw',
                'cosine_similarity_percent', 'keyword_intersection_ratio',
            ]), '',
            '### 상위 5개 학과별 공통 어휘', '',
            markdown_table(common[common['case_id'] == case_id], [
                'department_rank', '학과명', 'common_term_rank', 'common_term', 'common_score',
            ]), '',
            '### 보완 키워드 또는 비교 키워드', '',
            markdown_table(comparison[comparison['case_id'] == case_id], [
                'department_rank', '학과명', 'comparison_type', 'comparison_rank', 'term',
                'common_score', 'student_tfidf', 'department_tfidf', 'gap_score',
            ]), '',
        ])
    parts.extend([
        '## 계산 또는 추출하지 못한 항목', '',
        '- 공통 어휘별 학생 측·학과 측 개별 가중치 또는 빈도: 현재 앱 `gap_analysis`가 공통점수만 제공하므로 `지원하지 않음`으로 표시했다.',
        '- 현재 앱은 TF-IDF 특징어와 단어 빈도 가중치를 사용자가 선택하며 기본값은 TF-IDF다. 두 방식의 재현 JSON을 모두 생성했다.',
        '- 앱의 `공통핵심어` 표시 순서는 집합에서 추출되어 실행 간 순서가 고정되지 않을 수 있다. 분석 표는 `gap_analysis`의 공통점수 정렬 결과를 사용했다.', '',
        '## 생성 파일 목록', '',
        *[f'- `{name}`' for name in generated_files], '',
    ])
    return '\n'.join(parts)


def build_figure_inventory() -> str:
    parts = [
        '# 논문 그림 자료 목록', '',
        '> 이미지는 생성하지 않았다. 현재 앱은 TF-IDF 특징어와 단어 빈도 워드클라우드를 선택할 수 있으며 기본값은 TF-IDF다.',
        '> 두 가중치 방식의 재현 JSON을 모두 제공한다.', '',
    ]
    for case_id in ['S201', 'S050', 'S322', 'S296']:
        parts.extend([
            f'## {case_id}', '',
            f'- 빈도 워드클라우드 데이터: `wordcloud_{case_id}_PRIVATE.json`',
            f'- 현재 앱 워드클라우드 재현 데이터: `wordcloud_app_tfidf_{case_id}_PRIVATE.json`',
            '- TF-IDF 막대그래프 데이터: `case_tfidf_features_PRIVATE.csv`',
            '- 학과 유사도 상위 결과: `case_departments_PRIVATE.csv`',
            '- 특징어 근거 문장: `case_evidence_PRIVATE.csv`',
            '- 학과 공통·보완 어휘: `case_department_common_terms_PRIVATE.csv`, `case_keyword_comparison_PRIVATE.csv`',
            '- 탐색기 화면 위치: 학생 분석 탭 → TF-IDF 특징어/워드클라우드 → 관련 학과 정보 → 공통·보완 키워드', '',
        ])
    return '\n'.join(parts)


def git_ignored(path: Path) -> bool:
    completed = subprocess.run(
        ['git', 'check-ignore', '--quiet', str(path)], cwd=ROOT,
        check=False, capture_output=True, text=True,
    )
    return completed.returncode == 0


def write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> int:
    args = parse_args()
    source_before = {
        'student_db': {'sha256': file_sha256(args.student_db), 'size': args.student_db.stat().st_size},
        'major_db': {'sha256': file_sha256(args.major_db), 'size': args.major_db.stat().st_size},
        'app_py': {'sha256': file_sha256(ROOT / 'app.py'), 'size': (ROOT / 'app.py').stat().st_size},
    }
    cache = load_cache(args.student_db)
    major_df = load_major_corpus(args.major_db)
    mapping, metrics, candidates = read_sampling_inputs(args.sampling_dir)
    mapping_check = validate_mapping(cache, mapping, args.expected_students)
    if not all(value for key, value in mapping_check.items() if key not in {'row_count', 'expected_row_count'}):
        raise ValueError(f'case_id 대응표 검증 실패: {mapping_check}')
    missing_cases = sorted(set(CASE_ORDER) - set(mapping['case_id']))
    if missing_cases:
        raise ValueError(f'대응표에 대상 case_id가 없습니다: {missing_cases}')

    settings = read_analysis_settings(cache)
    overview = build_overview(mapping, metrics, candidates)
    features, evidence, evidence_payload, feature_details = build_features_and_evidence(cache, mapping)
    frequency = build_frequency(cache, mapping)
    records = selected_cache_table(cache['records'], mapping, CASE_ORDER)
    departments, common, comparison, department_checks = build_departments(records, major_df, settings)

    student_names = set(mapping['성명'].astype(str).str.strip())
    sensitive_terms = identify_sensitive_terms(feature_details, student_names)
    protected_terms = set(features['feature_term']) | set(frequency['term']) | set(common['common_term'])
    rare_entities = collect_rare_entities(evidence, protected_terms)
    overview_anon = anonymized_copy(overview, sensitive_terms, student_names, rare_entities)
    features_anon = anonymized_copy(features, sensitive_terms, student_names, rare_entities)
    frequency_anon = anonymized_copy(frequency, sensitive_terms, student_names, rare_entities)
    evidence_anon = anonymized_copy(evidence, sensitive_terms, student_names, rare_entities)
    departments_anon = anonymized_copy(departments, sensitive_terms, student_names, rare_entities)
    common_anon = anonymized_copy(common, sensitive_terms, student_names, rare_entities)
    comparison_anon = anonymized_copy(comparison, sensitive_terms, student_names, rare_entities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_frames = {
        'case_overview_PRIVATE.csv': overview,
        'case_tfidf_features_PRIVATE.csv': features,
        'case_frequency_terms_PRIVATE.csv': frequency,
        'case_evidence_PRIVATE.csv': evidence,
        'case_departments_PRIVATE.csv': departments,
        'case_department_common_terms_PRIVATE.csv': common,
        'case_keyword_comparison_PRIVATE.csv': comparison,
    }
    anonymized_frames = {
        'case_overview_anonymized.csv': overview_anon,
        'case_tfidf_features_anonymized.csv': features_anon,
        'case_frequency_terms_anonymized.csv': frequency_anon,
        'case_evidence_anonymized.csv': evidence_anon,
        'case_departments_anonymized.csv': departments_anon,
        'case_department_common_terms_anonymized.csv': common_anon,
    }
    for name, frame in {**private_frames, **anonymized_frames}.items():
        write_csv(frame, args.output_dir / name)
    write_json(evidence_payload, args.output_dir / 'case_evidence_PRIVATE.json')

    for case_id in CASE_ORDER:
        case_frequency = frequency[frequency['case_id'] == case_id]
        write_json(
            dict(zip(case_frequency['term'], case_frequency['frequency'].astype(int))),
            args.output_dir / f'wordcloud_{case_id}_PRIVATE.json',
        )
        case_features = features[features['case_id'] == case_id]
        write_json(
            dict(zip(case_features['feature_term'], case_features['tfidf_value'].astype(float))),
            args.output_dir / f'wordcloud_app_tfidf_{case_id}_PRIVATE.json',
        )

    sourcebook = build_sourcebook(
        overview, features, frequency, evidence, departments, common, comparison
    )
    (args.output_dir / 'case_analysis_sourcebook_PRIVATE.md').write_text(sourcebook, encoding='utf-8')
    (args.output_dir / 'figure_data_inventory.md').write_text(build_figure_inventory(), encoding='utf-8')

    generated_before_handoff = sorted({
        *(path.name for path in args.output_dir.iterdir() if path.is_file()),
        'chatgpt_case_analysis_handoff.md',
        'validation_report.json',
        'README_PRIVATE.md',
    })
    handoff = build_handoff(
        overview_anon, features_anon, frequency_anon, evidence_anon,
        departments_anon, common_anon, comparison_anon, generated_before_handoff,
    )
    (args.output_dir / 'chatgpt_case_analysis_handoff.md').write_text(handoff, encoding='utf-8')

    cache_tfidf = selected_cache_table(cache['tfidf'], mapping, CASE_ORDER)
    cache_tfidf = cache_tfidf[pd.to_numeric(cache_tfidf['순위']) <= 30].copy()
    expected_tfidf = ordered(cache_tfidf.rename(columns={'순위': 'feature_rank', '단어': 'feature_term', 'TF-IDF': 'tfidf_value'}), ['feature_rank'])
    tfidf_match = features[['case_id', 'feature_rank', 'feature_term', 'tfidf_value']].astype(str).equals(
        expected_tfidf[['case_id', 'feature_rank', 'feature_term', 'tfidf_value']].astype(str)
    )
    cache_frequency = selected_cache_table(cache['freq'], mapping, CASE_ORDER)
    cache_frequency = cache_frequency[pd.to_numeric(cache_frequency['순위']) <= 50].copy()
    expected_frequency = ordered(cache_frequency.rename(columns={'순위': 'frequency_rank', '단어': 'term', '빈도': 'frequency'}), ['frequency_rank'])
    frequency_match = frequency[['case_id', 'frequency_rank', 'term', 'frequency']].astype(str).equals(
        expected_frequency[['case_id', 'frequency_rank', 'term', 'frequency']].astype(str)
    )
    evidence_count_match = bool((
        features['total_evidence_sentence_count']
        == features[['creative_activity_evidence_count', 'subject_note_evidence_count', 'behavior_comment_evidence_count']].sum(axis=1)
    ).all())
    source_after = {
        'student_db': {'sha256': file_sha256(args.student_db), 'size': args.student_db.stat().st_size},
        'major_db': {'sha256': file_sha256(args.major_db), 'size': args.major_db.stat().st_size},
        'app_py': {'sha256': file_sha256(ROOT / 'app.py'), 'size': (ROOT / 'app.py').stat().st_size},
    }
    anonymous_paths = [args.output_dir / name for name in anonymized_frames] + [
        args.output_dir / 'chatgpt_case_analysis_handoff.md'
    ]
    anonymous_text = '\n'.join(path.read_text(encoding='utf-8-sig') for path in anonymous_paths)
    leaked_names = sorted(name for name in student_names if name and name in anonymous_text)
    forbidden_columns = {
        path.name: [column for column in PRIVATE_ID_COLUMNS if column in pd.read_csv(path, encoding='utf-8-sig', nrows=0).columns]
        for path in anonymous_paths if path.suffix == '.csv'
    }
    validation = {
        'mapping': mapping_check,
        'selected_case_count': len(overview),
        'selected_case_ids_in_requested_order': overview['case_id'].tolist() == list(CASE_ORDER),
        'tfidf_top30_matches_cache': tfidf_match,
        'frequency_top50_matches_cache': frequency_match,
        'frequency_scope_values': sorted(frequency['analysis_scope'].astype(str).unique().tolist()),
        'frequency_tie_rule': '캐시의 순위를 그대로 사용함. 캐시 생성 시 Counter.most_common의 빈도 내림차순 및 최초 출현 순서가 적용됨.',
        'frequency_top50_matches_current_app_frequency_wordcloud_input': True,
        'frequency_top50_matches_current_app_default_wordcloud_input': False,
        'wordcloud_mode_note': '현재 app.py의 기본값은 TF-IDF 특징어이며 사용자가 단어 빈도 방식을 선택하면 캐시의 빈도 상위 50개를 사용함.',
        'actual_app_wordcloud_tfidf_jsons_created': True,
        'feature_section_counts_sum_to_total': evidence_count_match,
        'feature_evidence_rows_derived_from_cache_evidence_table': True,
        'tfidf_rows_per_case': features.groupby('case_id').size().astype(int).to_dict(),
        'frequency_rows_per_case': frequency.groupby('case_id').size().astype(int).to_dict(),
        'department_rows_per_case': departments.groupby('case_id').size().astype(int).to_dict(),
        'department_top10_matches_app_similarity': all(item['matches'] for item in department_checks.values()),
        'department_app_checks': department_checks,
        'common_and_gap_use_app_gap_analysis': True,
        'common_individual_weights_supported': False,
        'output_directory_git_ignored': git_ignored(args.output_dir / 'case_overview_PRIVATE.csv'),
        'anonymized_forbidden_identity_columns': forbidden_columns,
        'anonymized_known_student_name_leaks': leaked_names,
        'anonymized_name_leak_free': not leaked_names,
        'source_files_unchanged': source_before == source_after,
        'source_hashes_before': source_before,
        'source_hashes_after': source_after,
        'notes': [
            '근거 문장 수는 각 특징어가 evidence 키워드목록에 정확히 포함된 행 수임.',
            '공통 어휘 전체 표는 상위 5개 학과에 대해 app.gap_analysis의 공통점수 정렬 결과를 저장함.',
            '익명화는 전체 학생 성명, 역할이 붙은 인명, 기관명 패턴 및 보호 어휘가 아닌 희귀 고유명사를 치환함.',
        ],
    }
    write_json(validation, args.output_dir / 'validation_report.json')
    readme = '\n'.join([
        '# 사례 분석 산출물 안내 (PRIVATE)', '',
        '- 모든 파일은 로컬 연구용이며 Git 추적에서 제외된다.',
        '- `*_PRIVATE.*`에는 학생 식별 정보 또는 원문이 포함될 수 있다.',
        '- `*_anonymized.*`와 `chatgpt_case_analysis_handoff.md`만 외부 대화 전달용이다.',
        '- 현재 앱 워드클라우드 기본값은 TF-IDF이며 사용자가 단어 빈도를 선택할 수 있다. 두 방식의 JSON을 제공한다.',
        '- 상세 검증 결과는 `validation_report.json`을 확인한다.', '',
    ])
    (args.output_dir / 'README_PRIVATE.md').write_text(readme, encoding='utf-8')
    print(json.dumps({
        'output_dir': str(args.output_dir),
        'case_count': len(overview),
        'files': sorted(path.name for path in args.output_dir.iterdir() if path.is_file()),
        'validation_summary': {
            'mapping': mapping_check['exact_cache_order_and_identity_match'],
            'tfidf': tfidf_match,
            'frequency_cache': frequency_match,
            'department': all(item['matches'] for item in department_checks.values()),
            'anonymous': not leaked_names and not any(forbidden_columns.values()),
            'git_ignored': validation['output_directory_git_ignored'],
            'sources_unchanged': validation['source_files_unchanged'],
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
