# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from contextlib import closing
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from careernet_corpus import enrich_existing_corpus, validate_corpus_schema


ID_COLUMNS = ['학년', '반', '번호', '성명']
SECTION_COLUMNS = ['창체', '교과세특', '행발']
REQUIRED_CACHE_TABLES = ['records', 'tfidf', 'freq', 'evidence', 'meta']
PRIVATE_FILES = {
    'case_id_mapping_PRIVATE.csv',
    'case_sampling_metrics_PRIVATE.csv',
    'candidate_cases_PRIVATE.csv',
    'candidate_evidence_PRIVATE.json',
}
BASE_METRICS = [
    'total_token_count',
    'creative_activity_token_count',
    'subject_note_token_count',
    'behavior_comment_token_count',
    'evidence_sentence_count',
    'tfidf_top1',
    'tfidf_top5',
    'tfidf_top1_top5_gap',
    'tfidf_top5_sum',
    'tfidf_top30_sum',
    'tfidf_top5_share',
    'tfidf_top30_normalized_entropy',
    'feature_terms_with_evidence',
    'cross_section_feature_count',
    'three_section_feature_count',
    'mean_sections_per_feature',
    'top_feature_evidence_section_count',
    'top_feature_evidence_sentence_count',
    'department_similarity_rank1',
    'department_similarity_rank2',
    'department_similarity_rank5',
    'department_similarity_rank10',
    'department_similarity_rank1_rank2_gap',
    'department_similarity_rank1_rank5_gap',
    'department_similarity_top5_mean',
    'department_similarity_top10_normalized_entropy',
    'top10_administrative_category_count',
    'available_feature_count',
    'available_department_count',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='학생부 탐색기 사례 표집용 정량 지표와 익명 후보군을 생성합니다.'
    )
    parser.add_argument('--student-db', type=Path, required=True)
    parser.add_argument('--major-db', type=Path, default=ROOT / 'data' / 'major_corpus.db')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT / 'local_outputs' / 'case_sampling',
    )
    parser.add_argument('--expected-students', type=int, default=376)
    parser.add_argument('--max-candidates', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def sqlite_uri(path: Path) -> str:
    return f'file:{path.resolve().as_posix()}?mode=ro'


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def load_cache(path: Path) -> dict[str, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f'학생 캐시 DB를 찾을 수 없습니다: {path}')
    result: dict[str, pd.DataFrame] = {}
    with closing(sqlite3.connect(sqlite_uri(path), uri=True)) as connection:
        available = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        missing = [table for table in REQUIRED_CACHE_TABLES if table not in available]
        if missing:
            raise ValueError(f'학생 캐시 DB에 필요한 테이블이 없습니다: {", ".join(missing)}')
        for table in [*REQUIRED_CACHE_TABLES, 'tfidf_metrics', 'student_metrics']:
            if table not in available:
                continue
            result[table] = pd.read_sql_query(
                f'SELECT * FROM {quote_identifier(table)}', connection
            ).fillna('')
    for table in ['records', 'tfidf', 'freq', 'evidence', 'tfidf_metrics', 'student_metrics']:
        if table not in result:
            continue
        missing_columns = [column for column in ID_COLUMNS if column not in result[table].columns]
        if missing_columns:
            raise ValueError(
                f'{table} 테이블에 학생 식별 열이 없습니다: {", ".join(missing_columns)}'
            )
        for column in ID_COLUMNS:
            result[table][column] = result[table][column].astype(str).str.strip()
    return result


def load_major_corpus(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'학과 말뭉치 DB를 찾을 수 없습니다: {path}')
    with closing(sqlite3.connect(sqlite_uri(path), uri=True)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if 'majors' not in tables:
            raise ValueError('학과 말뭉치 DB에 majors 테이블이 없습니다.')
        majors = pd.read_sql_query('SELECT * FROM majors', connection).fillna('')
    validate_corpus_schema(majors, require_integrated=True, require_raw=True)
    return majors


def natural_key(value: Any) -> tuple[int, Any]:
    text = str(value or '').strip()
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def student_key(row: pd.Series | dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(column, '')).strip() for column in ID_COLUMNS)


def sort_records(records: pd.DataFrame) -> pd.DataFrame:
    result = records.copy()
    result['_grade_sort'] = result['학년'].map(natural_key)
    result['_class_sort'] = result['반'].map(natural_key)
    result['_number_sort'] = result['번호'].map(natural_key)
    result = result.sort_values(
        ['_grade_sort', '_class_sort', '_number_sort', '성명'], kind='stable'
    )
    return result.drop(columns=['_grade_sort', '_class_sort', '_number_sort']).reset_index(drop=True)


def validate_input_identity(cache: dict[str, pd.DataFrame], expected_students: int) -> dict[str, Any]:
    records = cache['records']
    duplicate_count = int(records.duplicated(ID_COLUMNS, keep=False).sum())
    if duplicate_count:
        raise ValueError(f'records 테이블에 중복 학생 식별 행이 {duplicate_count}개 있습니다.')
    record_ids = set(map(tuple, records[ID_COLUMNS].astype(str).to_numpy()))
    coverage: dict[str, Any] = {}
    for table in ['records', 'tfidf', 'freq', 'evidence']:
        frame = cache[table]
        identities = set(map(tuple, frame[ID_COLUMNS].astype(str).drop_duplicates().to_numpy()))
        coverage[table] = {
            'students': len(identities),
            'missing_vs_records': len(record_ids - identities),
            'extra_vs_records': len(identities - record_ids),
        }
    return {
        'actual_students': len(records),
        'expected_students': expected_students,
        'student_count_matches_expected': len(records) == expected_students,
        'duplicate_identity_rows': duplicate_count,
        'table_identity_coverage': coverage,
    }


def make_case_mapping(records: pd.DataFrame) -> pd.DataFrame:
    mapping = records[ID_COLUMNS].copy()
    mapping.insert(0, 'case_id', [f'S{index:03d}' for index in range(1, len(mapping) + 1)])
    return mapping


def normalized_entropy(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array > 0)]
    if len(array) <= 1:
        return 0.0 if len(array) == 1 else math.nan
    probabilities = array / array.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(len(probabilities)))


def percentile_quartile(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors='coerce')
    percentile = numeric.rank(method='average', pct=True) * 100
    quartile = pd.Series(pd.NA, index=series.index, dtype='string')
    quartile.loc[percentile <= 25] = 'Q1'
    quartile.loc[(percentile > 25) & (percentile <= 50)] = 'Q2'
    quartile.loc[(percentile > 50) & (percentile <= 75)] = 'Q3'
    quartile.loc[percentile > 75] = 'Q4'
    return percentile, quartile


def metric_distribution(metrics: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metric_columns:
        values = pd.to_numeric(metrics[metric], errors='coerce')
        valid = values.dropna()
        rows.append({
            'metric': metric,
            'count': int(valid.count()),
            'missing': int(values.isna().sum()),
            'mean': float(valid.mean()) if not valid.empty else math.nan,
            'standard_deviation': float(valid.std(ddof=1)) if len(valid) > 1 else math.nan,
            'minimum': float(valid.min()) if not valid.empty else math.nan,
            'Q1': float(valid.quantile(0.25)) if not valid.empty else math.nan,
            'median': float(valid.quantile(0.50)) if not valid.empty else math.nan,
            'Q3': float(valid.quantile(0.75)) if not valid.empty else math.nan,
            'maximum': float(valid.max()) if not valid.empty else math.nan,
        })
    return pd.DataFrame(rows)


def read_analysis_settings(cache: dict[str, pd.DataFrame]) -> dict[str, Any]:
    meta = cache['meta'].iloc[0].to_dict() if not cache['meta'].empty else {}
    analyzer = app.analyzer_name(meta.get('형태소분석기', 'Kiwi'))
    min_len = int(pd.to_numeric(meta.get('최소단어길이', 2), errors='coerce') or 2)
    stopwords = app.read_stopwords()
    synonyms = app.read_synonyms()
    if not app.analyzer_available(analyzer):
        reason = app.analyzer_unavailable_reason(analyzer)
        raise RuntimeError(f'{analyzer} 형태소 분석기를 사용할 수 없습니다: {reason}')
    dictionary_status = '미확인'
    expected_dictionary = app.analysis_dictionary_metadata(stopwords, synonyms, min_len, analyzer)
    dictionary_keys = ['실효불용어SHA256', '실효표현통일규칙SHA256', '최소단어길이']
    if all(key in meta for key in dictionary_keys):
        dictionary_status = '일치' if all(
            str(meta.get(key, '')).strip() == str(expected_dictionary[key]).strip()
            for key in dictionary_keys
        ) else '불일치'
        if dictionary_status == '불일치':
            print('[warning] 현재 분석 사전과 학생 캐시 생성 당시 설정이 다릅니다. 재전처리를 권장합니다.', flush=True)
    return {
        'scope': str(meta.get('분석범위', '통합') or '통합'),
        'analyzer': analyzer,
        'min_len': min_len,
        'stopwords': stopwords,
        'synonyms': synonyms,
        'cache_created_at': str(meta.get('생성시각', '')),
        'dictionary_status': dictionary_status,
    }


def tokenize_records(
    records: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, list[str]]]]:
    rows: list[dict[str, Any]] = []
    token_documents: list[str] = []
    section_tokens: dict[str, dict[str, list[str]]] = {}
    for index, row in records.iterrows():
        case_id = str(row['case_id'])
        per_section: dict[str, list[str]] = {}
        for section in SECTION_COLUMNS:
            per_section[section] = app.tokenize(
                str(row.get(section, '')),
                settings['stopwords'],
                settings['synonyms'],
                settings['min_len'],
                settings['analyzer'],
            )
        combined_tokens = app.tokenize(
            str(row.get(settings['scope'], row.get('통합', ''))),
            settings['stopwords'],
            settings['synonyms'],
            settings['min_len'],
            settings['analyzer'],
        )
        section_tokens[case_id] = per_section
        token_documents.append(' '.join(combined_tokens))
        rows.append({
            'case_id': case_id,
            'total_token_count': len(combined_tokens),
            'creative_activity_token_count': len(per_section['창체']),
            'subject_note_token_count': len(per_section['교과세특']),
            'behavior_comment_token_count': len(per_section['행발']),
        })
        if (index + 1) % 50 == 0 or index + 1 == len(records):
            print(f'[tokenize] {index + 1}/{len(records)} students', flush=True)
    return pd.DataFrame(rows), token_documents, section_tokens


def prepare_tfidf_metrics(
    cache: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    tfidf = cache['tfidf'].merge(mapping, on=ID_COLUMNS, how='left', validate='many_to_one')
    tfidf['순위'] = pd.to_numeric(tfidf['순위'], errors='coerce')
    tfidf['TF-IDF'] = pd.to_numeric(tfidf['TF-IDF'], errors='coerce')
    tfidf = tfidf.sort_values(['case_id', '순위', '단어'], kind='stable')
    term_details: dict[str, list[dict[str, Any]]] = {}
    for case_id, group in tfidf.groupby('case_id', sort=False):
        group = group.dropna(subset=['TF-IDF']).head(30)
        term_details[str(case_id)] = [
            {'rank': int(row['순위']), 'term': str(row['단어']), 'value': float(row['TF-IDF'])}
            for _, row in group.iterrows()
        ]
    full = cache.get('tfidf_metrics')
    if isinstance(full, pd.DataFrame) and not full.empty:
        required = {
            'positive_tfidf_feature_count', 'tfidf_max_full', 'tfidf_rank5_full',
            'tfidf_top1_rank5_gap_full', 'tfidf_top5_sum_full', 'tfidf_top30_sum_full',
            'tfidf_top5_share_full', 'tfidf_normalized_entropy_full',
        }
        missing = sorted(required - set(full.columns))
        if missing:
            raise ValueError(
                'tfidf_metrics 테이블에 전체 벡터 지표가 없습니다: ' + ', '.join(missing)
            )
        result = full.merge(mapping, on=ID_COLUMNS, how='left', validate='one_to_one')
        for column in required:
            result[column] = pd.to_numeric(result[column], errors='coerce')
        result = result.rename(columns={
            'positive_tfidf_feature_count': 'available_feature_count',
            'tfidf_max_full': 'tfidf_top1',
            'tfidf_rank5_full': 'tfidf_top5',
            'tfidf_top1_rank5_gap_full': 'tfidf_top1_top5_gap',
            'tfidf_top5_sum_full': 'tfidf_top5_sum',
            'tfidf_top30_sum_full': 'tfidf_top30_sum',
            'tfidf_top5_share_full': 'tfidf_top5_share',
            'tfidf_normalized_entropy_full': 'tfidf_top30_normalized_entropy',
        })
        result['tfidf_metric_basis'] = 'full_positive_vector'
        columns = [
            'case_id', 'available_feature_count', 'tfidf_top1', 'tfidf_top5',
            'tfidf_top1_top5_gap', 'tfidf_top5_sum', 'tfidf_top30_sum',
            'tfidf_top5_share', 'tfidf_top30_normalized_entropy', 'tfidf_metric_basis',
        ]
        return result[columns], term_details

    rows: list[dict[str, Any]] = []
    for case_id, details in term_details.items():
        values = [float(item['value']) for item in details]
        top5_values = values[:5]
        top30_sum = float(sum(values))
        top5_sum = float(sum(top5_values))
        top5 = values[4] if len(values) >= 5 else math.nan
        rows.append({
            'case_id': case_id,
            'available_feature_count': len(values),
            'tfidf_top1': values[0] if values else math.nan,
            'tfidf_top5': top5,
            'tfidf_top1_top5_gap': values[0] - top5 if len(values) >= 5 else math.nan,
            'tfidf_top5_sum': top5_sum,
            'tfidf_top30_sum': top30_sum,
            'tfidf_top5_share': top5_sum / top30_sum if top30_sum > 0 else math.nan,
            'tfidf_top30_normalized_entropy': normalized_entropy(values),
            'tfidf_metric_basis': 'stored_top30_legacy',
        })
    return pd.DataFrame(rows), term_details


def prepare_evidence_metrics(
    cache: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    term_details: dict[str, list[dict[str, Any]]],
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    evidence = cache['evidence'].merge(mapping, on=ID_COLUMNS, how='left', validate='many_to_one')
    evidence['문장번호'] = pd.to_numeric(evidence['문장번호'], errors='coerce')
    top_terms = {
        case_id: {detail['term'] for detail in details}
        for case_id, details in term_details.items()
    }
    evidence_index: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {'sources': set(), 'sentence_keys': set(), 'sentences': []})
    )
    sentence_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unique_sentence_keys: dict[str, set[tuple[str, Any]]] = defaultdict(set)
    for _, row in evidence.iterrows():
        case_id = str(row['case_id'])
        source = str(row.get('출처', '')).strip()
        sentence_number = row.get('문장번호')
        sentence_key = (source, sentence_number)
        unique_sentence_keys[case_id].add(sentence_key)
        sentence_record = {
            'source': source,
            'sentence_number': None if pd.isna(sentence_number) else int(sentence_number),
            'text': str(row.get('원문', '')).strip(),
        }
        sentence_rows[case_id].append(sentence_record)
        keyword_field = str(row.get('키워드목록', ''))
        keywords = {value.strip() for value in keyword_field.split(',') if value.strip()}
        for term in keywords & top_terms.get(case_id, set()):
            item = evidence_index[case_id][term]
            item['sources'].add(source)
            item['sentence_keys'].add(sentence_key)
            if len(item['sentences']) < 3:
                item['sentences'].append(sentence_record)
    rows: list[dict[str, Any]] = []
    serializable_index: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id, details in term_details.items():
        term_index = evidence_index.get(case_id, {})
        with_evidence = [term for term in details if term['term'] in term_index]
        section_counts = [len(term_index[item['term']]['sources']) for item in with_evidence]
        top_term = details[0]['term'] if details else ''
        top_item = term_index.get(top_term, {'sources': set(), 'sentence_keys': set()})
        rows.append({
            'case_id': case_id,
            'evidence_sentence_count': len(unique_sentence_keys.get(case_id, set())),
            'feature_terms_with_evidence': len(with_evidence),
            'cross_section_feature_count': sum(count >= 2 for count in section_counts),
            'three_section_feature_count': sum(count == 3 for count in section_counts),
            'mean_sections_per_feature': (
                float(np.mean(section_counts)) if section_counts else math.nan
            ),
            'top_feature_evidence_section_count': len(top_item['sources']),
            'top_feature_evidence_sentence_count': len(top_item['sentence_keys']),
        })
        serializable_index[case_id] = {}
        for term, item in term_index.items():
            serializable_index[case_id][term] = {
                'sources': sorted(item['sources']),
                'sentence_count': len(item['sentence_keys']),
                'sentences': sorted(
                    item['sentences'], key=lambda value: (value['source'], value['sentence_number'] or 0)
                ),
            }
    return pd.DataFrame(rows), serializable_index, sentence_rows


def deterministic_top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    return scores.argsort()[::-1][:count]


def prepare_similarity_metrics(
    records: pd.DataFrame,
    token_documents: list[str],
    majors: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], Any, Any, Any]:
    corpus_documents = tuple(app.corpus_texts(majors, '통합'))
    vectorizer, major_matrix, terms = app.prepare_major_index(
        corpus_documents,
        tuple(sorted(settings['stopwords'])),
        tuple(sorted(settings['synonyms'].items())),
        settings['min_len'],
        settings['analyzer'],
    )
    if vectorizer is None or major_matrix is None:
        raise ValueError('학과 말뭉치 TF-IDF 인덱스를 생성하지 못했습니다.')
    student_matrix = vectorizer.transform(token_documents)
    similarity_matrix = cosine_similarity(student_matrix, major_matrix)
    rows: list[dict[str, Any]] = []
    details: dict[str, list[dict[str, Any]]] = {}
    for position, (_, row) in enumerate(records.iterrows()):
        case_id = str(row['case_id'])
        scores = similarity_matrix[position]
        top_indices = deterministic_top_indices(scores, 10)
        top_scores = [float(scores[index]) for index in top_indices]
        detail_rows = []
        for rank, index in enumerate(top_indices, start=1):
            detail_rows.append({
                'rank': rank,
                'major_index': int(index),
                'major_seq': str(majors.iloc[index].get('majorSeq', '')),
                'category': str(majors.iloc[index].get('계열', '')),
                'department': str(majors.iloc[index].get('학과명', '')),
                'similarity': float(scores[index]),
            })
        details[case_id] = detail_rows
        rows.append({
            'case_id': case_id,
            'available_department_count': int(np.isfinite(scores).sum()),
            'department_similarity_rank1': top_scores[0] if len(top_scores) >= 1 else math.nan,
            'department_similarity_rank2': top_scores[1] if len(top_scores) >= 2 else math.nan,
            'department_similarity_rank5': top_scores[4] if len(top_scores) >= 5 else math.nan,
            'department_similarity_rank10': top_scores[9] if len(top_scores) >= 10 else math.nan,
            'department_similarity_rank1_rank2_gap': (
                top_scores[0] - top_scores[1] if len(top_scores) >= 2 else math.nan
            ),
            'department_similarity_rank1_rank5_gap': (
                top_scores[0] - top_scores[4] if len(top_scores) >= 5 else math.nan
            ),
            'department_similarity_top5_mean': (
                float(np.mean(top_scores[:5])) if len(top_scores) >= 5 else math.nan
            ),
            'department_similarity_top10_normalized_entropy': normalized_entropy(top_scores[:10]),
            'top10_administrative_category_count': len({
                str(majors.iloc[index].get('계열', '')).strip()
                for index in top_indices
                if str(majors.iloc[index].get('계열', '')).strip()
            }),
        })
    return pd.DataFrame(rows), details, vectorizer, major_matrix, student_matrix


def recompute_student_tfidf_validation(
    records: pd.DataFrame,
    token_documents: list[str],
    cache: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    matrix = vectorizer.fit_transform(token_documents)
    terms = vectorizer.get_feature_names_out()
    cached = cache['tfidf'].merge(mapping, on=ID_COLUMNS, how='left', validate='many_to_one')
    cached['TF-IDF'] = pd.to_numeric(cached['TF-IDF'], errors='coerce')
    cached_groups = {case_id: group for case_id, group in cached.groupby('case_id')}
    term_mismatches = 0
    value_mismatches = 0
    checked_rows = 0
    for index, row in records.iterrows():
        case_id = str(row['case_id'])
        scores = matrix[index].toarray().ravel()
        top_indices = deterministic_top_indices(scores, 30)
        recomputed = {
            str(terms[term_index]): round(float(scores[term_index]), 4)
            for term_index in top_indices
            if scores[term_index] > 0
        }
        group = cached_groups.get(case_id, pd.DataFrame())
        cached_values = {
            str(item['단어']): round(float(item['TF-IDF']), 4)
            for _, item in group.iterrows()
            if pd.notna(item['TF-IDF'])
        }
        term_mismatches += int(set(recomputed) != set(cached_values))
        for term in set(recomputed) & set(cached_values):
            checked_rows += 1
            value_mismatches += int(recomputed[term] != cached_values[term])
    return {
        'students_checked': len(records),
        'tfidf_term_set_mismatch_students': term_mismatches,
        'tfidf_value_mismatch_rows': value_mismatches,
        'tfidf_value_rows_checked': checked_rows,
    }


def similarity_sample_validation(
    records: pd.DataFrame,
    majors: pd.DataFrame,
    settings: dict[str, Any],
    similarity_details: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    positions = sorted({0, len(records) // 2, len(records) - 1})
    mismatches = 0
    checked = 0
    for position in positions:
        row = records.iloc[position]
        case_id = str(row['case_id'])
        app_result = app.similarity(
            str(row.get(settings['scope'], row.get('통합', ''))),
            majors,
            settings['stopwords'],
            settings['synonyms'],
            settings['min_len'],
            settings['analyzer'],
            10,
            channel='통합',
        )
        ours = similarity_details[case_id]
        for index in range(min(10, len(app_result), len(ours))):
            checked += 1
            same_name = str(app_result.iloc[index]['학과명']) == ours[index]['department']
            same_score = float(app_result.iloc[index]['유사도']) == round(ours[index]['similarity'], 4)
            mismatches += int(not (same_name and same_score))
    return {'sample_students': len(positions), 'rows_checked': checked, 'mismatches': mismatches}


def evidence_sample_validation(
    records: pd.DataFrame,
    cache: dict[str, pd.DataFrame],
    mapping: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, Any]:
    evidence = cache['evidence'].merge(mapping, on=ID_COLUMNS, how='left', validate='many_to_one')
    positions = sorted({0, len(records) // 2, len(records) - 1})
    mismatches = 0
    for position in positions:
        row = records.iloc[position]
        case_id = str(row['case_id'])
        expected = 0
        for section in SECTION_COLUMNS:
            for sentence in app.split_record_sentences(str(row.get(section, ''))):
                if app.tokenize(
                    sentence,
                    settings['stopwords'],
                    settings['synonyms'],
                    settings['min_len'],
                    settings['analyzer'],
                ):
                    expected += 1
        actual = len(
            evidence.loc[evidence['case_id'] == case_id, ['출처', '문장번호']].drop_duplicates()
        )
        mismatches += int(expected != actual)
    return {'sample_students': len(positions), 'mismatches': mismatches}


def attach_quality_flags(metrics: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    record_lookup = records.set_index('case_id')
    token_values = pd.to_numeric(result['total_token_count'], errors='coerce')
    q1 = token_values.quantile(0.25)
    q3 = token_values.quantile(0.75)
    high_outlier = q3 + 1.5 * (q3 - q1)
    notes: list[str] = []
    missing_evidence_flags: list[bool] = []
    missing_similarity_flags: list[bool] = []
    for _, row in result.iterrows():
        case_id = str(row['case_id'])
        record = record_lookup.loc[case_id]
        issues: list[str] = []
        for section in SECTION_COLUMNS:
            if not str(record.get(section, '')).strip():
                issues.append(f'{section} 자료 비어 있음')
        if row['available_feature_count'] < 30:
            issues.append(f'TF-IDF 특징어 {int(row["available_feature_count"])}개')
        missing_evidence = bool(
            row['evidence_sentence_count'] <= 0 or row['feature_terms_with_evidence'] <= 0
        )
        if missing_evidence:
            issues.append('근거 문장 연결 누락')
        similarity_values = [
            row['department_similarity_rank1'],
            row['department_similarity_rank2'],
            row['department_similarity_rank5'],
            row['department_similarity_rank10'],
        ]
        missing_similarity = row['available_department_count'] <= 0 or any(
            pd.isna(value) for value in similarity_values
        )
        if missing_similarity:
            issues.append('학과 유사도 결과 누락')
        if any(np.isinf(pd.to_numeric(value, errors='coerce')) for value in similarity_values):
            issues.append('학과 유사도 무한값')
        if any(float(value) < -1e-12 for value in similarity_values if pd.notna(value)):
            issues.append('학과 유사도 음수')
        if float(row['total_token_count']) > high_outlier:
            issues.append('기록량 상단 IQR 이상치')
        notes.append('; '.join(issues))
        missing_evidence_flags.append(missing_evidence)
        missing_similarity_flags.append(missing_similarity)
    result['missing_evidence_flag'] = missing_evidence_flags
    result['missing_similarity_flag'] = missing_similarity_flags
    result['data_quality_note'] = notes
    return result


def add_relative_positions(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    for metric in BASE_METRICS:
        percentile, quartile = percentile_quartile(result[metric])
        result[f'{metric}_percentile'] = percentile
        result[f'{metric}_quartile'] = quartile
    return result


def quantiles(metrics: pd.DataFrame, metric: str) -> tuple[float, float]:
    values = pd.to_numeric(metrics[metric], errors='coerce')
    return float(values.quantile(0.25)), float(values.quantile(0.75))


def condition(value: bool, label: str) -> tuple[bool, str]:
    return bool(value), label


def build_candidate_rules(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    def low(row: pd.Series, metric: str) -> bool:
        return bool(row[f'{metric}_percentile'] <= 25)

    def high(row: pd.Series, metric: str) -> bool:
        return bool(row[f'{metric}_percentile'] >= 75)

    def concentration(row: pd.Series) -> tuple[bool, str]:
        passed = (
            high(row, 'tfidf_top5_share')
            or low(row, 'tfidf_top30_normalized_entropy')
        )
        return condition(
            passed,
            'tfidf_top5_share 상위 사분위 또는 tfidf_top30_normalized_entropy 하위 사분위',
        )

    def type1(row: pd.Series) -> tuple[bool, list[tuple[bool, str]]]:
        checks = [
            condition(not low(row, 'total_token_count'), 'total_token_count 하위 사분위 아님'),
            condition(high(row, 'cross_section_feature_count'), 'cross_section_feature_count 상위 사분위'),
            concentration(row),
            condition(high(row, 'department_similarity_rank1'), 'department_similarity_rank1 상위 사분위'),
            condition(high(row, 'department_similarity_rank1_rank5_gap'), 'department_similarity_rank1_rank5_gap 상위 사분위'),
        ]
        eligible = checks[0][0] and checks[2][0] and sum(item[0] for item in checks[1:]) >= 3
        return eligible, checks

    def type2(row: pd.Series) -> tuple[bool, list[tuple[bool, str]]]:
        checks = [
            condition(
                high(row, 'tfidf_top1') or high(row, 'tfidf_top5_share'),
                'tfidf_top1 또는 tfidf_top5_share 상위 사분위',
            ),
            condition(low(row, 'department_similarity_rank1_rank5_gap'), 'department_similarity_rank1_rank5_gap 하위 사분위'),
            condition(high(row, 'department_similarity_top10_normalized_entropy'), 'department_similarity_top10_normalized_entropy 상위 사분위'),
        ]
        return all(item[0] for item in checks), checks

    def type3(row: pd.Series) -> tuple[bool, list[tuple[bool, str]]]:
        checks = [
            condition(low(row, 'total_token_count'), 'total_token_count 하위 사분위'),
            condition(low(row, 'evidence_sentence_count'), 'evidence_sentence_count 하위 사분위'),
            condition(low(row, 'feature_terms_with_evidence'), 'feature_terms_with_evidence 하위 사분위'),
        ]
        return any(item[0] for item in checks), checks

    def type4(row: pd.Series) -> tuple[bool, list[tuple[bool, str]]]:
        checks = [
            concentration(row),
            condition(low(row, 'department_similarity_rank1'), 'department_similarity_rank1 하위 사분위'),
        ]
        return all(item[0] for item in checks), checks

    def type5(row: pd.Series) -> tuple[bool, list[tuple[bool, str]]]:
        checks = [
            condition(high(row, 'cross_section_feature_count'), 'cross_section_feature_count 상위 사분위'),
            condition(high(row, 'three_section_feature_count'), 'three_section_feature_count 상위 사분위'),
            condition(high(row, 'total_token_count'), 'total_token_count 상위 사분위 (기록량 효과 확인)'),
        ]
        return checks[0][0] or checks[1][0], checks

    def type6(row: pd.Series) -> tuple[bool, list[tuple[bool, str]]]:
        checks = [
            condition(high(row, 'tfidf_top1'), 'tfidf_top1 상위 사분위'),
            condition(low(row, 'cross_section_feature_count'), 'cross_section_feature_count 하위 사분위'),
            condition(
                row['top_feature_evidence_section_count'] <= 1
                or low(row, 'top_feature_evidence_sentence_count'),
                '최고 특징어 근거가 한 영역 또는 문장 수 하위 사분위',
            ),
        ]
        return all(item[0] for item in checks), checks

    def type7(row: pd.Series) -> tuple[bool, list[tuple[bool, str]]]:
        checks = [
            condition(high(row, 'department_similarity_top5_mean'), 'department_similarity_top5_mean 상위 사분위'),
            condition(low(row, 'department_similarity_rank1_rank5_gap'), 'department_similarity_rank1_rank5_gap 하위 사분위'),
            condition(high(row, 'department_similarity_top10_normalized_entropy'), 'department_similarity_top10_normalized_entropy 상위 사분위'),
        ]
        return all(item[0] for item in checks), checks

    def type8(row: pd.Series) -> tuple[bool, list[tuple[bool, str]]]:
        checks = [condition(bool(str(row['data_quality_note']).strip()), str(row['data_quality_note']))]
        return checks[0][0], checks

    return [
        {'code': '유형 1', 'name': '특징어와 관련 학과 결과가 비교적 집중된 사례', 'core_metric': 'department_similarity_rank1_rank5_gap', 'ascending': False, 'rule': type1},
        {'code': '유형 2', 'name': '특징어는 두드러지지만 학과 결과가 여러 방향으로 분산된 사례', 'core_metric': 'department_similarity_top10_normalized_entropy', 'ascending': False, 'rule': type2},
        {'code': '유형 3', 'name': '기록량이 희소한 사례', 'core_metric': 'total_token_count', 'ascending': True, 'rule': type3},
        {'code': '유형 4', 'name': '특징어는 두드러지지만 학과 말뭉치 유사도는 상대적으로 낮은 사례', 'core_metric': 'department_similarity_rank1', 'ascending': True, 'rule': type4},
        {'code': '유형 5', 'name': '여러 영역에서 반복되는 특징어가 많은 사례', 'core_metric': 'cross_section_feature_count', 'ascending': False, 'rule': type5},
        {'code': '유형 6', 'name': '특정 특징어는 강하지만 근거가 한 영역 또는 소수 문장에 집중된 사례', 'core_metric': 'tfidf_top1', 'ascending': False, 'rule': type6},
        {'code': '유형 7', 'name': '학과 유사도가 전반적으로 높지만 상위 결과가 분산된 사례', 'core_metric': 'department_similarity_top5_mean', 'ascending': False, 'rule': type7},
        {'code': '유형 8', 'name': '데이터 품질 또는 계산상 검토가 필요한 사례', 'core_metric': 'total_token_count', 'ascending': False, 'rule': type8},
    ]


def select_candidates(metrics: pd.DataFrame, max_candidates: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for definition in build_candidate_rules(metrics):
        rows: list[dict[str, Any]] = []
        for _, row in metrics.iterrows():
            eligible, checks = definition['rule'](row)
            if not eligible:
                continue
            matched = [label for passed, label in checks if passed and label]
            rows.append({
                'case_id': row['case_id'],
                'candidate_type': definition['code'],
                'candidate_type_name': definition['name'],
                'matched_condition_count': len(matched),
                'matched_conditions': '; '.join(matched),
                'core_metric': definition['core_metric'],
                'core_metric_value': row[definition['core_metric']],
                'core_metric_percentile': row[f'{definition["core_metric"]}_percentile'],
            })
        frame = pd.DataFrame(rows)
        eligible_count = len(frame)
        if not frame.empty:
            frame = frame.sort_values(
                ['matched_condition_count', 'core_metric_percentile', 'case_id'],
                ascending=[False, definition['ascending'], True],
                kind='stable',
            )
            if definition['code'] != '유형 8':
                frame = frame.head(max_candidates)
            frame.insert(2, 'candidate_rank', range(1, len(frame) + 1))
            selected_frames.append(frame)
        summaries.append({
            'candidate_type': definition['code'],
            'candidate_type_name': definition['name'],
            'eligible_student_count': eligible_count,
            'selected_candidate_count': len(frame),
        })
    candidates = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    if not candidates.empty:
        overlap = candidates.groupby('case_id')['candidate_type'].nunique()
        candidates['other_candidate_type_overlap_count'] = candidates['case_id'].map(overlap) - 1
    return candidates, summary


def sanitizable_names(mapping: pd.DataFrame) -> set[str]:
    return {
        value
        for value in mapping['성명'].astype(str).str.strip()
        if value
    }


def identify_sensitive_terms(
    term_details: dict[str, list[dict[str, Any]]],
    student_names: set[str],
) -> set[str]:
    document_frequency = Counter()
    all_terms: set[str] = set()
    for details in term_details.values():
        case_terms = {str(item['term']).strip() for item in details if str(item['term']).strip()}
        document_frequency.update(case_terms)
        all_terms.update(case_terms)
    sensitive = {
        term
        for term in all_terms
        if any(name == term or name in term for name in student_names)
    }
    kiwi = app.get_kiwi()
    if kiwi is None:
        return sensitive
    for term in sorted(all_terms):
        if document_frequency[term] > 2:
            continue
        tokens = kiwi.tokenize(term)
        if any(token.tag == 'NNP' for token in tokens):
            sensitive.add(term)
    return sensitive


def sanitize_feature_term(term: str, sensitive_terms: set[str]) -> str:
    text = str(term).strip()
    if text in sensitive_terms:
        return '[식별 가능 어휘 제거]'
    return text


def common_terms_for_pair(
    student_vector: Any,
    major_vector: Any,
    terms: np.ndarray,
    limit: int = 12,
) -> list[str]:
    student_scores = student_vector.toarray().ravel()
    major_scores = major_vector.toarray().ravel()
    common = np.flatnonzero((student_scores > 0) & (major_scores > 0))
    ranked = sorted(
        common,
        key=lambda index: (-min(student_scores[index], major_scores[index]), str(terms[index])),
    )
    return [str(terms[index]) for index in ranked[:limit]]


def add_candidate_descriptions(
    candidates: pd.DataFrame,
    metrics: pd.DataFrame,
    term_details: dict[str, list[dict[str, Any]]],
    similarity_details: dict[str, list[dict[str, Any]]],
    sensitive_terms: set[str],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    result = candidates.merge(metrics, on='case_id', how='left', validate='many_to_one')
    result['top_feature_terms'] = result['case_id'].map(
        lambda case_id: '; '.join(
            f'{sanitize_feature_term(item["term"], sensitive_terms)} ({item["value"]:.4f})'
            for item in term_details.get(str(case_id), [])[:10]
        )
    )
    result['top_departments'] = result['case_id'].map(
        lambda case_id: '; '.join(
            f'{item["department"]} ({item["similarity"]:.6f})'
            for item in similarity_details.get(str(case_id), [])[:5]
        )
    )
    return result


def candidate_evidence_payload(
    candidates: pd.DataFrame,
    term_details: dict[str, list[dict[str, Any]]],
    evidence_index: dict[str, dict[str, dict[str, Any]]],
    similarity_details: dict[str, list[dict[str, Any]]],
    student_matrix: Any,
    major_matrix: Any,
    terms: np.ndarray,
    records: pd.DataFrame,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'privacy_notice': 'PRIVATE: 학생 식별 대응표와 함께 외부 공유 금지. 원문 전체가 아닌 후보 검토용 최소 문장만 포함.',
        'candidates': [],
    }
    record_positions = {str(case_id): index for index, case_id in enumerate(records['case_id'])}
    for case_id in sorted(candidates['case_id'].unique()):
        top_features = []
        for item in term_details.get(case_id, [])[:10]:
            evidence = evidence_index.get(case_id, {}).get(item['term'], {})
            top_features.append({
                'term': item['term'],
                'tfidf': item['value'],
                'sources': evidence.get('sources', []),
                'sentence_count': evidence.get('sentence_count', 0),
                'evidence_sentences': evidence.get('sentences', [])[:2],
            })
        top_departments = []
        student_position = record_positions[case_id]
        for item in similarity_details.get(case_id, [])[:5]:
            top_departments.append({
                'rank': item['rank'],
                'department': item['department'],
                'category': item['category'],
                'similarity': item['similarity'],
                'common_terms': common_terms_for_pair(
                    student_matrix[student_position],
                    major_matrix[item['major_index']],
                    terms,
                ),
            })
        payload['candidates'].append({
            'case_id': case_id,
            'candidate_types': sorted(
                candidates.loc[candidates['case_id'] == case_id, 'candidate_type'].tolist()
            ),
            'top_features': top_features,
            'top_departments': top_departments,
        })
    return payload


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.10g')


def configure_korean_font() -> str:
    path = app.font_path()
    if path:
        family = fm.FontProperties(fname=path).get_name()
        plt.rcParams['font.family'] = family
    plt.rcParams['axes.unicode_minus'] = False
    return path or ''


def save_histogram(metrics: pd.DataFrame, metric: str, title: str, x_label: str, path: Path) -> None:
    values = pd.to_numeric(metrics[metric], errors='coerce').dropna()
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    axis.hist(values, bins='auto', color='#2563EB', alpha=0.82, edgecolor='white')
    axis.axvline(values.median(), color='#DC2626', linestyle='--', linewidth=1.5, label='중앙값')
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel('학생 수')
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches='tight', metadata={'Software': 'StudentRecordExplorer'})
    plt.close(figure)


def save_scatter(
    metrics: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    title: str,
    x_label: str,
    y_label: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    axis.scatter(
        pd.to_numeric(metrics[x_metric], errors='coerce'),
        pd.to_numeric(metrics[y_metric], errors='coerce'),
        color='#0F766E',
        alpha=0.65,
        s=28,
        edgecolors='none',
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    figure.tight_layout()
    figure.savefig(path, bbox_inches='tight', metadata={'Software': 'StudentRecordExplorer'})
    plt.close(figure)


def create_visualizations(metrics: pd.DataFrame, output_dir: Path) -> list[str]:
    configure_korean_font()
    histograms = [
        ('total_token_count', '학생별 전처리 토큰 수 분포', '전처리 토큰 수', 'distribution_total_token_count.png'),
        ('cross_section_feature_count', '영역 반복 특징어 수 분포', '둘 이상 영역에서 확인된 특징어 수', 'distribution_cross_section_feature_count.png'),
        ('tfidf_top5_share', '상위 5개 TF-IDF 가중치 비중 분포', '상위 5개 / 상위 30개 TF-IDF 합', 'distribution_tfidf_top5_share.png'),
        ('department_similarity_rank1', '최고 학과 말뭉치 유사도 분포', '1위 코사인 유사도', 'distribution_department_similarity_rank1.png'),
        ('department_similarity_rank1_rank5_gap', '학과 유사도 1위-5위 격차 분포', '1위-5위 코사인 유사도 차이', 'distribution_department_similarity_rank1_rank5_gap.png'),
    ]
    for metric, title, label, filename in histograms:
        save_histogram(metrics, metric, title, label, output_dir / filename)
    save_scatter(
        metrics,
        'total_token_count',
        'cross_section_feature_count',
        '기록량과 영역 반복 특징어의 관계',
        '전처리 토큰 수',
        '둘 이상 영역에서 확인된 특징어 수',
        output_dir / 'scatter_record_volume_vs_cross_section_features.png',
    )
    save_scatter(
        metrics,
        'tfidf_top5_share',
        'department_similarity_top10_normalized_entropy',
        'TF-IDF 가중치 집중과 학과 결과 분산의 관계',
        '상위 5개 TF-IDF 비중',
        '상위 10개 학과 유사도 정규화 엔트로피',
        output_dir / 'scatter_tfidf_concentration_vs_department_dispersion.png',
    )
    return [filename for *_, filename in histograms] + [
        'scatter_record_volume_vs_cross_section_features.png',
        'scatter_tfidf_concentration_vs_department_dispersion.png',
    ]


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    subset = frame[columns].copy()
    headers = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    rows = []
    for _, row in subset.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append('' if math.isnan(value) else f'{value:.6g}')
            else:
                values.append(str(value).replace('|', '/').replace('\n', ' '))
        rows.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join([headers, separator, *rows])


def build_handoff(
    metrics: pd.DataFrame,
    distribution: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    validation: dict[str, Any],
    settings: dict[str, Any],
    majors: pd.DataFrame,
    generated_files: list[str],
) -> str:
    focus_metrics = [
        'total_token_count',
        'evidence_sentence_count',
        'cross_section_feature_count',
        'tfidf_top5_share',
        'department_similarity_rank1',
        'department_similarity_rank1_rank5_gap',
        'department_similarity_top5_mean',
    ]
    quartiles = distribution.loc[
        distribution['metric'].isin(focus_metrics), ['metric', 'Q1', 'median', 'Q3']
    ]
    candidate_sections = []
    for candidate_type, group in candidates.groupby('candidate_type', sort=True):
        candidate_sections.append(f'### {candidate_type} · {group.iloc[0]["candidate_type_name"]}')
        candidate_sections.append(
            markdown_table(
                group.head(10),
                ['candidate_rank', 'case_id', 'matched_conditions', 'top_feature_terms', 'top_departments'],
            )
        )
    overlap = (
        candidates.groupby('case_id')['candidate_type'].nunique().sort_values(ascending=False)
        if not candidates.empty else pd.Series(dtype=int)
    )
    overlap_rows = pd.DataFrame({
        'case_id': overlap.index,
        'included_type_count': overlap.values,
    }).head(20)
    quality_count = int(metrics['data_quality_note'].astype(str).str.strip().ne('').sum())
    return f"""# 학생부 사례 표집 분석 전달 문서

## 1. 실제 입력 구조

- 학생 캐시 SQLite: `records`, `tfidf`, `freq`, `evidence`, `meta` 테이블
- 학생 수: {len(metrics)}명
- 학과 말뭉치 SQLite: `majors` 테이블, {len(majors)}개 문서
- 분석 범위: {settings['scope']}
- 학생 캐시는 읽기 전용으로 사용했으며 원본 파일을 수정하지 않았다.

## 2. 기존 코드에서 확인한 설정

- Python {validation['software_versions']['python']}
- Kiwi/kiwipiepy {validation['software_versions']['kiwipiepy']}
- scikit-learn {validation['software_versions']['scikit-learn']}
- 추출 품사: Kiwi 태그가 `N`으로 시작하는 명사 및 `SL`, `SH`
- 최소 토큰 길이: {settings['min_len']}
- 불용어: {len(settings['stopwords'])}개, `data/stopwords.txt`와 앱 기본 목록의 합집합
- 표현 통일 규칙: {len(settings['synonyms'])}개, `data/synonyms.txt`
- TF-IDF: `sklearn.feature_extraction.text.TfidfVectorizer(token_pattern=r'(?u)\\b\\w+\\b')`
- 별도 지정이 없으므로 `norm='l2'`, `use_idf=True`, `smooth_idf=True`, `sublinear_tf=False`, `min_df=1`, `max_df=1.0` 기본값을 사용한다.
- TF는 원시 빈도, IDF는 평활화된 역문서빈도, 벡터는 L2 정규화이다.
- 학생 특징어는 전체 {len(metrics)}명 통합 문서의 TF-IDF에서 학생별 상위 30개가 캐시에 저장되어 있다.
- 학과 유사도는 501개 `말뭉치_통합` 문서로 만든 TF-IDF 공간에 학생 문서를 변환한 뒤 코사인 유사도를 계산한다.
- 앱 화면은 유사도 원값에 100을 곱해 백분율로 표시하지만, 이번 산출물은 0–1 원값을 저장한다.
- 근거 문장은 `출처`, `문장번호`, `원문`, `키워드목록`으로 색인되어 있다.
- 학생부 영역은 `창체`, `교과세특`, `행발` 세 값이다.

## 3. 산출 지표

- 기록량: 통합 및 세 영역의 전처리 토큰 수, 근거 문장 수
- TF-IDF: 상위 1·5위 값, 격차, 상위 5·30개 합, 상위 5개 비중, 상위 30개 정규화 엔트로피
- 영역 반복성: 근거가 있는 특징어 수, 둘 이상·세 영역 특징어 수, 특징어당 평균 영역 수
- 학과 비교: 1·2·5·10위 유사도, 순위 간 격차, 상위 5개 평균, 상위 10개 정규화 엔트로피
- 품질 관리: 특징어·학과 결과 가용 수, 근거·유사도 누락 플래그, 검토 메모
- 백분위는 동점에 평균순위를 부여한 `rank(method='average', pct=True)` 방식이며, 사분위 집단은 이 백분위 순위로 구분했다.

## 4. 데이터 품질과 검증

- 실제 학생 수: {len(metrics)}명
- 데이터 품질 검토 대상: {quality_count}명
- 학생 TF-IDF 재계산 불일치 학생: {validation['tfidf_recalculation']['tfidf_term_set_mismatch_students']}명
- 학생 TF-IDF 값 불일치 행: {validation['tfidf_recalculation']['tfidf_value_mismatch_rows']}행
- 앱 학과 유사도 표본 불일치: {validation['similarity_sample']['mismatches']}행
- 근거 문장 표본 불일치: {validation['evidence_sample']['mismatches']}명
- 학생 식별자 중복: {validation['input']['duplicate_identity_rows']}행

## 5. 주요 지표 사분위수

{markdown_table(quartiles, ['metric', 'Q1', 'median', 'Q3'])}

## 6. 유형별 후보 수

{markdown_table(candidate_summary, ['candidate_type', 'candidate_type_name', 'eligible_student_count', 'selected_candidate_count'])}

## 7. 유형별 익명 후보와 선정 이유

{chr(10).join(candidate_sections)}

## 8. 후보 유형 중복

{markdown_table(overlap_rows, ['case_id', 'included_type_count']) if not overlap_rows.empty else '후보 유형 중복 없음'}

## 9. 기존 논문 사례 A–E 관련

기존 사례 A–E의 원래 학생 식별자 또는 익명 대응표가 저장소와 입력 자료에 별도로 표시되어 있지 않아 직접 대응 여부를 계산하지 않았다. 최종 사례 검토 단계에서 `case_id_mapping_PRIVATE.csv`와 기존 A–E 식별정보를 로컬에서 대조해야 한다.

## 10. 최종 선정 전 사람의 확인 사항

- 후보의 상위 특징어가 실제 근거 문장에서 어떤 맥락으로 사용되었는지 확인
- 희귀 기관명·인명·고유명사가 익명 파일에 남지 않았는지 추가 점검
- 기록량 차이가 교사 기록 방식이나 과목 편성 차이에서 비롯되었는지 확인
- 유사도가 진로 적합성이나 추천 확률로 해석되지 않도록 질적 서술 검토
- 동일 학생이 여러 후보 유형에 포함된 이유와 최종 사례 간 대비가 충분한지 확인

## 11. 계산하지 못한 항목

- 현재 상담 정보와 희망 진로는 의도적으로 사용하지 않았다.
- 기존 A–E 사례와의 직접 대응은 식별 자료가 없어 계산하지 못했다.
- 익명 파일에서는 학생 성명과 일치하는 특징어 및 1~2명에게만 나타난 Kiwi 고유명사 특징어 {validation['privacy']['redacted_feature_term_count']}개를 치환했다. 기관명 등 자동 판별이 어려운 표현은 사람이 추가 확인해야 한다.

## 12. 생성 파일

{chr(10).join(f'- `{name}`' for name in generated_files)}

## 13. 다음 ChatGPT 대화에 함께 제공할 파일

- `chatgpt_handoff.md`
- `candidate_cases_anonymized.csv`
- 필요 시 `metric_distribution_summary.csv`, `candidate_type_summary.csv`

이 분석은 정량 지표에 기반한 체계적 후보군 구성 단계이며, 학생의 진로·역량·우수성을 측정하거나 최종 사례를 확정한 결과가 아니다.
"""


def build_method_summary(
    metrics: pd.DataFrame,
    distribution: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    quality_count: int,
) -> str:
    token_row = distribution.loc[distribution['metric'] == 'total_token_count'].iloc[0]
    evidence_row = distribution.loc[distribution['metric'] == 'evidence_sentence_count'].iloc[0]
    candidate_counts = ', '.join(
        f"{row['candidate_type']} {int(row['selected_candidate_count'])}명"
        for _, row in candidate_summary.iterrows()
    )
    return f"""# 2.5. 사례 선정 및 분석 초안

본 연구는 전체 분석 대상 {len(metrics)}명의 학생부 기록에 대해 정량 지표를 산출한 뒤, 지표의 상대적 분포에 따라 서로 다른 기록 양상을 나타내는 후보군을 구성하는 층화 목적표집 절차를 적용하였다. 이 절차는 최종 사례를 자동으로 결정하거나 학생의 진로·역량·우수성을 측정하기 위한 것이 아니라, 사례 선정 과정의 근거를 명시하고 상반된 양상의 사례를 함께 검토하기 위한 것이다.

학생별 기록량은 불용어 제거와 표현 통일을 거친 전처리 토큰 수 및 근거 문장 수로 확인하였다. 통합 문서의 전처리 토큰 수는 Q1 {token_row['Q1']:.1f}, 중앙값 {token_row['median']:.1f}, Q3 {token_row['Q3']:.1f}이었고, 근거 문장 수는 Q1 {evidence_row['Q1']:.1f}, 중앙값 {evidence_row['median']:.1f}, Q3 {evidence_row['Q3']:.1f}이었다. TF-IDF 관련 지표는 학생별 상위 30개 특징어의 가중치 분포를 이용하여 상위값, 상위 5개 비중, 정규화 엔트로피 등을 산출하였다. TF-IDF 특징어는 전체 학생 문서 집합에서 해당 학생의 문서를 상대적으로 구별하는 어휘 지표이며, 관심의 강도나 진로 구체성을 직접 측정하는 값으로 해석하지 않았다.

영역 반복성은 상위 특징어가 창의적 체험활동, 교과 세부능력 및 특기사항, 행동특성 및 종합의견 중 몇 개 영역의 근거 문장에서 확인되는지를 기준으로 산출하였다. 학과 말뭉치 비교는 학생부 통합 문서와 커리어넷 학과 통합 말뭉치의 TF-IDF 가중 어휘 분포 간 코사인 유사도를 사용하였다. 유사도는 전공 적합성, 진학 가능성, 추천 확률 또는 정확도로 해석하지 않았으며 절대 임계값을 설정하지 않았다.

각 연속형 지표에는 동점 평균순위 방식의 백분위와 사분위 집단을 부여하였다. 후보군은 특징어 및 학과 결과의 상대적 집중, 특징어와 학과 결과의 분산 차이, 기록량 희소성, 낮은 학과 말뭉치 유사도, 여러 기록 영역에서의 어휘 반복, 소수 영역·문장에 집중된 강한 특징어, 전반적으로 높은 동시에 분산된 학과 유사도, 데이터 품질 검토 필요의 여덟 유형으로 구성하였다. 유형별 선정 인원은 {candidate_counts}이었다. 후보 순위는 별도의 합성점수를 만들지 않고 유형 조건 충족 개수, 유형별 대표 지표의 백분위, 익명 ID 순으로 정하였다. 동일 학생은 서로 다른 분석적 의미를 가질 경우 여러 후보 유형에 중복 포함하였다.

데이터 품질 또는 계산상 검토가 필요한 학생은 {quality_count}명이었다. 최종 사례는 본 정량 산출만으로 확정하지 않으며, 후보별 특징어의 실제 근거 문장, 기록 영역의 맥락, 학과 말뭉치의 포착 범위와 분석상 한계를 사람이 질적으로 검토한 뒤 선정할 예정이다. 동료 검토는 최종 사례 선정 과정에서 수행할 예정이며, 본 연구 단계에서 사용성 조사나 효과 검증을 실시한 것으로 간주하지 않는다.
"""


def software_versions() -> dict[str, str]:
    import platform
    from importlib.metadata import version

    return {
        'python': platform.python_version(),
        'pandas': version('pandas'),
        'scikit-learn': version('scikit-learn'),
        'kiwipiepy': version('kiwipiepy'),
        'matplotlib': version('matplotlib'),
        'streamlit': version('streamlit'),
    }


def write_private_readme(output_dir: Path) -> None:
    text = """# 비공개 로컬 분석 파일

이 디렉터리의 `*_PRIVATE.*` 파일에는 학생 식별 대응정보 또는 학생부 근거 문장이 포함될 수 있다. 학교가 승인한 로컬 환경에서만 보관하고 외부 공유, Git 커밋, 원격 업로드를 금지한다.

외부 검토에는 `chatgpt_handoff.md`, `candidate_cases_anonymized.csv`, `metric_distribution_summary.csv`, `candidate_type_summary.csv`만 사용하되, 희귀 고유명사가 남아 있지 않은지 사람이 한 번 더 확인한다.
"""
    (output_dir / 'README_PRIVATE.md').write_text(text, encoding='utf-8')


def hash_files(output_dir: Path, filenames: list[str]) -> dict[str, str]:
    hashes = {}
    for filename in filenames:
        path = output_dir / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_csv_encoding(output_dir: Path, filenames: list[str]) -> dict[str, bool]:
    result = {}
    for filename in filenames:
        path = output_dir / filename
        result[filename] = path.read_bytes().startswith(b'\xef\xbb\xbf')
        pd.read_csv(path, encoding='utf-8-sig')
    return result


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.student_db)
    majors = load_major_corpus(args.major_db)
    input_validation = validate_input_identity(cache, args.expected_students)
    settings = read_analysis_settings(cache)

    records = sort_records(cache['records'])
    mapping = make_case_mapping(records)
    records = records.merge(mapping, on=ID_COLUMNS, how='left', validate='one_to_one')

    token_metrics, token_documents, _ = tokenize_records(records, settings)
    tfidf_metrics, term_details = prepare_tfidf_metrics(cache, mapping)
    evidence_metrics, evidence_index, _ = prepare_evidence_metrics(cache, mapping, term_details)
    similarity_metrics, similarity_details, major_vectorizer, major_matrix, student_matrix = prepare_similarity_metrics(
        records, token_documents, majors, settings
    )

    metrics = token_metrics.merge(tfidf_metrics, on='case_id', how='outer', validate='one_to_one')
    metrics = metrics.merge(evidence_metrics, on='case_id', how='outer', validate='one_to_one')
    metrics = metrics.merge(similarity_metrics, on='case_id', how='outer', validate='one_to_one')
    metrics = attach_quality_flags(metrics, records)
    metrics = add_relative_positions(metrics)
    metrics = metrics.sort_values('case_id', kind='stable').reset_index(drop=True)

    distribution = metric_distribution(metrics, BASE_METRICS)
    candidates, candidate_summary = select_candidates(metrics, args.max_candidates)
    names = sanitizable_names(mapping)
    sensitive_terms = identify_sensitive_terms(term_details, names)
    candidate_output = add_candidate_descriptions(
        candidates, metrics, term_details, similarity_details, sensitive_terms
    )
    private_candidate_output = add_candidate_descriptions(
        candidates, metrics, term_details, similarity_details, set()
    )

    private_metrics = mapping.merge(metrics, on='case_id', how='left', validate='one_to_one')
    anonymized_metrics = metrics.copy()
    private_candidates = private_candidate_output.merge(
        mapping, on='case_id', how='left', validate='many_to_one'
    ) if not private_candidate_output.empty else private_candidate_output

    quality_issues = metrics.loc[
        metrics['data_quality_note'].astype(str).str.strip().ne(''),
        ['case_id', 'missing_evidence_flag', 'missing_similarity_flag', 'data_quality_note'],
    ].copy()

    evidence_payload = candidate_evidence_payload(
        candidates,
        term_details,
        evidence_index,
        similarity_details,
        student_matrix,
        major_matrix,
        major_vectorizer.get_feature_names_out(),
        records,
    )

    write_csv(mapping, output_dir / 'case_id_mapping_PRIVATE.csv')
    write_csv(private_metrics, output_dir / 'case_sampling_metrics_PRIVATE.csv')
    write_csv(private_candidates, output_dir / 'candidate_cases_PRIVATE.csv')
    (output_dir / 'candidate_evidence_PRIVATE.json').write_text(
        json.dumps(evidence_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    write_csv(anonymized_metrics, output_dir / 'case_sampling_metrics_anonymized.csv')
    write_csv(distribution, output_dir / 'metric_distribution_summary.csv')
    write_csv(candidate_output, output_dir / 'candidate_cases_anonymized.csv')
    write_csv(candidate_summary, output_dir / 'candidate_type_summary.csv')
    write_csv(quality_issues, output_dir / 'data_quality_issues.csv')
    write_private_readme(output_dir)
    visualization_files = create_visualizations(metrics, output_dir)

    validation = {
        'input': input_validation,
        'software_versions': software_versions(),
        'privacy': {'redacted_feature_term_count': len(sensitive_terms)},
        'tfidf_recalculation': recompute_student_tfidf_validation(
            records, token_documents, cache, mapping
        ),
        'similarity_sample': similarity_sample_validation(
            records, majors, settings, similarity_details
        ),
        'evidence_sample': evidence_sample_validation(records, cache, mapping, settings),
    }
    csv_files = [
        'case_id_mapping_PRIVATE.csv',
        'case_sampling_metrics_PRIVATE.csv',
        'candidate_cases_PRIVATE.csv',
        'case_sampling_metrics_anonymized.csv',
        'metric_distribution_summary.csv',
        'candidate_cases_anonymized.csv',
        'candidate_type_summary.csv',
        'data_quality_issues.csv',
    ]
    validation['csv_utf8_sig'] = validate_csv_encoding(output_dir, csv_files)

    generated_files = sorted([
        *csv_files,
        'candidate_evidence_PRIVATE.json',
        'README_PRIVATE.md',
        *visualization_files,
        'validation_report.json',
        'chatgpt_handoff.md',
        'method_summary_draft.md',
    ])
    handoff = build_handoff(
        metrics,
        distribution,
        candidate_output,
        candidate_summary,
        validation,
        settings,
        majors,
        generated_files,
    )
    method_summary = build_method_summary(
        metrics,
        distribution,
        candidate_summary,
        len(quality_issues),
    )
    (output_dir / 'chatgpt_handoff.md').write_text(handoff, encoding='utf-8')
    (output_dir / 'method_summary_draft.md').write_text(method_summary, encoding='utf-8')
    (output_dir / 'validation_report.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    deterministic_files = [
        filename
        for filename in generated_files
        if filename != 'validation_report.json'
    ]
    validation['output_sha256'] = hash_files(output_dir, deterministic_files)
    (output_dir / 'validation_report.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True),
        encoding='utf-8',
    )

    print(f'[complete] students={len(metrics)} majors={len(majors)}')
    print(f'[complete] candidates={len(candidate_output)} quality_issues={len(quality_issues)}')
    print(f'[complete] output={output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
