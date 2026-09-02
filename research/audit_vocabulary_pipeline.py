"""학과 원자료의 어휘가 통합 문서와 유사도 입력까지 전달되는지 감사한다.

출력은 집계값만 포함하며 학생 원문이나 식별정보를 읽거나 저장하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from careernet_corpus import CORPUS_TEXT_COLUMNS, validate_corpus_schema
from research.case_sampling_analysis import load_major_corpus


FIELD_COLUMNS = ['학과명', '세부학과명', *CORPUS_TEXT_COLUMNS]


def _token_stats(text: str, stop: set[str], syn: dict[str, str], min_len: int, analyzer: Any):
    before = app.tokenize(text, set(), syn, 1, analyzer)
    after = app.tokenize(text, stop, syn, min_len, analyzer)
    return before, after


def audit_major_fields(
    majors: pd.DataFrame,
    stop: set[str],
    syn: dict[str, str],
    min_len: int = 2,
    analyzer: Any = '간이 토큰화',
) -> pd.DataFrame:
    """필드별 원자료·토큰·통합 전달 집계를 반환한다."""
    validate_corpus_schema(majors, require_integrated=True, require_raw=True)
    field_tokens: dict[str, list[set[str]]] = {}
    field_before: dict[str, list[list[str]]] = {}
    field_after: dict[str, list[list[str]]] = {}
    rows: list[dict[str, Any]] = []

    for field in FIELD_COLUMNS:
        before_rows: list[list[str]] = []
        after_rows: list[list[str]] = []
        token_sets: list[set[str]] = []
        value_count = 0
        character_count = 0
        for _, row in majors.iterrows():
            text = app.clean(row.get(field, ''))
            if text:
                value_count += 1
                character_count += len(text)
            before, after = _token_stats(text, stop, syn, min_len, analyzer)
            before_rows.append(before)
            after_rows.append(after)
            token_sets.append(set(after))
        field_before[field] = before_rows
        field_after[field] = after_rows
        field_tokens[field] = token_sets

    vectorizer, integrated_matrix, _ = app.prepare_major_index(
        tuple(majors['말뭉치_통합'].astype(str)), tuple(sorted(stop)),
        tuple(sorted(syn.items())), min_len, analyzer,
    )
    integrated_token_rows = [
        app.tokenize(app.clean(value), stop, syn, min_len, analyzer)
        for value in majors['말뭉치_통합'].tolist()
    ]
    integrated_values = [app.clean(value) for value in majors['말뭉치_통합'].tolist()]
    integrated_vocabulary = set().union(*(set(tokens) for tokens in integrated_token_rows))
    for field in FIELD_COLUMNS:
        before_rows = field_before[field]
        after_rows = field_after[field]
        values = [app.clean(value) for value in majors[field].tolist()]
        before_tokens = [token for row in before_rows for token in row]
        after_tokens = [token for row in after_rows for token in row]
        field_matrix = vectorizer.transform([' '.join(tokens) for tokens in after_rows]) if vectorizer is not None else None
        other_vocab = set().union(*(
            set().union(*field_tokens[other])
            for other in FIELD_COLUMNS if other != field
        ))
        dropped = Counter(before_tokens)
        dropped.subtract(after_tokens)
        dropped = {term: count for term, count in dropped.items() if count > 0}
        reasons = Counter()
        for term, count in dropped.items():
            if term in stop:
                reasons['불용어'] += count
            elif not app.token_length_allowed(term, min_len):
                reasons['최소길이'] += count
            else:
                reasons['형태소/비명사 필터'] += count
        rows.append({
            '필드': field,
            '값있는학과수': sum(bool(value) for value in values),
            '공란학과수': sum(not value for value in values),
            '전체문자수': sum(len(value) for value in values),
            '형태소분석전토큰수': len(before_tokens),
            '불용어제거후토큰수': len(after_tokens),
            '고유어휘수': len(set(after_tokens)),
            '해당필드만고유어휘수': len(set(after_tokens) - other_vocab),
            '값있지만최종토큰0인학과수': sum(bool(value) and not after for value, after in zip(values, after_rows)),
            '통합문서전달토큰수': len(set(after_tokens) & integrated_vocabulary),
            '원문통합전달학과수': sum(
                bool(value) and value in integrated_value
                for value, integrated_value in zip(values, integrated_values)
            ),
            '코사인입력비영벡터학과수': int(
                (field_matrix.getnnz(axis=1) > 0).sum()
            ) if vectorizer is not None else 0,
            '같은행TFIDF겹침학과수': int(
                (field_matrix.multiply(integrated_matrix).sum(axis=1).A1 > 0).sum()
            ) if vectorizer is not None else 0,
            '대표탈락어휘': ', '.join(term for term, _ in sorted(dropped.items(), key=lambda item: (-item[1], item[0]))[:10]),
            '대표탈락사유': ', '.join(f'{key}:{value}' for key, value in reasons.most_common()),
        })
    return pd.DataFrame(rows)


def validate_field_delivery(
    majors: pd.DataFrame,
    stop: set[str],
    syn: dict[str, str],
    min_len: int = 2,
    analyzer: Any = '간이 토큰화',
) -> None:
    """각 원자료 필드가 통합 문서의 유사도 입력과 연결되는지 검증한다."""
    validate_corpus_schema(majors, require_integrated=True, require_raw=True)
    for field in FIELD_COLUMNS:
        for row_index, value in enumerate(majors[field].tolist()):
            source_text = app.clean(value)
            if source_text and source_text not in app.clean(majors.iloc[row_index]['말뭉치_통합']):
                major_seq = majors.iloc[row_index].get('majorSeq', row_index)
                raise ValueError(
                    f'{field}의 원문이 같은 학과의 말뭉치_통합에 전달되지 않았습니다: {major_seq}'
                )
        field_docs = [
            app.tokenize(app.clean(value), stop, syn, min_len, analyzer)
            for value in majors[field].tolist()
        ]
        if any(app.clean(value) for value in majors[field].tolist()) and not any(field_docs):
            raise ValueError(f'{field}는 값이 있지만 최종 토큰이 모두 0개입니다.')
    vectorizer, integrated_matrix, _ = app.prepare_major_index(
        tuple(majors['말뭉치_통합'].astype(str)), tuple(sorted(stop)),
        tuple(sorted(syn.items())), min_len, analyzer,
    )
    if vectorizer is None or integrated_matrix is None:
        raise ValueError('통합 말뭉치 TF-IDF 벡터를 만들 수 없습니다.')
    for field in FIELD_COLUMNS:
        field_docs = [
            app.tokenized(app.clean(value), stop, syn, min_len, analyzer)
            for value in majors[field].tolist()
        ]
        field_matrix = vectorizer.transform(field_docs)
        if any(field_docs) and field_matrix.nnz == 0:
            raise ValueError(f'{field}의 어휘가 통합 TF-IDF 공간에 전달되지 않았습니다.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='학과 어휘 전달 경로를 집계 감사합니다.')
    parser.add_argument('--major-db', type=Path, default=ROOT / 'data' / 'major_corpus.db')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'local_outputs' / 'tfidf_audit')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    majors = load_major_corpus(args.major_db)
    stop = app.read_stopwords()
    syn = app.read_synonyms()
    settings = audit_major_fields(majors, stop, syn, 2, '간이 토큰화')
    validate_field_delivery(majors, stop, syn, 2, '간이 토큰화')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings.to_csv(args.output_dir / 'vocabulary_audit.csv', index=False, encoding='utf-8-sig')
    summary = {
        'major_count': len(majors),
        'field_count': len(FIELD_COLUMNS),
        'delivery_check': '통과',
        'analyzer': '간이 토큰화',
    }
    (args.output_dir / 'vocabulary_audit_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
