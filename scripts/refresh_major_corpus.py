"""검증된 CareerNet 원자료 DB로 정식 학과 말뭉치 DB를 재생성한다."""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from careernet_corpus import RAW_COLUMNS, enrich_existing_corpus, validate_corpus_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CareerNet 학과 원자료 DB를 말뭉치 DB로 갱신합니다.')
    parser.add_argument('source', type=Path, help='검증된 원자료 SQLite DB')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'major_corpus.db')
    parser.add_argument('--expected-count', type=int, default=501)
    return parser.parse_args()


def read_source(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'원자료 DB를 찾을 수 없습니다: {path}')
    with closing(sqlite3.connect(path)) as connection:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if 'majors' not in tables:
            raise ValueError('원자료 DB에 majors 테이블이 없습니다.')
        return pd.read_sql_query('SELECT * FROM majors', connection).fillna('')


def refresh(source: Path, output: Path, expected_count: int = 501) -> dict[str, int | str]:
    source_df = read_source(source)
    validate_corpus_schema(source_df, require_integrated=False, require_raw=True)
    if len(source_df) != expected_count:
        raise ValueError(f'원자료 행 수가 {expected_count}가 아닙니다: {len(source_df)}')
    if source_df['majorSeq'].astype(str).nunique() != expected_count:
        raise ValueError('majorSeq가 고유하지 않습니다.')
    result = enrich_existing_corpus(source_df)
    for column in RAW_COLUMNS:
        if not result[column].astype(str).equals(source_df[column].astype(str)):
            raise ValueError(f'원자료 필드가 변경되었습니다: {column}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'{output.name}.tmp')
    try:
        connection = sqlite3.connect(temporary)
        try:
            result.fillna('').to_sql('majors', connection, if_exists='replace', index=False)
            integrity = connection.execute('PRAGMA integrity_check').fetchone()[0]
        finally:
            connection.close()
        if integrity != 'ok':
            raise ValueError(f'SQLite 무결성 검사 실패: {integrity}')
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        'rows': len(result),
        'majorSeq_unique': int(result['majorSeq'].astype(str).nunique()),
        'integrity_check': 'ok',
        'schema_version': str(result['스키마버전'].iloc[0]) if not result.empty else '',
    }


if __name__ == '__main__':
    args = parse_args()
    print(refresh(args.source, args.output, args.expected_count))
