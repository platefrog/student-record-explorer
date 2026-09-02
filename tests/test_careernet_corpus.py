import unittest
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import pandas as pd
import requests

from scripts.refresh_major_corpus import refresh

from careernet_corpus import (
    CHANNEL_COLUMNS,
    build_corpus_channels,
    corpus_quality,
    enrich_existing_corpus,
    format_careernet_error,
    redact_api_key,
    _request_json,
    parse_major_detail,
)


class CareernetCorpusTests(unittest.TestCase):
    def test_parse_major_detail_builds_raw_fields_and_channels(self):
        base = {
            'majorSeq': '100',
            '계열': '공학계열',
            '학과명': '컴퓨터공학과',
            '세부학과명': '소프트웨어학과',
        }
        payload = {
            'dataSearch': {
                'content': {
                    'major': '컴퓨터공학과',
                    'summary': '<b>컴퓨터</b> 시스템을 연구합니다.',
                    'interest': '논리적 사고에 관심이 있는 학생에게 적합합니다.',
                    'property': '소프트웨어와 하드웨어를 함께 학습합니다.',
                    'relate_subject': {
                        'content': [
                            {'subject_name': '일반선택', 'subject_description': '수학, 물리학'},
                            {'subject_name': '진로선택', 'subject_description': '인공지능 기초'},
                        ]
                    },
                    'career_act': {
                        'content': {'act_name': '코딩 활동', 'act_description': '프로그램을 직접 제작합니다.'}
                    },
                    'job': '소프트웨어개발자, 데이터과학자',
                    'qualifications': '정보처리기사',
                    'enter_field': {
                        'content': {'gradeuate': 'IT 기업', 'description': '소프트웨어 개발 분야'}
                    },
                    'main_subject': {
                        'content': [
                            {'SBJECT_NM': '자료구조', 'SBJECT_SUMRY': '자료를 효율적으로 저장하고 처리합니다.'},
                            {'SBJECT_NM': '알고리즘', 'SBJECT_SUMRY': '문제 해결 절차를 설계합니다.'},
                        ]
                    },
                }
            }
        }

        row = parse_major_detail(base, payload)

        self.assertIn('컴퓨터 시스템을 연구합니다.', row['학과개요'])
        self.assertIn('인공지능 기초', row['관련고교교과'])
        self.assertIn('프로그램을 직접 제작합니다.', row['진로탐색활동'])
        self.assertIn('자료구조', row['대학주요교과목'])
        self.assertIn('소프트웨어 개발 분야', row['말뭉치_진로'])
        self.assertIn('알고리즘', row['말뭉치_학업'])
        self.assertEqual(row['말뭉치'], row['말뭉치_통합'])

    def test_enrich_existing_corpus_preserves_compatibility(self):
        source = pd.DataFrame([{
            'majorSeq': '1',
            '계열': '자연계열',
            '학과명': '수학과',
            '학과개요': '수학의 이론을 연구합니다.',
            '대학주요교과목': '해석학 대수학',
        }])

        enriched = enrich_existing_corpus(source)

        for column in CHANNEL_COLUMNS:
            self.assertIn(column, enriched.columns)
        self.assertEqual(enriched.iloc[0]['말뭉치'], enriched.iloc[0]['말뭉치_통합'])
        self.assertIn('해석학', enriched.iloc[0]['말뭉치_학업'])

    def test_quality_counts_errors_and_filled_fields(self):
        rows = pd.DataFrame([
            {'majorSeq': '1', '학과명': '수학과', '학과개요': '설명', '수집오류': ''},
            {'majorSeq': '2', '학과명': '물리학과', '학과개요': '', '수집오류': 'timeout'},
        ])
        quality = corpus_quality(enrich_existing_corpus(rows))
        self.assertEqual(quality['total'], 2)
        self.assertEqual(quality['complete'], 1)
        self.assertEqual(quality['errors'], 1)
        self.assertEqual(quality['field_counts']['학과개요'], 1)

    def test_integrated_corpus_uses_each_raw_field_once(self):
        row = {
            'majorSeq': '10146', '계열': '공학계열', '학과명': '학과명고유', '세부학과명': '세부학과명고유',
            '학과개요': '개요고유', '흥미와적성': '흥미적성고유', '학과특성': '특성고유',
            '관련고교교과': '고교교과고유', '진로탐색활동': '진로활동고유', '관련직업': '관련직업고유',
            '관련자격': '관련자격고유', '졸업후진출분야': '진출분야고유', '대학주요교과목': '대학과목고유',
        }
        channels = build_corpus_channels(row)
        self.assertIn('대학과목고유', channels['말뭉치_학업'])
        self.assertIn('대학과목고유', channels['말뭉치_교과'])
        self.assertEqual(channels['말뭉치_통합'].count('대학과목고유'), 1)
        for field in [
            '학과명', '세부학과명', '학과개요', '흥미와적성', '학과특성', '관련고교교과',
            '진로탐색활동', '관련직업', '관련자격', '졸업후진출분야', '대학주요교과목',
        ]:
            self.assertIn(row[field], channels['말뭉치_통합'])

    def test_enrich_is_idempotent_and_legacy_mirror_is_equal(self):
        source = pd.DataFrame([{
            'majorSeq': '1', '계열': '자연계열', '학과명': '수학과', '세부학과명': '수학전공',
            '학과개요': '수학 개요', '대학주요교과목': '해석학 대수학',
        }])
        first = enrich_existing_corpus(source)
        second = enrich_existing_corpus(first)
        self.assertEqual(first.iloc[0]['말뭉치'], first.iloc[0]['말뭉치_통합'])
        self.assertEqual(first.iloc[0]['말뭉치_통합'], second.iloc[0]['말뭉치_통합'])
        self.assertEqual(first.iloc[0]['말뭉치_통합'].count('해석학'), 1)

    def test_api_key_is_redacted_in_raw_and_encoded_urls(self):
        secret = 'fake-key/only-for-test'
        message = 'GET https://example.test?foo=1&apiKey=fake-key%2Fonly-for-test&x=2'
        safe = redact_api_key(message, secret)
        self.assertNotIn(secret, safe)
        self.assertNotIn('fake-key%2Fonly-for-test', safe)
        self.assertIn('apiKey=[REDACTED]', safe)

    def test_request_error_does_not_store_api_key(self):
        secret = 'fake-key/only-for-request-test'

        class FailingSession:
            def get(self, *args, **kwargs):
                raise requests.exceptions.HTTPError(
                    f'GET https://example.test?apiKey={secret}&svcType=api'
                )

        with self.assertRaises(RuntimeError) as context:
            _request_json(FailingSession(), secret)
        self.assertNotIn(secret, str(context.exception))
        self.assertIn('[REDACTED]', str(context.exception))

    def test_ssl_error_is_distinguished_from_api_key_errors(self):
        message = format_careernet_error(
            requests.exceptions.SSLError('CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate chain'),
            'fake-key',
        )
        self.assertIn('HTTPS 인증서 검증', message)
        self.assertIn('API 키 오류가 아니라', message)

    def test_refresh_script_preserves_raw_fields_and_rebuilds_schema(self):
        row = {
            'majorSeq': '1', '계열': '공학', '학과명': '테스트학과', '세부학과명': '테스트전공',
            '학과개요': '개요', '흥미와적성': '흥미', '학과특성': '특성', '관련고교교과': '교과',
            '진로탐색활동': '활동', '관련직업': '직업', '관련자격': '자격',
            '졸업후진출분야': '진출', '대학주요교과목': '과목',
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.db'
            output = Path(directory) / 'result.db'
            with closing(sqlite3.connect(source)) as connection:
                pd.DataFrame([row]).to_sql('majors', connection, index=False)
                connection.commit()
            result = refresh(source, output, expected_count=1)
            self.assertEqual(result['rows'], 1)
            self.assertEqual(result['majorSeq_unique'], 1)
            with closing(sqlite3.connect(output)) as connection:
                saved = pd.read_sql_query('select * from majors', connection).iloc[0]
                self.assertEqual(saved['대학주요교과목'], '과목')
                self.assertEqual(saved['말뭉치_통합'].count('과목'), 1)
                self.assertEqual(saved['스키마버전'], '3')

    def test_legacy_integrated_text_is_preserved_when_only_names_are_present(self):
        source = pd.DataFrame([{
            'majorSeq': '1', '계열': '공학', '학과명': '테스트학과',
            '말뭉치': '기존 말뭉치의 고유 설명',
        }])
        enriched = enrich_existing_corpus(source)
        self.assertIn('기존 말뭉치의 고유 설명', enriched.iloc[0]['말뭉치_통합'])


if __name__ == '__main__':
    unittest.main()
