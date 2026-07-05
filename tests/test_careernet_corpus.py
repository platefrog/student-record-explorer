import unittest

import pandas as pd

from careernet_corpus import (
    CHANNEL_COLUMNS,
    build_corpus_channels,
    corpus_quality,
    enrich_existing_corpus,
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


if __name__ == '__main__':
    unittest.main()
