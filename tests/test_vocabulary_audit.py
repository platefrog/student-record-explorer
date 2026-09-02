import unittest

import pandas as pd

import app
from research.audit_vocabulary_pipeline import audit_major_fields, validate_field_delivery


class VocabularyAuditTests(unittest.TestCase):
    def setUp(self):
        self.majors = pd.DataFrame([{
            'majorSeq': '1', '계열': '공학계열', '학과명': '감사학과', '세부학과명': '감사전공',
            '학과개요': '개요어휘', '흥미와적성': '흥미어휘', '학과특성': '특성어휘',
            '관련고교교과': '교과어휘', '진로탐색활동': '활동어휘', '관련직업': '직업어휘',
            '관련자격': '자격어휘', '졸업후진출분야': '진출어휘', '대학주요교과목': '과목어휘',
            '말뭉치_통합': '감사학과 감사전공 개요어휘 흥미어휘 특성어휘 교과어휘 활동어휘 직업어휘 자격어휘 진출어휘 과목어휘',
        }])

    def test_audit_reports_all_fields_and_delivery(self):
        report = audit_major_fields(self.majors, set(), {}, 2, '간이 토큰화')
        self.assertEqual(len(report), 11)
        self.assertEqual(set(report['필드']), {
            '학과명', '세부학과명', '학과개요', '흥미와적성', '학과특성', '관련고교교과',
            '진로탐색활동', '관련직업', '관련자격', '졸업후진출분야', '대학주요교과목',
        })
        self.assertTrue((report['코사인입력비영벡터학과수'] == 1).all())
        validate_field_delivery(self.majors, set(), {}, 2, '간이 토큰화')

    def test_corpus_texts_does_not_join_metadata_or_arbitrary_columns(self):
        legacy = pd.DataFrame([{'majorSeq': '1', '말뭉치': '정상 텍스트', '수집오류': 'secret-metadata'}])
        self.assertEqual(app.corpus_texts(legacy, '통합'), ['정상 텍스트'])
        with self.assertRaises(ValueError):
            app.corpus_texts(legacy, '학업')

    def test_app_rejects_metadata_only_corpus_input(self):
        with self.assertRaisesRegex(ValueError, '사용 가능한 학과 텍스트 필드'):
            app.validate_app_corpus_input(pd.DataFrame([{
                'majorSeq': '1', '수집오류': '수집시각과 오류만 있는 파일', '수집시각': '2026-09-01',
            }]))

    def test_delivery_rejects_raw_field_missing_from_same_major(self):
        broken = self.majors.copy()
        broken.loc[0, '말뭉치_통합'] = broken.loc[0, '말뭉치_통합'].replace('과목어휘', '')
        with self.assertRaisesRegex(ValueError, '같은 학과'):
            validate_field_delivery(broken, set(), {}, 2, '간이 토큰화')


if __name__ == '__main__':
    unittest.main()
