import unittest

from research.case_analysis_extraction import (
    CASES,
    anonymize_text,
    anonymize_term,
    split_keywords,
)


class CaseAnalysisExtractionTests(unittest.TestCase):
    def test_target_case_order_is_fixed(self):
        self.assertEqual(
            [item[0] for item in CASES],
            ['S201', 'S050', 'S322', 'S296', 'S065', 'S209', 'S269', 'S223'],
        )

    def test_split_keywords_strips_and_drops_blanks(self):
        self.assertEqual(split_keywords('응급, 의료, , 구조'), ['응급', '의료', '구조'])

    def test_anonymization_replaces_names_institutions_and_sensitive_terms(self):
        text = anonymize_text(
            '김민수 학생이 한빛고등학교와 새봄센터를 방문함.',
            {'김민수'},
        )
        self.assertNotIn('김민수', text)
        self.assertNotIn('한빛고등학교', text)
        self.assertNotIn('새봄센터', text)
        self.assertIn('[인명 제거]', text)
        self.assertIn('[기관명 제거]', text)
        self.assertEqual(anonymize_term('희귀어', {'희귀어'}), '[희귀 고유명사 제거]')


if __name__ == '__main__':
    unittest.main()
