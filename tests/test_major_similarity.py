import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import app

from app import (
    analyzer_name,
    calibrate_similarity_thresholds,
    gap_analysis,
    highlight_keyword_html,
    label_similarity_results,
    prepare_major_index,
    similarity,
)
from careernet_corpus import enrich_existing_corpus


class MajorSimilarityTests(unittest.TestCase):
    def setUp(self):
        prepare_major_index.clear()
        self.majors = enrich_existing_corpus(pd.DataFrame([
            {
                'majorSeq': '1', '계열': '공학계열', '학과명': '컴퓨터공학과',
                '학과개요': '컴퓨터 소프트웨어 프로그래밍 알고리즘 데이터 분석',
                '관련고교교과': '수학 정보 인공지능 기초',
                '진로탐색활동': '코딩 프로그램 제작 데이터 분석 프로젝트',
            },
            {
                'majorSeq': '2', '계열': '예체능계열', '학과명': '체육학과',
                '학과개요': '운동 스포츠 체력 훈련 경기 지도',
                '관련고교교과': '체육 운동과 건강',
                '진로탐색활동': '스포츠 경기 참여 체력 훈련',
            },
        ]))

    def test_similarity_reuses_prepared_channel_index(self):
        result = similarity(
            '프로그래밍으로 데이터를 분석하고 알고리즘을 구현함',
            self.majors, set(), {}, 2, False, 2, channel='학업',
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['학과명'], '컴퓨터공학과')

    def test_similarity_uses_full_student_document_not_display_top_terms(self):
        majors = enrich_existing_corpus(pd.DataFrame([
            {'majorSeq': '1', '계열': '공학', '학과명': '희귀학과', '학과개요': '희귀어휘'},
            {'majorSeq': '2', '계열': '공학', '학과명': '다른학과', '학과개요': '다른어휘'},
        ]))
        student_text = ('공통어휘 ' * 100) + '희귀어휘'
        result = similarity(student_text, majors, set(), {}, 2, '간이 토큰화', 2, channel='통합')
        self.assertEqual(result.iloc[0]['학과명'], '희귀학과')
        self.assertGreater(result.iloc[0]['유사도'], 0)

    def test_gap_analysis_uses_selected_channel(self):
        gap, common, _, metrics = gap_analysis(
            '정보 수업에서 코딩 활동을 수행함',
            self.majors.iloc[0], self.majors, set(), {}, 2, False, 20,
            channel='활동',
        )
        self.assertFalse(gap.empty)
        self.assertIn('cosine', metrics)
        self.assertGreaterEqual(len(common), 1)

    def test_analyzer_names_support_old_and_new_cache_values(self):
        self.assertEqual(analyzer_name(True), 'Kiwi')
        self.assertEqual(analyzer_name(False), '간이 토큰화')
        self.assertEqual(analyzer_name('MeCab'), 'MeCab')
        self.assertEqual(analyzer_name('간이토큰화'), '간이 토큰화')

    def test_synonym_rules_produce_one_target_token(self):
        synonyms = {
            'S/W': '소프트웨어',
            '머신러닝': '기계학습',
            '딥러닝': '심층학습',
            '아이오티': '사물인터넷',
            '생기부': '학교생활기록부',
        }
        for source, target in synonyms.items():
            tokens = app.tokenize(source, set(), synonyms, 2, '간이 토큰화')
            self.assertEqual(tokens, [target], source)

    def test_synonym_alias_does_not_match_inside_english_word(self):
        tokens = app.tokenize('RAID와 AI를 사용함', set(), {'AI': '인공지능'}, 2, '간이 토큰화')
        self.assertIn('RAID', tokens)
        self.assertIn('인공지능', tokens)

    def test_simple_tokenizer_applies_minimum_length_to_single_characters(self):
        tokens = app.tokenize('꿈 삶 것', set(), {}, 2, '간이 토큰화')
        self.assertEqual(tokens, [])

    def test_kiwi_applies_minimum_length_to_single_character_nouns(self):
        fake_kiwi = SimpleNamespace(tokenize=lambda _text: [
            SimpleNamespace(form='꿈', tag='NNG'),
            SimpleNamespace(form='것', tag='NNB'),
        ])
        with patch.object(app, 'get_kiwi', return_value=fake_kiwi):
            self.assertEqual(app.tokenize('ignored', set(), {}, 2, 'Kiwi'), [])

    def test_mecab_applies_minimum_length_to_single_character_nouns(self):
        fake_mecab = SimpleNamespace(pos=lambda _text: [('꿈', 'NNG'), ('것', 'NNB')])
        with patch.object(app, 'get_mecab', return_value=fake_mecab), patch.object(
            app, 'MECAB_BACKEND', 'python_mecab_ko'
        ):
            self.assertEqual(app.tokenize('ignored', set(), {}, 2, 'MeCab'), [])

    def test_keyword_highlight_escapes_html_and_includes_synonym_surface(self):
        rendered = highlight_keyword_html('<script>AI 분석 활동</script>', '인공지능', {'AI': '인공지능'})
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)
        self.assertIn('sre-keyword-highlight', rendered)
        self.assertIn('AI', rendered)

    def test_similarity_labels_use_empirical_distribution(self):
        student_documents = ('프로그래밍 알고리즘 데이터', '운동 스포츠 체력')
        thresholds = calibrate_similarity_thresholds(
            student_documents,
            tuple(self.majors['말뭉치_통합'].astype(str)),
            tuple(), tuple(), 2, '간이 토큰화',
        )
        raw = similarity(
            student_documents[0], self.majors, set(), {}, 2,
            '간이 토큰화', 2, channel='통합',
        )
        labeled = label_similarity_results(raw, thresholds)
        self.assertIn('연계수준', labeled.columns)
        self.assertIn('유사도(%)', labeled.columns)
        self.assertGreaterEqual(thresholds['fit'], thresholds['high'])
        self.assertGreaterEqual(thresholds['high'], thresholds['medium'])
        self.assertTrue(
            set(labeled['연계수준']).issubset(
                {'적합', '높은 연계', '보통 연계', '낮은 연계'}
            )
        )

        boundary = pd.DataFrame([{
            '순위': 1,
            '학과명': '경계값 학과',
            '계열': '테스트',
            '유사도': thresholds['fit'],
            '공통핵심어': '테스트',
        }])
        boundary_labeled = label_similarity_results(boundary, thresholds)
        self.assertEqual(boundary_labeled.iloc[0]['연계수준'], '적합')


if __name__ == '__main__':
    unittest.main()
