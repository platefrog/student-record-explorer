import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

import app


class UploadedBytes(io.BytesIO):
    def __init__(self, name: str, content: bytes):
        super().__init__(content)
        self.name = name

    def getvalue(self):
        return super().getvalue()


def unique_words(count: int) -> str:
    words = []
    for number in range(count):
        value = number
        letters = ''
        while True:
            letters = chr(ord('a') + value % 26) + letters
            value = value // 26 - 1
            if value < 0:
                break
        words.append('term' + letters)
    return ' '.join(words)


def sample_records(text: str = '사과 바나나 사과') -> pd.DataFrame:
    return pd.DataFrame([
        {
            '학년': '1', '반': '1', '번호': '1', '성명': '가명학생',
            '창체': text, '교과세특': '바나나 포도', '행발': '사과 포도',
            '통합': f'{text} 바나나 포도 사과 포도',
        }
    ])


def sample_major() -> pd.DataFrame:
    return pd.DataFrame([
        {
            'majorSeq': '1', '계열': '테스트계열', '학과명': '테스트학과',
            '말뭉치_통합': '사과 바나나 전공', '말뭉치': '사과 바나나 전공',
        }
    ])


class FullTfidfMetricTests(unittest.TestCase):
    def test_one_vectorizer_fit_builds_top_terms_and_metrics(self):
        original = app.TfidfVectorizer
        calls = []

        def factory(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        with patch.object(app, 'TfidfVectorizer', side_effect=factory):
            result = app.prepare_student_tfidf(
                sample_records(), '통합', set(), {}, 2, False, cache_term_limit=120
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result.full_metrics), 1)
        self.assertFalse(result.top_terms.empty)

    def test_full_metrics_follow_formula(self):
        result = app.prepare_student_tfidf(
            sample_records(), '통합', set(), {}, 2, False, cache_term_limit=2
        )
        values = np.sort(result.matrix.getrow(0).data)[::-1]
        metric = result.full_metrics.iloc[0]
        self.assertAlmostEqual(metric['tfidf_max_full'], values[0])
        self.assertAlmostEqual(metric['tfidf_top3_sum_full'], values[:3].sum())
        self.assertAlmostEqual(
            metric['tfidf_top3_share_full'], values[:3].sum() / values.sum()
        )

    def test_full_metrics_do_not_depend_on_cache_limit(self):
        records = sample_records(unique_words(150))
        thirty = app.prepare_student_tfidf(records, '통합', set(), {}, 2, False, 30)
        sixty = app.prepare_student_tfidf(records, '통합', set(), {}, 2, False, 60)
        one_twenty = app.prepare_student_tfidf(records, '통합', set(), {}, 2, False, 120)
        columns = app.TFIDF_FULL_METRIC_COLUMNS
        pd.testing.assert_frame_equal(
            thirty.full_metrics[columns], sixty.full_metrics[columns]
        )
        pd.testing.assert_frame_equal(
            sixty.full_metrics[columns], one_twenty.full_metrics[columns]
        )

    def test_equal_weights_use_alphabetical_tie_break(self):
        records = sample_records('alpha beta')
        result = app.prepare_student_tfidf(records, '창체', set(), {}, 2, False, 10)
        self.assertEqual(result.top_terms['단어'].tolist()[:2], ['alpha', 'beta'])

    def test_empty_documents_are_safe(self):
        records = sample_records('')
        records[['창체', '교과세특', '행발', '통합']] = ''
        result = app.prepare_student_tfidf(records, '통합', set(), {}, 2, False, 120)
        self.assertEqual(result.matrix.shape, (1, 0))
        self.assertEqual(result.full_metrics.iloc[0]['positive_tfidf_feature_count'], 0)

    def test_top_term_extraction_never_densifies_matrix(self):
        with patch.object(sparse.csr_matrix, 'toarray', side_effect=AssertionError('dense')):
            result = app.prepare_student_tfidf(
                sample_records(), '통합', set(), {}, 2, False, 120
            )
        self.assertFalse(result.top_terms.empty)


class StudentMetricAndCacheTests(unittest.TestCase):
    def build_cache(self, top_n=10, major=None):
        return app.build_student_cache(
            sample_records(), '통합', {'사과'}, {}, 2, False,
            top_n=top_n, major_df=major,
        )

    def test_record_composition_sums_and_ratios(self):
        cache = self.build_cache(major=sample_major())
        metric = cache['student_metrics'].iloc[0]
        self.assertEqual(
            metric['token_count_total'],
            metric['token_count_creative']
            + metric['token_count_subject']
            + metric['token_count_behavior'],
        )
        self.assertEqual(
            metric['evidence_unit_count_total'],
            metric['evidence_unit_count_creative']
            + metric['evidence_unit_count_subject']
            + metric['evidence_unit_count_behavior'],
        )
        self.assertEqual(
            metric['removed_token_count_total'],
            metric['token_count_before_stopwords_total'] - metric['token_count_total'],
        )
        self.assertAlmostEqual(
            metric['creative_token_ratio']
            + metric['subject_token_ratio']
            + metric['behavior_token_ratio'],
            1.0,
        )
        self.assertGreaterEqual(metric['department_vocab_token_coverage'], 0)
        self.assertLessEqual(metric['department_vocab_token_coverage'], 1)

    def test_character_count_uses_cleaned_sections(self):
        records = sample_records('  사과   바나나  ')
        cache = app.build_student_cache(records, '통합', set(), {}, 2, False)
        metric = cache['student_metrics'].iloc[0]
        expected = sum(len(app.clean(records.iloc[0][section])) for section in ['창체', '교과세특', '행발'])
        self.assertEqual(metric['character_count_total'], expected)

    def test_display_top_n_does_not_change_cache_storage_or_full_metrics(self):
        words = unique_words(150)
        records = sample_records(words)
        first = app.build_student_cache(records, '통합', set(), {}, 2, False, top_n=10)
        second = app.build_student_cache(records, '통합', set(), {}, 2, False, top_n=100)
        self.assertEqual(len(first['tfidf']), 120)
        self.assertEqual(len(second['tfidf']), 120)
        pd.testing.assert_frame_equal(first['tfidf_metrics'], second['tfidf_metrics'])

    def test_zero_department_vector_is_safe(self):
        cache = app.build_student_cache(
            sample_records('오렌지 레몬'), '통합', set(), {}, 2, False,
            major_df=sample_major(),
        )
        metric = cache['student_metrics'].iloc[0]
        self.assertIn(metric['department_student_vector_is_zero'], [True, False])
        self.assertGreaterEqual(metric['department_vocab_token_coverage'], 0)

    def test_new_tables_survive_zip_excel_and_sqlite(self):
        cache = self.build_cache(major=sample_major())
        zip_loaded = app.load_student_cache(
            UploadedBytes('cache.zip', app.cache_to_zip(cache))
        )
        excel_loaded = app.load_student_cache(
            UploadedBytes('cache.xlsx', app.cache_to_excel(cache))
        )
        db_loaded = app.load_student_cache(
            UploadedBytes('cache.db', app.cache_to_db(cache))
        )
        for loaded in [zip_loaded, excel_loaded, db_loaded]:
            self.assertIn('tfidf_metrics', loaded)
            self.assertIn('student_metrics', loaded)
            self.assertIn('meta', loaded)

    def test_old_cache_is_loaded_without_inventing_metrics(self):
        old_cache = {
            'records': sample_records(),
            'tfidf': pd.DataFrame([
                {'학년': '1', '반': '1', '번호': '1', '성명': '가명학생', '순위': 1, '단어': '사과', 'TF-IDF': 1.0}
            ]),
            'freq': pd.DataFrame(),
            'evidence': pd.DataFrame(),
            'meta': pd.DataFrame([{'분석범위': '통합'}]),
        }
        status = app.student_cache_compatibility(old_cache)
        self.assertNotEqual(status['status'], '정상 최신 캐시')
        self.assertFalse(status['has_student_metrics'])
        self.assertTrue(app.student_cache_rows(old_cache, 'student_metrics', sample_records().iloc[0]).empty)

    def test_new_cache_reports_latest_schema(self):
        cache = self.build_cache(major=sample_major())
        status = app.student_cache_compatibility(cache)
        self.assertEqual(status['status'], '정상 최신 캐시')
        self.assertEqual(status['schema_version'], app.STUDENT_CACHE_SCHEMA_VERSION)

    def test_cache_meta_contains_full_dictionary_configuration(self):
        cache = self.build_cache(major=sample_major())
        meta = cache['meta'].iloc[0]
        for column in [
            '불용어목록JSON', '실효불용어SHA256', '표현통일규칙JSON',
            '실효표현통일규칙SHA256', '최소단어길이',
            '형태소분석기', '형태소분석기버전',
        ]:
            self.assertIn(column, meta.index)
        self.assertEqual(meta['형태소분석기버전'], '내장')

    def test_cache_dictionary_mismatch_is_reported_when_settings_are_supplied(self):
        cache = self.build_cache(major=sample_major())
        status = app.student_cache_compatibility(
            cache, stop={'다른불용어'}, syn={}, min_len=2, analyzer=False
        )
        self.assertTrue(status['dictionary_mismatch'])
        self.assertIn('다시 전처리', status['warning'])

    def test_private_research_outputs_are_git_ignored(self):
        ignore_text = Path('.gitignore').read_text(encoding='utf-8')
        self.assertIn('/local_outputs/', ignore_text)
        self.assertIn('*_PRIVATE.*', ignore_text)
        self.assertIn('*.npz', ignore_text)
        self.assertIn('/major_corpus_2026-09-01.db', ignore_text)


class WordcloudSettingTests(unittest.TestCase):
    def setUp(self):
        self.tfidf = pd.DataFrame({
            '단어': ['alpha', 'beta', 'gamma'],
            'TF-IDF': [0.9, 0.5, 0.2],
        })
        self.frequency = pd.DataFrame({
            '단어': ['alpha', 'beta', 'gamma'],
            '빈도': [2, 9, 4],
        })

    def test_tfidf_wordcloud_uses_only_selected_top_n(self):
        weights = app.wordcloud_weights(self.tfidf, self.frequency, 'TF-IDF 특징어', 2)
        self.assertEqual(weights, {'alpha': 0.9, 'beta': 0.5})

    def test_frequency_wordcloud_uses_frequency_not_tfidf(self):
        weights = app.wordcloud_weights(self.tfidf, self.frequency, '단어 빈도', 2)
        self.assertEqual(weights, {'alpha': 2, 'beta': 9})

    def test_wordcloud_controls_do_not_rebuild_cache(self):
        with patch.object(app, 'build_student_cache') as build:
            app.wordcloud_weights(self.tfidf, self.frequency, 'TF-IDF 특징어', 2)
            app.wordcloud_weights(self.tfidf, self.frequency, '단어 빈도', 3)
        build.assert_not_called()

    def test_same_wordcloud_settings_are_reproducible(self):
        weights = app.wordcloud_weights(self.tfidf, self.frequency, 'TF-IDF 특징어', 3)
        first = app.wordcloud_fig(weights, 'TF-IDF 특징어', 3, 500, 300, '')
        app.wordcloud_fig.clear()
        second = app.wordcloud_fig(weights, 'TF-IDF 특징어', 3, 500, 300, '')
        first_image = np.asarray(first.axes[0].images[0].get_array())
        second_image = np.asarray(second.axes[0].images[0].get_array())
        np.testing.assert_array_equal(first_image, second_image)
        plt.close(first)
        plt.close(second)


if __name__ == '__main__':
    unittest.main()
