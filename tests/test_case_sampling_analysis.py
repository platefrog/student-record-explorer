import math
import unittest

import pandas as pd

from research.case_sampling_analysis import (
    BASE_METRICS,
    add_relative_positions,
    normalized_entropy,
    percentile_quartile,
    prepare_tfidf_metrics,
    sanitize_feature_term,
    select_candidates,
)


class CaseSamplingAnalysisTests(unittest.TestCase):
    def test_full_vector_metrics_are_preferred_when_available(self):
        identity = {'학년': '1', '반': '1', '번호': '1', '성명': '가명학생'}
        cache = {
            'tfidf': pd.DataFrame([
                {**identity, '순위': 1, '단어': 'alpha', 'TF-IDF': 0.8},
                {**identity, '순위': 2, '단어': 'beta', 'TF-IDF': 0.2},
            ]),
            'tfidf_metrics': pd.DataFrame([{
                **identity,
                'positive_tfidf_feature_count': 10,
                'tfidf_max_full': 0.8,
                'tfidf_rank5_full': 0.1,
                'tfidf_top1_rank5_gap_full': 0.7,
                'tfidf_top5_sum_full': 1.5,
                'tfidf_top30_sum_full': 2.0,
                'tfidf_top5_share_full': 0.25,
                'tfidf_normalized_entropy_full': 0.9,
            }]),
        }
        mapping = pd.DataFrame([{**identity, 'case_id': 'S001'}])
        metrics, _details = prepare_tfidf_metrics(cache, mapping)

        self.assertEqual(metrics.iloc[0]['available_feature_count'], 10)
        self.assertEqual(metrics.iloc[0]['tfidf_top5_share'], 0.25)
        self.assertEqual(metrics.iloc[0]['tfidf_metric_basis'], 'full_positive_vector')

    def test_normalized_entropy_distinguishes_even_and_concentrated_values(self):
        self.assertAlmostEqual(normalized_entropy([1, 1, 1, 1]), 1.0)
        self.assertLess(normalized_entropy([10, 1, 1, 1]), 1.0)
        self.assertTrue(math.isnan(normalized_entropy([])))

    def test_percentile_uses_average_rank_for_ties(self):
        percentile, quartile = percentile_quartile(pd.Series([1, 1, 3, 4]))

        self.assertEqual(percentile.iloc[0], percentile.iloc[1])
        self.assertEqual(quartile.iloc[3], 'Q4')

    def test_known_student_name_is_removed_from_anonymized_terms(self):
        self.assertEqual(
            sanitize_feature_term('김민수', {'김민수'}),
            '[식별 가능 어휘 제거]',
        )
        self.assertEqual(sanitize_feature_term('알고리즘', {'김민수'}), '알고리즘')

    def test_tied_feature_count_does_not_make_every_student_sparse(self):
        frame = pd.DataFrame({'case_id': ['S001', 'S002', 'S003', 'S004']})
        for metric in BASE_METRICS:
            frame[metric] = [1.0, 2.0, 3.0, 4.0]
        frame['available_feature_count'] = 30.0
        frame['feature_terms_with_evidence'] = 30.0
        frame['missing_evidence_flag'] = False
        frame['missing_similarity_flag'] = False
        frame['data_quality_note'] = ''
        frame = add_relative_positions(frame)

        _, summary = select_candidates(frame, max_candidates=10)
        sparse_count = int(
            summary.loc[summary['candidate_type'] == '유형 3', 'eligible_student_count'].iloc[0]
        )

        self.assertEqual(sparse_count, 1)


if __name__ == '__main__':
    unittest.main()
