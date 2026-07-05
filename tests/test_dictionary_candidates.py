import unittest

import pandas as pd

import app


class DictionaryCandidateTests(unittest.TestCase):
    def test_repeated_words_are_ranked_without_existing_stopwords(self):
        records = pd.DataFrame([
            {'학년': '3', '반': '1', '번호': '1', '성명': '가나다'},
            {'학년': '3', '반': '1', '번호': '2', '성명': '라마바'},
            {'학년': '3', '반': '1', '번호': '3', '성명': '사아자'},
        ])
        freq = pd.DataFrame([
            {'학년': '3', '반': '1', '번호': '1', '성명': '가나다', '단어': '문제해결', '빈도': 3},
            {'학년': '3', '반': '1', '번호': '2', '성명': '라마바', '단어': '문제해결', '빈도': 2},
            {'학년': '3', '반': '1', '번호': '3', '성명': '사아자', '단어': '협업', '빈도': 4},
            {'학년': '3', '반': '1', '번호': '1', '성명': '가나다', '단어': '활동', '빈도': 8},
            {'학년': '3', '반': '1', '번호': '2', '성명': '라마바', '단어': '활동', '빈도': 7},
        ])

        result = app.dictionary_stopword_candidates(
            {'records': records, 'freq': freq}, {'활동'}, min_student_ratio=0.5
        )

        self.assertEqual(result['단어'].tolist(), ['문제해결'])
        self.assertEqual(result.iloc[0]['등장 학생'], 2)
        self.assertEqual(result.iloc[0]['전체 빈도'], 5)
        self.assertAlmostEqual(result.iloc[0]['학생 비율(%)'], 66.7)


if __name__ == '__main__':
    unittest.main()
