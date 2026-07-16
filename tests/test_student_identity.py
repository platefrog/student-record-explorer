import io
import unittest

import pandas as pd

from app import (
    build_student_cache,
    find_student_by_identity,
    merge_records,
    parse_excel_files,
    student_identity,
    student_mask,
    tfidf_table,
)


class NamedExcel(io.BytesIO):
    def __init__(self, name: str, rows: list[list[str]]):
        buffer = io.BytesIO()
        pd.DataFrame(rows).to_excel(buffer, index=False, header=False, engine='openpyxl')
        super().__init__(buffer.getvalue())
        self.name = name


def student_file(name: str, detail: str) -> NamedExcel:
    return NamedExcel(name, [
        ['번 호', '성 명', '특기사항'],
        ['1', '김민수', detail],
    ])


class StudentIdentityTests(unittest.TestCase):
    def test_tfidf_table_keeps_richer_wordcloud_terms(self):
        terms = [chr(97 + index // 26) + chr(97 + index % 26) for index in range(130)]
        records = pd.DataFrame([{
            '학년': '3', '반': '1', '번호': '1', '성명': '가학생', '통합': ' '.join(terms)
        }])

        tfidf, _, _ = tfidf_table(records, '통합', set(), {}, 2, False, top_n=120)

        self.assertEqual(len(tfidf), 120)

    def test_committed_student_identity_finds_the_same_class_record(self):
        records = pd.DataFrame([
            {'학년': '3', '반': '1', '번호': '7', '성명': '김민수', '통합': '1반 기록'},
            {'학년': '3', '반': '2', '번호': '7', '성명': '김민수', '통합': '2반 기록'},
        ])

        identity = student_identity(records.iloc[1])
        matched = find_student_by_identity(records, identity)

        self.assertIsNotNone(matched)
        self.assertEqual(matched['반'], '2')
        self.assertEqual(matched['통합'], '2반 기록')

    def test_same_name_and_number_in_different_classes_stay_separate(self):
        files = [
            student_file('3-1반 창체.xlsx', '과학 탐구 활동에 성실하게 참여함'),
            student_file('3-2반 창체.xlsx', '독서 토론 활동에 주도적으로 참여함'),
        ]

        parsed, _ = parse_excel_files(files, '창체')

        self.assertEqual(len(parsed), 2)
        self.assertEqual(set(zip(parsed['학년'], parsed['반'])), {('3', '1'), ('3', '2')})

    def test_three_sources_merge_by_grade_class_number_and_name(self):
        source_frames = []
        for source in ['창체', '교과세특', '행발']:
            files = [
                student_file(f'3학년 1반 {source}.xlsx', f'{source} 첫 번째 반 학생 기록 내용입니다'),
                student_file(f'3학년 2반 {source}.xlsx', f'{source} 두 번째 반 학생 기록 내용입니다'),
            ]
            parsed, _ = parse_excel_files(files, source)
            source_frames.append(parsed)

        merged = merge_records(source_frames)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[['학년', '반', '번호', '성명']].values.tolist(), [
            ['3', '1', '1', '김민수'],
            ['3', '2', '1', '김민수'],
        ])
        self.assertTrue((merged[['창체', '교과세특', '행발']].map(len) > 0).all().all())

        cache = build_student_cache(merged, '통합', set(), {}, 2, False, top_n=10)
        for key in ['records', 'tfidf', 'freq', 'evidence']:
            self.assertTrue({'학년', '반', '번호', '성명'}.issubset(cache[key].columns))

        first_student = cache['records'].iloc[0]
        self.assertEqual(student_mask(cache['freq'], first_student).sum() > 0, True)
        self.assertTrue((cache['freq'].loc[student_mask(cache['freq'], first_student), '반'] == '1').all())


if __name__ == '__main__':
    unittest.main()
