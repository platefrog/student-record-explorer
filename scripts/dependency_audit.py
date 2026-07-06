#!/usr/bin/env python3
"""Dependency checks for source/runtime and PyInstaller build outputs."""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CRITICAL_IMPORTS = (
    'streamlit',
    'pandas',
    'sklearn',
    'wordcloud',
    'matplotlib',
    'kiwipiepy',
    'kiwipiepy_model',
    'requests',
    'openpyxl',
)
REQUIRED_SPEC_PACKAGES = (
    'streamlit',
    'pandas',
    'sklearn',
    'wordcloud',
    'matplotlib',
    'kiwipiepy',
    'kiwipiepy_model',
    'requests',
    'openpyxl',
)
REQUIRED_REQUIREMENTS = (
    'streamlit',
    'pandas',
    'scikit-learn',
    'wordcloud',
    'matplotlib',
    'kiwipiepy',
    'kiwipiepy-model',
    'requests',
    'openpyxl',
    'pyinstaller',
)


def check_requirements() -> list[str]:
    errors: list[str] = []
    text = (ROOT / 'requirements.txt').read_text(encoding='utf-8').lower()
    for item in REQUIRED_REQUIREMENTS:
        if item.lower() not in text:
            errors.append(f'requirements.txt missing dependency marker: {item}')
    return errors


def check_spec() -> list[str]:
    errors: list[str] = []
    text = (ROOT / 'StudentRecordExplorer.spec').read_text(encoding='utf-8')
    for item in REQUIRED_SPEC_PACKAGES:
        if f"'{item}'" not in text:
            errors.append(f'StudentRecordExplorer.spec missing collect_all package: {item}')
    return errors


def check_imports() -> list[str]:
    errors: list[str] = []
    for module in CRITICAL_IMPORTS:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f'Cannot import {module}: {type(exc).__name__}: {exc}')

    if not any(err.startswith('Cannot import kiwipiepy') for err in errors):
        try:
            from kiwipiepy import Kiwi  # type: ignore

            kiwi = Kiwi()
            kiwi.tokenize('학생 기록 분석 테스트')
        except Exception as exc:
            errors.append(f'Kiwi runtime check failed: {type(exc).__name__}: {exc}')
    return errors


def check_build(version: str) -> list[str]:
    errors: list[str] = []
    dist_root = ROOT / 'dist' / f'StudentRecordExplorer-{version}'
    internal = dist_root / '_internal'
    expected_paths = (
        dist_root / 'StudentRecordExplorer.exe',
        internal / 'app.py',
        internal / 'VERSION',
        internal / 'data' / 'stopwords.txt',
        internal / 'data' / 'synonyms.txt',
    )
    for path in expected_paths:
        if not path.exists():
            errors.append(f'Missing build artifact: {path.relative_to(ROOT)}')

    expected_module_files = {
        'streamlit': internal / 'streamlit' / '__init__.py',
        'pandas': internal / 'pandas' / '__init__.py',
        'sklearn': internal / 'sklearn' / '__init__.py',
        'wordcloud': internal / 'wordcloud' / '__init__.py',
        'matplotlib': internal / 'matplotlib' / '__init__.py',
        'kiwipiepy': internal / 'kiwipiepy' / '__init__.py',
        'kiwipiepy_model': internal / 'kiwipiepy_model' / '__init__.py',
        'requests': internal / 'requests' / '__init__.py',
        'openpyxl': internal / 'openpyxl' / '__init__.py',
    }
    for module, marker in expected_module_files.items():
        if not marker.exists():
            errors.append(f'Bundled dependency marker missing: {module} ({marker.relative_to(ROOT)})')
    return errors


def run_source_checks() -> list[str]:
    errors: list[str] = []
    errors.extend(check_requirements())
    errors.extend(check_spec())
    errors.extend(check_imports())
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-source', action='store_true')
    parser.add_argument('--check-build', action='store_true')
    parser.add_argument('--version', default='')
    args = parser.parse_args()

    if not args.check_source and not args.check_build:
        args.check_source = True

    errors: list[str] = []
    if args.check_source:
        errors.extend(run_source_checks())
    if args.check_build:
        if not args.version:
            errors.append('--check-build requires --version')
        else:
            errors.extend(check_build(args.version))

    if errors:
        print('Dependency audit failed:', file=sys.stderr)
        for item in errors:
            print(f'- {item}', file=sys.stderr)
        return 1
    print('Dependency audit passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
