#!/usr/bin/env python3
"""Validate, clean, and describe Windows release artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.bat', '.iss', '.md', '.py', '.txt'}
IGNORED_DIRS = {'.git', '.venv', '__pycache__', 'build', 'dist', 'release'}
MOJIBAKE_MARKERS = ('?숈', '?먯', '?섏', '?ㅽ', '濡쒓', '援먯', '諛고룷', '짤 Park')


def version() -> str:
    value = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
    if not re.fullmatch(r'\d+\.\d+\.\d+', value):
        raise ValueError(f'Invalid VERSION value: {value!r}')
    return value


def text_files() -> list[Path]:
    return [
        path for path in ROOT.rglob('*')
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and not any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)
    ]


def check_source() -> None:
    current = version()
    errors: list[str] = []

    for path in text_files():
        relative = path.relative_to(ROOT)
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError as exc:
            errors.append(f'{relative}: invalid UTF-8 ({exc})')
            continue
        if '\ufffd' in text:
            errors.append(f'{relative}: contains Unicode replacement characters')
        if path.resolve() != Path(__file__).resolve():
            for marker in MOJIBAKE_MARKERS:
                if marker in text:
                    errors.append(f'{relative}: contains suspected mojibake marker {marker!r}')

    expected = {
        'README.md': (f'# 학생부 탐색기 v{current}', f'현재 버전: **v{current}**'),
        'CHANGELOG.md': (f'## v{current}',),
        'DEPLOYMENT.md': (f'StudentRecordExplorer-Portable-{current}.zip',),
        'installer_info_ko.txt': (f'학생부 탐색기 v{current}',),
        'version_info.txt': (
            f"filevers=({', '.join(current.split('.'))}, 0)",
            f"StringStruct('FileVersion', '{current}')",
            f"StringStruct('ProductVersion', '{current}')",
        ),
    }
    for name, needles in expected.items():
        text = (ROOT / name).read_text(encoding='utf-8')
        for needle in needles:
            if needle not in text:
                errors.append(f'{name}: missing version marker {needle!r}')

    app_text = (ROOT / 'app.py').read_text(encoding='utf-8')
    if '<div class="sre-footer">' in app_text:
        errors.append('app.py: footer must not depend on raw HTML')

    if errors:
        raise RuntimeError('Release source verification failed:\n- ' + '\n- '.join(errors))
    print(f'Source verification passed for v{current}.')


def safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise RuntimeError(f'Refusing to remove unsafe path: {resolved}')
    if resolved.exists():
        shutil.rmtree(resolved)


def clean() -> None:
    for name in ('build', 'dist', 'release'):
        safe_rmtree(ROOT / name)
    print('Removed prior build, dist, and release directories.')


def archive() -> None:
    current = version()
    dist_root = ROOT / 'dist'
    source = dist_root / f'StudentRecordExplorer-{current}'
    destination = ROOT / 'release' / f'StudentRecordExplorer-Portable-{current}.zip'
    if not source.is_dir():
        raise FileNotFoundError(f'Missing application directory: {source.relative_to(ROOT)}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(source.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(dist_root))
    print(f'Created {destination.relative_to(ROOT)} ({destination.stat().st_size:,} bytes).')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finalize() -> None:
    current = version()
    release_dir = ROOT / 'release'
    portable = release_dir / f'StudentRecordExplorer-Portable-{current}.zip'
    setup = release_dir / f'StudentRecordExplorer-Setup-{current}.exe'
    app_exe = ROOT / 'dist' / f'StudentRecordExplorer-{current}' / 'StudentRecordExplorer.exe'

    for path in (portable, app_exe):
        if not path.is_file():
            raise FileNotFoundError(f'Missing release artifact: {path.relative_to(ROOT)}')

    with zipfile.ZipFile(portable) as archive:
        version_name = f'StudentRecordExplorer-{current}/_internal/VERSION'
        app_name = f'StudentRecordExplorer-{current}/_internal/app.py'
        archived_version = archive.read(version_name).decode('utf-8').strip()
        archived_app = archive.read(app_name).decode('utf-8')
        if archived_version != current:
            raise RuntimeError(f'ZIP version mismatch: {archived_version!r} != {current!r}')
        if '<div class="sre-footer">' in archived_app:
            raise RuntimeError('ZIP contains the obsolete raw-HTML footer')
        bundled_data = {
            name.rsplit('/', 1)[-1]
            for name in archive.namelist()
            if name.startswith(f'StudentRecordExplorer-{current}/_internal/data/') and not name.endswith('/')
        }
        if bundled_data != {'stopwords.txt', 'synonyms.txt'}:
            raise RuntimeError(f'Unexpected bundled application data: {sorted(bundled_data)}')

    assets = [portable]
    if setup.is_file():
        assets.append(setup)
    asset_rows = [
        {
            'name': path.name,
            'bytes': path.stat().st_size,
            'sha256': sha256(path),
        }
        for path in assets
    ]

    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ['git', 'status', '--porcelain', '--untracked-files=no'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip())
    manifest = {
        'product': 'Student Record Explorer',
        'version': current,
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': commit,
        'source_dirty': dirty,
        'assets': asset_rows,
    }
    info_path = release_dir / 'BUILD-INFO.json'
    info_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    checksum_paths = assets + [info_path]
    checksum_text = ''.join(f'{sha256(path).upper()}  {path.name}\n' for path in checksum_paths)
    (release_dir / 'SHA256SUMS.txt').write_text(checksum_text, encoding='utf-8')
    print(f'Artifact verification passed for v{current}.')


def main() -> int:
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--check', action='store_true')
    actions.add_argument('--clean', action='store_true')
    actions.add_argument('--archive', action='store_true')
    actions.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    if args.check:
        check_source()
    elif args.clean:
        clean()
    elif args.archive:
        archive()
    else:
        finalize()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
