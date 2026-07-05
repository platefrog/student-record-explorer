# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

datas = [
    ('app.py', '.'),
    ('careernet_corpus.py', '.'),
    ('VERSION', '.'),
    ('data', 'data'),
]
hiddenimports = []
binaries = []

for pkg in ['streamlit', 'pandas', 'sklearn', 'wordcloud', 'matplotlib', 'kiwipiepy', 'requests', 'openpyxl']:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

try:
    datas += collect_data_files('streamlit')
except Exception:
    pass


a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StudentRecordExplorer',
    version='version_info.txt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StudentRecordExplorer-1.0.0',
)
