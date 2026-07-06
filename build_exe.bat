@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUTF8=1
set "PYTHON=.venv\Scripts\python.exe"
set /p APP_VERSION=<VERSION
set "DIST_NAME=StudentRecordExplorer-%APP_VERSION%"

if not exist "%PYTHON%" (
    echo [Setup] Creating the build virtual environment...
    py -3.14 -m venv .venv
    if errorlevel 1 goto ERROR
)

echo [1/6] Installing build requirements...
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto ERROR

echo [2/6] Auditing dependency readiness...
"%PYTHON%" scripts\dependency_audit.py --check-source
if errorlevel 1 goto ERROR

echo [3/6] Verifying source and cleaning old build artifacts...
"%PYTHON%" scripts\release_verify.py --check
if errorlevel 1 goto ERROR
"%PYTHON%" scripts\release_verify.py --clean
if errorlevel 1 goto ERROR

echo [4/6] Building StudentRecordExplorer.exe...
"%PYTHON%" -m PyInstaller StudentRecordExplorer.spec --clean --noconfirm
if errorlevel 1 goto ERROR

echo [5/6] Validating packaged dependencies and creating portable ZIP...
"%PYTHON%" scripts\dependency_audit.py --check-build --version %APP_VERSION%
if errorlevel 1 goto ERROR
"%PYTHON%" scripts\release_verify.py --archive
if errorlevel 1 goto ERROR

echo [6/6] Creating installer and verifying artifacts...
set "ISCC="
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if defined ISCC (
    "%ISCC%" "/DMyAppVersion=%APP_VERSION%" installer.iss
    if errorlevel 1 goto ERROR
) else (
    echo Inno Setup is not installed. The portable ZIP was created; installer compilation was skipped.
)

"%PYTHON%" scripts\release_verify.py --finalize
if errorlevel 1 goto ERROR

echo.
echo Build complete.
echo App: dist\%DIST_NAME%\StudentRecordExplorer.exe
echo Release: release
goto END

:ERROR
echo.
echo Build failed. Copy the error message and send it to ChatGPT.
:END
if not defined SRE_BUILD_AUTOMATED pause
