@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUTF8=1
set "PYTHON=.venv\Scripts\python.exe"
set "APP_VERSION=1.0.1"
set "DIST_NAME=StudentRecordExplorer-%APP_VERSION%"

if not exist "%PYTHON%" (
    echo [Setup] Creating the build virtual environment...
    py -3.14 -m venv .venv
    if errorlevel 1 goto ERROR
)

echo [1/4] Installing build requirements...
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto ERROR

echo [2/4] Building StudentRecordExplorer.exe...
"%PYTHON%" -m PyInstaller StudentRecordExplorer.spec --clean --noconfirm
if errorlevel 1 goto ERROR

echo [3/4] Creating portable ZIP...
if not exist release mkdir release
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path 'dist\%DIST_NAME%' -DestinationPath 'release\StudentRecordExplorer-Portable-%APP_VERSION%.zip' -CompressionLevel Optimal -Force"
if errorlevel 1 goto ERROR

echo [4/4] Looking for Inno Setup...
set "ISCC="
if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if defined ISCC (
    "%ISCC%" installer.iss
    if errorlevel 1 goto ERROR
) else (
    echo Inno Setup is not installed. The portable ZIP was created; installer compilation was skipped.
)

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
