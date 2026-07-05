@echo off
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONUTF8=1
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if not exist ".venv\Scripts\python.exe" (
    echo [Setup] Creating the project virtual environment...
    py -3.14 -m venv .venv
    if errorlevel 1 goto :error
)

echo [Setup] Checking project dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless=false
if errorlevel 1 goto :error
goto :end

:error
echo.
echo Failed to start StudentRecord Explorer.

:end
pause
