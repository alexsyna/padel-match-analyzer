@echo off
rem Double-click to start the Padel Analyzer (Windows). First run sets everything up.
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
  echo During setup tick "Add python.exe to PATH", then double-click this file again.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up the analyzer. This takes a few minutes...
  %PY% -m venv .venv
)
if not exist ".venv\installed.flag" (
  ".venv\Scripts\python" -m pip install --upgrade pip >nul
  ".venv\Scripts\python" -m pip install -e ".[yolo,llm]" || ".venv\Scripts\python" -m pip install -e ".[llm]"
  if errorlevel 1 (
    echo Installation failed. See the messages above.
    pause
    exit /b 1
  )
  echo ok> ".venv\installed.flag"
)
".venv\Scripts\python" -m padelfusion ui
pause
