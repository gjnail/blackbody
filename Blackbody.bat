@echo off
rem Blackbody launcher for Windows: sets up a private Python environment on first run, then opens the app.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Setting up Blackbody for the first time. This downloads about 400 MB and takes a few minutes...
    where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
    if errorlevel 1 (
        echo Python 3.10 or newer is required: https://www.python.org/downloads/
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing the dependencies failed. See the messages above.
        pause
        exit /b 1
    )
)
if "%~1"=="render" (
    ".venv\Scripts\python.exe" -m blackbody %*
) else (
    start "" ".venv\Scripts\pythonw.exe" -m blackbody %*
)
