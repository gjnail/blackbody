@echo off
rem Blackbody launcher for Windows: sets up a private Python environment on first run (and again when
rem requirements.txt changes), then opens the app. Commands such as render and --version run in this window.
setlocal
cd /d "%~dp0" || exit /b 1
rem (.venv\.installed is a copy of requirements.txt, written only once an install has worked, so a failed or
rem interrupted install is tried again next time instead of leaving an app that cannot start)
fc /b requirements.txt .venv\.installed >nul 2>nul || call :install || exit /b 1
set "console="
for %%c in (render simulate presets settings info precompile -h --help --version) do if /i "%~1"=="%%c" set "console=1"
if defined console (
    ".venv\Scripts\python.exe" -m blackbody %*
) else (
    start "" ".venv\Scripts\pythonw.exe" -m blackbody %*
)
exit /b %errorlevel%

:install
rem (the pinned packages have ready-built wheels for 64-bit Python 3.12 and 3.13 only)
set "check=import sys; sys.exit(not ((3, 12) <= sys.version_info[:2] <= (3, 13) and sys.maxsize > 2**32))"
if exist .venv\.installed del .venv\.installed
if not exist .venv goto :new_venv
".venv\Scripts\python.exe" -c "%check%" >nul 2>nul && ".venv\Scripts\python.exe" -c "import pip" >nul 2>nul && goto :packages
echo The Python environment in .venv cannot be used, so Blackbody is making a new one.
:new_venv
for %%p in ("py -3.13" "py -3.12" "py -3" python) do %%~p -c "%check%" >nul 2>nul && %%~p -m venv --clear .venv && goto :packages
echo.
echo Blackbody needs 64-bit Python 3.12 or 3.13: https://www.python.org/downloads/
echo Newer versions are not supported yet, as some of the packages it uses have no builds for them.
echo Python found on this computer:
py -0 2>nul || python --version 2>nul || echo   none
pause
exit /b 1
:packages
echo Setting up Blackbody. The first time, this downloads about 400 MB and takes a few minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed
rem (the app runs without a console, so a package that cannot load is reported here, not silently)
".venv\Scripts\python.exe" -c "import numpy, wgpu, PySide6.QtWidgets, av, OpenEXR, PIL.Image, mujoco, pxr.Usd, PyOpenColorIO" || goto :broken
copy /y requirements.txt .venv\.installed >nul || goto :failed
exit /b 0
:failed
echo.
echo Installing the dependencies failed: see the messages above. Starting Blackbody again tries again.
pause
exit /b 1
:broken
echo.
echo The dependencies installed, but Python cannot load them: see the messages above. If it says a DLL
echo failed to load, installing the latest Microsoft Visual C++ Redistributable (x64) usually fixes it.
pause
exit /b 1
