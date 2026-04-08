@echo off
setlocal

echo ============================================
echo   EyeLab - Phase 1 Webcam MVP Launcher
echo ============================================
echo.

REM ── Check Python ─────────────────────────────────────────────────────
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Verify Python version >= 3.9
python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python 3.9+ is required.
    python --version
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM ── Set paths ────────────────────────────────────────────────────────
set "PROJECT_DIR=%~dp0"
set "PYTHON_DIR=%PROJECT_DIR%python"
set "VENV_DIR=%PROJECT_DIR%venv"
set "REQ_FILE=%PYTHON_DIR%\requirements.txt"

REM ── Create venv if missing ──────────────────────────────────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
    echo.
)

REM ── Activate venv ────────────────────────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"
echo [OK] Virtual environment activated.
echo.

REM ── Install / update dependencies ───────────────────────────────────
echo Checking dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install --quiet -r "%REQ_FILE%"
pip install --quiet -r "%REQ_FILE%"
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.
echo.

REM ── Launch the GUI ──────────────────────────────────────────────────
echo Starting EyeLab GUI...
echo ============================================
echo.
python "%PYTHON_DIR%\eyelab_gui.py"

REM ── Deactivate on exit ──────────────────────────────────────────────
deactivate 2>nul
endlocal
