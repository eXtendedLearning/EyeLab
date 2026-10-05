@echo off
setlocal
REM EyeLab validation campaign - menu for every step of the measurement
REM campaign (calibration, plate measurements, sessions, analysis).
REM Guide: docs\validation-campaign.md

echo ============================================
echo   EyeLab validation campaign
echo ============================================
echo.

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found on PATH.
    pause
    exit /b 1
)

set "PROJECT_DIR=%~dp0"
set "PYTHON_DIR=%PROJECT_DIR%python"
set "VENV_DIR=%PROJECT_DIR%venv"
set "REQ_FILE=%PYTHON_DIR%\requirements.txt"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)
call "%VENV_DIR%\Scripts\activate.bat"

echo Checking dependencies...
python -m pip install --quiet -r "%REQ_FILE%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

cd /d "%PYTHON_DIR%"
python -m validation campaign
set "RC=%ERRORLEVEL%"

deactivate 2>nul
if not "%RC%"=="0" pause
endlocal & exit /b %RC%
