@echo off
setlocal
REM EyeLab campaign 0.5 - hits GUI: watches the campaign data folder (default python\campaign_runs),
REM analyses every new scan, photo or Beam Pro trial and shows the results as they arrive.
REM Guide: docs\tasks\T7.0-campaign-0.5.md section 14.5

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
python -m campaign gui %*
set "RC=%ERRORLEVEL%"

deactivate 2>nul
if not "%RC%"=="0" pause
endlocal & exit /b %RC%
