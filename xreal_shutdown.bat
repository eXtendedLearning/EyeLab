@echo off
rem Stop everything that talks to the XREAL glasses, verify, and confirm unplug.
rem Logic lives in tools\xreal_shutdown.ps1 (pass -NoWait to skip the unplug wait).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\xreal_shutdown.ps1" %*
set RC=%ERRORLEVEL%
echo.
pause
exit /b %RC%
