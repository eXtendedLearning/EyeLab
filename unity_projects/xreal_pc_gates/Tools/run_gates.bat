@echo off
rem EyeLab XREAL PC-beta gate launcher (docs/XREAL-PC-GATES-G1-G3.md section 6).
rem Copied into Builds\Win64\ by BuildGates.BuildAll. Unzip the build to C:\EyeLabGates\.
setlocal EnableExtensions
cd /d "%~dp0"

set "EXE=%~dp0GateHarness\XrealPcGates.exe"
set "VENDOR=%~dp0VendorRGBCamera\VendorRGBCamera.exe"
rem Application.persistentDataPath for company EyeLab / product XrealPcGates.
set "RUNROOT=%USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates\GateRuns"
if not exist "%RUNROOT%" mkdir "%RUNROOT%"

:menu
echo.
echo ===== EyeLab XREAL PC gates =====
echo  Close Nebula and every XREAL app first. Decline any firmware update.
echo  Run folder: %RUNROOT%
echo.
echo  1  G1   camera only (no XR)
echo  2  G2   6DoF (Enter XR)
echo  3  G3   camera + 6DoF, XR first
echo  4  G3b  camera + 6DoF, camera first
echo  5  Manual (keyboard)
echo  6  Vendor RGBCamera sample (control, unmodified)
echo  7  Open run folder
echo  8  Zip all runs for transfer
echo  Q  Quit
echo.
choice /c 12345678Q /n /m "Select: "
set "SEL=%errorlevel%"
if "%SEL%"=="1" call :gate G1 g1 & goto menu
if "%SEL%"=="2" call :gate G2 g2 & goto menu
if "%SEL%"=="3" call :gate G3 g3_xrfirst -order xrfirst & goto menu
if "%SEL%"=="4" call :gate G3 g3b_camfirst -order camfirst & goto menu
if "%SEL%"=="5" call :gate manual manual & goto menu
if "%SEL%"=="6" call :vendor & goto menu
if "%SEL%"=="7" start "" explorer "%RUNROOT%" & goto menu
if "%SEL%"=="8" call :zipruns & goto menu
goto :eof

rem %1 gate, %2 label, %3 %4 optional extra args (-order ...)
:gate
call :stamp
set "PLOG=%RUNROOT%\player_%2_%TS%.log"
echo Running %1 (%2) ... player log: %PLOG%
start "" /wait "%EXE%" -gate %1 -label %2 %3 %4 -logFile "%PLOG%"
echo %1 (%2) finished, exit code %errorlevel%.
exit /b 0

:vendor
call :stamp
set "PLOG=%RUNROOT%\player_vendor_%TS%.log"
echo Running vendor RGBCamera sample ... player log: %PLOG%
echo  Note: its Enter XR flow may restart the exe (vendor second-window behaviour).
start "" /wait "%VENDOR%" -logFile "%PLOG%"
echo Vendor sample finished, exit code %errorlevel%.
exit /b 0

:zipruns
call :stamp
set "ZIP=%~dp0GateRuns_%COMPUTERNAME%_%TS%.zip"
powershell -NoProfile -Command "Compress-Archive -Path '%RUNROOT%\*' -DestinationPath '%ZIP%' -Force"
if exist "%ZIP%" (
  echo Wrote %ZIP%
  start "" explorer /select,"%ZIP%"
) else (
  echo Zip failed.
)
exit /b 0

:stamp
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
exit /b 0
