@echo off
rem EyeLab XREAL PC-beta gate launcher (docs/XREAL-PC-GATES-G1-G3.md sections 6 and 13).
rem Copied into Builds\Win64\ by BuildGates.BuildAll. Every gate is guided on screen; do not click in the app.
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
echo  Before each run: Nebula and XREAL apps closed, glasses on the Surface USB-C, mains power.
echo  Follow the yellow instruction on screen / in the glasses. A beep marks every new step.
echo  Run folder: %RUNROOT%
echo.
echo  1  G1        camera only, no XR                          (about 1.5 min)
echo  2  G2        6DoF tracking, needs tape marks A/B          (about 4 min)
echo  3  G3c       camera + 6DoF, camera starts after READY     (about 3 min)
echo  4  G3c-drain same as 3 with the latest-frame reader       (about 3 min)
echo  5  G3-3DoF   camera + XR in 3DoF (no 6DoF tracking)       (about 2.5 min)
echo  6  G3-xr     camera right after the 6DoF switch (13 Oct)  (about 2.5 min)
echo  7  Manual    keyboard control
echo  8  Vendor    XREAL RGBCamera sample (control, unmodified)
echo  O  Open run folder
echo  Z  Zip all runs for transfer
echo  Q  Quit
echo.
choice /c 12345678OZQ /n /m "Select: "
set "SEL=%errorlevel%"
if "%SEL%"=="1" call :gate G1 g1 & goto menu
if "%SEL%"=="2" call :gate G2 g2 & goto menu
if "%SEL%"=="3" call :gate G3 g3c -order tracked & goto menu
if "%SEL%"=="4" call :gate G3 g3c_drain -order tracked -reader drain & goto menu
if "%SEL%"=="5" call :gate G3 g3_3dof -order 3dof & goto menu
if "%SEL%"=="6" call :gate G3 g3_xrfirst -order xrfirst & goto menu
if "%SEL%"=="7" call :gate manual manual & goto menu
if "%SEL%"=="8" call :vendor & goto menu
if "%SEL%"=="9" start "" explorer "%RUNROOT%" & goto menu
if "%SEL%"=="10" call :zipruns & goto menu
goto :eof

rem %1 gate, %2 label, %3..%6 optional extra args
:gate
call :stamp
set "PLOG=%RUNROOT%\player_%2_%TS%.log"
echo Running %1 (%2) ... player log: %PLOG%
start "" /wait "%EXE%" -gate %1 -label %2 %3 %4 %5 %6 -logFile "%PLOG%"
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
