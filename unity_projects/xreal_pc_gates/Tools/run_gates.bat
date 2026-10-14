@echo off
rem EyeLab XREAL PC-beta gate launcher (docs/XREAL-PC-GATES-G1-G3.md sections 13-15).
rem Copied into Builds\Win64\ by BuildGates.BuildAll. Every gate is guided by voice, beeps and a
rem large banner in the glasses; do not click in the app.
setlocal EnableExtensions
cd /d "%~dp0"

set "EXE=%~dp0GateHarness\XrealPcGates.exe"
set "VENDOR=%~dp0VendorRGBCamera\VendorRGBCamera.exe"
rem Application.persistentDataPath for company EyeLab / product XrealPcGates.
set "RUNROOT=%USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates\GateRuns"
if not exist "%RUNROOT%" mkdir "%RUNROOT%"
set "BUILD=unknown"
for /f "tokens=2 delims== " %%a in ('findstr /b /c:"utc_build" "%~dp0BUILD_INFO.txt" 2^>nul') do set "BUILD=%%a"

rem ---- Is this the newest build? Compare with LATEST_BUILD.txt in the pulled repo.
set "REPO="
for %%r in ("%EYELAB_REPO%" "C:\github_repos\EyeLab" "C:\GitHub_Uni\EyeLab" "%USERPROFILE%\source\repos\EyeLab") do (
  if not defined REPO if exist "%%~r\unity_projects\xreal_pc_gates\LATEST_BUILD.txt" set "REPO=%%~r"
)
set "LATEST=unknown"
if defined REPO for /f "tokens=2 delims== " %%a in ('findstr /b /c:"utc_build" "%REPO%\unity_projects\xreal_pc_gates\LATEST_BUILD.txt"') do set "LATEST=%%a"

:menu
echo.
echo ===== EyeLab XREAL PC gates =====  build %BUILD%
if not defined REPO (
  echo  [?] Repo not found, cannot check for a newer build. Set EYELAB_REPO to your EyeLab checkout.
) else if "%LATEST%"=="%BUILD%" (
  echo  [OK] This is the newest build per %REPO% ^(after git pull^).
) else (
  echo.
  echo  ##########################################################################
  echo  ##  WARNING: this build is %BUILD%
  echo  ##  the repo says the newest build is %LATEST%
  echo  ##  git pull, then copy the newest Builds\XrealPcGates_*.zip from the desktop
  echo  ##  and unzip it over this folder before running gates.
  echo  ##########################################################################
)
echo  Check: the same build time is shown small at the bottom left of the app.
echo  Before each run: Nebula and XREAL apps closed, glasses on the Surface USB-C, mains power,
echo  Windows power mode "Best performance", Surface standing in a cool place, sound on.
echo  If the app shows "SURFACE THROTTLED", let it cool for a few minutes before the next run.
echo  Run folder: %RUNROOT%
echo.
echo  1  G1         camera only, no XR                          (about 2 min)
echo  2  G3c        camera + 6DoF, camera after TRACKING READY  (about 3 min)
echo  3  G3c-30fps  as 2, XR rendered at 30 fps (lighter load)  (about 3 min)
echo  4  G3c-flush  as 2, camera restarts when lag exceeds 1 s  (about 3 min)
echo  5  G2         6DoF tracking only                          (about 4.5 min)
echo  6  G3-3DoF    camera + XR in 3DoF (no 6DoF tracking)      (about 3 min)
echo  7  G3c-drain  as 2, latest-frame reader                   (about 3 min)
echo  8  Manual     keyboard control
echo  V  Vendor     XREAL RGBCamera sample (control, unmodified)
echo  O  Open run folder
echo  Z  Zip NEW runs for transfer (exported runs move to _exported)
echo  Q  Quit
echo.
choice /c 12345678VOZQ /n /m "Select: "
set "SEL=%errorlevel%"
if "%SEL%"=="1" call :gate G1 g1 & goto menu
if "%SEL%"=="2" call :gate G3 g3c -order tracked & goto menu
if "%SEL%"=="3" call :gate G3 g3c_30fps -order tracked -renderFps 30 & goto menu
if "%SEL%"=="4" call :gate G3 g3c_flush -order tracked -flushLag 1 & goto menu
if "%SEL%"=="5" call :gate G2 g2 & goto menu
if "%SEL%"=="6" call :gate G3 g3_3dof -order 3dof & goto menu
if "%SEL%"=="7" call :gate G3 g3c_drain -order tracked -reader drain & goto menu
if "%SEL%"=="8" call :gate manual manual & goto menu
if "%SEL%"=="9" call :vendor & goto menu
if "%SEL%"=="10" start "" explorer "%RUNROOT%" & goto menu
if "%SEL%"=="11" call :zipruns & goto menu
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

rem Zip only what has not been exported yet, then move it to _exported so the next zip is small.
:zipruns
call :stamp
set "ZIP=%~dp0GateRuns_%COMPUTERNAME%_%TS%.zip"
powershell -NoProfile -Command "$r='%RUNROOT%'; $items = @(Get-ChildItem -LiteralPath $r | Where-Object { $_.Name -ne '_exported' }); if ($items.Count -eq 0) { Write-Host 'Nothing new to export.'; exit 1 }; Compress-Archive -LiteralPath $items.FullName -DestinationPath '%ZIP%' -Force; $e = Join-Path $r '_exported'; New-Item -ItemType Directory -Force -Path $e | Out-Null; $items | Move-Item -Destination $e; Write-Host ('Exported ' + $items.Count + ' items.')"
if exist "%ZIP%" (
  echo Wrote %ZIP%
  echo Copy it to the desktop into unity_projects\xreal_pc_gates\GateRuns\_inbox\
  start "" explorer /select,"%ZIP%"
) else (
  echo No zip written.
)
exit /b 0

:stamp
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
exit /b 0
