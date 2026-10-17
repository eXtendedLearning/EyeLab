@echo off
rem EyeLab XREAL PC-beta gate launcher (docs/XREAL-PC-GATES-G1-G3.md sections 13-17).
rem Copied into Builds\Win64\ by BuildGates.BuildAll. Every gate is guided by voice, beeps and a
rem large banner in the glasses; do not click in the app.
setlocal EnableExtensions
cd /d "%~dp0"

set "EXE=%~dp0GateHarness\XrealPcGates.exe"
set "VENDOR=%~dp0VendorRGBCamera\VendorRGBCamera.exe"
rem Application.persistentDataPath for company EyeLab / product XrealPcGates.
set "RUNROOT=%USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates\GateRuns"
if not exist "%RUNROOT%" mkdir "%RUNROOT%"
rem Eye calibration written by C and read by the G5 pose service (outside RUNROOT so Z never moves it).
set "CALIB=%USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates\calibration\eye_calibration.yaml"
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
echo  Windows power mode "Best performance", Surface standing in a cool place, sound on,
echo  Rotation lock ON (Settings, System, Display): the 14 Oct G2 showed the XR view upside down.
echo  Between XR runs use K (cool-down check): 20 s load + 20 s rest per check, "Cool enough" when ready.
echo  Run folder: %RUNROOT%
echo.
if exist "%CALIB%" (echo  Eye calibration: %CALIB%) else (echo  Eye calibration: none yet - run C first, G5 would use the vendor values)
echo.
echo  C  Calibrate    capture with the 25 mm board, then compute       (about 2 min + 1 min)
echo  A  G5-plateA    grid locked on plate A by the camera              (about 4 min)
echo  L  G5-plateL    grid locked on plate L by the camera              (about 4 min)
echo  B  G5-board25   grid locked on the 25 mm ChArUco board             (about 4 min)
echo  1  G1           camera only, no XR                                 (about 2 min)
echo  2  G4-6DoF      light: grid on the board, camera burst every 2 s   (about 6 min)
echo  3  G4-3DoF      as 2 in 3DoF (lightest tracking)                   (about 6 min)
echo  4  G4-nocam     as 2, camera never started (control)               (about 6 min)
echo  5  G2           6DoF tracking only                                 (about 4.5 min)
echo  6  G3c          camera + 6DoF, full video, after TRACKING READY    (about 3 min)
echo  7  G3c-30fps    as 6, XR rendered at 30 fps                        (about 3 min)
echo  8  G3c-flush    as 6, camera restarts when lag exceeds 1 s         (about 3 min)
echo  9  G3-3DoF      full video + XR in 3DoF                            (about 3 min)
echo  D  G3c-drain    as 6, latest-frame reader
echo  M  Manual       keyboard control
echo  K  Cool-down    sustained CPU clock check, run between XR gates   (until cool, max 10 min)
echo  V  Vendor       XREAL RGBCamera sample (control, unmodified)
echo  O  Open run folder
echo  Z  Zip NEW runs for transfer (exported runs move to _exported)
echo  Q  Quit
echo.
choice /c CALB123456789DMKVOZQ /n /m "Select: "
set "SEL=%errorlevel%"
if "%SEL%"=="1" call :calib & goto menu
if "%SEL%"=="2" call :g5 plateA & goto menu
if "%SEL%"=="3" call :g5 plateL & goto menu
if "%SEL%"=="4" call :g5 board25 & goto menu
if "%SEL%"=="5" call :gate G1 g1 & goto menu
if "%SEL%"=="6" call :gate G4 g4_6dof -order tracked -burstEvery 2 & goto menu
if "%SEL%"=="7" call :gate G4 g4_3dof -order 3dof -burstEvery 2 & goto menu
if "%SEL%"=="8" call :gate G4 g4_nocam -order tracked -burstEvery 0 & goto menu
if "%SEL%"=="9" call :gate G2 g2 & goto menu
if "%SEL%"=="10" call :gate G3 g3c -order tracked & goto menu
if "%SEL%"=="11" call :gate G3 g3c_30fps -order tracked -renderFps 30 & goto menu
if "%SEL%"=="12" call :gate G3 g3c_flush -order tracked -flushLag 1 & goto menu
if "%SEL%"=="13" call :gate G3 g3_3dof -order 3dof & goto menu
if "%SEL%"=="14" call :gate G3 g3c_drain -order tracked -reader drain & goto menu
if "%SEL%"=="15" call :gate manual manual & goto menu
if "%SEL%"=="16" call :gate cooldown cooldown & goto menu
if "%SEL%"=="17" call :vendor & goto menu
if "%SEL%"=="18" start "" explorer "%RUNROOT%" & goto menu
if "%SEL%"=="19" call :zipruns & goto menu
goto :eof

rem ---- Eye calibration: capture (camera only), then python -m xreal_gates calibrate on the newest run.
:calib
call :gate calib calib
call :findpython
if not defined PY exit /b 0
echo.
echo Computing the Eye calibration (python -m xreal_gates calibrate) ...
pushd "%REPO%\python"
"%PY%" -m xreal_gates calibrate --latest-in "%RUNROOT%" --copy-to "%CALIB%"
set "CALRC=%errorlevel%"
popd
if "%CALRC%"=="0" (echo Calibration ready. G5 will use it.) else (echo Calibration FAILED - see above. Repeat C, covering the corners and edges.)
pause
exit /b 0

rem ---- G5: start the pose service in its own window, then the gate. %1 = plateA, plateL or board25.
:g5
call :findpython
if not defined PY exit /b 0
echo Starting the pose service for %1 (window "EyeLab pose service") ...
if exist "%CALIB%" (set "CALARG=--calib "%CALIB%"") else (set "CALARG=")
start "EyeLab pose service" /min /d "%REPO%\python" "%PY%" -m xreal_gates serve --target %1 --idle-timeout 120 %CALARG%
call :gate G5 g5_%1 -target %1
exit /b 0

rem Python with OpenCV from the pulled repo: its venv first, else the one on PATH.
:findpython
set "PY="
if not defined REPO (
  echo  Repo not found: set EYELAB_REPO to your EyeLab checkout. Calibration and G5 need its python folder.
  pause
  exit /b 0
)
for %%p in ("%REPO%\venv\Scripts\python.exe" "%REPO%\.venv\Scripts\python.exe" "%REPO%\python\venv\Scripts\python.exe" "%REPO%\python\.venv\Scripts\python.exe") do (
  if not defined PY if exist "%%~p" set "PY=%%~p"
)
if not defined PY for /f "delims=" %%p in ('where python 2^>nul') do if not defined PY set "PY=%%p"
if not defined PY (
  echo  Python not found. Install Python 3 and: pip install -r "%REPO%\python\requirements.txt"
  pause
)
exit /b 0

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
