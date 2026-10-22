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
if exist "%CALIB%" (echo  Eye calibration: %CALIB%) else (echo  Eye calibration: none yet - run C first, G5 and the demo would use the vendor values)
echo.
echo  D  DEMO         world grid + plate A and 25 mm board locked by the camera + Eye view, 3DoF (until Esc)
echo  C  Calibrate    capture with the 25 mm board, then compute       (about 2 min + 1 min)
echo  1  G1           camera only, no XR                                 (about 2 min)
echo  2  G2           6DoF tracking only                                 (about 4.5 min)
echo  3  G3 ...       camera + XR with full video (5 variants)
echo  4  G4 ...       light overlay, camera bursts (3 variants)
echo  5  G5 ...       overlay locked on a target by the camera, ALIGN, 6DoF demo
echo  K  Cool-down    sustained CPU clock check, run between XR gates   (until cool, max 10 min)
echo  M  Manual       keyboard control
echo  V  Vendor       XREAL RGBCamera sample (control, unmodified)
echo  O  Open run folder
echo  Z  Zip NEW runs for transfer (exported runs move to _exported)
echo  Q  Quit
echo.
choice /c DC12345KMVOZQ /n /m "Select: "
set "SEL=%errorlevel%"
if "%SEL%"=="1" call :demo 3dof & goto menu
if "%SEL%"=="2" call :calib & goto menu
if "%SEL%"=="3" call :gate G1 g1 & goto menu
if "%SEL%"=="4" call :gate G2 g2 & goto menu
if "%SEL%"=="5" goto menu_g3
if "%SEL%"=="6" goto menu_g4
if "%SEL%"=="7" goto menu_g5
if "%SEL%"=="8" call :gate cooldown cooldown & goto menu
if "%SEL%"=="9" call :gate manual manual & goto menu
if "%SEL%"=="10" call :vendor & goto menu
if "%SEL%"=="11" start "" explorer "%RUNROOT%" & goto menu
if "%SEL%"=="12" call :zipruns & goto menu
goto :eof

:menu_g3
echo.
echo ----- G3: camera + XR with full video -----
echo  1  G3c          camera + 6DoF, full video, after TRACKING READY    (about 3 min)
echo  2  G3c-30fps    as 1, XR rendered at 30 fps                        (about 3 min)
echo  3  G3c-flush    as 1, camera restarts when lag exceeds 1 s         (about 3 min)
echo  4  G3-3DoF      full video + XR in 3DoF                            (about 3 min)
echo  5  G3c-drain    as 1, latest-frame reader                          (about 3 min)
echo  B  Back
choice /c 12345B /n /m "Select: "
set "SEL=%errorlevel%"
if "%SEL%"=="1" call :gate G3 g3c -order tracked & goto menu_g3
if "%SEL%"=="2" call :gate G3 g3c_30fps -order tracked -renderFps 30 & goto menu_g3
if "%SEL%"=="3" call :gate G3 g3c_flush -order tracked -flushLag 1 & goto menu_g3
if "%SEL%"=="4" call :gate G3 g3_3dof -order 3dof & goto menu_g3
if "%SEL%"=="5" call :gate G3 g3c_drain -order tracked -reader drain & goto menu_g3
goto menu

:menu_g4
echo.
echo ----- G4: light overlay, camera bursts -----
echo  1  G4-6DoF      world grid in front of you, camera burst every 2 s (about 6 min)
echo  2  G4-3DoF      as 1 in 3DoF (lightest tracking)                   (about 6 min)
echo  3  G4-nocam     as 1, camera never started (control)               (about 6 min)
echo  B  Back
choice /c 123B /n /m "Select: "
set "SEL=%errorlevel%"
if "%SEL%"=="1" call :gate G4 g4_6dof -order tracked -burstEvery 2 & goto menu_g4
if "%SEL%"=="2" call :gate G4 g4_3dof -order 3dof -burstEvery 2 & goto menu_g4
if "%SEL%"=="3" call :gate G4 g4_nocam -order tracked -burstEvery 0 & goto menu_g4
goto menu

:menu_g5
echo.
echo ----- G5: overlay locked on a target by the camera (about 3 min each) -----
echo  Target about 70 cm in front of A; only one board with IDs 0-16 in view.
echo  1  plate A, 6DoF
echo  2  plate L, 6DoF
echo  3  25 mm board, 6DoF
echo  4  plate A, 3DoF       (no position tracking: immune to its drift)
echo  5  25 mm board, 3DoF
echo  6  plate A + 25 mm board together, 6DoF
echo  7  ALIGN on plate A    move the drawing onto the plate by eye, Enter saves (3DoF, until Esc)
echo  8  ALIGN on 25 mm board
echo  9  Demo in 6DoF        as D, with position tracking
echo  B  Back
echo  Saved view offset: %USERPROFILE%\AppData\LocalLow\EyeLab\XrealPcGates\calibration\view_offset.json
choice /c 123456789B /n /m "Select: "
set "SEL=%errorlevel%"
if "%SEL%"=="1" call :g5 plateA tracked & goto menu_g5
if "%SEL%"=="2" call :g5 plateL tracked & goto menu_g5
if "%SEL%"=="3" call :g5 board25 tracked & goto menu_g5
if "%SEL%"=="4" call :g5 plateA 3dof & goto menu_g5
if "%SEL%"=="5" call :g5 board25 3dof & goto menu_g5
if "%SEL%"=="6" call :g5 plateA+board25 tracked & goto menu_g5
if "%SEL%"=="7" call :align plateA & goto menu_g5
if "%SEL%"=="8" call :align board25 & goto menu_g5
if "%SEL%"=="9" call :demo tracked & goto menu_g5
goto menu

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
if "%CALRC%"=="0" (echo Calibration ready. G5 and the demo will use it.) else (echo Calibration FAILED - see above. Repeat C, covering the corners and edges.)
pause
exit /b 0

rem ---- Pose service in its own minimised window. %1 = targets joined by + (batch splits on commas).
:service
call :findpython
if not defined PY exit /b 1
set "TGT=%~1"
set "TGT=%TGT:+=,%"
echo Starting the pose service for %TGT% (window "EyeLab pose service") ...
if exist "%CALIB%" (set "CALARG=--calib "%CALIB%"") else (set "CALARG=")
start "EyeLab pose service" /min /d "%REPO%\python" "%PY%" -m xreal_gates serve --target %TGT% --idle-timeout 120 %CALARG%
exit /b 0

rem ---- G5: %1 targets (plateA, plateL, board25, or joined by +), %2 order (tracked = 6DoF, 3dof).
:g5
call :service %1
if errorlevel 1 exit /b 0
set "G5LABEL=g5_%~1"
if "%2"=="3dof" set "G5LABEL=g5_%~1_3dof"
set "G5LABEL=%G5LABEL:+=_%"
call :gate G5 %G5LABEL% -target "%TGT%" -order %2
exit /b 0

rem ---- Demo: plate A and the 25 mm board until Esc. %1 = 3dof or tracked (6DoF). Keys: G grid, P place grid, E Eye view, T targets.
:demo
call :service plateA+board25
if errorlevel 1 exit /b 0
echo  Demo keys on this keyboard while it runs: G grid on/off, P place grid where the wearer looks,
echo  E Eye view on/off, T target overlays on/off, arrows/PgUp/PgDn view offset (Enter saves), Esc end.
set "DLABEL=demo"
if "%1"=="3dof" set "DLABEL=demo_3dof"
call :gate demo %DLABEL% -target "%TGT%" -order %1
exit /b 0

rem ---- Align: move the overlay onto the target by eye (head-frame view offset), Enter saves it. %1 = target.
:align
call :service %1
if errorlevel 1 exit /b 0
echo  Align keys: arrows move, PgUp/PgDn farther/nearer, Ctrl+arrows and Ctrl+PgUp/PgDn rotate,
echo  Shift for big steps, Backspace zero, Enter SAVE, Esc end.
call :gate align align_%1 -target "%TGT%"
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
