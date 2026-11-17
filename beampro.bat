@echo off
setlocal EnableDelayedExpansion
REM EyeLab Beam Pro probes (T6.0 gate BS, T6.1 gate B1) - menu.
REM Guide: docs\tasks\T6.0-beampro-bringup.md and docs\tasks\T6.1-eye-chain-latency.md
REM Every adb command here is read-only on the device, or acts on the EyeLab app's own storage.

set "PROJECT_DIR=%~dp0"
set "PYTHON_DIR=%PROJECT_DIR%python"
set "VENV_PY=%PROJECT_DIR%venv\Scripts\python.exe"
set "RUNS_DIR=%PYTHON_DIR%\beampro_runs"
set "PKG=it.univpm.eyelab"
set "APP_FILES=/sdcard/Android/data/%PKG%/files"
set "UNITY_ADB=%ProgramFiles%\Unity\Hub\Editor\6000.4.5f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe"

set "ADB="
for /f "delims=" %%A in ('where adb 2^>nul') do if not defined ADB set "ADB=%%A"
if not defined ADB if exist "%UNITY_ADB%" set "ADB=%UNITY_ADB%"
if not defined ADB (
    echo ERROR: adb not found on PATH nor in the Unity 6000.4.5f1 Android SDK.
    pause
    exit /b 1
)
if not exist "%VENV_PY%" set "VENV_PY=python"
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"

:menu
echo.
echo ============================================
echo   EyeLab Beam Pro probes
echo   adb: %ADB%
echo ============================================
echo   1  Check the connection (adb devices)
echo   2  Inventory over adb (versions, cameras, sensors, displays) - read-only
echo   3  Install the newest APK from eyelab_xreal\build
echo   4  Choose the probe the app runs at next start (probe.json)
echo   5  Pull the runs from the Beam Pro
echo   6  Analyse the newest run (python -m xreal_gates bp)
echo   7  Analyse every pulled run
echo   Q  Quit
set "CH="
set /p "CH=Choice: "
if /i "%CH%"=="1" goto devices
if /i "%CH%"=="2" goto inventory
if /i "%CH%"=="3" goto install
if /i "%CH%"=="4" goto probe
if /i "%CH%"=="5" goto pull
if /i "%CH%"=="6" goto analyse_latest
if /i "%CH%"=="7" goto analyse_all
if /i "%CH%"=="Q" exit /b 0
goto menu

:devices
"%ADB%" devices -l
goto menu

:inventory
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "STAMP=%%T"
set "OUT=%RUNS_DIR%\adb_inventory_%STAMP%"
mkdir "%OUT%"
echo Writing to %OUT% ...
"%ADB%" devices -l > "%OUT%\adb_devices.txt"
"%ADB%" shell getprop > "%OUT%\getprop.txt"
"%ADB%" shell uname -a > "%OUT%\uname.txt"
"%ADB%" shell dumpsys media.camera > "%OUT%\dumpsys_media_camera.txt"
"%ADB%" shell dumpsys sensorservice > "%OUT%\dumpsys_sensorservice.txt"
"%ADB%" shell dumpsys display > "%OUT%\dumpsys_display.txt"
"%ADB%" shell dumpsys usb > "%OUT%\dumpsys_usb.txt"
"%ADB%" shell "pm list packages | grep -iE 'xreal|nreal|nebula|glasses|eyelab'" > "%OUT%\packages.txt"
"%ADB%" shell "for p in $(pm list packages | grep -iE 'xreal|nreal|nebula|glasses|eyelab' | cut -d: -f2); do echo == $p; dumpsys package $p | grep -E 'versionName|versionCode|lastUpdateTime'; done" > "%OUT%\package_versions.txt"
echo Done. Record the glasses firmware from MyGlasses by hand in %OUT%\glasses_firmware.txt
type "%OUT%\package_versions.txt"
goto menu

:install
set "APK="
for /f "delims=" %%F in ('dir /b /o-d "%PROJECT_DIR%eyelab_xreal\build\*.apk" 2^>nul') do if not defined APK set "APK=%PROJECT_DIR%eyelab_xreal\build\%%F"
if not defined APK (
    echo No APK in eyelab_xreal\build. Build it in Unity first ^(File - Build Profiles - Android^).
    goto menu
)
echo Installing %APK%
"%ADB%" install -r "%APK%"
goto menu

:probe
echo   a  inventory + IMU 20 s
echo   b  Eye chain 5 min
echo   c  timecode 60 s ^(Eye on the Beam Pro screen^)
echo   d  all three in a row
echo   x  none ^(the app shows buttons^)
set "P="
set /p "P=Probe: "
set "NAME="
if /i "%P%"=="a" set "NAME=inventory"
if /i "%P%"=="b" set "NAME=chain"
if /i "%P%"=="c" set "NAME=timecode"
if /i "%P%"=="d" set "NAME=all"
if /i "%P%"=="x" set "NAME="
> "%TEMP%\eyelab_probe.json" echo {"probe":"%NAME%"}
"%ADB%" push "%TEMP%\eyelab_probe.json" "%APP_FILES%/probe.json"
echo Next start of EyeLab runs: "%NAME%" ^(start it from MyGlasses^).
goto menu

:pull
"%ADB%" pull "%APP_FILES%/BeamProRuns" "%RUNS_DIR%"
goto menu

:analyse_latest
set "LATEST="
for /f "delims=" %%D in ('dir /b /ad /o-n "%RUNS_DIR%\BeamProRuns" 2^>nul') do if not defined LATEST set "LATEST=%RUNS_DIR%\BeamProRuns\%%D"
if not defined LATEST (
    echo No pulled runs. Use 5 first.
    goto menu
)
pushd "%PYTHON_DIR%"
"%VENV_PY%" -m xreal_gates bp "%LATEST%"
popd
goto menu

:analyse_all
pushd "%PYTHON_DIR%"
for /f "delims=" %%D in ('dir /b /ad /o-n "%RUNS_DIR%\BeamProRuns" 2^>nul') do "%VENV_PY%" -m xreal_gates bp "%RUNS_DIR%\BeamProRuns\%%D"
popd
goto menu
