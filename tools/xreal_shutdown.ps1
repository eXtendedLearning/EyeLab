<#
.SYNOPSIS
    Stop everything that talks to the XREAL glasses, verify it, then confirm
    the glasses are physically unplugged.

.DESCRIPTION
    1. Kills EyeLab Python processes (xreal_native_probe, xreal_probe,
       display_check, eyelab_gui, webcam_pipeline) and any Nebula process.
    2. Kills any other process that still has an XREAL DLL loaded
       (libnr_api / libnr_loader / libnr_glasses_api / libnr_plugin_6dof /
       ov_utils), found with `tasklist /m`.
    3. Re-checks both, and reports PASS or FAIL.
    4. Waits for the glasses (USB VID 3318) to disappear from Windows PnP,
       i.e. for you to unplug them, and confirms it.

    Read-only towards the glasses: nothing is sent to them, no device is
    disabled or removed in Device Manager. Unplugging is the only reliable
    "disconnect" - a software disable would persist until re-enabled.

    Run via xreal_shutdown.bat (double-click). No admin needed, except to
    kill processes that were themselves started elevated.

.PARAMETER NoWait
    Do not wait for the glasses to be unplugged; just report.
.PARAMETER UnplugTimeoutS
    How long to wait for the unplug. Default 120 s.
#>
param(
    [switch]$NoWait,
    [int]$UnplugTimeoutS = 120
)

$ErrorActionPreference = 'Continue'
$XrealModules = @('libnr_api.dll', 'libnr_loader.dll', 'libnr_glasses_api.dll',
                  'libnr_plugin_6dof.dll', 'ov_utils.dll')
$EyeLabScripts = 'xreal_native_probe|xreal_probe|display_check|eyelab_gui|webcam_pipeline'
$VendorProcesses = @('Nebula for Windows')

function Get-EyeLabPython {
    Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match $EyeLabScripts } |
        ForEach-Object { [pscustomobject]@{ Id = [int]$_.ProcessId; Name = $_.Name; Why = 'EyeLab ' + [regex]::Match($_.CommandLine, $EyeLabScripts).Value } }
}

function Get-ModuleHolders {
    $found = @{}
    foreach ($m in $XrealModules) {
        foreach ($row in (tasklist /m $m /fo csv /nh 2>$null)) {
            if ($row -match '^"([^"]+)","(\d+)"') {
                $procId = [int]$Matches[2]
                if (-not $found.ContainsKey($procId)) {
                    $found[$procId] = [pscustomobject]@{ Id = $procId; Name = $Matches[1]; Why = "has $m loaded" }
                }
            }
        }
    }
    $found.Values
}

function Get-VendorProcesses {
    Get-Process -Name $VendorProcesses -ErrorAction SilentlyContinue |
        ForEach-Object { [pscustomobject]@{ Id = $_.Id; Name = $_.ProcessName; Why = 'XREAL vendor app' } }
}

function Get-Offenders {
    $all = @(Get-EyeLabPython) + @(Get-ModuleHolders) + @(Get-VendorProcesses) | Where-Object { $_ }
    $all | Sort-Object Id -Unique
}

function Get-Glasses {
    Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceId -like '*VID_3318*' }
}

Write-Host '== XREAL shutdown ==' -ForegroundColor Cyan

# --- 1-2. find and stop ------------------------------------------------------
$offenders = @(Get-Offenders)
if ($offenders.Count -eq 0) {
    Write-Host 'No EyeLab / XREAL processes running.'
} else {
    foreach ($p in $offenders) {
        Write-Host ("Stopping PID {0,-6} {1,-22} ({2})" -f $p.Id, $p.Name, $p.Why)
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 800
}

# --- 3. verify ---------------------------------------------------------------
$left = @(Get-Offenders)
$processesOk = ($left.Count -eq 0)
if ($processesOk) {
    Write-Host 'PASS  no process has an XREAL DLL loaded; no EyeLab probe running.' -ForegroundColor Green
} else {
    Write-Host 'FAIL  still running (try again from an elevated prompt):' -ForegroundColor Red
    $left | ForEach-Object { Write-Host ("      PID {0,-6} {1,-22} ({2})" -f $_.Id, $_.Name, $_.Why) }
}

# --- 4. glasses --------------------------------------------------------------
$glasses = @(Get-Glasses)
$glassesOk = ($glasses.Count -eq 0)
if ($glassesOk) {
    Write-Host 'PASS  glasses not present on USB.' -ForegroundColor Green
} elseif ($NoWait) {
    Write-Host ("INFO  glasses still plugged in ({0} PnP nodes). Unplug the cable to disconnect." -f $glasses.Count) -ForegroundColor Yellow
} else {
    Write-Host ("Glasses still plugged in ({0} PnP nodes). Unplug the USB-C cable now - waiting up to {1} s..." -f $glasses.Count, $UnplugTimeoutS) -ForegroundColor Yellow
    $deadline = (Get-Date).AddSeconds($UnplugTimeoutS)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 1
        if (@(Get-Glasses).Count -eq 0) { $glassesOk = $true; break }
    }
    if ($glassesOk) {
        Write-Host 'PASS  glasses disconnected.' -ForegroundColor Green
    } else {
        Write-Host 'FAIL  glasses still enumerated after the timeout.' -ForegroundColor Red
    }
}

if ($processesOk -and $glassesOk) {
    Write-Host 'All clear.' -ForegroundColor Green
    exit 0
}
exit 1
