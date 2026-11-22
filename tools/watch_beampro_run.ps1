<#
Waits for the next Beam Pro probe run and pulls it, file by file, into python/beampro_runs/BeamProRuns/<run>.

Used by an agent as a background job: it costs nothing while it waits, and the agent is woken when it exits.
It only touches the EyeLab app's own storage (AGENTS.md, Unity). Exit 0 = run pulled, 2 = timed out.

  powershell -File tools/watch_beampro_run.ps1 -Port 35833 [-Ip 192.168.1.14] [-TimeoutMin 30]
#>
param(
    [Parameter(Mandatory = $true)][int]$Port,
    [string]$Ip = "192.168.1.14",
    [int]$TimeoutMin = 30,
    [int]$PollS = 10
)
$ErrorActionPreference = "Continue"
$adb = "$env:ProgramFiles\Unity\Hub\Editor\6000.4.5f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe"
$dev = "${Ip}:$Port"
$remote = "/sdcard/Android/data/it.univpm.eyelab/files/BeamProRuns"
$local = Join-Path $PSScriptRoot "..\python\beampro_runs\BeamProRuns"
$known = @(Get-ChildItem $local -Directory | ForEach-Object Name)
$deadline = (Get-Date).AddMinutes($TimeoutMin)

function Write-Stamp($m) { Write-Output ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $m) }
function Invoke-Adb { & $adb -s $dev @args 2>&1 }

$state = ""
while ((Get-Date) -lt $deadline) {
    $online = (& $adb devices) -match [regex]::Escape($dev) + "\s+device"
    if (-not $online) {
        & $adb connect $dev 2>&1 | Out-Null
        $online = (& $adb devices) -match [regex]::Escape($dev) + "\s+device"
    }
    if (-not $online) {
        if ($state -ne "offline") { Write-Stamp "waiting for $dev"; $state = "offline" }
        Start-Sleep -Seconds $PollS; continue
    }
    $runs = @(Invoke-Adb shell ls $remote | ForEach-Object { "$_".Trim() } |
              Where-Object { $_ -match '^\d{8}_\d{6}_\w+$' -and $known -notcontains $_ } | Sort-Object)
    if ($runs.Count -eq 0) {
        if ($state -ne "norun") { Write-Stamp "connected, no new run yet"; $state = "norun" }
        Start-Sleep -Seconds $PollS; continue
    }
    $run = $runs[-1]
    $ended = "$(Invoke-Adb shell grep -c '^cue,end' "$remote/$run/events.csv")".Trim()
    $json = "$(Invoke-Adb shell ls "$remote/$run/run.json")" -notmatch "No such file"
    if ($ended -notmatch '^[1-9]' -or -not $json) {
        if ($state -ne "running $run") { Write-Stamp "run $run in progress"; $state = "running $run" }
        Start-Sleep -Seconds $PollS; continue
    }
    Start-Sleep -Seconds 3  # let the last writes land
    $dst = Join-Path $local $run
    New-Item -ItemType Directory -Force (Join-Path $dst "frames") | Out-Null
    $files = @(Invoke-Adb shell ls "$remote/$run" | ForEach-Object { "$_".Trim() } | Where-Object { $_ -and $_ -ne "frames" })
    foreach ($f in $files) { Invoke-Adb pull "$remote/$run/$f" (Join-Path $dst $f) | Out-Null }
    $frames = @(Invoke-Adb shell ls "$remote/$run/frames" | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
    foreach ($f in $frames) { Invoke-Adb pull "$remote/$run/frames/$f" (Join-Path $dst "frames\$f") | Out-Null }
    $logs = @(Invoke-Adb shell ls $remote | ForEach-Object { "$_".Trim() } | Where-Object { $_ -like "app_log_*" })
    foreach ($l in $logs) { if (-not (Test-Path (Join-Path $local $l))) { Invoke-Adb pull "$remote/$l" (Join-Path $local $l) | Out-Null } }
    $got = @(Get-ChildItem $dst -File).Count
    $gotFrames = @(Get-ChildItem (Join-Path $dst "frames") -File).Count
    Write-Stamp "pulled $run : $got of $($files.Count) files, $gotFrames of $($frames.Count) frames"
    Write-Output "RUN=$run"
    exit 0
}
Write-Stamp "timed out after $TimeoutMin min"
exit 2
