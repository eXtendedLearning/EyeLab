param(
    [string]$PackagePath = "eyelab_xreal\Packages\com.xreal.xr"
)

# Renames the manifest package of the XREAL SDK 3.1.0 nr_common.aar from "nrsdk.pack" to "nrsdk.pack.common"
# (it clashes with another SDK AAR's package). Everything else in the AAR is copied byte for byte.
#
# The AAR is rebuilt entry by entry from the untouched backup (nr_common.aar.original), so entry names keep
# their forward slashes. Up to 0.3.012 this script used ZipFile.CreateFromDirectory, which on Windows
# PowerShell 5.1 writes "libs\framework.jar": the Android build then drops libs/framework.jar and the APK
# aborts at XR start with jni::NameResolutionException com/xreal/framework/net/binder/Connection
# (Beam Pro, 2026-11-17). Re-running this script repairs an AAR patched by the old version.

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedPackage = (Resolve-Path (Join-Path $repoRoot $PackagePath)).Path
$aarPath = Join-Path $resolvedPackage "Runtime\Plugins\Android\nr_common.aar"
$backupPath = "$aarPath.original"

if (-not (Test-Path $aarPath)) {
    throw "Missing XREAL AAR: $aarPath"
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Read-ManifestText([System.IO.Compression.ZipArchive]$zip) {
    $entry = $zip.GetEntry("AndroidManifest.xml")
    if ($null -eq $entry) { throw "AndroidManifest.xml missing in AAR" }
    $reader = New-Object System.IO.StreamReader($entry.Open(), [System.Text.Encoding]::UTF8)
    try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
}

function Test-PatchedAar([string]$path) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($path)
    try {
        $bad = @($zip.Entries | Where-Object { $_.FullName.Contains('\') })
        $text = Read-ManifestText $zip
        return ($bad.Count -eq 0) -and ($text -match 'package="nrsdk\.pack\.common"') -and ($null -ne $zip.GetEntry("libs/framework.jar"))
    } finally { $zip.Dispose() }
}

if (-not (Test-Path $backupPath)) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($aarPath)
    try { $text = Read-ManifestText $zip } finally { $zip.Dispose() }
    if ($text -notmatch 'package="nrsdk\.pack"') {
        throw "No backup and $aarPath is not the vendor original; reinstall XREAL XR Plugin 3.1.0 and run again."
    }
    Copy-Item -LiteralPath $aarPath -Destination $backupPath
}

if (Test-PatchedAar $aarPath) {
    Write-Output "Already patched: $aarPath"
    exit 0
}

$tmp = "$aarPath.tmp"
if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Force }

$src = [System.IO.Compression.ZipFile]::OpenRead($backupPath)
try {
    $text = Read-ManifestText $src
    if ($text -notmatch 'package="nrsdk\.pack"') {
        throw 'Expected package="nrsdk.pack" not found in the backup manifest.'
    }
    $patched = $text -replace 'package="nrsdk\.pack"', 'package="nrsdk.pack.common"'

    $fs = [System.IO.File]::Open($tmp, [System.IO.FileMode]::CreateNew)
    $dst = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($e in $src.Entries) {
            $n = $dst.CreateEntry($e.FullName, [System.IO.Compression.CompressionLevel]::Optimal)
            $n.LastWriteTime = $e.LastWriteTime
            if ($e.FullName.EndsWith("/")) { continue }   # directory entry
            $out = $n.Open()
            try {
                if ($e.FullName -eq "AndroidManifest.xml") {
                    $bytes = (New-Object System.Text.UTF8Encoding($false)).GetBytes($patched)
                    $out.Write($bytes, 0, $bytes.Length)
                } else {
                    $in = $e.Open()
                    try { $in.CopyTo($out) } finally { $in.Dispose() }
                }
            } finally { $out.Dispose() }
        }
    } finally {
        $dst.Dispose()
        $fs.Dispose()
    }
} finally {
    $src.Dispose()
}

Move-Item -LiteralPath $tmp -Destination $aarPath -Force
if (-not (Test-PatchedAar $aarPath)) { throw "Patched AAR failed its own check: $aarPath" }

Write-Output "Patched: $aarPath"
Write-Output "Backup:  $backupPath"
