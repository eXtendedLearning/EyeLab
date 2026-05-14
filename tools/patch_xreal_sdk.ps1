param(
    [string]$PackagePath = "eyelab_xreal\Packages\com.xreal.xr"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedPackage = (Resolve-Path (Join-Path $repoRoot $PackagePath)).Path
$aarPath = Join-Path $resolvedPackage "Runtime\Plugins\Android\nr_common.aar"
$backupPath = "$aarPath.original"

if (-not (Test-Path $aarPath)) {
    throw "Missing XREAL AAR: $aarPath"
}

$work = Join-Path $repoRoot ("tmp_xreal_patch_" + [guid]::NewGuid().ToString("N"))
$extract = Join-Path $work "extract"
$out = Join-Path $work "nr_common.patched.aar"

New-Item -ItemType Directory -Path $extract | Out-Null

try {
    if (-not (Test-Path $backupPath)) {
        Copy-Item -LiteralPath $aarPath -Destination $backupPath
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($aarPath, $extract)

    $manifest = Join-Path $extract "AndroidManifest.xml"
    $text = Get-Content -LiteralPath $manifest -Raw

    if ($text -match 'package="nrsdk\.pack\.common"') {
        Write-Output "Already patched: $aarPath"
        exit 0
    }

    if ($text -notmatch 'package="nrsdk\.pack"') {
        throw 'Expected package="nrsdk.pack" not found in nr_common.aar manifest.'
    }

    $text = $text -replace 'package="nrsdk\.pack"', 'package="nrsdk.pack.common"'
    Set-Content -LiteralPath $manifest -Value $text -NoNewline -Encoding UTF8

    [System.IO.Compression.ZipFile]::CreateFromDirectory($extract, $out)
    Copy-Item -LiteralPath $out -Destination $aarPath -Force

    Write-Output "Patched: $aarPath"
    Write-Output "Backup:  $backupPath"
}
finally {
    if (Test-Path $work) {
        Remove-Item -LiteralPath $work -Recurse -Force
    }
}
