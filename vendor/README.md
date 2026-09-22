# vendor/ — local, gitignored vendor binaries

Nothing under `vendor/` is committed except this file. The directory is a
staging area so the Python code can `LoadLibrary` a stable path instead of
reaching into an unpacked Nebula build.

Rationale is the same one already applied to `eyelab_xreal/Packages/com.xreal.xr/`:
XREAL publishes no Windows SDK, these DLLs came from a private beta channel,
and no redistribution licence accompanies them. Each developer stages their own
copy.

## Layout

```
vendor/xreal/win-x64/
  libnr_glasses_api.dll      Tier A — USB/HID glasses control. 1 file, 3.5 MB.
```

## Staging (Windows, from an unpacked Nebula for Windows build)

Source: `<NebulaBuild>/Nebula for Windows_Data/Plugins/x86_64/`

Tier A — everything `python/xreal_glasses.py` needs. Imports only
`KERNEL32` / `msvcrt` / `WS2_32`; `hid.dll` and `setupapi.dll` are resolved at
runtime via `LoadLibrary`. No Unity, no Nebula, no C runtime redistributable.

```powershell
mkdir -Force vendor\xreal\win-x64
copy "<NebulaBuild>\Nebula for Windows_Data\Plugins\x86_64\libnr_glasses_api.dll" vendor\xreal\win-x64\
```

Tier B — full native NRSDK (IMU, 6DoF pose, grayscale/RGB cameras, factory
intrinsics, DP/EDID control). 12 files, 97.5 MB. Only stage this once Tier A has
answered the open questions in the audit.

```powershell
$src = "<NebulaBuild>\Nebula for Windows_Data\Plugins\x86_64"
$tier_b = "libnr_api.dll","libnr_loader.dll","libnr_plugin_6dof.dll","ov_utils.dll",
          "avcodec-60.dll","avformat-60.dll","avutil-58.dll","swresample-4.dll",
          "swscale-7.dll","vcruntime140.dll","vcruntime140_1.dll"
$tier_b | ForEach-Object { copy "$src\$_" vendor\xreal\win-x64\ }
```

`XREALXRPlugin.dll` is Unity-subsystem glue and is **not** needed by the Python
path. The 454 MB `StreamingAssets/NROTA/` firmware tree must never be staged.

## Provenance of the current copy

| | |
|---|---|
| Build | `Nebula_Windows_20260205` (private beta, XREAL Discord) |
| `libnr_glasses_api.dll` SHA-256 | `ffd4af6885cc3992d79c3deb9a0f5fc1b22f0113a6db29fa9c0f0ff52c97fa5f` |
| PE timestamp | 2026-01-22 14:05:45 UTC |
| Authenticode | **unsigned** — integrity rests on the distribution channel |

Audit: [`docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md`](../docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md)
