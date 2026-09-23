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
  libnr_loader.dll ...       Tier B — native NRSDK, 11 more files, 94 MB.
                             Entry point for S2 (python/xreal_native.py).
```

The authoritative Tier B file list is `xreal_native.TIER_B_FILES`; the
PowerShell below repeats it. `python python/xreal_native_probe.py --level load`
reports anything missing, including the two system DLLs `libnr_api.dll`
imports but which are never staged: `vulkan-1.dll` (GPU driver) and
`D3DCOMPILER_47.dll` (Windows).

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
intrinsics, DP/EDID control). 11 files beside Tier A, 94 MB. ADR-003 puts
Tier B on the critical path (Stage S2), so it is staged ahead of the Tier A
firmware questions rather than after them.

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

Tier B, staged 2026-09-22 from the same build (SHA-256):

```
3586c14119c75a17fe2853123cc2e8874692b7fa80111707496e526d32bb1d69  libnr_api.dll
faa64e1687f82399996736f6e89e522959953e72e08fbb1251f5a1b77b683a63  libnr_loader.dll
37cb22858d0b6855d2eed3e4db8a1e5cf3646f7ae765d1ed422e70bca8995861  libnr_plugin_6dof.dll
f38328fd089976f9c2194ac4a1e8236065cbaf203622a46cac38a61d6cf1d97d  ov_utils.dll
f38f2248d15fb079fc4a896fc822fd65cc09f540d240bf6c122380ec5f001804  avcodec-60.dll
09906e293f15c7df02fae4f4395504da693130122145f061473a183b983ec73d  avformat-60.dll
75fa6621f95ac0e0974c31f7d67df76752e11cf1949b0910dfb9d71a42b05acc  avutil-58.dll
92d89f92c69079792ba3fa2244980bf798be2f59449e5813397398b09547a24e  swresample-4.dll
f059d64ffdb353b0fae89160f9451bb76ed3f8604a7353d6e56150628a4382f2  swscale-7.dll
a8f950b4357ec12cfccddc9094cca56a3d5244b95e09ea6e9a746489f2d58736  vcruntime140.dll
e4b533a94e02c574780e4b333fcf0889f65ed00d39e32c0fbbda2116f185873f  vcruntime140_1.dll
```

Audit: [`docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md`](../docs/NEBULA-WINDOWS-AUDIT-2026-09-21.md)
