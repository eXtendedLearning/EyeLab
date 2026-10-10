# Versioning — `MAJOR.MINOR.BUILD`

Introduced 2026-10-10 (build 0.2.001). Binding for every contributor, human or agent, of any provider.

## The scheme

`0.2.014` = MAJOR `0` · MINOR `2` · BUILD `014` (always 3 digits).

| part | changes when | resets |
|---|---|---|
| BUILD | **every commit** (`tools/bump_version.py`) | → `000` when MINOR changes |
| MINOR | a milestone cut: a coherent capability set is done and verified (`... minor`) | BUILD → `000` |
| MAJOR | a breaking change of the on-disk or wire formats, or the first validated release (`... major`) | MINOR, BUILD → `0` |

`0.2.000` is the `v 0.2.0` commit of 2026-07-11 (`f7dba8e`). Everything after it up to
the introduction of this scheme is collapsed into `0.2.001`; there are no per-commit numbers
before that. `0.1.1` (`3898425`) and `0.1.0` predate the scheme and keep their names.

## Single source of truth

**`python/eyelab_version.py` → `__version__`.** Nothing else holds the number as an original.
It is stdlib-only and runnable: `python python/eyelab_version.py` prints the version
(`--string` → `EyeLab v0.2.001`, `--package` → `0.2.1`).

Every other occurrence is one of:

| where | how it follows the source |
|---|---|
| Python banners, GUI title, validation session metadata | import `VERSION_STRING` |
| `run_eyelab.bat`, `validation.bat` | run `eyelab_version.py --string` at launch |
| `README.md` title, `CHANGELOG.md` "Current build" | stamped between `<!-- eyelab-version -->` markers by `tools/bump_version.py` |
| `eyelab_xreal` Unity `bundleVersion` | stamped by `tools/bump_version.py` |
| `CONTEXT.md` "Last updated: DATE, build X.Y.ZZZ" | **by hand** — it records the build at which facts were last verified, so it must not auto-follow |
| commit subjects | prefix, enforced by the `commit-msg` hook |

`python tools/bump_version.py check` (and `python/test_version_scheme.py` in CI) fails when a stamp drifts
or a marker was deleted. Do not "fix" drift by editing the stamp; run `sync`.

## Every commit

```
git pull --rebase                      # first: two machines bumping at once is the one real hazard
python tools/bump_version.py           # BUILD+1, stamps README/CHANGELOG/Unity, prints the prefix
git add -A
git commit -m "0.2.014 geometry bugfix"   # plain lowercase text after the prefix, as before
```

- Subject = `<version> <text>`. Merges, reverts and `fixup!`/`squash!` are exempt.
- Enable the guard once per clone (desktop, Surface, agent sandbox): `git config core.hooksPath tools/githooks`.
  It rejects a prefix that does not match the staged version, a commit that did not bump, and a number already used in history.
- Amending a commit keeps its number: `EYELAB_ALLOW_SAME_VERSION=1 git commit --amend`.
- Docs-only commits bump too. One rule, no judgement calls.

## Cutting a MINOR

```
python tools/bump_version.py minor     # 0.2.047 -> 0.3.000
# write the [0.3.000] heading in CHANGELOG.md: move [Unreleased] under it
git commit -m "0.3.000 <milestone name>"
git tag v0.3.000                       # tags only at MINOR/MAJOR cuts, never per build
```

## Agents (any provider)

1. Read this file and `AGENTS.md` first. The version is in `python/eyelab_version.py`; never infer it from the README or a commit message.
2. **Sandbox agents cannot commit** (`AGENTS.md`). Run `python tools/bump_version.py` once at the **end** of your change set (one bump per hand-off, not per file edit) and tell Jack the exact prefix to commit with.
3. When you touch a CONTEXT.md fact, update its `Last updated` line and write `build X.Y.ZZZ` in it.
4. New CHANGELOG bullets go under `[Unreleased]`; tag them with the build, e.g. `(0.2.014)`.
5. Cite a build, not a date, when you refer to "the version where X changed": `git log --grep "^0.2.014 "`.
6. A number in a doc that is *not* stamped and not a historical reference is a bug: replace it with a pointer to the source.

## Known limits

- **Not strict SemVer / PEP 440.** `0.2.014` has a leading zero (SemVer forbids it) and PEP 440 normalises it to `0.2.14`. Anything a packaging tool parses (pip, a future portable-bundle name, installers) must use `PACKAGE_VERSION`. The padded form is for humans and logs.
- **999 builds per MINOR.** At `0.2.999` the tool refuses to bump; cut a MINOR.
- **The number does not identify a dirty tree.** Uncommitted edits still read as the last build. Validation sessions record the git commit and a dirty flag (`python/validation/capture.py: git_state`) for exactly this.
- **Android `versionCode` is not automated.** `AndroidBundleVersionCode` in `eyelab_xreal` must be a monotonically increasing integer for every APK installed over another one. Proposed mapping when the first real APK ships: `MAJOR*1_000_000 + MINOR*1_000 + BUILD` (0.2.001 → 2001). Until then it stays `1`.
- **`unity_projects/xreal_pc_gates/` is deliberately not stamped.** It is a throwaway gate harness with its own `bundleVersion`; gate run logs identify the harness by its own build, not by EyeLab's.
- **Bypass.** `git commit --no-verify` skips the hook. Use it for history surgery only; an unnumbered commit breaks the "find a feature by build" property.
