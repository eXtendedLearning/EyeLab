# EyeLab — committed engineering docs

This folder holds the **committed, public-safe** engineering documentation for EyeLab. Anything an AI agent or new contributor needs to start contributing should be here. Siemens-confidential collaboration plans (`PROJECT.md`, `TASKS.md`, full `THEORY.md`, original task specs) live in `.docs/` and are gitignored.

## Where to start (in order)

Project wiki (EN/IT): **`wiki/index.html`** — one navigable overview of the system, the
validation campaign, the XREAL work and the decision record, with equations and references.
It summarises the documents below and does not replace them; sources in `wiki/src/`,
rebuild with `python docs/wiki/build_wiki.py`.

XREAL gate experiment: **`XREAL-PC-GATES-G1-G3.md`** is the executable guide
(desktop Unity CLI build, Surface runs) for G1 Eye frames, G2 6DoF and G3
both together, with acceptance criteria and the outcome matrix.

XREAL update: **`XREAL-PC-BETA-2026-10-09.md`** records the downloaded PC beta,
Standalone camera source, current USB/network preflight, isolated Unity test
procedure and corrected WSL/emulator assessment. Actual beta camera/pose
delivery remains unverified; four core DLLs match the September runtime while
the Unity wrapper differs. The main project still uses SDK 3.1.0.

Current review: **`AUDIT-2026-10-08.md`**, revised in place on **2026-10-09**,
records reproduced defects and their correction status. Its revised XREAL
assessment reconciles all three ADRs, previous blockers and the conditional
Windows Unity/Python architecture with the optical-overlay objective. Applied
2026-10-08: F01, F02, F03, F06, F11
(code) and F05, F08, F13, F14 (documentation: the checklist, ADR-003 amendments,
`CAPABILITIES.md`). Still open: F04, F07, F09, F10, F12. For what is known to
work on the glasses, today, read **`CAPABILITIES.md`** (dated, with evidence).

1. **`../CONTEXT.md`** — repo-root domain context: what EyeLab does, the verified XREAL XR Plugin 3.1.0 surface, all resolved decisions, doc-visibility scheme, the open-issues board.
2. **`adr/0001-opencv-bridging-staircase.md`** — the foundational architectural decision: why the OpenCV-on-Unity path is a staircase **C → A → B** (off-device Python service → DIY .aar → OpenCV for Unity), with explicit pivot criteria.
3. **`tasks/T2.6-implementation-checklist.md`** — the actionable, phase-by-phase implementation plan for the XREAL bring-up. Self-contained: file paths, code stubs, acceptance criteria, "done when" gates.

## Folder structure

```
docs/
├── README.md                                    # this file
├── AUDIT-2026-10-08.md                          # current project audit (F01-F14, XREAL development sequence)
├── CAPABILITIES.md                              # dated capability matrix: host/firmware/transport/mode/evidence
├── HARDWARE.md                                  # machines and the glasses
├── AUDIT-2026-09-18.md                          # historical code/doc audit (findings S1-S8, D1-D9)
├── NEBULA-WINDOWS-AUDIT-2026-09-21.md           # XREAL Windows runtime: tiers, S0 result, S2 signatures (§9)
├── ar-freeze-diagnostics.md                     # AR loop freeze investigation
├── aruco_marker_best_practices.pdf
├── validation.md                                # baseline validation protocol (python/validation/)
├── validation-campaign.md                       # step-by-step campaign guide (validation.bat)
├── validation_printables/                       # plate A/L PDFs (targets, markers, UNV: python/config/)
├── audit_verify.py                              # behavioural cross-check of the 2026-09-18 findings (historical)
├── adr/
│   ├── 0001-opencv-bridging-staircase.md        # ADR-001 (accepted)
│   ├── 0002-registration-strategy.draft.md      # ADR-002 (proposed)
│   └── 0003-eye-pov-rigid-coupling.draft.md     # ADR-003 (proposed) — S0-S6 stage ladder
└── tasks/
    └── T2.6-implementation-checklist.md         # T2.6 v0 plan
```

ADRs are numbered sequentially (`NNNN-kebab-case-title.md`). The format is Michael Nygard's classic — context, decision, consequences. Add new ADRs with the next available number.

Implementation guides under `tasks/` follow the work-package numbering from the (private) `.docs/TASKS.md`. Each has an "Acceptance" or "Done when" section that defines completion.

## Status snapshot (HISTORICAL, 2026-05-14; current status: `CAPABILITIES.md`, `HARDWARE.md`, `AUDIT-2026-10-08.md`)

| Item | Status |
|---|---|
| ADR-001 — OpenCV bridging staircase | Accepted |
| Pose-timing strategy | Decided: option (iii) `EwmaStructureAnchor`, with option (i) AR Foundation `XRAnchor` as the future upgrade path. Captured in CONTEXT.md "Resolved decisions" §3 and in T2.6-implementation-checklist Phase 5. |
| T2.6 implementation | Specified (this file's `tasks/`); Phase 0 repo/Unity hygiene and empty Android APK build gate complete. Local XREAL SDK copy required an `nr_common.aar` manifest-package patch to avoid duplicate `nrsdk.pack` namespace with `nr_loader.aar`; reapply with `tools/patch_xreal_sdk.ps1`. |
| Hardware | XREAL One Pro + Eye received 2026-05-08; Beam Pro 8 GB / 256 GB ordered 2026-05-14, expected mid-to-late June 2026 (**superseded: not arrived as of 2026-10-08**) |

## Conventions

- **CONTEXT.md** is the single source of truth for verified facts (SDK surface, resolved decisions). If you observe a discrepancy between docs and runtime behaviour, **the runtime wins** — update CONTEXT.md and continue.
- **ADRs are append-only**. If a decision is reversed, write a new ADR that supersedes the old one; mark the old one's `Status: Superseded by ADR-NNNN` rather than editing the prose.
- **No code in CONTEXT.md** beyond brief snippets. Code stubs live in the implementation checklist.
- **Inline citations**: when a doc references a source file, use the form ``[`Packages/com.xreal.xr/Runtime/Scripts/XREALRGBCamera.cs`](...)`` so the path is greppable even when the link rendering fails.

## For AI coding agents

If you're an AI coding agent working on this repo:

1. Read `../CONTEXT.md` then this file's "Where to start" list before writing any code.
2. The implementation checklist at `tasks/T2.6-implementation-checklist.md` is your work plan. Tackle one phase per PR. Use the "Done when" gates as PR-acceptance criteria.
3. The `IArucoPoseBridge` interface (Phase 2) is the load-bearing abstraction in the codebase — never make non-bridge code depend on the WebSocket implementation directly.
4. If a phase tells you a specific class name (e.g. `EwmaStructureAnchor`), use that name. The names are referenced from CONTEXT.md and other docs; renaming silently breaks the cross-references.
5. If the SDK behaves differently from how it's documented in CONTEXT.md, your observation wins. Update CONTEXT.md "Verified XREAL XR Plugin 3.1.0 surface" with the corrected facts as part of your PR.
