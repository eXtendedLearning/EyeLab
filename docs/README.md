# EyeLab — committed engineering docs

This folder holds the **committed, public-safe** engineering documentation for EyeLab. Siemens-confidential collaboration plans (`PROJECT.md`, `TASKS.md`, full `THEORY.md`, original task specs) live in `.docs/` and are gitignored. Working rules and document precedence: [`../AGENTS.md`](../AGENTS.md).

## Where to start (in order)

1. **`../CONTEXT.md`** — verified facts: the XREAL XR Plugin 3.1.0 surface, resolved decisions, Android settings and why, the open-issues board.
2. **`HARDWARE.md`** and **`CAPABILITIES.md`** — what hardware is in hand and what is known to work on it (dated, with evidence).
3. **`tasks/T7.0-campaign-0.5.md`** — the current work: campaign 0.5 (objectives, literature, gates C0-C8, sheets, hit analysis, the 0.5.000 build, lab materials, hand-off to Claude Code §14).
4. **`BEAMPRO-HANDOVER-2026-11-10.md`** — the Beam Pro plan and its gate table (B0, BS, B1, …); the `tasks/T6.*` files execute those gates.
5. **`adr/`** — architecture decisions; ADR-0001 (bridging staircase) and ADR-0004 (pose on the Beam Pro) are accepted.
6. **`tasks/T2.6-implementation-checklist.md`** — the XREAL application plan (bridge, `EwmaStructureAnchor`, calibration scene). Its class names are contracts; its code stubs are intent.

Project wiki (EN/IT): **`wiki/index.html`**, a navigable summary of the system, the validation campaign, the XREAL work and the decision record. Sources in `wiki/src/`, rebuild with `python docs/wiki/build_wiki.py`. Where it disagrees with the documents here, the documents win.

## Folder structure

```
docs/
├── README.md                                    # this file
├── HARDWARE.md                                  # machines and the glasses, what each can do
├── CAPABILITIES.md                              # dated capability matrix with evidence labels
├── VERSIONING.md                                # build numbering policy
├── BEAMPRO-HANDOVER-2026-11-10.md               # Beam Pro plan and gate table (current)
├── XREAL-PC-GATES-G1-G3.md                      # PC-beta gate project guide and run log §1-26 (paused 2026-11-10)
├── NEBULA-WINDOWS-AUDIT-2026-09-21.md           # Windows native route (historical; cited by python/xreal_native*.py)
├── validation.md                                # baseline validation protocol (python/validation/)
├── validation-campaign.md                       # step-by-step campaign guide (validation.bat)
├── validation_printables/                       # plate A/L PDFs
├── campaign_printables/                         # campaign 0.5 sheets: A40, T, P, M PDFs (python -m campaign sheets)
├── aruco_marker_best_practices.pdf
├── wiki/                                        # EN/IT project wiki
├── adr/
│   ├── 0001-opencv-bridging-staircase.md        # accepted
│   ├── 0002-registration-strategy.draft.md      # proposed
│   ├── 0003-eye-pov-rigid-coupling.draft.md     # proposed (S0-S6 stage ladder)
│   └── 0004-on-device-pose-beampro.md           # accepted 2026-11-19
└── tasks/
    ├── T2.6-implementation-checklist.md         # XREAL application plan
    ├── T6.0 … T6.13                             # Beam Pro gates, one file per session, results inside
    └── T7.0-campaign-0.5.md                     # campaign 0.5: plan, gates, build, hand-off (current)
```

ADRs are numbered sequentially (`NNNN-kebab-case-title.md`), Nygard format, append-only: to reverse one, write a new ADR and mark the old one `Status: Superseded by ADR-NNNN`.

## Removed documents

Retired on 2026-11-19 because their work is finished or superseded. Each is still in git; read one with `git show 5e9a386:<path>` (5e9a386 = build 0.4.004, the last commit that holds all of them). Their conclusions live on where the right column says.

| Path | What it was | Where its content lives now |
|---|---|---|
| `docs/AUDIT-2026-09-18.md` | first code/science/doc audit (S1-S8, D1-D9) | S1-S3 fixed and guarded by tests (`python/test_scale_contracts.py`, `pose_lock`, `registration`); superseded by the 2026-10-08 audit |
| `docs/audit_verify.py` | behavioural cross-check of the 2026-09-18 findings | superseded by the unit tests above |
| `docs/AUDIT-2026-10-08.md` | project audit F01-F14, revised 9 Oct, with a Surface development sequence | F01-F03, F05, F06, F08, F11, F13, F14 corrected on 2026-10-08 (CHANGELOG); F04, F07, F09, F10, F12 on the `CONTEXT.md` open-issues board; the Surface sequence was done as the PC gates and then superseded by the Beam Pro (ADR-0004) |
| `docs/S3-HANDOVER-2026-09-25.md` | Windows native head-pose route, S3/S3b state | dead end: licence-gated IMU, identity poses; `CAPABILITIES.md` rows, `NEBULA-WINDOWS-AUDIT` §§11-14 |
| `docs/XREAL-PC-BETA-2026-10-09.md` | first look at the XREAL 3.1.2 PC beta | followed up by `XREAL-PC-GATES-G1-G3.md`; capabilities in `CAPABILITIES.md` |
| `docs/ar-freeze-diagnostics.md` | 2026-09-18/19 GUI freeze on `Start AR` | fixed 2026-09-19 (`camera_utils.CameraOpener`, paced reader); remaining item on the open-issues board |
| `docs/tasks/T5.1-eye-latency.md` | Eye latency plan for the PC gate project | closed as paused at 0.3.010; results in `XREAL-PC-GATES-G1-G3.md` §23-26, continued as T6.1 on the Beam Pro |

## Conventions

- **CONTEXT.md** is the single source of truth for verified facts. If a doc disagrees with runtime behaviour, **the runtime wins** — correct the doc in the same change.
- **No code in CONTEXT.md** beyond brief snippets.
- **Task files** (`tasks/T*.md`) are written before a session, carry its "Done when" gates, and get a dated Results section after it. When a task is finished and its results are reflected in `CAPABILITIES.md` / `CONTEXT.md`, it may be retired into the table above.
- **Inline citations**: reference source files as `` `path/to/file.cs` `` so the path stays greppable.
