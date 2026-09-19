# AR freeze diagnostics

**Symptom under investigation (2026-09-18).** `Start AR` renders a couple of
frames in the AR View tab, then the view goes black, the window stops
repainting, Windows marks the process *not responding*, and it has to be
killed. Nothing appears in the console and nothing appears in
`python/.logs/eyelab_*.jsonl` — no exception is raised, so the existing
session logger has nothing to record.

A blocked process cannot report on itself, so the diagnosis is written from
outside the blocked thread by `python/ar_watchdog.py`.

## What is instrumented

Everything on the tkinter main thread that the AR path touches, as named
*phases*. The watchdog writes a record whenever the main thread goes silent
for longer than `stall_s` (2 s), naming the phase it was in:

| Phase | Covers |
| --- | --- |
| `start.pipeline_init` | `ArucoPipeline(...)` construction, calibration + board build |
| `start.camera_open` | `ThreadedCapture` open, resolution check, reader start |
| `frame.process` | `pipeline.process_frame()` — CLAHE, ArUco, ROI recovery, carryover, PnP, pose lock |
| `frame.registration` | Kabsch registration update over detected markers |
| `frame.diagnostics` | status text and the detection-diagnostics widgets |
| `frame.overlay` | `draw_overlay`, marker roles, wireframe projection and drawing |
| `frame.display` | BGR→RGB, resize, `ImageTk.PhotoImage`, label update |
| `frame.filtered` | FLT tab rendering |
| `stop.camera_release` | `ThreadedCapture.stop()` → thread join + `cap.release()` |
| `close.stop_ar` / `close.matplotlib` / `close.session_log` | shutdown path |

Two observers, because one of them cannot work during a hard freeze:

1. A daemon thread samples a liveness timestamp and writes a `stall` record
   with the Python stack of **every** thread. Works whenever the blocking
   call releases the GIL (all OpenCV calls, all I/O).
2. A `faulthandler` timer, re-armed only while the main thread is healthy.
   During a freeze nothing re-arms it, so it fires from its own C thread and
   dumps all stacks without needing the GIL. This is the only mechanism that
   survives a native deadlock or a GIL-holding C call.

The instrumentation is **passive**: one daemon thread and two files. It
changes no runtime behaviour, so a freeze that reproduces without it
reproduces with it.

## Reproducing and collecting

1. Run the GUI as usual (`run_eyelab.bat` or `python eyelab_gui.py`).
2. Reproduce the freeze. **Leave it frozen for at least 30 s** before
   killing it — the stall records repeat every 10 s and the faulthandler
   dump needs 10 s of silence.
3. Kill it from Task Manager. Everything is already flushed to disk.
4. Send the newest pair from `python/.logs/`:
   `ar_debug_<ts>.jsonl` and `ar_stacks_<ts>.log`.

## Reading the log

```
python - <<EOF
import json
for line in open(r".logs\ar_debug_<ts>.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["kind"] in ("stall", "resumed", "phase_slow", "ar_start_failed", "ar_loop_exception"):
        print(r["ts"], r["kind"], r.get("phase"), r.get("blocked_ms"))
EOF
```

- `heartbeat` — every ~2 s while AR runs: per-stage `avg_ms` / `max_ms`, UI
  frame rate, RSS, thread count and the capture-thread counters
  (`reads`, `failures`, `fail_streak`, `max_read_ms`, `last_ok_age_ms`).
- `stall` — main thread silent > 2 s, with `phase` and all thread stacks.
- `resumed` — how long it was blocked, if it ever came back.
- A stall while you are dragging or resizing the window, or holding a menu
  open, is expected: Windows runs its own modal loop there and tk callbacks
  do not fire. Those stalls show the main thread inside `mainloop`.
- `ar_stacks_*.log` — raw faulthandler output. `Timeout (0:00:10)!` blocks are
  the hard-freeze dumps; a `Windows fatal exception` block means a native
  crash rather than a hang.

## Findings from the first capture (2026-09-19)

`.logs/ar_debug_20260919_113447.jsonl` answered it. Two independent defects,
both now fixed; the diagnostics stay in place because they are what made the
difference between "it freezes sometimes" and the numbers below.

**1. Opening a camera blocked the GUI thread for ~32 s.** Every stall record
in that run has the same shape — the main thread inside
`cv2.VideoCapture(index, CAP_DSHOW)`:

| When | Blocked | Call site |
| --- | --- | --- |
| startup | 14.7 s | `_refresh_cameras` → `list_cameras()` (8 indices x 2 backends) |
| calibration | 41.2 s | `CalibrationWindow.__init__` → `open_camera` |
| `Start AR` | 32.2 s | `ArucoPipeline.start` → `ThreadedCapture.__init__` → `open_camera` |

That is the "black screen, not responding" symptom exactly: the window cannot
repaint while the main thread is inside the open call, and Windows paints the
stale window dark and marks the process unresponsive.

**2. The capture it returned never delivered a frame, and the reader thread
spun on it.** At teardown the counters read:

```
reads 70 905 456 | failures 70 905 456 | fail_streak 70 905 456
max_read_ms 5132.1 | last_ok_age_ms null
```

70.9 million failed reads in ~30 s: `cv2.read()` was failing instantly and
`ThreadedCapture._reader` had no pause and no give-up, so it consumed a core
and hammered the driver. `last_ok_age_ms: null` means not one frame ever
arrived — the same condition the calibration window showed as "Waiting for
camera frame..." forever. The first read took 5.1 s and failed; every one
after it failed immediately.

## What changed as a result

- `camera_utils.open_camera_result()` accepts a backend only after it has
  actually delivered a frame, and falls through to the next backend (DSHOW →
  MSMF) otherwise, reporting every attempt. `EYELAB_CAPTURE_BACKEND=dshow|msmf|any`
  forces one backend when a machine only works with one.
- `camera_utils.CameraOpener` / `CameraScanner` run opens and index scans on a
  worker thread. The GUI shows elapsed seconds and a Cancel button, and a
  capture that arrives after a cancel is released rather than leaked — a leaked
  handle is what leaves the webcam unusable until it is replugged.
- `ThreadedCapture` pauses 10 ms between failed reads and gives up after
  `CAPTURE_GIVE_UP_S` (5 s) with an explanatory error, which the AR loop
  surfaces and then stops cleanly.
- `CalibrationWindow` opens asynchronously and gives up on a silent camera
  after 5 s with the reason on screen, instead of waiting forever.

## Remaining candidates, if it freezes again

1. **A native call on the main thread not returning** — most likely inside
   `frame.process` (detection / LK / solvePnP) or `frame.display` (Tk
   PhotoImage upload). *Signature:* `stall` naming that phase, with the main
   thread's stack pointing at the exact call.
2. **Device release deadlock at teardown.** `cap.release()` while the reader
   is inside `cap.read()`. *Signature:* phase `stop.camera_release` stalled
   and `capture.join_timed_out: true`.
3. **Shutdown hang after AR stops.** *Signature:* `close.matplotlib` or
   `close.session_log` stalled.
4. **Native crash rather than a hang.** *Signature:* a `Windows fatal
   exception` block in `ar_stacks_*.log` and no further JSONL records.
