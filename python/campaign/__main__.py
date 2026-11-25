"""python -m campaign <command> (T7.0). Commands: sheets, set, hits, summary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def cmd_sheets(a) -> int:
    from . import sheets as S
    from validation.plates import build_target, plate_a40
    from validation.printable import export_derived
    a40, sheets = S.build_all()
    errs = S.check(a40) + [e for sh in sheets for e in S.check(sh)]
    if errs:
        print("\n".join(errs))
        import cv2
        if int(cv2.__version__.split(".")[0]) < 5:
            print(f"\nOpenCV {cv2.__version__}: its Hershey text is wider than OpenCV 5's, which rendered the printed "
                  "PDFs (0.5.000), so the scale-bar caption would run into marker 33. The committed PDFs and sheet "
                  "JSONs are the campaign's; do not regenerate them here.")
        return 1
    out_pdf = Path(a.pdf_dir)
    out_json = Path(a.json_dir)
    out_pdf.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)
    # sheet JSONs: all numbers, printed or not (the app and the analysis read them)
    index = []
    for sh in [a40] + sheets:
        d = S.sheet_json(sh)
        (out_json / f"{sh.name}.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        index.append({"name": sh.name, "kind": sh.kind, "number": sh.number, "digest": d["digest"],
                      "targets": len(sh.targets)})
    (out_json / "index.json").write_text(json.dumps({"schema": "eyelab.campaign.sheets/1", "sheets": index},
                                                    indent=1), encoding="utf-8")
    # validation target, marker config and geometry of A40 (the frame's tracking target for every sheet)
    paths = export_derived(build_target(plate_a40()), "plate_A40")
    print("A40 derived:", ", ".join(str(p) for p in paths.values()))
    if a.no_pdf:
        return 0
    ppm = a.dpi / 25.4
    want = {k: set(range(*map(int, v.split("-"))) if "-" in v else [int(v)]) for k, v in
            (x.split(":") for x in a.print)}
    S.write_pdf([S.render(a40, ppm)], out_pdf / "campaign_A40.pdf", a.dpi)
    print("campaign_A40.pdf")
    for kind in ("T", "P", "M"):
        sel = [sh for sh in sheets if sh.kind == kind and sh.number in want.get(kind, set())]
        if sel:
            S.write_pdf([S.render(sh, ppm) for sh in sel], out_pdf / f"campaign_{kind}.pdf", a.dpi)
            print(f"campaign_{kind}.pdf: {sel[0].name}-{sel[-1].name} ({len(sel)} pages)")
    return 0


def cmd_set(a) -> int:
    """Put plate_A40 (the sheets' F40 frame) into the Beam Pro frame set, keeping the Eye calibration."""
    from xreal_gates import b3a
    calib = Path(a.calib)
    s = b3a.update_set(b3a.DEFAULT_SET, calib)
    names = [t["name"] for t in s.get("targets", [])]
    print(f"frame set {b3a.DEFAULT_SET}: calibration {s['calibration']}, targets {', '.join(names)}")
    pkg = "/sdcard/Android/data/it.univpm.eyelab/files"
    print("push to the Beam Pro (from python/):")
    print(f"  adb push beampro_runs/b3a_set/. {pkg}/b3a/")
    print(f"  adb push config/campaign/sheets/. {pkg}/campaign/sheets/      (optional: the APK carries them)")
    print(f"  adb push <protocol.json> {pkg}/campaign/protocol.json        (optional: overrides the APK's)")
    return 0


def cmd_hits(a) -> int:
    import cv2
    from . import hits as Hm
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for img in a.images:
        try:
            res, rect = Hm.analyse(img, a.sheet, a.ppmm, a.max_mm, a.calib)
        except Exception as e:  # keep going over a batch
            print(f"{img}: FAILED {e}")
            rc = 1
            continue
        stem = f"{res.sheet}_{Path(img).stem}"
        Hm.write_csv(res, out / f"{stem}.csv")
        cv2.imwrite(str(out / f"{stem}_check.jpg"), Hm.annotate(rect, res, a.ppmm), [cv2.IMWRITE_JPEG_QUALITY, 85])
        st = Hm.stats([h.err for h in res.hits if h.err is not None])
        miss = [h.n for h in res.hits if h.hit is None]
        print(f"{res.sheet} <- {Path(img).name}: {st.get('n', 0)}/{len(res.hits)} hits, "
              f"frame fit {res.fit_points} pts rms {res.fit_rms_mm:.3f} mm, {res.mm_per_px:.3f} mm/px")
        if st.get("n"):
            print(f"   bias ({st['bias_x_mm']:+.2f}, {st['bias_y_mm']:+.2f}) mm, median {st['median_mm']:.2f}, "
                  f"R95 {st['r95_mm']:.2f}, max {st['max_mm']:.2f} mm")
        if miss:
            print(f"   no dot for targets {miss}")
        if res.extra_dots:
            print(f"   {len(res.extra_dots)} unassigned dots: check {stem}_check.jpg")
        for n in res.notes:
            print("   note:", n)
    return rc


def cmd_summary(a) -> int:
    import csv
    from collections import defaultdict
    from . import hits as Hm
    meta = {}
    if a.map and Path(a.map).exists():
        with open(a.map, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                meta[r["sheet"]] = r
    groups = defaultdict(list)
    for p in sorted(Path(a.hits_dir).glob("*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r["dx_mm"]:
                    continue
                m = meta.get(r["sheet"], {})
                key = (m.get("participant", "?"), m.get("condition", r["sheet"][0]), m.get("display", ""),
                       Path(r["source"]).suffix.lower())
                groups[key].append((float(r["dx_mm"]), float(r["dy_mm"])))
    print(f"{'participant':12}{'cond':6}{'display':9}{'src':6}{'n':>4}{'bias x':>8}{'bias y':>8}{'sd x':>7}"
          f"{'sd y':>7}{'median':>8}{'R95':>7}")
    for k in sorted(groups):
        st = Hm.stats(groups[k])
        print(f"{k[0]:12}{k[1]:6}{k[2]:9}{k[3]:6}{st['n']:4d}{st['bias_x_mm']:8.2f}{st['bias_y_mm']:8.2f}"
              f"{st['sd_x_mm']:7.2f}{st['sd_y_mm']:7.2f}{st['median_mm']:8.2f}{st['r95_mm']:7.2f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets", help="check, export sheet JSONs and A40 target, write the print PDFs")
    s.add_argument("--pdf-dir", default=str(REPO / "docs" / "campaign_printables"))
    s.add_argument("--json-dir", default=str(HERE.parent / "config" / "campaign" / "sheets"))
    s.add_argument("--dpi", type=int, default=600)
    s.add_argument("--print", nargs="*", default=["T:1-9", "P:11-27", "M:41-81"],
                   help="kind:first-end(exclusive) to put in the PDFs")
    s.add_argument("--no-pdf", action="store_true")
    st = sub.add_parser("set", help="add plate_A40 to the Beam Pro frame set (then adb push it)")
    st.add_argument("--calib", default=str(HERE.parent / "config" / "cameras" / "beampro_eye.json"))
    h = sub.add_parser("hits", help="hit positions from scans, photos or Eye frames of used sheets")
    h.add_argument("images", nargs="+")
    h.add_argument("--sheet", help="sheet name (default: read the sheet-ID marker)")
    h.add_argument("--out", default=str(REPO / "python" / "campaign_runs" / "hits"))
    h.add_argument("--ppmm", type=float, default=10.0, help="rectified pixels per mm")
    h.add_argument("--max-mm", type=float, default=40.0, help="largest hit error assigned to a target")
    h.add_argument("--calib", help="camera YAML (camera_matrix, dist_coeffs) to undistort a photo first")
    m = sub.add_parser("summary", help="hit errors pooled per participant, condition, display and source")
    m.add_argument("--hits-dir", default=str(REPO / "python" / "campaign_runs" / "hits"))
    m.add_argument("--map", default=str(REPO / "python" / "campaign_runs" / "sheets_used.csv"),
                   help="CSV sheet,participant,condition,display (written from the app's trial logs)")
    a = ap.parse_args(argv)
    return {"sheets": cmd_sheets, "set": cmd_set, "hits": cmd_hits, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
