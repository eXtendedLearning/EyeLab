"""Printable plates and the files derived from the same layout.

``write_plate_bundle`` writes, for one layout:

- ``plate_<L>.pdf``            A4 landscape, 1:1, lossless bilevel raster;
- ``plate_<L>.target.json``    nominal ground truth (``apply-measurements`` refines it);
- ``plate_<L>.markers.json``   EyeLab marker config (Markers tab / capture);
- ``plate_<L>.unv``            reference points as UNV geometry (Geometry tab).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .plates import PageLayout, build_target
from .render import render_page
from .target import Target, save_target


def write_pdf(layout: PageLayout, out_pdf, dpi: int = 600) -> None:
    from PIL import Image
    page = render_page(layout, dpi / 25.4)
    # Mode "1" keeps the PDF lossless (Pillow would JPEG-encode a greyscale page).
    Image.fromarray(np.where(page >= 128, 255, 0).astype(np.uint8)).convert("1").save(
        str(out_pdf), "PDF", resolution=float(dpi))


def export_derived(target: Target, out_dir, stem: str) -> dict:
    out = Path(out_dir)
    paths = {"target": out / f"{stem}.target.json", "markers": out / f"{stem}.markers.json",
             "unv": out / f"{stem}.unv"}
    save_target(target, paths["target"])
    target.export_marker_config(paths["markers"])
    target.export_unv(paths["unv"])
    return paths


def write_plate_bundle(layout: PageLayout, out_dir, dpi: int = 600) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"plate_{layout.name}"
    write_pdf(layout, out / f"{stem}.pdf", dpi)
    paths = export_derived(build_target(layout), out, stem)
    paths["pdf"] = out / f"{stem}.pdf"
    return paths
