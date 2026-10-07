"""Printable plates and the files derived from the same layout.

``write_plate_bundle`` writes, for one layout:

- ``plate_<L>.pdf``            A4 landscape, 1:1, lossless bilevel raster
                               (``docs/validation_printables``);
- ``plate_<L>.target.json``    nominal ground truth, ``config/targets``
                               (``apply-measurements`` refines it);
- ``plate_<L>.markers.json``   EyeLab marker config, ``config/markers``;
- ``plate_<L>.unv``            reference points as UNV geometry, ``config/geometry``.
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


def export_derived(target: Target, stem: str, out_dir=None) -> dict:
    """Write target, marker config and UNV. Default: the shared config tree
    (``config/targets``, ``config/markers``, ``config/geometry``) that the GUI
    and the validation both read; ``out_dir`` puts all three in one folder."""
    import eyelab_paths
    if out_dir is None:
        eyelab_paths.ensure_dirs()
        dirs = (eyelab_paths.TARGETS_DIR, eyelab_paths.MARKER_CONFIGS_DIR, eyelab_paths.GEOMETRY_DIR)
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        dirs = (Path(out_dir),) * 3
    paths = {"target": dirs[0] / f"{stem}.target.json", "markers": dirs[1] / f"{stem}.markers.json",
             "unv": dirs[2] / f"{stem}.unv"}
    save_target(target, paths["target"])
    target.export_marker_config(paths["markers"])
    target.export_unv(paths["unv"])
    return paths


def write_plate_bundle(layout: PageLayout, out_dir, dpi: int = 600, flat: bool = False) -> dict:
    """PDF into ``out_dir``; the derived files into the config tree (``flat``:
    also into ``out_dir``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"plate_{layout.name}"
    write_pdf(layout, out / f"{stem}.pdf", dpi)
    paths = export_derived(build_target(layout), stem, out if flat else None)
    paths["pdf"] = out / f"{stem}.pdf"
    return paths
