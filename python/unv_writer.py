#!/usr/bin/env python3
"""
UNV geometry writer — the inverse of unv_to_json.py for the datasets EyeLab
uses. Produces Siemens Universal Files with:

    164  — units (SI by default)
    2411 — nodes
    82   — trace lines (one dataset, ``0`` used as pen-up separator)

Written via pyuff so the output is guaranteed to round-trip through
``unv_to_json.UNVParser`` with the same pyuff version.

Kept UI-free so it can be used by the geometry editor GUI, scripts, and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pyuff


def segments_to_polylines(segments: Sequence[Sequence[int]]) -> list[list[int]]:
    """
    Greedily chain 2-node segments into polylines to keep dataset 82 compact.

    ``[[1,2],[2,3],[7,8]] -> [[1,2,3],[7,8]]``. Segments that are already
    polylines (>2 nodes) are passed through unchanged.
    """
    chains: list[list[int]] = []
    pending: list[tuple[int, int]] = []
    for seg in segments:
        ids = [int(n) for n in seg]
        if len(ids) > 2:
            chains.append(ids)
        elif len(ids) == 2:
            pending.append((ids[0], ids[1]))

    for a, b in pending:
        for chain in chains:
            if chain[-1] == a:
                chain.append(b)
                break
            if chain[0] == b:
                chain.insert(0, a)
                break
        else:
            chains.append([a, b])
    return chains


def polylines_to_trace_sequence(polylines: Sequence[Sequence[int]]) -> list[int]:
    """Flatten polylines into a dataset-82 node sequence with 0 pen-up separators."""
    seq: list[int] = []
    for line in polylines:
        if len(line) < 2:
            continue
        if seq:
            seq.append(0)
        seq.extend(int(n) for n in line)
    return seq


def write_unv(
    path: str | Path,
    nodes: Sequence[dict],
    trace_lines: Sequence[Sequence[int]] | None = None,
    units_code: int = 1,
    description: str = "EyeLab geometry editor export",
) -> Path:
    """
    Write geometry to a UNV file.

    Args:
        path: output file (overwritten if it exists).
        nodes: list of ``{"id", "x", "y", "z"}`` dicts (coordinates in metres
            for ``units_code=1``). Optional keys ``exportCS``/``displacementCS``.
        trace_lines: list of 2-node segments and/or longer polylines.
        units_code: UNV dataset 164 units code (1 = SI).
        description: id line stored in dataset 82.

    Returns:
        The written path.
    """
    path = Path(path)
    if not nodes:
        raise ValueError("Cannot write UNV without nodes")

    node_nums = np.array([int(n["id"]) for n in nodes], dtype=int)
    if len(set(node_nums.tolist())) != len(node_nums):
        raise ValueError("Duplicate node IDs")

    d164 = pyuff.prepare_164(
        units_code=int(units_code),
        units_description="SI - mks" if units_code == 1 else "user units",
        temp_mode=1,
        length=1.0,
        force=1.0,
        temp=1.0,
        temp_offset=273.15,
    )
    d2411 = pyuff.prepare_2411(
        node_nums=node_nums,
        def_cs=np.array([int(n.get("exportCS", 0)) for n in nodes], dtype=int),
        disp_cs=np.array([int(n.get("displacementCS", 0)) for n in nodes], dtype=int),
        color=np.zeros(len(nodes), dtype=int),
        x=np.array([float(n["x"]) for n in nodes]),
        y=np.array([float(n["y"]) for n in nodes]),
        z=np.array([float(n["z"]) for n in nodes]),
    )
    sets = [d164, d2411]

    if trace_lines:
        known = set(node_nums.tolist())
        for seg in trace_lines:
            bad = [n for n in seg if int(n) not in known]
            if bad:
                raise ValueError(f"Trace line references unknown node IDs: {bad}")
        seq = polylines_to_trace_sequence(segments_to_polylines(trace_lines))
        if seq:
            sets.append(
                pyuff.prepare_82(
                    trace_num=1,
                    n_nodes=len(seq),
                    color=0,
                    id=description[:80],
                    nodes=np.array(seq, dtype=int),
                )
            )

    if path.exists():
        path.unlink()
    pyuff.UFF(str(path)).write_sets(sets, mode="add")
    return path
