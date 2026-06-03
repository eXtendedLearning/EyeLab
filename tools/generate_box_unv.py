#!/usr/bin/env python3
"""Generate an EyeLab-compatible surface-grid cuboid UNV file."""

from __future__ import annotations

import argparse
from pathlib import Path


def fnum(value: float) -> str:
    return f"{value: .5e}"


def trace_sequence(polylines: list[list[int]]) -> list[int]:
    sequence: list[int] = []
    for polyline in polylines:
        clean = [int(v) for v in polyline]
        if len(clean) >= 2:
            if sequence:
                sequence.append(0)
            sequence.extend(clean)
    return sequence


def write_nodes(lines: list[str], nodes: list[tuple[int, float, float, float]]) -> None:
    lines += ["    -1", "    15"]
    for node_id, x, y, z in nodes:
        lines.append(
            f"{node_id:10d}{0:10d}{0:10d}{8:10d}"
            f" {fnum(x)} {fnum(y)} {fnum(z)}"
        )


def write_trace(lines: list[str], trace_id: int, name: str, sequence: list[int]) -> None:
    padded = sequence + [0] * ((8 - len(sequence) % 8) % 8)
    lines += ["    -1", "    -1", "    82"]
    lines.append(f"{trace_id:10d}{len(sequence):10d}{8:10d}")
    lines.append(name[:80] or f"TRACE{trace_id}")
    for i in range(0, len(padded), 8):
        lines.append("".join(f"{v:10d}" for v in padded[i:i + 8]))


def write_units(lines: list[str]) -> None:
    lines += [
        "    -1",
        "    -1",
        "   164",
        "         1",
        "       1.0       1.0       1.0",
        "    -1",
    ]


def linspace(length: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("Grid counts must be at least 2.")
    return [length * i / (count - 1) for i in range(count)]


def make_surface_box(
    length_x: float,
    length_y: float,
    length_z: float,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[list[tuple[int, float, float, float]], list[tuple[str, list[int]]]]:
    xs = linspace(length_x, nx)
    ys = linspace(length_y, ny)
    zs = linspace(length_z, nz)

    node_ids: dict[tuple[int, int, int], int] = {}
    nodes: list[tuple[int, float, float, float]] = []

    def on_surface(ix: int, iy: int, iz: int) -> bool:
        return (
            ix in (0, nx - 1)
            or iy in (0, ny - 1)
            or iz in (0, nz - 1)
        )

    next_id = 1
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                if not on_surface(ix, iy, iz):
                    continue
                node_ids[(ix, iy, iz)] = next_id
                nodes.append((next_id, xs[ix], ys[iy], zs[iz]))
                next_id += 1

    xy_polylines: list[list[int]] = []
    xz_polylines: list[list[int]] = []
    yz_polylines: list[list[int]] = []

    # Bottom and top face grids.
    for iz in (0, nz - 1):
        for iy in range(ny):
            xy_polylines.append([node_ids[(ix, iy, iz)] for ix in range(nx)])
        for ix in range(nx):
            xy_polylines.append([node_ids[(ix, iy, iz)] for iy in range(ny)])

    # Front and back face grids.
    for iy in (0, ny - 1):
        for iz in range(nz):
            xz_polylines.append([node_ids[(ix, iy, iz)] for ix in range(nx)])
        for ix in range(nx):
            xz_polylines.append([node_ids[(ix, iy, iz)] for iz in range(nz)])

    # Left and right face grids.
    for ix in (0, nx - 1):
        for iz in range(nz):
            yz_polylines.append([node_ids[(ix, iy, iz)] for iy in range(ny)])
        for iy in range(ny):
            yz_polylines.append([node_ids[(ix, iy, iz)] for iz in range(nz)])

    traces = [
        ("XY_BOTTOM_TOP", trace_sequence(xy_polylines)),
        ("XZ_FRONT_BACK", trace_sequence(xz_polylines)),
        ("YZ_LEFT_RIGHT", trace_sequence(yz_polylines)),
    ]
    return nodes, traces


def write_unv(
    path: Path,
    nodes: list[tuple[int, float, float, float]],
    traces: list[tuple[str, list[int]]],
) -> None:
    lines: list[str] = []
    write_nodes(lines, nodes)
    for trace_id, (name, sequence) in enumerate(traces, start=1):
        write_trace(lines, trace_id, name, sequence)
    write_units(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an EyeLab-compatible cuboid surface-grid UNV file."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="test_assets/xrealprobox.unv",
        help="Output UNV file path.",
    )
    parser.add_argument(
        "--size-cm",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(8.7, 8.5, 18.5),
        help="Cuboid dimensions in centimetres.",
    )
    parser.add_argument("--nx", type=int, default=5, help="Grid points along X.")
    parser.add_argument("--ny", type=int, default=5, help="Grid points along Y.")
    parser.add_argument("--nz", type=int, default=7, help="Grid points along Z.")
    args = parser.parse_args()

    sx, sy, sz = (v / 100.0 for v in args.size_cm)
    nodes, traces = make_surface_box(sx, sy, sz, args.nx, args.ny, args.nz)
    write_unv(Path(args.output), nodes, traces)
    edge_count = sum(len([v for v in seq if v != 0]) for _, seq in traces)
    print(
        f"Wrote {args.output}: {len(nodes)} nodes, "
        f"{len(traces)} trace blocks, {edge_count} trace entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
