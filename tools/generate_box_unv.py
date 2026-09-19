#!/usr/bin/env python3
"""Generate an EyeLab-compatible surface-grid cuboid UNV file (+ CSV node table).

Nodes are placed on the six faces only (no interior points): interior nodes are
neither measurable nor visible in the AR overlay. The origin is the cuboid
corner, coordinates are written in metres (UNV dataset 164 = SI).

Examples:
    # 10 cm cube, one node every 5 cm
    python tools/generate_box_unv.py --size-cm 10 10 10 --pitch-cm 5 \
        -o test_assets/cube100mm_grid50mm.unv

    # anisotropic pitch (X, Y, Z)
    python tools/generate_box_unv.py --size-cm 8.7 8.5 18.5 --pitch-cm 2.9 1.7 3.7

    # explicit node counts instead of a pitch
    python tools/generate_box_unv.py --nx 5 --ny 5 --nz 7
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}


# ── grid helpers ───────────────────────────────────────────────────────────────

def linspace(length: float, count: int) -> list[float]:
    if count < 2:
        raise ValueError("Grid counts must be at least 2.")
    return [length * i / (count - 1) for i in range(count)]


def count_from_pitch(length: float, pitch: float, axis: str, tol: float = 1e-6) -> int:
    """Node count along one axis for a given spacing; pitch must divide length."""
    if pitch <= 0:
        raise ValueError(f"Pitch on {axis} must be positive (got {pitch}).")
    if pitch > length:
        raise ValueError(f"Pitch on {axis} ({pitch}) exceeds the edge length ({length}).")
    intervals = length / pitch
    nearest = round(intervals)
    if abs(intervals - nearest) > tol:
        raise ValueError(
            f"Pitch on {axis} ({pitch}) does not divide the edge length ({length}): "
            f"{intervals:.6f} intervals. Use a divisor such as {length / nearest:.4f}."
        )
    return nearest + 1


def dedupe_polylines(polylines: list[list[int]]) -> list[list[int]]:
    """Drop polylines already present (in either direction) — cuboid edges are
    shared by two faces, so each of the 12 edges would otherwise be traced twice."""
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for line in polylines:
        key = tuple(line)
        if key in seen or tuple(reversed(key)) in seen:
            continue
        seen.add(key)
        unique.append(line)
    return unique


def trace_sequence(polylines: list[list[int]]) -> list[int]:
    """Flatten polylines into a dataset-82 node sequence with 0 pen-up separators."""
    sequence: list[int] = []
    for polyline in polylines:
        clean = [int(v) for v in polyline]
        if len(clean) >= 2:
            if sequence:
                sequence.append(0)
            sequence.extend(clean)
    return sequence


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

    drawn: list[list[int]] = []
    traces: list[tuple[str, list[int]]] = []
    for name, polylines in (
        ("XY_BOTTOM_TOP", xy_polylines),
        ("XZ_FRONT_BACK", xz_polylines),
        ("YZ_LEFT_RIGHT", yz_polylines),
    ):
        unique = dedupe_polylines(drawn + polylines)[len(drawn):]
        drawn += unique
        traces.append((name, trace_sequence(unique)))
    return nodes, traces


# ── UNV writing ────────────────────────────────────────────────────────────────

def fnum(value: float) -> str:
    return f"{value: .5e}"


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
        "         1                  SI         1",
        "  1.00000000000000000D+00  1.00000000000000000D+00  1.00000000000000000D+00",
        "  0.00000000000000000D+00",
        "    -1",
    ]


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


def write_csv(
    path: Path,
    nodes: list[tuple[int, float, float, float]],
    unit: str = "m",
) -> None:
    """Node table: id,x,y,z in the requested unit (UNV itself stays in metres)."""
    scale = UNIT_SCALE[unit]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", f"x_{unit}", f"y_{unit}", f"z_{unit}"])
        for node_id, x, y, z in nodes:
            writer.writerow([node_id, f"{x * scale:.6g}", f"{y * scale:.6g}", f"{z * scale:.6g}"])


# ── CLI ────────────────────────────────────────────────────────────────────────

def resolve_counts(args: argparse.Namespace, sizes_cm: tuple[float, float, float]) -> tuple[int, int, int]:
    if args.pitch_cm:
        pitches = args.pitch_cm * 3 if len(args.pitch_cm) == 1 else args.pitch_cm
        if len(pitches) != 3:
            raise ValueError("--pitch-cm takes either 1 value (isotropic) or 3 (X Y Z).")
        return tuple(  # type: ignore[return-value]
            count_from_pitch(size, pitch, axis)
            for size, pitch, axis in zip(sizes_cm, pitches, "XYZ")
        )
    return (args.nx, args.ny, args.nz)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an EyeLab-compatible cuboid surface-grid UNV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", "-o", default="test_assets/xrealprobox.unv",
                        help="Output UNV file path.")
    parser.add_argument("--size-cm", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=(8.7, 8.5, 18.5), help="Cuboid dimensions in centimetres.")
    parser.add_argument("--pitch-cm", nargs="+", type=float, metavar="P",
                        help="Grid spacing in cm: one value (isotropic) or three (X Y Z). "
                             "Must divide the corresponding edge. Overrides --nx/--ny/--nz.")
    parser.add_argument("--nx", type=int, default=5, help="Grid points along X (ignored with --pitch-cm).")
    parser.add_argument("--ny", type=int, default=5, help="Grid points along Y (ignored with --pitch-cm).")
    parser.add_argument("--nz", type=int, default=7, help="Grid points along Z (ignored with --pitch-cm).")
    parser.add_argument("--csv", metavar="PATH",
                        help="CSV node table path (default: output path with .csv suffix).")
    parser.add_argument("--csv-unit", choices=sorted(UNIT_SCALE), default="m",
                        help="Unit for the CSV coordinates (default: m).")
    parser.add_argument("--no-csv", action="store_true", help="Skip the CSV node table.")
    args = parser.parse_args()

    sizes_cm = tuple(args.size_cm)
    try:
        nx, ny, nz = resolve_counts(args, sizes_cm)
    except ValueError as exc:
        parser.error(str(exc))

    sx, sy, sz = (v / 100.0 for v in sizes_cm)
    nodes, traces = make_surface_box(sx, sy, sz, nx, ny, nz)

    output = Path(args.output)
    write_unv(output, nodes, traces)
    edge_count = sum(len([v for v in seq if v != 0]) for _, seq in traces)
    print(f"Wrote {output}: {len(nodes)} nodes ({nx}x{ny}x{nz} lattice, surface only), "
          f"{len(traces)} trace blocks, {edge_count} trace entries")

    if not args.no_csv:
        csv_path = Path(args.csv) if args.csv else output.with_suffix(".csv")
        write_csv(csv_path, nodes, args.csv_unit)
        print(f"Wrote {csv_path}: {len(nodes)} rows in {args.csv_unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
