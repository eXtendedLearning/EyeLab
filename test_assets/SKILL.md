---
name: create-unv-geometry
description: Create EyeLab-compatible Siemens UNV geometry files for webcam or XREAL overlay tests. Use when the user wants to generate, replace, or adapt a geometry.unv file from shape parameters, dimensions, node counts, trace-line topology, or the reference test_assets/geometry.unv file.
---

# Create UNV Geometry

Create a new EyeLab-compatible `.unv` geometry file, normally `geometry.unv`, using the reference file in this directory as the style guide.

## Quick Workflow

1. Read the existing `geometry.unv` in the same directory if the target format is unclear.
2. Ask only for missing parameters that affect geometry:
   - Output path, default `test_assets/geometry.unv`.
   - Shape: L-flange, plate/grid, box/frame, cylinder/ring, custom nodes, or another simple wireframe.
   - Units and dimensions. Default to metres; convert millimetres to metres before writing.
   - Origin and axes. Default: origin at the lower-left/front reference corner; X right, Y length/depth, Z up.
   - Discretization: number of points or spacing along each active dimension.
   - Trace topology: outline only, grid rows/columns, ribs, closed loops, or custom edges.
   - Registration needs: marker/node locations that must exist as node IDs.
3. If the target file already exists and the user did not explicitly ask to replace it, confirm before overwriting.
4. Generate unique nodes, trace polylines, and a minimal UNV file.
5. Validate by loading with EyeLab or by running `python/unv_to_json.py` when `pyuff` is available.

## Reference Pattern

The sample `geometry.unv` is an L-shaped flange-like test structure:

- Dataset `15`: legacy node records, one node per line.
- Dataset `82`: trace-line/wireframe blocks. Node IDs are stored as polylines; `0` means pen-up/break.
- Dataset `164`: units.
- Coordinates are in metres.
- Nodes use coordinate systems `0` and display color `8`.
- The sample has a horizontal XY grid at `z=0` and a vertical XZ grid at `y=0`, sharing the base edge.

Prefer this simple legacy style for generated files because EyeLab's parser supports it and it is easy to inspect by hand.

## Geometry Rules

- Use one-based integer node IDs.
- Keep all coordinates in metres in the final file.
- Reuse a node ID for shared physical points; do not create duplicate coincident nodes unless the user explicitly wants separated parts.
- Keep node spacing realistic for physical markers and camera overlay. For small lab coupons, 10-30 mm spacing is usually readable.
- Make marker registration positions actual nodes whenever possible.
- Avoid self-edges and edges that reference missing node IDs.
- Use `0` only as a trace separator or trailing padding in Dataset `82`.
- Pad Dataset `82` node sequences with trailing zeros so rows can be written in groups of eight integers.

## Minimal UNV Structure

Write the file as ASCII text. A practical EyeLab file can contain only Dataset `15`, one or more Dataset `82` blocks, and Dataset `164`:

```text
    -1
    15
         1         0         0         8  0.00000e+00  0.00000e+00  0.00000e+00
         2         0         0         8  1.00000e-02  0.00000e+00  0.00000e+00
    -1
    -1
    82
         1         2         8
EDGE
         1         2         0         0         0         0         0         0
    -1
    -1
   164
         1
       1.0       1.0       1.0
    -1
```

Dataset `15` node record:

```text
<node_id> <definition_cs=0> <displacement_cs=0> <color=8> <x> <y> <z>
```

Dataset `82` trace record:

```text
    -1
    82
<trace_id> <meaningful_entry_count> <color=8>
<trace_name>
<node IDs and 0 separators, padded to 8 integers per row>
    -1
```

`meaningful_entry_count` is the number of node IDs plus intentional `0` separators before trailing padding.

## L-Flange Recipe

Use this when the user wants something inspired by the reference file:

Required parameters:

- `flange_width_x`
- `flange_length_y`
- `web_height_z`
- `nx`: points across X
- `ny`: points along Y on the horizontal flange
- `nz`: points up Z on the vertical web, excluding the shared base row if using the XY grid's `y=0,z=0` nodes

Default proportions, if the user only asks for a test part:

- `flange_width_x = 0.048 m`
- `flange_length_y = 0.128 m`
- `web_height_z = 0.098 m`
- `nx = 4`
- `ny = 9`
- `nz = 7`

Generate nodes:

1. Horizontal flange: all `(x_i, y_j, 0)` for `i=0..nx-1`, `j=0..ny-1`.
2. Vertical web: all `(x_i, 0, z_k)` for `i=0..nx-1`, `k=1..nz`, sharing the base row from the horizontal flange.
3. Assign IDs in a stable order: horizontal grid first by X column then Y row, then vertical web by X column then Z row.

Generate traces:

- Horizontal grid rows: each constant-X column along Y.
- Horizontal grid columns: each constant-Y row along X.
- Vertical web columns: each constant-X column along Z, starting from the shared base node.
- Vertical web rows: each constant-Z row along X.
- Optional diagonal/rib traces only if requested.

Use separate Dataset `82` blocks when it makes the file easier to inspect, such as one block named `FLANGE`, one named `WEB`, and one named `RIBS`.

## Writer Skeleton

Adapt this logic when creating a `.unv`. It is a reference, not a required separate file.

```python
from pathlib import Path

def fnum(value):
    return f"{value: .5e}"

def write_nodes(lines, nodes):
    lines += ["    -1", "    15"]
    for node_id, x, y, z in nodes:
        lines.append(
            f"{node_id:10d}{0:10d}{0:10d}{8:10d}"
            f" {fnum(x)} {fnum(y)} {fnum(z)}"
        )

def write_trace(lines, trace_id, name, sequence):
    meaningful = list(sequence)
    padded = meaningful + [0] * ((8 - len(meaningful) % 8) % 8)
    lines += ["    -1", "    -1", "    82"]
    lines.append(f"{trace_id:10d}{len(meaningful):10d}{8:10d}")
    lines.append(name[:80] or f"TRACE{trace_id}")
    for i in range(0, len(padded), 8):
        lines.append("".join(f"{v:10d}" for v in padded[i:i + 8]))

def write_units(lines):
    lines += ["    -1", "    -1", "   164", "         1", "       1.0       1.0       1.0", "    -1"]

def write_unv(path, nodes, traces):
    lines = []
    write_nodes(lines, nodes)
    for trace_id, (name, sequence) in enumerate(traces, start=1):
        write_trace(lines, trace_id, name, sequence)
    write_units(lines)
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
```

## Trace Construction

Build a trace sequence from a list of polylines:

```python
def trace_sequence(polylines):
    sequence = []
    for polyline in polylines:
        clean = [int(v) for v in polyline]
        if len(clean) >= 2:
            if sequence:
                sequence.append(0)
            sequence.extend(clean)
    return sequence
```

EyeLab converts each polyline into consecutive edges, so `[1, 2, 3, 0, 4, 5]` becomes edges `(1,2)`, `(2,3)`, and `(4,5)`.

## Validation Checklist

- The file is ASCII and ends with `-1`.
- Every Dataset `82` node ID exists in Dataset `15`.
- Every intentional trace segment has at least two nodes.
- There are no duplicate node IDs.
- There are no unintended duplicate coordinates.
- Units are metres in the coordinates and Dataset `164` uses SI code `1`.
- EyeLab GUI can load the `.unv` without a parse error.
- If using the CLI converter, run:

```bash
python python/unv_to_json.py test_assets/geometry.unv --pretty -o test_assets/geometry.json
```

Do not keep generated validation JSON unless the user asks for it.
