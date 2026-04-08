#!/usr/bin/env python3
"""
UNV to JSON Converter

Converts Siemens Simcenter Testlab UNV geometry files to JSON format
for consumption by Unity AR application.

Parses four critical datasets:
  2411 — Nodes (X/Y/Z coordinates, displacement CS) — modern
  15   — Nodes (legacy single-precision format) — fallback
  82   — Trace Lines (wireframe edge connectivity, polyline with 0-separators)
  2420 — Coordinate Systems (4×3 transformation matrices)
  164  — Units (SI, MM/N/S, etc.)

Usage:
    python unv_to_json.py input.unv -o output.json
    python unv_to_json.py input.unv --validate-cs --verbose
    python unv_to_json.py input.unv --pretty --skip-validation
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import pyuff
except ImportError:
    print("ERROR: pyuff not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class UNVParseError(Exception):
    """Raised when UNV file parsing fails."""


class UNVParser:
    """Parser for UNV (Universal File Format) geometry files."""

    DATASET_NODES = 2411
    DATASET_NODES_LEGACY = 15
    DATASET_TRACE_LINES = 82
    DATASET_COORD_SYSTEMS = 2420
    DATASET_UNITS = 164

    UNIT_CODES: Dict[int, Dict[str, str]] = {
        1: {"name": "SI",        "description": "meters, newtons, seconds, kelvin"},
        2: {"name": "MM_N_S",    "description": "millimeters, newtons, seconds"},
        4: {"name": "MM_KG_S",   "description": "millimeters, kilograms, seconds"},
        6: {"name": "IN_LBF_S",  "description": "inches, pounds-force, seconds"},
    }

    def __init__(self, unv_file: Path, validate_cs: bool = True, verbose: bool = False):
        """
        Initialize parser.

        Args:
            unv_file:    Path to UNV file.
            validate_cs: If True, validate that all displacement CS IDs in Dataset 2411
                         are defined in Dataset 2420.  Missing IDs cause Testlab to
                         fail silently during mode-shape visualization.
            verbose:     Enable DEBUG logging.
        """
        self.unv_file = Path(unv_file)
        self.validate_cs = validate_cs
        self.verbose = verbose

        if self.verbose:
            logger.setLevel(logging.DEBUG)

        if not self.unv_file.exists():
            raise FileNotFoundError(f"UNV file not found: {self.unv_file}")

        logger.info(f"Initialized parser for: {self.unv_file.name}")

    # ── Public API ────────────────────────────────────────────────────────────

    def parse(self) -> Dict[str, Any]:
        """
        Parse UNV file and return a JSON-serialisable dictionary.

        Returns:
            Dict with keys: metadata, nodes, traceLines, coordinateSystems, units.

        Raises:
            UNVParseError: If critical datasets are missing or malformed.
        """
        try:
            logger.info(f"Reading UNV file: {self.unv_file}")
            uff = pyuff.UFF(str(self.unv_file))
            datasets = uff.read_sets()
            logger.info(f"Read {len(datasets)} dataset(s) from UNV file")
        except Exception as e:
            raise UNVParseError(f"Failed to read UNV file: {e}") from e

        nodes_ds  = self._get_dataset(datasets, self.DATASET_NODES)
        legacy_nodes_ds = None
        if nodes_ds is None:
            legacy_nodes_ds = self._get_dataset(datasets, self.DATASET_NODES_LEGACY)
            if legacy_nodes_ds is not None:
                logger.info("Found legacy nodes dataset 15 (no 2411 present)")
        lines_ds_all = self._get_all_datasets(datasets, self.DATASET_TRACE_LINES)
        cs_ds     = self._get_dataset(datasets, self.DATASET_COORD_SYSTEMS)
        units_ds  = self._get_dataset(datasets, self.DATASET_UNITS)

        try:
            if nodes_ds:
                nodes = self._parse_nodes(nodes_ds)
            elif legacy_nodes_ds:
                nodes = self._parse_nodes_legacy(legacy_nodes_ds)
            else:
                nodes = []
            trace_lines: List[List[int]] = []
            for lds in lines_ds_all:
                trace_lines.extend(self._parse_trace_lines(lds))
            coord_systems = self._parse_coord_systems(cs_ds)  if cs_ds     else []
            units        = self._parse_units(units_ds)        if units_ds  else self._default_units()
        except UNVParseError:
            raise
        except Exception as e:
            raise UNVParseError(f"Failed to parse datasets: {e}") from e

        if self.validate_cs and coord_systems:
            self._validate_cs_references(nodes, coord_systems)

        metadata = {
            "sourceFile": self.unv_file.name,
            "parseDate": datetime.utcnow().isoformat() + "Z",
            "nodeCount": len(nodes),
            "lineCount": len(trace_lines),
            "coordinateSystemCount": len(coord_systems),
        }

        result = {
            "metadata": metadata,
            "nodes": nodes,
            "traceLines": trace_lines,
            "coordinateSystems": coord_systems,
            "units": units,
        }

        logger.info(
            f"Parse complete: {len(nodes)} nodes, "
            f"{len(trace_lines)} lines, {len(coord_systems)} CS"
        )
        return result

    # ── Dataset parsers ───────────────────────────────────────────────────────

    def _parse_nodes(self, dataset: Dict) -> List[Dict[str, Any]]:
        """Parse Dataset 2411 (Nodes)."""
        node_ids  = dataset.get("node_nums", [])
        export_cs = dataset.get("coord_sys", [])
        disp_cs   = dataset.get("disp_coord_sys", [])
        x_coords  = dataset.get("x", [])
        y_coords  = dataset.get("y", [])
        z_coords  = dataset.get("z", [])

        if not node_ids:
            raise UNVParseError("No nodes found in Dataset 2411")

        nodes = []
        for i, node_id in enumerate(node_ids):
            nodes.append({
                "id":             int(node_id),
                "x":              float(x_coords[i]) if i < len(x_coords) else 0.0,
                "y":              float(y_coords[i]) if i < len(y_coords) else 0.0,
                "z":              float(z_coords[i]) if i < len(z_coords) else 0.0,
                "exportCS":       int(export_cs[i])  if i < len(export_cs) else 0,
                "displacementCS": int(disp_cs[i])    if i < len(disp_cs)   else 0,
            })

        logger.debug(f"Parsed {len(nodes)} nodes from Dataset 2411")
        return nodes

    def _parse_nodes_legacy(self, dataset: Dict) -> List[Dict[str, Any]]:
        """
        Parse Dataset 15 (legacy Nodes).

        Record format (per UNV spec):
            node_label, def_cs, disp_cs, color, x, y, z
        pyuff exposes these as separate arrays (names may vary across versions).
        """
        # pyuff field-name candidates seen across versions
        def _first(*keys):
            for k in keys:
                if k in dataset and dataset[k] is not None and len(dataset[k]) > 0:
                    return dataset[k]
            return []

        node_ids  = _first("node_nums", "node_label", "node_labels")
        x_coords  = _first("x")
        y_coords  = _first("y")
        z_coords  = _first("z")
        export_cs = _first("def_cs", "coord_sys")
        disp_cs   = _first("disp_cs", "disp_coord_sys")

        if len(node_ids) == 0:
            raise UNVParseError("No nodes found in Dataset 15 (legacy)")

        nodes = []
        for i, node_id in enumerate(node_ids):
            nodes.append({
                "id":             int(node_id),
                "x":              float(x_coords[i]) if i < len(x_coords) else 0.0,
                "y":              float(y_coords[i]) if i < len(y_coords) else 0.0,
                "z":              float(z_coords[i]) if i < len(z_coords) else 0.0,
                "exportCS":       int(export_cs[i])  if i < len(export_cs) else 0,
                "displacementCS": int(disp_cs[i])    if i < len(disp_cs)   else 0,
            })

        logger.debug(f"Parsed {len(nodes)} nodes from Dataset 15 (legacy)")
        return nodes

    def _parse_trace_lines(self, dataset: Dict) -> List[List[int]]:
        """
        Parse Dataset 82 (Trace Lines).

        Dataset 82 stores a polyline as a flat node-ID sequence where ``0``
        acts as a "pen-up" separator (break to a new segment).  We convert
        that into explicit edge pairs.

        pyuff can expose the sequence under ``nodes`` (modern) or
        ``node_nums`` (older), and it may be a single flat array or a list
        of per-line arrays.
        """
        node_seq = dataset.get("nodes")
        if node_seq is None or (hasattr(node_seq, "__len__") and len(node_seq) == 0):
            node_seq = dataset.get("node_nums", [])

        if node_seq is None or (hasattr(node_seq, "__len__") and len(node_seq) == 0):
            logger.warning("No trace lines found in Dataset 82")
            return []

        # Flatten to a single list of ints
        flat: List[int] = []
        try:
            arr = np.asarray(node_seq)
            if arr.dtype == object:
                for sub in node_seq:
                    flat.extend(int(v) for v in np.asarray(sub).ravel().tolist())
            else:
                flat.extend(int(v) for v in arr.ravel().tolist())
        except Exception:
            for item in node_seq:
                if isinstance(item, (list, tuple, np.ndarray)):
                    flat.extend(int(v) for v in list(item))
                else:
                    flat.append(int(item))

        # Split on 0 (pen-up) and build consecutive edges
        lines: List[List[int]] = []
        segment: List[int] = []

        def _flush():
            for a, b in zip(segment, segment[1:]):
                if a != b:
                    lines.append([a, b])

        for v in flat:
            if v == 0:
                _flush()
                segment = []
            else:
                segment.append(v)
        _flush()

        logger.debug(
            f"Parsed {len(lines)} trace-line edges from Dataset 82 "
            f"({len(flat)} raw entries)"
        )
        return lines

    def _parse_coord_systems(self, dataset: Dict) -> List[Dict[str, Any]]:
        """Parse Dataset 2420 (Coordinate Systems)."""
        cs_ids   = dataset.get("cs_id",   [])
        cs_types = dataset.get("cs_type", [])
        origins  = dataset.get("origin",  [])
        x_axes   = dataset.get("x_axis",  [])
        y_axes   = dataset.get("y_axis",  [])
        z_axes   = dataset.get("z_axis",  [])

        if not cs_ids:
            logger.warning("No coordinate systems found in Dataset 2420")
            return []

        type_map = {0: "rectangular", 1: "cylindrical", 2: "spherical"}
        coord_systems = []
        for i, cs_id in enumerate(cs_ids):
            raw_type = int(cs_types[i]) if i < len(cs_types) else 0
            coord_systems.append({
                "id":     int(cs_id),
                "type":   type_map.get(raw_type, "unknown"),
                "origin": self._safe_vector(origins, i, [0.0, 0.0, 0.0]),
                "xAxis":  self._safe_vector(x_axes,  i, [1.0, 0.0, 0.0]),
                "yAxis":  self._safe_vector(y_axes,  i, [0.0, 1.0, 0.0]),
                "zAxis":  self._safe_vector(z_axes,  i, [0.0, 0.0, 1.0]),
            })

        logger.debug(f"Parsed {len(coord_systems)} coordinate systems from Dataset 2420")
        return coord_systems

    def _parse_units(self, dataset: Dict) -> Dict[str, Any]:
        """Parse Dataset 164 (Units)."""
        unit_code = int(dataset.get("unit_code", 1))
        factors   = list(dataset.get("factors", [1.0, 1.0, 1.0]))
        factors  += [1.0] * (3 - len(factors))   # pad to 3 if shorter

        unit_info = self.UNIT_CODES.get(unit_code, {"name": "UNKNOWN", "description": "unknown"})
        logger.debug(f"Parsed units: {unit_info['name']}")

        return {
            "code":                unit_code,
            "name":                unit_info["name"],
            "description":         unit_info["description"],
            "lengthFactor":        float(factors[0]),
            "forceFactor":         float(factors[1]),
            "temperatureFactor":   float(factors[2]),
        }

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_cs_references(
        self,
        nodes: List[Dict],
        coord_systems: List[Dict],
    ) -> None:
        """
        Validate that every displacementCS in Dataset 2411 is defined in Dataset 2420.

        Displacement CS 0 (global frame) is always valid and never requires an entry.

        Raises:
            UNVParseError: Lists all undefined CS IDs so the user can fix the file.
        """
        defined_ids: Set[int] = {cs["id"] for cs in coord_systems}
        missing: Set[int] = set()

        for node in nodes:
            d_cs = node["displacementCS"]
            if d_cs != 0 and d_cs not in defined_ids:
                missing.add(d_cs)

        if missing:
            missing_str = ", ".join(str(c) for c in sorted(missing))
            raise UNVParseError(
                f"CRITICAL: Displacement coordinate system(s) not defined in Dataset 2420: "
                f"{missing_str}.  Siemens Testlab will fail silently when visualizing mode "
                f"shapes.  Verify that Dataset 2420 contains these CS IDs, or use "
                f"--skip-validation to bypass this check."
            )

        logger.info("Coordinate system validation passed")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_dataset(datasets: List[Dict], dataset_type: int) -> Optional[Dict]:
        """Return the first dataset matching the given type number, or None."""
        return next((d for d in datasets if d.get("type") == dataset_type), None)

    @staticmethod
    def _get_all_datasets(datasets: List[Dict], dataset_type: int) -> List[Dict]:
        """Return every dataset matching the given type number (e.g. multiple 82 blocks)."""
        return [d for d in datasets if d.get("type") == dataset_type]

    @staticmethod
    def _safe_vector(vectors: List, index: int, default: List[float]) -> List[float]:
        """Safely extract a 3-element float list from a list-of-vectors."""
        if index < len(vectors) and vectors[index] is not None:
            try:
                v = vectors[index]
                if isinstance(v, (list, tuple)):
                    return [float(x) for x in v[:3]]
                elif hasattr(v, "__iter__"):
                    return [float(x) for x in list(v)[:3]]
            except (TypeError, ValueError):
                pass
        return list(default)

    @staticmethod
    def _default_units() -> Dict[str, Any]:
        return {
            "code": 1,
            "name": "SI",
            "description": "meters, newtons, seconds, kelvin",
            "lengthFactor": 1.0,
            "forceFactor": 1.0,
            "temperatureFactor": 1.0,
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert UNV geometry files to JSON for Unity AR application.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input UNV file path")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file path (default: same name as input with .json extension)",
    )
    parser.add_argument(
        "--validate-cs",
        action="store_true",
        default=True,
        help="Validate displacement CS cross-references (default: enabled)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip coordinate system validation (use only for debugging)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with 2-space indentation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    # Resolve output path
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    validate_cs = not args.skip_validation and args.validate_cs

    try:
        p = UNVParser(input_path, validate_cs=validate_cs, verbose=args.verbose)
        result = p.parse()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2 if args.pretty else None)

        logger.info(f"JSON output written to: {output_path}")
        print(f"Success: {output_path}")
        return 0

    except UNVParseError as e:
        logger.error(f"Parse error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
