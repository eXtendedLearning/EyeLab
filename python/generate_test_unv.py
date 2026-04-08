#!/usr/bin/env python3
"""
Generate synthetic UNV files for testing the unv_to_json.py parser.

Usage:
    python generate_test_unv.py --minimal  -o test_minimal.unv
    python generate_test_unv.py --multi-cs -o test_multi_cs.unv
    python generate_test_unv.py --missing-cs -o test_missing_cs.unv
    python generate_test_unv.py --large 2000 -o test_large.unv

Test workflow:
    python generate_test_unv.py --minimal  -o test_minimal.unv
    python unv_to_json.py test_minimal.unv --pretty
    # Expected: 4 nodes, 3 lines, 1 CS in output JSON
"""

import argparse
import sys
from pathlib import Path


def generate_minimal_unv(output_path: Path) -> None:
    """Minimal valid UNV: 4 nodes, 3 trace lines, 1 coordinate system, SI units."""
    content = """\
    -1
  2411
       1         0         0         0
  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
       2         0         0         0
  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
       3         0         0         0
  1.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00
       4         0         0         0
  0.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00
    -1
    82
       1       0       1       0       2
       2       0       1       0       3
       3       0       1       0       4
    -1
  2420
         0         0
  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
  0.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00
  0.0000000000000D+00  0.0000000000000D+00  1.0000000000000D+00
    -1
   164
         1
       1.0       1.0       1.0
    -1
"""
    output_path.write_text(content)
    print(f"Generated minimal UNV: {output_path}  (4 nodes, 3 lines, 1 CS)")


def generate_multi_cs_unv(output_path: Path) -> None:
    """UNV with multiple coordinate systems (nodes reference CS 1 and 2)."""
    content = """\
    -1
  2411
       1         0         1         0
  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
       2         0         1         0
  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
       3         0         2         0
  1.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00
       4         0         2         0
  0.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00
    -1
    82
       1       0       1       0       2
       2       0       1       0       3
       3       0       1       0       4
    -1
  2420
         1         0
  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
  0.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00
  0.0000000000000D+00  0.0000000000000D+00  1.0000000000000D+00
         2         0
  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
  0.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00
  0.0000000000000D+00  0.0000000000000D+00  1.0000000000000D+00
    -1
   164
         1
       1.0       1.0       1.0
    -1
"""
    output_path.write_text(content)
    print(f"Generated multi-CS UNV: {output_path}  (4 nodes, 3 lines, 2 CS)")


def generate_missing_cs_unv(output_path: Path) -> None:
    """Invalid UNV: nodes reference displacement CS 5 which is not defined in Dataset 2420."""
    content = """\
    -1
  2411
       1         0         5         0
  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
       2         0         5         0
  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
    -1
  2420
         1         0
  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00
  0.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00
  0.0000000000000D+00  0.0000000000000D+00  1.0000000000000D+00
    -1
   164
         1
       1.0       1.0       1.0
    -1
"""
    output_path.write_text(content)
    print(f"Generated missing-CS UNV: {output_path}  (CS 5 referenced but undefined)")
    print("  Expected: unv_to_json.py --validate-cs should raise UNVParseError")


def generate_large_unv(output_path: Path, num_nodes: int = 1000) -> None:
    """Large UNV with `num_nodes` nodes for performance testing."""
    with open(output_path, "w") as f:
        # Dataset 2411
        f.write("    -1\n")
        f.write("  2411\n")
        for i in range(1, num_nodes + 1):
            x = float(i % 100) * 0.1
            y = float((i // 100) % 10) * 0.1
            z = float(i // 1000) * 0.1
            f.write(f"{i:8d}         0         0         0\n")
            # Use Fortran D-notation (replace E with D)
            f.write(
                f"  {x:.13E}  {y:.13E}  {z:.13E}\n".replace("E", "D")
            )

        # Dataset 82: connect adjacent nodes
        f.write("    -1\n")
        f.write("    82\n")
        for i in range(1, num_nodes):
            f.write(f"{i:8d}       0       1       0       {i}       {i + 1}\n")

        # Dataset 2420
        f.write("    -1\n")
        f.write("  2420\n")
        f.write("         0         0\n")
        f.write("  0.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00\n")
        f.write("  1.0000000000000D+00  0.0000000000000D+00  0.0000000000000D+00\n")
        f.write("  0.0000000000000D+00  1.0000000000000D+00  0.0000000000000D+00\n")
        f.write("  0.0000000000000D+00  0.0000000000000D+00  1.0000000000000D+00\n")

        # Dataset 164
        f.write("    -1\n")
        f.write("   164\n")
        f.write("         1\n")
        f.write("       1.0       1.0       1.0\n")
        f.write("    -1\n")

    print(f"Generated large UNV: {output_path}  ({num_nodes} nodes, {num_nodes - 1} lines)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic UNV files for testing unv_to_json.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--minimal",    action="store_true", help="4 nodes, 3 lines, 1 CS (default)")
    mode.add_argument("--multi-cs",   action="store_true", help="4 nodes, 3 lines, 2 CS")
    mode.add_argument("--missing-cs", action="store_true", help="Invalid: undefined displacement CS")
    mode.add_argument("--large", type=int, metavar="N",    help="N nodes, N-1 lines, 1 CS")

    parser.add_argument(
        "-o", "--output",
        default="test.unv",
        help="Output UNV file path (default: test.unv)",
    )

    args = parser.parse_args()
    output = Path(args.output)

    if args.multi_cs:
        generate_multi_cs_unv(output)
    elif args.missing_cs:
        generate_missing_cs_unv(output)
    elif args.large:
        generate_large_unv(output, args.large)
    else:
        generate_minimal_unv(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
