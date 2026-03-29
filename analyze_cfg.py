#!/usr/bin/env python3
"""
analyze_cfg.py — Relax and evaluate bulk modulus for a directory of CIF files.

Relaxes all structures with MatterSim BatchRelaxer (EXPCELLFILTER, fmax=0.1),
then estimates the bulk modulus via a static E(V) curve fit.

Usage:
    python analyze_cfg.py -d /path/to/results
    python analyze_cfg.py -d /path/to/results --no-relax
    python analyze_cfg.py -d /path/to/results --n-points 7 --strain 0.04
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

# Reuse existing helpers
sys.path.insert(0, str(Path(__file__).parent))
from calc_bulk_modulus_single import calc_bulk_modulus_value
from mattergen.evaluation.utils.relaxation import relax_atoms


MATTERSIM_MAX_Z = 94


def load_cif_files(directory: Path) -> tuple[list, list[str]]:
    """
    Load CIF files from a directory. Checks for generated_crystals_cif.zip first;
    falls back to loose *.cif files. Returns (atoms_list, names).
    """
    zip_path = directory / "generated_crystals_cif.zip"

    if zip_path.exists():
        print(f"Found {zip_path.name} — extracting to {directory}...")
        with zipfile.ZipFile(zip_path) as zf:
            cif_members = [n for n in zf.namelist() if n.endswith(".cif")]
            if not cif_members:
                print("Archive contains no CIF files.")
                sys.exit(1)
            zf.extractall(directory)
        print(f"Extracted {len(cif_members)} CIF file(s).")

    # fallback: loose CIF files in the directory
    cif_paths = sorted(directory.glob("*.cif"))
    if not cif_paths:
        print(f"No CIF files or generated_crystals_cif.zip found in {directory}")
        sys.exit(1)
    print(f"Found {len(cif_paths)} CIF file(s) in {directory}")

    atoms_list, names = [], []
    for path in cif_paths:
        try:
            atoms = ase_read(str(path))
            atoms.pbc = True
            atoms_list.append(atoms)
            names.append(path.name)
        except Exception as e:
            print(f"  [SKIP] {path.name}: failed to load ({e})")

    return atoms_list, names


def filter_valid(atoms_list: list, names: list[str]):
    """Remove structures with atomic numbers outside MatterSim range [1, 94]."""
    valid_atoms, valid_names, skipped = [], [], []
    for atoms, name in zip(atoms_list, names):
        nums = atoms.get_atomic_numbers()
        bad = nums[(nums <= 0) | (nums > MATTERSIM_MAX_Z)]
        if len(bad) > 0:
            print(f"  [SKIP] {name}: atomic numbers {np.unique(bad)} outside MatterSim range")
            skipped.append(name)
        else:
            valid_atoms.append(atoms)
            valid_names.append(name)
    return valid_atoms, valid_names, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Relax CIF structures and compute bulk modulus via MatterSim + E(V) fit"
    )
    parser.add_argument("-d", "--directory", required=True, help="Directory containing CIF files")
    parser.add_argument(
        "--no-relax",
        action="store_true",
        help="Skip BatchRelaxer relaxation (faster but less accurate)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=5,
        help="Number of volume-strain points for E(V) fit (default: 5)",
    )
    parser.add_argument(
        "--strain",
        type=float,
        default=0.03,
        help="Max volumetric strain fraction for E(V) fit (default: 0.03)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: <directory>/bulk_modulus.csv)",
    )
    args = parser.parse_args()

    results_dir = Path(args.directory).resolve()
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory")
        sys.exit(1)

    output_csv = Path(args.output) if args.output else results_dir / "bulk_modulus.csv"

    # ── Load ──────────────────────────────────────────────────────────────────
    atoms_list, names = load_cif_files(results_dir)
    atoms_list, names, skipped_load = filter_valid(atoms_list, names)

    if not atoms_list:
        print("No valid structures to process.")
        sys.exit(1)

    # ── Relax (optional) ──────────────────────────────────────────────────────
    if not args.no_relax:
        print(f"\nRelaxing {len(atoms_list)} structure(s) with BatchRelaxer (EXPCELLFILTER, fmax=0.1)...")
        try:
            relaxed_list, _ = relax_atoms(atoms_list, fmax=0.1)
        except TypeError:
            # fmax may not be a relax_atoms kwarg — fall back without it
            relaxed_list, _ = relax_atoms(atoms_list)
        print("Relaxation complete.")
    else:
        print("\nSkipping relaxation (--no-relax).")
        relaxed_list = atoms_list

    # ── Bulk modulus E(V) sweep ───────────────────────────────────────────────
    print(f"\nCalculating bulk modulus ({args.n_points} E(V) points, strain={args.strain})...")
    print(f"{'File':<45} {'Formula':<20} {'BM (GPa)':>10}")
    print("-" * 78)

    rows = []
    for atoms, name in zip(relaxed_list, names):
        formula = atoms.get_chemical_formula()
        try:
            bm = calc_bulk_modulus_value(atoms, n_points=args.n_points, strain=args.strain)
            bm = max(0.0, float(bm))
            status = f"{bm:10.2f}"
        except Exception as e:
            bm = float("nan")
            status = f"{'ERROR':>10}"
            print(f"  {name:<43} {formula:<20} {status}  ({e})")
            rows.append({"file": name, "formula": formula, "bulk_modulus_gpa": bm, "error": str(e)})
            continue

        print(f"  {name:<43} {formula:<20} {status}")
        rows.append({"file": name, "formula": formula, "bulk_modulus_gpa": bm, "error": ""})

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_bms = [r["bulk_modulus_gpa"] for r in rows if not np.isnan(r["bulk_modulus_gpa"])]
    print("-" * 78)
    if valid_bms:
        print(f"  {'Structures evaluated:':<42} {len(valid_bms):>10}")
        print(f"  {'Mean BM (GPa):':<42} {np.mean(valid_bms):>10.2f}")
        print(f"  {'Median BM (GPa):':<42} {np.median(valid_bms):>10.2f}")
        print(f"  {'Max BM (GPa):':<42} {np.max(valid_bms):>10.2f}")
        print(f"  {'Min BM (GPa):':<42} {np.min(valid_bms):>10.2f}")
    if skipped_load:
        print(f"  Skipped (invalid Z): {skipped_load}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "formula", "bulk_modulus_gpa", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
