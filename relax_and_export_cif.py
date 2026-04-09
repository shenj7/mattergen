#!/usr/bin/env python3
"""
relax_and_export_cif.py

Loads all structures from a generation output directory, relaxes them with
BatchRelaxer (via MatterSim), and writes individual CIF files to an output
directory.

Usage:
    python relax_and_export_cif.py <generation_dir> [--output-dir DIR]
                                   [--fmax FLOAT] [--device DEVICE]
                                   [--potential PATH]
"""

import argparse
import sys
from pathlib import Path

import ase.io
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.common.utils.eval_utils import load_structures
from mattergen.evaluation.utils.relaxation import relax_structures


def parse_args():
    parser = argparse.ArgumentParser(description="Relax generated structures and export CIF files.")
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="Path to the generation output directory (containing generated_crystals.extxyz or generated_crystals_cif.zip).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write relaxed CIF files. Defaults to <generation_dir>/relaxed_cif/.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence threshold for relaxation (default: 0.05 eV/Å).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for MatterSim (e.g. 'cuda', 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--potential",
        type=str,
        default=None,
        help="Path to a custom MatterSim potential checkpoint. Uses default if not specified.",
    )
    return parser.parse_args()


def find_structures_path(generation_dir: Path) -> Path:
    """Return the best available structures file in generation_dir."""
    # Prefer extxyz (single file, faster to load)
    extxyz = generation_dir / "generated_crystals.extxyz"
    if extxyz.exists():
        return extxyz
    # Fall back to zip of CIFs
    zip_path = generation_dir / "generated_crystals_cif.zip"
    if zip_path.exists():
        return zip_path
    # Fall back to CIF directory
    cif_dir = generation_dir / "generated_crystals_cif"
    if cif_dir.is_dir():
        return cif_dir
    raise FileNotFoundError(
        f"No recognised generation output found in {generation_dir}. "
        "Expected 'generated_crystals.extxyz', 'generated_crystals_cif.zip', "
        "or 'generated_crystals_cif/'."
    )


def main():
    args = parse_args()

    generation_dir = args.generation_dir.resolve()
    if not generation_dir.is_dir():
        print(f"ERROR: {generation_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or generation_dir / "relaxed_cif"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load structures ---
    structures_path = find_structures_path(generation_dir)
    print(f"Loading structures from: {structures_path}")
    structures = list(load_structures(structures_path))
    print(f"Loaded {len(structures)} structures.")

    if not structures:
        print("No structures found. Exiting.")
        sys.exit(0)

    # --- Relax ---
    relax_kwargs = {"fmax": args.fmax}
    device_kwargs = {}
    if args.device:
        device_kwargs["device"] = args.device

    print(f"Relaxing {len(structures)} structures (fmax={args.fmax}) ...")
    relaxed_structures, total_energies = relax_structures(
        structures,
        potential_load_path=args.potential,
        **device_kwargs,
        **relax_kwargs,
    )
    print("Relaxation complete.")

    # --- Export CIF files ---
    print(f"Writing CIF files to: {output_dir}")
    for i, structure in enumerate(relaxed_structures):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        # Attach energy as info so it appears in the CIF comment
        atoms.info["total_energy_eV"] = float(total_energies[i])
        cif_path = output_dir / f"relaxed_{i:04d}.cif"
        ase.io.write(str(cif_path), atoms, format="cif")

    print(f"Done. {len(relaxed_structures)} CIF files written to {output_dir}.")


if __name__ == "__main__":
    main()
