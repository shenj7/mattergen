#!/usr/bin/env python3
"""
analyze_structures.py

For each CIF file in a given directory or zip, computes:
  1. Bulk modulus (MatterSim + E(V) curve fitting, unrelaxed)
  2. Energy above convex hull (unrelaxed, using MP reference entries)
  3. Relaxes the structure with BatchRelaxer (EXPCELLFILTER)
  4. Bulk modulus after relaxation
  5. Energy above convex hull after relaxation

NOTE: MatterSim energies are not on the same absolute scale as MP DFT energies,
so hull distances are indicative (relative ranking is meaningful, absolute values
are not directly comparable to published MP hull distances).

Usage:
    python analyze_structures.py -d /path/to/results [--n-points 5] [--strain 0.03]
"""

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Element
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.ase import AseAtomsAdaptor

load_dotenv()
MP_API_KEY = os.environ.get("MP_API_KEY")
if not MP_API_KEY:
    print("ERROR: MP_API_KEY not found in environment / .env file.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from calc_bulk_modulus_single import _fit_bulk_modulus, calc_bulk_modulus_value
from mattergen.evaluation.utils.relaxation import relax_atoms


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cif_files(directory: Path) -> tuple[list[Atoms], list[str]]:
    """
    Load CIF files from a directory. If generated_crystals_cif.zip is present,
    extract it first, then read all *.cif files in the directory.
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

    cif_paths = sorted(directory.glob("*.cif"))
    if not cif_paths:
        print(f"No CIF files found in {directory}.")
        sys.exit(1)
    print(f"Found {len(cif_paths)} CIF file(s).")

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


# ---------------------------------------------------------------------------
# Bulk modulus
# ---------------------------------------------------------------------------

def compute_bulk_modulus(atoms: Atoms, calc, n_points: int, strain: float) -> float:
    """E(V) sweep + quadratic fit on a single structure using a shared calculator."""
    atoms = atoms.copy()
    atoms.calc = calc
    vol_scales = np.linspace(1.0 - strain, 1.0 + strain, n_points)
    len_scales = vol_scales ** (1.0 / 3.0)
    base_cell = atoms.get_cell().array.copy()
    base_pos = atoms.get_positions().copy()
    ev_rows = []
    for scale in len_scales:
        atoms.set_cell(base_cell * scale, scale_atoms=False)
        atoms.set_positions(base_pos * scale)
        with torch.enable_grad():
            E = float(atoms.get_potential_energy())
        V = float(atoms.get_volume())
        ev_rows.append((V, E))
    return _fit_bulk_modulus(ev_rows)


# ---------------------------------------------------------------------------
# Energy above hull
# ---------------------------------------------------------------------------

def get_mp_reference_entries(elements: list[str]) -> list:
    """Fetch all stable MP entries for the given chemical system."""
    with MPRester(MP_API_KEY) as mpr:
        entries = mpr.get_entries_in_chemsys(elements)
    return entries


def compute_e_above_hull(atoms: Atoms, mp_entries: list) -> float:
    """
    Compute energy above convex hull (eV/atom) using MatterSim total energy
    and MP reference entries.

    Note: MatterSim and MP DFT energies are on different absolute scales;
    the returned value is indicative, not directly comparable to MP hull distances.
    """
    with torch.enable_grad():
        energy_ev = float(atoms.get_potential_energy())
    n_atoms = len(atoms)
    structure = AseAtomsAdaptor.get_structure(atoms)
    composition = structure.composition
    entry = ComputedEntry(composition, energy_ev)
    all_entries = mp_entries + [entry]
    pd = PhaseDiagram(all_entries)
    return pd.get_e_above_hull(entry)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze generated crystal structures")
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="Directory containing CIF files or generated_crystals_cif.zip")
    parser.add_argument("--n-points", type=int, default=5,
                        help="Number of E(V) points for bulk modulus fit (default: 5)")
    parser.add_argument("--strain", type=float, default=0.03,
                        help="Max volumetric strain for E(V) sweep (default: 0.03)")
    parser.add_argument("--fmax", type=float, default=0.1,
                        help="Force convergence threshold for relaxation in eV/Å (default: 0.1)")
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        print(f"ERROR: {directory} is not a directory.")
        sys.exit(1)

    # Load structures
    atoms_list, names = load_cif_files(directory)
    if not atoms_list:
        print("No structures loaded.")
        sys.exit(1)

    # Load MatterSim calculator once
    print("Loading MatterSim calculator...")
    from mattersim.forcefield import MatterSimCalculator
    calc = MatterSimCalculator()
    for atoms in atoms_list:
        atoms.calc = calc

    # Fetch MP reference entries per unique chemical system
    print("Fetching MP reference entries...")
    chemsys_cache: dict[frozenset, list] = {}

    def get_mp_entries_for(atoms: Atoms) -> list:
        elements = sorted({el.symbol for el in AseAtomsAdaptor.get_structure(atoms).composition.elements})
        key = frozenset(elements)
        if key not in chemsys_cache:
            print(f"  Fetching MP entries for system: {'-'.join(elements)}")
            chemsys_cache[key] = get_mp_reference_entries(elements)
        return chemsys_cache[key]

    results = []

    # -----------------------------------------------------------------------
    # Step 1 & 2: unrelaxed bulk modulus + hull distance
    # -----------------------------------------------------------------------
    print("\n--- Unrelaxed analysis ---")
    unrelaxed_bm, unrelaxed_hull = [], []
    for i, (atoms, name) in enumerate(zip(atoms_list, names)):
        print(f"[{i+1}/{len(atoms_list)}] {name}")
        bm = None
        try:
            bm = compute_bulk_modulus(atoms, calc, args.n_points, args.strain)
            print(f"  Bulk modulus (unrelaxed): {bm:.2f} GPa")
        except Exception as e:
            print(f"  Bulk modulus (unrelaxed): FAILED ({e})")
        hull = None
        try:
            mp_entries = get_mp_entries_for(atoms)
            hull = compute_e_above_hull(atoms, mp_entries)
            print(f"  E above hull (unrelaxed): {hull:.4f} eV/atom")
        except Exception as e:
            print(f"  E above hull (unrelaxed): FAILED ({e})")
        unrelaxed_bm.append(bm)
        unrelaxed_hull.append(hull)

    # -----------------------------------------------------------------------
    # Step 3: batch relaxation
    # -----------------------------------------------------------------------
    print("\n--- Relaxing structures (BatchRelaxer) ---")
    try:
        relaxed_list, _ = relax_atoms(atoms_list, fmax=args.fmax)
        for atoms in relaxed_list:
            atoms.calc = calc
    except Exception as e:
        print(f"BatchRelaxer failed ({e}); using unrelaxed structures for post-relaxation metrics.")
        relaxed_list = atoms_list

    # -----------------------------------------------------------------------
    # Step 4 & 5: relaxed bulk modulus + hull distance
    # -----------------------------------------------------------------------
    print("\n--- Relaxed analysis ---")
    relaxed_bm, relaxed_hull = [], []
    for i, (atoms, name) in enumerate(zip(relaxed_list, names)):
        print(f"[{i+1}/{len(relaxed_list)}] {name}")
        bm = None
        try:
            bm = compute_bulk_modulus(atoms, calc, args.n_points, args.strain)
            print(f"  Bulk modulus (relaxed):   {bm:.2f} GPa")
        except Exception as e:
            print(f"  Bulk modulus (relaxed):   FAILED ({e})")
        hull = None
        try:
            mp_entries = get_mp_entries_for(atoms)
            hull = compute_e_above_hull(atoms, mp_entries)
            print(f"  E above hull (relaxed):   {hull:.4f} eV/atom")
        except Exception as e:
            print(f"  E above hull (relaxed):   FAILED ({e})")
        relaxed_bm.append(bm)
        relaxed_hull.append(hull)

    # -----------------------------------------------------------------------
    # Output CSV
    # -----------------------------------------------------------------------
    out_path = directory / "analysis_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "bulk_modulus_unrelaxed_GPa",
            "e_above_hull_unrelaxed_eV_per_atom",
            "bulk_modulus_relaxed_GPa",
            "e_above_hull_relaxed_eV_per_atom",
        ])
        for name, bm_u, hull_u, bm_r, hull_r in zip(
            names, unrelaxed_bm, unrelaxed_hull, relaxed_bm, relaxed_hull
        ):
            writer.writerow([name, bm_u, hull_u, bm_r, hull_r])

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
