#!/usr/bin/env python3
# batch_qha_from_zip.py
#
# Usage:
#   python batch_qha_from_zip.py path/to/structures.zip qha_outputs --t-report 300
#
# What it does:
#   - Unzips CIFs from a zip file
#   - For each CIF, makes: out_root/<cif_stem>/
#       * e-v.dat
#       * thermal_properties.yaml-00..NN
#       * runs: phonopy-qha e-v.dat thermal_properties.yaml-{00..NN}
#       * parses bulk_modulus-temperature.dat to report B_T at T_report (closest row)
#   - Writes a summary CSV: out_root/summary_bulk_modulus.csv

import os, sys, csv, math, shutil, zipfile, subprocess
from pathlib import Path
import numpy as np

from ase.io import read as ase_read
from ase import Atoms
from mattersim.forcefield import MatterSimCalculator

from phonopy.structure.atoms import PhonopyAtoms
from phonopy import Phonopy

# ----------------- tiny converters -----------------
def ase_to_phonopy(a: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=a.get_chemical_symbols(),
        cell=a.get_cell().array,
        scaled_positions=a.get_scaled_positions()
    )

def phonopy_to_ase(p: PhonopyAtoms) -> Atoms:
    return Atoms(
        symbols=p.get_chemical_symbols(),
        cell=p.get_cell(),
        scaled_positions=p.get_scaled_positions(),
        pbc=True
    )

# ----------------- core per-structure worker -----------------
def make_qha_inputs_for_cif(
    cif_path: Path,
    outdir: Path,
    n_points: int = 7,
    strain: float = 0.06,
    supercell=(2,2,2),
    disp: float = 0.03,
    t_min: float = 0.0,
    t_max: float = 1000.0,
    t_step: float = 10.0,
    qmesh=(20,20,20),
    device: str | None = None,
    checkpoint: str | None = None,
):
    """Create e-v.dat and thermal_properties.yaml-* for a single CIF."""
    outdir.mkdir(parents=True, exist_ok=True)

    base = ase_read(str(cif_path))
    # calculator
    calc_kwargs = {}
    if device:
        calc_kwargs["device"] = device
    try:
        import torch
        if not device:
            calc_kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        calc_kwargs.setdefault("device", "cpu")
    if checkpoint:
        calc_kwargs["load_path"] = checkpoint
    calc = MatterSimCalculator(**calc_kwargs)

    vol_scales = np.linspace(1.0 - strain, 1.0 + strain, n_points)
    len_scales = vol_scales ** (1.0/3.0)
    suffixes = [f"{i:02d}" for i in range(n_points)]

    ev_rows = []
    for i, s in enumerate(len_scales):
        tag = suffixes[i]
        atoms = base.copy()
        atoms.set_cell(base.get_cell() * s, scale_atoms=True)
        atoms.calc = calc

        # static energy
        E = float(atoms.get_potential_energy())
        V = float(atoms.get_volume())
        ev_rows.append((V, E))
        print(f"    [{i+1}/{n_points}] V={V:.4f} Å^3  E={E:.6f} eV")

        # phonons & thermal props
        ph_base = ase_to_phonopy(atoms)
        phonon = Phonopy(ph_base, supercell_matrix=np.diag(supercell))
        phonon.generate_displacements(distance=disp)

        forces = []
        for scell in phonon.supercells_with_displacements:
            sc_ase = phonopy_to_ase(scell)
            sc_ase.calc = calc
            forces.append(sc_ase.get_forces())   # eV/Å

        phonon.forces = forces
        phonon.produce_force_constants()
        phonon.run_mesh(list(qmesh), is_gamma_center=True)
        phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)

        yaml_path = outdir / f"thermal_properties.yaml-{tag}"
        phonon.write_yaml_thermal_properties(filename=str(yaml_path))

    # write e-v.dat (same order as suffixes)
    with (outdir / "e-v.dat").open("w") as f:
        f.write("# volume(Ang^3)   energy(eV)\n")
        for V, E in ev_rows:
            f.write(f"{V:20.10f} {E:20.10f}\n")

    return suffixes

def run_phonopy_qha(outdir: Path, suffixes: list[str]) -> None:
    """Shell out to phonopy-qha for this folder."""
    # Prefer brace expansion 00..NN; otherwise list explicitly
    last = suffixes[-1]
    yaml_glob = [f"thermal_properties.yaml-{s}" for s in suffixes]
    cmd = ["phonopy-qha", "e-v.dat"] + yaml_glob
    print("    Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(outdir), check=True)

def parse_bulk_modulus_temperature(outdir: Path, t_report: float) -> float | None:
    """Return B_T (GPa) at temperature closest to t_report from bulk_modulus-temperature.dat."""
    cand = None
    path = outdir / "bulk_modulus-temperature.dat"
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            T = float(parts[0])
            B = float(parts[1])
            if cand is None or abs(T - t_report) < abs(cand[0] - t_report):
                cand = (T, B)
    return cand[1] if cand else None

# ----------------- batch driver -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Batch QHA from a zip of CIFs using MatterSim + Phonopy.")
    ap.add_argument("zip_path", help="Path to .zip containing CIF files.")
    ap.add_argument("out_root", help="Output root directory.")
    ap.add_argument("--n-points", type=int, default=7, help="Number of volumes (>=3). Default 7")
    ap.add_argument("--strain", type=float, default=0.06, help="±strain on volume range. Default 0.06")
    ap.add_argument("--supercell", nargs=3, type=int, default=[2,2,2], help="Supercell (e.g., 2 2 2)")
    ap.add_argument("--disp", type=float, default=0.03, help="Displacement distance (Å). Default 0.03")
    ap.add_argument("--t-min", type=float, default=0.0, help="Min T (K). Default 0")
    ap.add_argument("--t-max", type=float, default=1000.0, help="Max T (K). Default 1000")
    ap.add_argument("--t-step", type=float, default=10.0, help="T step (K). Default 10")
    ap.add_argument("--qmesh", nargs=3, type=int, default=[20,20,20], help="q-point mesh (e.g., 20 20 20)")
    ap.add_argument("--t-report", type=float, default=300.0, help="Report B_T at this T (K). Default 300")
    ap.add_argument("--device", default=None, help="Forcefield device: cpu|cuda|auto (default auto)")
    ap.add_argument("--checkpoint", default=None, help="Path to MatterSim checkpoint (optional).")
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # unzip to temp folder
    tmp_extract = out_root / "_unzipped"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    tmp_extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_extract)

    # find all CIFs
    cif_files = sorted(tmp_extract.rglob("*.cif"))
    if not cif_files:
        print("No CIF files found in the zip.")
        sys.exit(1)

    summary_rows = []
    print(f"Found {len(cif_files)} CIF(s). Starting…\n")

    for idx, cif in enumerate(cif_files, 1):
        stem = cif.stem
        outdir = out_root / stem
        print(f"[{idx}/{len(cif_files)}] Processing {cif.name} → {outdir.name}/")

        try:
            suffixes = make_qha_inputs_for_cif(
                cif_path=cif,
                outdir=outdir,
                n_points=args.n_points,
                strain=args.strain,
                supercell=tuple(args.supercell),
                disp=args.disp,
                t_min=args.t_min,
                t_max=args.t_max,
                t_step=args.t_step,
                qmesh=tuple(args.qmesh),
                device=args.device,
                checkpoint=args.checkpoint,
            )

            # run phonopy-qha
            run_phonopy_qha(outdir, suffixes)

            # parse B_T at T_report
            B_T = parse_bulk_modulus_temperature(outdir, args.t_report)
            summary_rows.append({
                "material": stem,
                "outdir": str(outdir),
                "T_report_K": args.t_report,
                "B_T_GPa": f"{B_T:.4f}" if B_T is not None else ""
            })

        except subprocess.CalledProcessError as e:
            print(f"    ERROR: phonopy-qha failed for {stem}: {e}")
            summary_rows.append({
                "material": stem,
                "outdir": str(outdir),
                "T_report_K": args.t_report,
                "B_T_GPa": ""
            })
        except Exception as e:
            print(f"    ERROR while processing {stem}: {e}")
            summary_rows.append({
                "material": stem,
                "outdir": str(outdir),
                "T_report_K": args.t_report,
                "B_T_GPa": ""
            })

    # write summary CSV
    summary_csv = out_root / "summary_bulk_modulus.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["material", "outdir", "T_report_K", "B_T_GPa"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nAll done.")
    print(f"Summary: {summary_csv}")
    print("Per-structure QHA outputs live in their respective folders.")
    print("Tip: open *_/bulk_modulus-temperature.dat to see the full B_T(T) curve.")
    
if __name__ == "__main__":
    main()

