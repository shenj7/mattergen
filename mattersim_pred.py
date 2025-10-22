#!/usr/bin/env python3
# barebones_qha_inputs.py (fixed for run_mesh)
# Usage: python barebones_qha_inputs.py your_structure.cif [optional_outdir]

import sys
from pathlib import Path
import numpy as np
from ase.io import read as ase_read
from ase import Atoms
from mattersim.forcefield import MatterSimCalculator
from phonopy.structure.atoms import PhonopyAtoms
from phonopy import Phonopy

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

def main(cif_path: str, outdir: str = "qha_out"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    base = ase_read(cif_path)
    calc = MatterSimCalculator()

    n_points = 7
    strain = 0.06
    vol_scales = np.linspace(1.0 - strain, 1.0 + strain, n_points)
    len_scales = vol_scales ** (1.0/3.0)
    suffixes = [f"{i:02d}" for i in range(n_points)]

    supercell = np.diag([2,2,2])
    disp = 0.03
    t_min, t_max, t_step = 0.0, 1000.0, 10.0

    # choose a q-mesh (keep the same across volumes)
    qmesh = [20, 20, 20]  # adjust for speed/accuracy

    ev_rows = []
    for i, s in enumerate(len_scales):
        tag = suffixes[i]
        atoms = base.copy()
        atoms.set_cell(base.get_cell() * s, scale_atoms=True)
        atoms.calc = calc

        E = float(atoms.get_potential_energy())
        V = float(atoms.get_volume())
        ev_rows.append((V, E))
        print(f"[{i+1}/{n_points}] V={V:.4f} Å^3  E={E:.6f} eV")

        ph_base = ase_to_phonopy(atoms)
        phonon = Phonopy(ph_base, supercell_matrix=supercell)
        phonon.generate_displacements(distance=disp)

        forces = []
        for scell in phonon.supercells_with_displacements:
            sc_ase = phonopy_to_ase(scell)
            sc_ase.calc = calc
            forces.append(sc_ase.get_forces())  # eV/Å

        phonon.forces = forces
        phonon.produce_force_constants()

        # >>> REQUIRED before thermal properties:
        phonon.run_mesh(qmesh, is_gamma_center=True)

        phonon.run_thermal_properties(t_min=t_min, t_max=t_max, t_step=t_step)

        yaml_path = out / f"thermal_properties.yaml-{tag}"
        phonon.write_yaml_thermal_properties(filename=str(yaml_path))

    with (out / "e-v.dat").open("w") as f:
        f.write("# volume(Ang^3)   energy(eV)\n")
        for V, E in ev_rows:
            f.write(f"{V:20.10f} {E:20.10f}\n")

    last = suffixes[-1]
    print("\nFiles written:")
    print(f"  {out/'e-v.dat'}")
    for s in suffixes:
        print(f"  {out/f'thermal_properties.yaml-{s}'}")
    print("\nNext step:")
    print(f"  cd {out}")
    print(f"  phonopy-qha e-v.dat thermal_properties.yaml-{{00..{last}}}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python barebones_qha_inputs.py your_structure.cif [optional_outdir]")
        sys.exit(1)
    cif = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) == 3 else "qha_out"
    main(cif, outdir)

