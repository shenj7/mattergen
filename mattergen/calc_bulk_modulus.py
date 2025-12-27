#!/usr/bin/env python3
# barebones_qha_inputs.py (fixed for run_mesh)
# Usage: python barebones_qha_inputs.py your_structure.cif [optional_outdir]

import sys
from pathlib import Path
import numpy as np
from ase.io import read as ase_read
from ase import Atoms
import torch
from mattersim.forcefield import MatterSimCalculator
from phonopy.structure.atoms import PhonopyAtoms
from phonopy import Phonopy
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

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

def calc_bulk_modulus(cif_path: str, outdir: str = "qha_out"):
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

def _fit_bulk_modulus(ev_rows: list[tuple[float, float]]) -> float:
    """
    Fit a polynomial to the E(V) curve and estimate the bulk modulus via
    B = V * d^2E/dV^2 evaluated at the minimum-energy volume.
    Returns the bulk modulus in GPa.
    """
    volumes, energies = np.array(ev_rows).T
    # Quadratic is sufficient for small strains, keeps things stable
    poly = np.poly1d(np.polyfit(volumes, energies, 2))
    d1 = poly.deriv(1)
    # critical point of quadratic is unique
    vol0 = float(-d1[1] / d1[0]) if d1[0] != 0 else float(volumes.mean())
    # clamp to sampled range to avoid extrapolation instability
    vol0 = float(np.clip(vol0, volumes.min(), volumes.max()))
    curvature = float(poly.deriv(2)(vol0))
    # 1 eV / Ang^3 = 160.21766208 GPa
    return vol0 * curvature * 160.21766208

def calc_bulk_modulus_value(
    atoms: Atoms,
    n_points: int = 5,
    strain: float = 0.03,
) -> float:
    """
    Lightweight bulk modulus estimator from a single ASE Atoms object.
    Uses static energy-volume sampling (no phonon calculation) and fits a quadratic E(V).
    """
    atoms = atoms.copy()
    if atoms.calc is None:
        atoms.calc = MatterSimCalculator()

    vol_scales = np.linspace(1.0 - strain, 1.0 + strain, n_points)
    len_scales = vol_scales ** (1.0 / 3.0)

    ev_rows: list[tuple[float, float]] = []
    base_cell = atoms.get_cell()
    base_positions = atoms.get_positions()
    for scale in len_scales:
        atoms.set_cell(base_cell * scale, scale_atoms=False)
        atoms.set_positions(base_positions * scale)
        E = float(atoms.get_potential_energy())
        V = float(atoms.get_volume())
        ev_rows.append((V, E))
    return _fit_bulk_modulus(ev_rows)

def _as_atoms(obj: Atoms | Structure) -> Atoms:
    if isinstance(obj, Atoms):
        return obj
    if isinstance(obj, Structure):
        return AseAtomsAdaptor.get_atoms(obj)
    raise TypeError(f"Unsupported structure type: {type(obj)}")

def risk_seeking_bulk_modulus_gradient(
    structure: Atoms | Structure,
    risk_beta: float = 1.0,
    pos_eps: float = 1e-3,
    cell_eps: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """
    Finite-difference estimator of the risk-seeking policy gradient for bulk modulus.

    Objective: U = exp(risk_beta * B), where B is the bulk modulus (GPa).
    Returns gradients of U w.r.t. atomic positions and cell entries, packaged for
    external diffusion guidance (keys: 'pos', 'cell').
    """
    atoms = _as_atoms(structure)

    def utility(a: Atoms) -> float:
        return float(np.exp(risk_beta * calc_bulk_modulus_value(a)))

    base_pos = atoms.get_positions()
    base_cell = atoms.get_cell().array

    grad_pos = np.zeros_like(base_pos)
    grad_cell = np.zeros_like(base_cell)

    # Position gradients
    for i in range(len(atoms)):
        for d in range(3):
            atoms_pos_plus = atoms.copy()
            atoms_pos_minus = atoms.copy()
            pos_plus = base_pos.copy(); pos_plus[i, d] += pos_eps
            pos_minus = base_pos.copy(); pos_minus[i, d] -= pos_eps
            atoms_pos_plus.set_positions(pos_plus)
            atoms_pos_minus.set_positions(pos_minus)
            u_plus = utility(atoms_pos_plus)
            u_minus = utility(atoms_pos_minus)
            grad_pos[i, d] = (u_plus - u_minus) / (2 * pos_eps)

    # Cell gradients (applied with atom scaling to preserve fractional coords)
    for i in range(3):
        for j in range(3):
            cell_plus = base_cell.copy()
            cell_minus = base_cell.copy()
            cell_plus[i, j] += cell_eps
            cell_minus[i, j] -= cell_eps

            atoms_cell_plus = atoms.copy(); atoms_cell_plus.set_cell(cell_plus, scale_atoms=True)
            atoms_cell_minus = atoms.copy(); atoms_cell_minus.set_cell(cell_minus, scale_atoms=True)

            u_plus = utility(atoms_cell_plus)
            u_minus = utility(atoms_cell_minus)
            grad_cell[i, j] = (u_plus - u_minus) / (2 * cell_eps)

    return {
        "pos": torch.tensor(grad_pos, dtype=torch.float32),
        "cell": torch.tensor(grad_cell[None, ...], dtype=torch.float32),
    }

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python barebones_qha_inputs.py your_structure.cif [optional_outdir]")
        sys.exit(1)
    cif = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) == 3 else "qha_out"
    calc_bulk_modulus(cif, outdir)
