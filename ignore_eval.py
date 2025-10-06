import os
from ase.io import read
from mattersim import MatterSim

# Directory containing generated materials
materials_dir = "results/ml_bulk_modulus/large_batch_novelty"   # change to your folder path

# Initialize MatterSim
sim = MatterSim()

# Loop through files in the directory
for fname in os.listdir(materials_dir):
    if fname.endswith((".traj", ".xyz", ".json")):  # ASE supported formats
        path = os.path.join(materials_dir, fname)

        # Read material with ASE
        try:
            atoms = read(path)  # ASE Atoms object
        except Exception as e:
            print(f"⚠️ Could not read {fname}: {e}")
            continue

        print(f"\n=== {fname} ===")
        print(f"Formula: {atoms.get_chemical_formula()}")

        # Predict properties
        bulk_modulus = sim.predict_bulk_modulus(atoms)
        magnetic_density = sim.predict_magnetic_density(atoms)
        band_gap = sim.predict_band_gap(atoms)

        print(f"Bulk Modulus: {bulk_modulus:.2f} GPa")
        print(f"Magnetic Density: {magnetic_density:.4f} µB/Å³")
        print(f"Band Gap: {band_gap:.3f} eV")

