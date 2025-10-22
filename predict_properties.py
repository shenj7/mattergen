import os
import sys
import zipfile
import tempfile
from datasets import load_dataset
from collections import defaultdict, Counter
from ase.io import read
from datasets import load_dataset
import torch
from loguru import logger
from ase.build import bulk
from ase.units import GPa
from ase.io import read
from mattersim.forcefield import MatterSimCalculator

def predict_bulk(material_file):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    si = read(material_file)
    si.calc = MatterSimCalculator(device=device)
    print(si.calc)
    print(f"Energy (eV)                 = {si.get_potential_energy()}")
    print(f"Stress[0][0] (GPa)          = {si.get_stress(voigt=False)[0][0] / GPa}")

def extract_zip_to_temp(zip_path: str) -> str:
    """
    Extracts a zip file into a temporary directory and returns its path.
    """
    temp_dir = tempfile.mkdtemp(prefix="cif_unzipped_")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        print(f"Extracted {zip_path} to temporary directory: {temp_dir}")

    return temp_dir


def get_cif_directory(input_path: str) -> str:
    """
    Determines if the input is a zip file or directory, returning the CIF directory path.
                """
    if os.path.isfile(input_path) and input_path.lower().endswith(".zip"):
        return extract_zip_to_temp(input_path)
    elif os.path.isdir(input_path):
        return input_path
    else:
        raise ValueError("Input must be a directory or a .zip file containing CIFs.")



# --- Step 1: get atomic numbers directly from ASE Atoms ---
def get_atomic_numbers_from_cif(file_path):
    atoms = read(file_path)
    return sorted(atoms.get_atomic_numbers())  # already integers

# --- Step 3: process CIF folder against dataset ---
def process_cif_directory(cif_dir, dataset):
    results = defaultdict(list)

    for fname in os.listdir(cif_dir):
        if fname.endswith(".cif"):
            path = os.path.join(cif_dir, fname)
            try:
                atom_list = get_atomic_numbers_from_cif(path)
                bulk = predict_bulk(path)
                results[fname].append({
                    "atomic_numbers": atom_list,
                    "bulk_modulus": bulk
                    })
            except Exception as e:
                print(f"Error with {fname}: {e}")
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <cif_directory_or_zip>")
        sys.exit(1)

    input_path = sys.argv[1]
    # Load your dataset (replace with actual dataset ID or local path)
    dataset = load_dataset("OMatG/Alex-MP-20")

    # Point to your CIF directory
    cif_directory = get_cif_directory(input_path)

    bulks = process_cif_directory(cif_directory, dataset["train"])

    for cif_file, info in bulks.items():
        for entry in info:
            print(f"\nCIF: {cif_file}")
            print(f" Atomic Numbers: {entry['atomic_numbers']}")
            print(f" Predicted Bulk Modulus: {entry['bulk_modulus']}")
            #print(" Matches in dataset:")
            #print(entry["matches"])

