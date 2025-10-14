import os
import sys
import zipfile
import tempfile
from datasets import load_dataset
from collections import defaultdict
from collections import Counter
from ase.io import read
from datasets import load_dataset
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


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
    return atoms.get_atomic_numbers()  # already integers

def get_space_group_from_cif(file_path):
    structure = Structure.from_file(file_path)
    analyzer = SpacegroupAnalyzer(structure)
    space_group_symbol = analyzer.get_space_group_symbol()
    return space_group_symbol
    

# --- Step 2: search HF dataset for matching atomic_numbers ---
def find_matches(dataset, target_composition, space_group):
    return dataset.filter(lambda row: Counter(row["atomic_numbers"]) == Counter(target_composition) and
                                        row["space_group"] == space_group)

# --- Step 3: process CIF folder against dataset ---
def process_cif_directory(cif_dir, dataset):
    results = defaultdict(list)

    for fname in os.listdir(cif_dir):
        if fname.endswith(".cif"):
            path = os.path.join(cif_dir, fname)
            try:
                atom_list = get_atomic_numbers_from_cif(path)
                space_group = get_space_group_from_cif(path)
                print(atom_list)
                print(space_group)
                matches = find_matches(dataset, atom_list, space_group)
                print(matches)
                if len(matches) > 0:
                    results[fname].append({
                        "atomic_numbers": atom_list,
                        "space_group": space_group,
                        "matches": matches
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

    matched = process_cif_directory(cif_directory, dataset["train"])

    for cif_file, info in matched.items():
        for entry in info:
            print(f"\nCIF: {cif_file}")
            print(f" Atomic Numbers: {entry['atomic_numbers']}")
            print(f" Space Group: {entry['space_group']}")
            print(f" Bulk Modulus: {entry['matches']['ml_bulk_modulus']}")
            print(f" Number of matches: {len(entry['matches'])}")
            #print(" Matches in dataset:")
            #print(entry["matches"])

