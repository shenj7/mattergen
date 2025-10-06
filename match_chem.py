import os
from collections import defaultdict
from ase.io import read
from datasets import load_dataset

# --- Step 1: get atomic numbers directly from ASE Atoms ---
def get_atomic_numbers_from_cif(file_path):
    atoms = read(file_path)
    return sorted(atoms.get_atomic_numbers())  # already integers

# --- Step 2: search HF dataset for matching atomic_numbers ---
def find_matches(dataset, target_list):
    return dataset.filter(lambda row: set(row["atomic_numbers"]) == set(target_list))

# --- Step 3: process CIF folder against dataset ---
def process_cif_directory(cif_dir, dataset):
    results = defaultdict(list)

    for fname in os.listdir(cif_dir):
        if fname.endswith(".cif"):
            path = os.path.join(cif_dir, fname)
            try:
                atom_list = get_atomic_numbers_from_cif(path)
                matches = find_matches(dataset, atom_list)
                if len(matches) > 0:
                    results[fname].append({
                        "atomic_numbers": atom_list,
                        "matches": matches
                    })
            except Exception as e:
                print(f"Error with {fname}: {e}")
    return results

if __name__ == "__main__":
    # Load your dataset (replace with actual dataset ID or local path)
    dataset = load_dataset("OMatG/Alex-MP-20")

    # Point to your CIF directory
    cif_directory = "./results/ml_bulk_modulus/large_batch_novelty/generated_crystals_cif/"

    matched = process_cif_directory(cif_directory, dataset["train"])

    for cif_file, info in matched.items():
        for entry in info:
            print(f"\nCIF: {cif_file}")
            print(f" Atomic Numbers: {entry['atomic_numbers']}")
            print(f" Bulk Modulus: {entry['matches']['ml_bulk_modulus']}")
            print(f" Number of matches: {len(entry['matches'])}")
            #print(" Matches in dataset:")
            #print(entry["matches"])

