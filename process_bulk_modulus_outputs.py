#!/usr/bin/env python3
import os
import sys
import zipfile
import csv
import shutil
from pathlib import Path
from ase.io import read as ase_read
from calc_bulk_modulus_single import calc_bulk_modulus

def process_bulk_modulus_outputs(directory: str):
    """
    Process outputs in the given directory to calculate bulk modulus.
    1. Unzip generated_crystals_cif.zip
    2. Calculate bulk modulus for each CIF
    3. Save results to bulk_modulus_results.csv
    """
    directory = Path(directory).resolve()
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist.")
        return

    zip_file = directory / "generated_crystals_cif.zip"
    if not zip_file.exists():
        print(f"Error: {zip_file} does not exist.")
        return

    # Unzip
    extract_dir = directory / "generated_crystals_cif"
    print(f"Unzipping {zip_file} to {extract_dir}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    results = []
    cif_files = sorted(list(extract_dir.glob("*.cif")))
    
    if not cif_files:
        print("No CIF files found in the extracted directory.")
        return

    print(f"Found {len(cif_files)} CIF files. Starting calculations...")

    # Create a temporary directory for QHA calculations to keep things clean
    qha_outdir = directory / "qha_temp"
    qha_outdir.mkdir(exist_ok=True)

    success_count = 0
    fail_count = 0

    for i, cif_path in enumerate(cif_files):
        filename = cif_path.name
        print(f"Processing ({i+1}/{len(cif_files)}): {filename}")

        try:
            # Get chemical composition
            atoms = ase_read(cif_path)
            composition = atoms.get_chemical_formula()

            # Calculate bulk modulus
            # We use a try-except block here because calc_bulk_modulus might fail 
            # (e.g., if phonopy fails or structure is invalid)
            bulk_modulus = calc_bulk_modulus(str(cif_path), outdir=str(qha_outdir))
            
            results.append({
                "filename": filename,
                "composition": composition,
                "bulk_modulus": bulk_modulus
            })
            success_count += 1
            print(f"  -> Bulk Modulus: {bulk_modulus:.4f} GPa")

        except Exception as e:
            print(f"  -> Failed: {e}")
            fail_count += 1
            results.append({
                "filename": filename,
                "composition": "Error",
                "bulk_modulus": None,
                "error": str(e)
            })

    # Cleanup temp dir
    # shutil.rmtree(qha_outdir)
    print(f"Cleaning up {qha_outdir}...") 
    # Only remove if you want to save space, but maybe useful for debugging?
    # For now, let's keep it or remove it? The user didn't specify. 
    # I'll leave it but maybe I should have asked? 
    # Actually, the user asked for a script that "runs calc_bulk_modulus_single", 
    # presumably they might want to inspect if things fail. 
    # But for a batch processing script, cleaning up is usually polite.
    # Let's clean it up to avoid cluttering the user's workspace with processed files.
    if qha_outdir.exists():
        shutil.rmtree(qha_outdir)

    # Save to CSV
    csv_file = directory / "bulk_modulus_results.csv"
    print(f"Saving results to {csv_file}...")
    
    fieldnames = ["filename", "composition", "bulk_modulus", "error"]
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Done! processed {len(cif_files)} files.")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_bulk_modulus_outputs.py <directory_path>")
        sys.exit(1)
    
    process_bulk_modulus_outputs(sys.argv[1])
