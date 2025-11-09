import os
import sys
import zipfile
import tempfile

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser

from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)

import pandas as pd

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

mp2020_path="datasets/mp_20"

from mattergen.evaluation.reference.presets import ReferenceMP2020Correction
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer

ref_mp2020 = ReferenceMP2020Correction()

def find_match(cif_file):
    global ref_mp2020
    #matcher = StructureMatcher()
    matcher = DefaultDisorderedStructureMatcher()
    
    this_struct = Structure.from_file(cif_file)
    #print('this_struct=', this_struct)

    num_matched = 0
    matchedlist = []
    index = 0
    for item in ref_mp2020:
        if matcher.fit(item.structure, this_struct):
            num_matched += 1
            matchedlist.append(item)
        index += 1
        if index % 1000 == 0:
            print("      [processed "+str(index)+" materials, #(matching)="+str(len(matchedlist))+']')
            
    return matchedlist
    

# --- Step 3: process CIF folder against dataset ---
def process_cif_directory(cif_dir):

    for fname in os.listdir(cif_dir):
        if fname.endswith(".cif"):
            path = os.path.join(cif_dir, fname)
            try:
                print('[G]  '+fname)
                matchedlist = find_match(path)
                outfile=path+'.mp2020-disordered-match.gz'
                if len(matchedlist) > 0:
                    dataset_to_save = ReferenceDataset.from_entries(name=fname+".mp2020-disordered-match", entries=matchedlist)
                    LMDBGZSerializer().serialize(dataset_to_save, outfile)
            except Exception as e:
                print(f"Error with {fname}: {e}")
                


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <cif_directory_or_zip>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Point to your CIF directory
    cif_directory = get_cif_directory(input_path)
    print('cif_directory=', cif_directory)

    process_cif_directory(cif_directory)

    
