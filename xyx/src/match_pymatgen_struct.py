import os
import sys
import zipfile
import tempfile

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser

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

def find_match(cif_file, dataset_file):
    matcher = StructureMatcher()
    
    this_struct = Structure.from_file(cif_file)
    #print('this_struct=', this_struct)

    df = pd.read_csv(dataset_file)
    matchedlist = pd.DataFrame()
    for index, material in df.iterrows():
        parser = CifParser.from_str(material['cif'])
        #print('parser=', parser)
        struct = parser.get_structures()[0]
        #print('struct=', struct)
        if matcher.fit(struct, this_struct):
            if len(matchedlist) == 0:
                matchedlist = pd.DataFrame(material)
            else:
                matchedlist.loc[len(matchedlist)] = material

        if index % 1000 == 0:
            print("      [processed "+str(index)+" materials, #(matching)="+str(len(matchedlist))+']')

    return matchedlist
    

# --- Step 3: process CIF folder against dataset ---
def process_cif_directory(cif_dir):
    global mp2020_path

    for fname in ['gen_9.cif', 'gen_12.cif']: #os.listdir(cif_dir):
        if fname.endswith(".cif"):
            path = os.path.join(cif_dir, fname)
            try:
                print('[G]  '+fname)
                matchlist = find_match(path, mp2020_path+'/train.csv')
                print('[G]   num matched materials (after train)=', len(matchlist))
                matchlist = pd.concat([matchlist, find_match(path, mp2020_path+'/val.csv')], ignore_index=True)
                print('[G]   num matched materials (after train, val)=', len(matchlist))
                matchlist = pd.concat([matchlist, find_match(path, mp2020_path+'/test.csv')], ignore_index=True)
                print('[G]   num matched materials (total)=', len(matchlist))
                
                outfile=path+'.mp20-match.csv'
                matchlist.to_csv(outfile, index=False, sep=',')
                
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

    
