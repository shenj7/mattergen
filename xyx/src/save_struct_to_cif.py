from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer

from pymatgen.io.cif import CifWriter

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save the list of pymatgen structures into a series of CIF files in a folder.")
    parser.add_argument("input_file", type=str, help="The file containing Pymatgen structures (generated using LMDBGZSerializer).")
    parser.add_argument("output_folder", type=str, help="The output folder to save the CIF files.")

    args = parser.parse_args()

    structures = LMDBGZSerializer().deserialize(args.input_file)

    for idx, struct in enumerate(structures):
        cif_filename = args.output_folder+"/struct_"+str(idx)+".cif"

        cif_writer = CifWriter(struct.structure)
        cif_writer.write_file(cif_filename)
