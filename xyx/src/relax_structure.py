import argparse
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

from mattergen.evaluation.utils.relaxation import relax_structures
from mattergen.common.utils.globals import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relax a structure given in a CIF file."
    )
    parser.add_argument(
        "input_cif_file",
        type=str,
        help="Input CIF file name.",
    )
    parser.add_argument(
        "output_cif_file",
        type=str,
        help="Output CIF file name.",
    )

    args = parser.parse_args()

    this_struct = Structure.from_file(args.input_cif_file)

    relaxed_structures, energies = relax_structures(
            this_struct, device=str(get_device()), potential_load_path=None, output_path=None
    )

    cif_writer = CifWriter(relaxed_structures[0])
    cif_writer.write_file(args.output_cif_file)
