#!/bin/bash

cd ../..

input_path=xyx/results/ml_bulk_modulus400/generated_crystals_cif

#python xyx/src/match_pymatgen_struct.py $input_path 1> $input_path.mp20-match.log 2> $input_path.mp20-match.err

python xyx/src/match_pymatgen_struct_mp2020.py $input_path 1> $input_path.relaxed.mp2020-disordered-match.log 2> $input_path.relaxed.mp2020-disordered-match.err

