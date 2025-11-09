#!/bin/bash

cd ../..

input_path=xyx/results/ml_bulk_modulus400/generated_crystals_cif

for num in {0..15}; 
do
    struct_file=$input_path/gen_${num}.cif
    relaxed_struct_file=$input_path/gen_${num}.relaxed.cif
    echo "Processing $struct_file"
    
    python xyx/src/relax_structure.py $struct_file $relaxed_struct_file
done