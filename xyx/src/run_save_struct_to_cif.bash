#!/bin/bash

cd ../..

input_path=xyx/results/ml_bulk_modulus400/generated_crystals_cif

for struct_file in $input_path/gen*mp2020-disordered-match.gz; 
do
    echo "Processing $struct_file"
    folder=${struct_file}-cif
    mkdir -p $folder
    
    python xyx/src/save_struct_to_cif.py $struct_file $folder
    
done