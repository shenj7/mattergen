export MODEL_NAME=ml_bulk_modulus

for i in $(seq 40 40 400);
do
	export RESULTS_PATH="results/$MODEL_NAME/$i"  # Samples will be written to this directory, e.g., `results/dft_mag_density`
	python match_chem_space_group.py "$RESULTS_PATH/generated_crystals_cif.zip" > "$RESULTS_PATH/properties_space_group.txt"
done
