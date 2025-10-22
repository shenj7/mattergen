export MODEL_NAME=ml_bulk_modulus

for i in $(seq 40 40 400);
do
	export RESULTS_PATH="results/$MODEL_NAME/$i"  # Samples will be written to this directory, e.g., `results/dft_mag_density`
	python mattersim_pred_by_zip.py "$RESULTS_PATH/generated_crystals_cif.zip" "$RESULTS_PATH/qha_outputs" --t-report 300 > "$RESULTS_PATH/mattersim_pred_results.txt"
done
