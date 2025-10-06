export MODEL_NAME=ml_bulk_modulus

for i in $(seq 40 40 400);
do
	export RESULTS_PATH="results/$MODEL_NAME/$i"  # Samples will be written to this directory, e.g., `results/dft_mag_density`
	mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=16 --properties_to_condition_on="{'ml_bulk_modulus': $i}" --diffusion_guidance_factor=2.0
	mattergen-evaluate --structures_path=$RESULTS_PATH --relax=True --structure_matcher='disordered' --save_as="$RESULTS_PATH/metrics.json"
done
