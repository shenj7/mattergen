export MODEL_NAME=dft_band_gap

for i in $(seq 40 40 400);
do
	export RESULTS_PATH="results/$MODEL_NAME/$i"  # Samples will be written to this directory, e.g., `results/dft_mag_density`
	mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=16 --properties_to_condition_on="{'dft_band_gap': $i}" --diffusion_guidance_factor=2.0
done
