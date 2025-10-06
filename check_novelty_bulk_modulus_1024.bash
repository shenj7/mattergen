export MODEL_NAME=ml_bulk_modulus
export RESULTS_PATH="results/$MODEL_NAME/large_batch_novelty"  # Samples will be written to this directory, e.g., `results/dft_mag_density`
mattergen-generate $RESULTS_PATH --pretrained-name=$MODEL_NAME --batch_size=1000 --properties_to_condition_on="{'ml_bulk_modulus': 400}" --diffusion_guidance_factor=2.0
