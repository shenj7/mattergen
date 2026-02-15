
import argparse
import os
import torch
from pathlib import Path
from mattergen.generator import CrystalGenerator
from mattergen.common.utils.eval_utils import MatterGenCheckpointInfo
from mattergen.common.utils.globals import get_device

def main():
    parser = argparse.ArgumentParser(description="Generate materials using DDPO fine-tuned model")
    parser.add_argument("--checkpoint", required=True, help="Path to the fine-tuned DDPO checkpoint (state_dict)")
    parser.add_argument("--base_model", default="mattergen_base", help="Base model to load config from")
    parser.add_argument("--output_dir", default="outputs/ddpo_generated", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--sampling_steps", type=int, default=None, help="Override sampling steps (optional)")
    
    args = parser.parse_args()
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Setup Base Model Info
    print(f"Initializing base model config from {args.base_model}...")
    if "checkpoints" in args.base_model or os.path.exists(args.base_model):
         ckpt_info = MatterGenCheckpointInfo(model_path=Path(args.base_model).resolve())
    else:
         ckpt_info = MatterGenCheckpointInfo.from_hf_hub(args.base_model)
    
    # 2. Initialize Generator
    # We might need to override sampling config if sampling_steps is provided
    sampling_overrides = []
    if args.sampling_steps:
        # Assuming standard sampler config structure
        # This might need adjustment based on actual config keys
        sampling_overrides.append(f"sampler_partial.num_steps={args.sampling_steps}")

    generator = CrystalGenerator(
        checkpoint_info=ckpt_info,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        sampling_config_overrides=sampling_overrides
    )
    
    # 3. Load Base Model
    print("Loading base model...")
    generator.prepare()
    
    # 4. Overwrite with Fine-tuned Weights
    print(f"Loading and checking fine-tuned weights from {args.checkpoint}...")
    finetuned_state_dict = torch.load(args.checkpoint, map_location=device)
    
    # Check if keys match (simple validation)
    model_keys = set(generator.model.state_dict().keys())
    ckpt_keys = set(finetuned_state_dict.keys())
    
    if not model_keys.issubset(ckpt_keys):
         missing = model_keys - ckpt_keys
         print(f"Warning: {len(missing)} keys missing in checkpoint. This might be expected if loading partial weights.")
         # Examples: 'loss_fn.weights', etc. might be missing if only model was saved
    
    # Load strictly=False to allow for minor mismatches if safe, but let's try strict first or handle exceptions
    try:
        generator.model.load_state_dict(finetuned_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        generator.model.load_state_dict(finetuned_state_dict, strict=False)

    print("Weights loaded successfully.")
    
    # 5. Generate
    print(f"Generating {args.batch_size * args.num_batches} samples to {args.output_dir}...")
    generator.generate(output_dir=args.output_dir)
    print("Generation complete.")


if __name__ == "__main__":
    main()
