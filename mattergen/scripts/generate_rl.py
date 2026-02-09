#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate materials using the DDPO-trained model.

Usage:
    python -m mattergen.scripts.generate_rl --batch-size 8 --num-batches 2 --output-dir outputs_rl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from pymatgen.core.structure import Structure

from mattergen.common.utils.eval_utils import (
    MatterGenCheckpointInfo,
    save_structures,
)
from mattergen.common.utils.globals import get_device
from mattergen.generator import CrystalGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate materials with MatterGen")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned RL model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="checkpoints/mattergen_base",
        help="Path to base model directory (containing .hydra config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of samples per batch (default: 8)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=2,
        help="Number of batches to generate (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_rl",
        help="Output directory for generated structures (default: outputs_rl)",
    )
    parser.add_argument(
        "--bulk-modulus-classifier",
        type=str,
        default=None,
        help="Path to bulk modulus classifier checkpoint for guidance",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Bulk modulus guidance scale (default: 0.0, no guidance)",
    )
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Don't record intermediate trajectories",
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.batch_size * args.num_batches} materials...")
    print(f"Output directory: {output_path}")

    # Load base model checkpoint info (for config and architecture)
    base_ckpt_path = Path(args.base_checkpoint).resolve()
    if not base_ckpt_path.exists():
         # Fallback logic or error
         print(f"Base checkpoint not found at {base_ckpt_path}. Trying HF Hub...")
         checkpoint_info = MatterGenCheckpointInfo.from_hf_hub("mattergen_base")
    else:
        checkpoint_info = MatterGenCheckpointInfo(model_path=base_ckpt_path)

    # Create generator with base model
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        record_trajectories=not args.no_trajectories,
        bulk_modulus_classifier_ckpt=Path(args.bulk_modulus_classifier).resolve() if args.bulk_modulus_classifier else None,
        bulk_modulus_guidance_scale=args.guidance_scale,
    )
    
    # Pre-load the base model so we can overwrite weights
    generator.prepare()
    
    # Load fine-tuned weights if provided
    if args.checkpoint:
        rl_ckpt_path = Path(args.checkpoint).resolve()
        print(f"Loading fine-tuned weights from {rl_ckpt_path}...")
        
        # DDPO trainer saves state dict in a dict under 'actor_state_dict' or 'state_dict'
        # Or sometimes just the model state dict if user saved it that way.
        # Let's inspect what train_ddpo.py saves. 
        # It saves: {'epoch': ..., 'actor_state_dict': ..., ...}
        
        checkpoint = torch.load(rl_ckpt_path, map_location=get_device())
        
        if "actor_state_dict" in checkpoint:
            state_dict = checkpoint["actor_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint # Assume it's the state dict itself
            
        # The actor wrapper might have prefixes.
        # MatterGenActor wraps the denoiser. 
        # If train_ddpo.py saved actor.state_dict(), and actor has self.denoiser...
        # Wait, MatterGenActor inherits from nn.Module and has self.denoiser = denoiser.
        # So keys will be "denoiser.xxx".
        # But generator.model is the DiffusionLightningModule.
        # generator.model.model IS the denoiser (GemNetTDenoiser).
        
        # We need to map "denoiser.xxx" -> "model.xxx" (if loading into PL module)
        # OR "denoiser.xxx" -> "xxx" (if loading into denoiser directly).
        
        # generator.model is DiffusionLightningModule
        # generator.model.model is GemNetTDenoiser
        
        # Let's clean the keys.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("denoiser."):
                # Remove "denoiser." prefix to get raw GemNet keys
                new_key = k[len("denoiser."):]
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
                
        # Load into the DENOISER component of the Lightning Module
        # generator.model -> DiffusionLightningModule
        # .diffusion_module -> DiffusionModule
        # .model -> GemNetTDenoiser
        missing, unexpected = generator.model.diffusion_module.model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(unexpected) > 0:
            print(f"Unexpected keys sample: {unexpected[:5]}")

    # Generate structures
    structures = generator.generate(output_dir=str(output_path))

    print(f"\nGenerated {len(structures)} structures")
    print(f"Saved to: {output_path}/generated_crystals.extxyz")
    print(f"Saved to: {output_path}/generated_crystals.zip")

    return structures


if __name__ == "__main__":
    main()
