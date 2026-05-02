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

        checkpoint = torch.load(rl_ckpt_path, map_location=get_device())

        if "actor_state_dict" in checkpoint:
            state_dict = checkpoint["actor_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip the "denoiser." prefix that MatterGenActor adds, and ignore the
        # "diffusion_module.*" subtree (duplicate of denoiser weights stored by
        # MatterGenActor.__init__'s self.diffusion_module reference).
        denoiser_sd = {}
        for k, v in state_dict.items():
            if k.startswith("denoiser."):
                denoiser_sd[k[len("denoiser."):]] = v

        if not denoiser_sd:
            # Checkpoint has no "denoiser." prefix — assume it's already a raw denoiser dict.
            denoiser_sd = {k: v for k, v in state_dict.items()
                           if not k.startswith("diffusion_module.")}

        # Detect whether this is a LoRA checkpoint (keys contain ".base_layer." or ".lora_A.").
        # MatterGenActor wraps fc_atom / out_energy / out_forces with LoRALayer which renames
        #   fc_atom.weight  ->  fc_atom.base_layer.weight  (+ fc_atom.lora_A/B.weight)
        # The base model has no LoRALayer, so we merge the update back into the base weight:
        #   W_merged = W_base + scaling * lora_B @ lora_A   (scaling = alpha / rank)
        is_lora = any(".base_layer." in k for k in denoiser_sd)

        if is_lora:
            print("Detected LoRA checkpoint — merging LoRA weights into base weights...")
            # Collect all LoRA root prefixes (e.g. "fc_atom", "out_energy.linear", …)
            import re
            lora_roots = set(
                re.sub(r"\.(base_layer|lora_A|lora_B)\..*$", "", k)
                for k in denoiser_sd if ".base_layer." in k or ".lora_A." in k
            )
            merged_sd = {}
            for root in lora_roots:
                W = denoiser_sd[f"{root}.base_layer.weight"]
                lora_A = denoiser_sd[f"{root}.lora_A.weight"]  # (rank, in)
                lora_B = denoiser_sd[f"{root}.lora_B.weight"]  # (out, rank)
                # Recover scaling from the rank dimension (alpha=rank=16 in MatterGenActor)
                rank = lora_A.shape[0]
                alpha = float(rank)  # MatterGenActor hardcodes alpha == rank == 16
                scaling = alpha / rank
                merged_sd[f"{root}.weight"] = W + scaling * (lora_B @ lora_A)
                # Copy bias if present
                bias_key = f"{root}.base_layer.bias"
                if bias_key in denoiser_sd:
                    merged_sd[f"{root}.bias"] = denoiser_sd[bias_key]
            # Add all non-LoRA keys (everything that isn't base_layer / lora_A / lora_B)
            for k, v in denoiser_sd.items():
                if not any(tag in k for tag in (".base_layer.", ".lora_A.", ".lora_B.")):
                    merged_sd[k] = v
            denoiser_sd = merged_sd
            print(f"Merged {len(lora_roots)} LoRA layers.")

        denoiser = generator.model.diffusion_module.model
        missing, unexpected = denoiser.load_state_dict(denoiser_sd, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys sample: {missing[:5]}")
        if unexpected:
            print(f"  Unexpected keys sample: {unexpected[:5]}")
        if missing or unexpected:
            print("  WARNING: key mismatches above mean some weights weren't loaded — check checkpoint source.")

    # Generate structures
    structures = generator.generate(output_dir=str(output_path))

    print(f"\nGenerated {len(structures)} structures")
    print(f"Saved to: {output_path}/generated_crystals.extxyz")
    print(f"Saved to: {output_path}/generated_crystals.zip")

    return structures


if __name__ == "__main__":
    main()
