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

        # Unwrap {"state_dict": ddpo_ckpt} added by the conversion step — the inner dict
        # is the real DDPO checkpoint containing actor_state_dict, epoch, etc.
        inner = checkpoint.get("state_dict", checkpoint)
        if isinstance(inner, dict) and "actor_state_dict" in inner:
            checkpoint = inner

        if "actor_state_dict" in checkpoint:
            state_dict = checkpoint["actor_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Keys are "diffusion_module.model.gemnet.xxx" — same namespace as generator.model
        # (DiffusionLightningModule), so load directly into generator.model, not the denoiser.
        #
        # LoRA layers appear as "xxx.linear.base_layer.weight" + lora_A/lora_B siblings.
        # Merge them back: W_merged = W_base + (alpha/rank) * lora_B @ lora_A
        # MatterGenActor hardcodes rank=16, alpha=16 so scaling=1.
        is_lora = any(".base_layer." in k for k in state_dict)

        if is_lora:
            print("Detected LoRA checkpoint — merging LoRA weights into base weights...")
            import re
            lora_roots = set(
                re.sub(r"\.(base_layer|lora_A|lora_B)\..*$", "", k)
                for k in state_dict if ".base_layer." in k
            )
            merged_sd = dict(state_dict)
            for root in lora_roots:
                W      = state_dict[f"{root}.base_layer.weight"]
                lora_A = state_dict[f"{root}.lora_A.weight"]   # (rank, in_features)
                lora_B = state_dict[f"{root}.lora_B.weight"]   # (out_features, rank)
                rank   = lora_A.shape[0]
                scaling = 1.0  # alpha == rank == 16 in MatterGenActor
                merged_sd[f"{root}.weight"] = W + scaling * (lora_B @ lora_A)
                for tag in ("base_layer.weight", "lora_A.weight", "lora_B.weight"):
                    merged_sd.pop(f"{root}.{tag}", None)
                bias_key = f"{root}.base_layer.bias"
                if bias_key in merged_sd:
                    merged_sd[f"{root}.bias"] = merged_sd.pop(bias_key)
            state_dict = merged_sd
            print(f"Merged {len(lora_roots)} LoRA layers.")

        # Drop the duplicate "denoiser.*" subtree — MatterGenActor stores self.denoiser and
        # self.diffusion_module pointing to the same module, producing duplicate keys.
        # The "diffusion_module.*" keys already cover everything we need.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("denoiser.")}

        missing, unexpected = generator.model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys sample: {missing[:5]}")
            print("  WARNING: missing keys mean some weights weren't loaded — check checkpoint source.")
        if unexpected:
            print(f"  Unexpected keys sample: {unexpected[:5]}")
            print("  WARNING: unexpected keys were ignored.")

    # Generate structures
    structures = generator.generate(output_dir=str(output_path))

    print(f"\nGenerated {len(structures)} structures")
    print(f"Saved to: {output_path}/generated_crystals.extxyz")
    print(f"Saved to: {output_path}/generated_crystals.zip")

    return structures


if __name__ == "__main__":
    main()
