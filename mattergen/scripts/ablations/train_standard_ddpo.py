#!/usr/bin/env python3
"""Train MatterGen with standard DDPO (ablation: single combined ratio, no decoupling).

Key difference from train_ddpo.py: continuous and discrete log-probs are summed
into one importance ratio with a single clip epsilon, rather than treated separately.

Usage:
    python -m mattergen.scripts.ablations.train_standard_ddpo \
        --checkpoint checkpoints/mattergen_base \
        --critic-checkpoint checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt \
        --output-dir checkpoints/standard_ddpo_bulk_modulus \
        --num-epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from hydra.utils import instantiate

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.common.utils.globals import get_device
from mattergen.generator import CrystalGenerator
from mattergen.rl import MatterGenActor, ValueNetwork
from mattergen.rl.standard_ddpo_trainer import StandardDDPOConfig, StandardDDPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from mattergen.property_predictors.bulk_modulus_time_lora_mlp_predictor import BulkModulusLoRAMLPTimePredictor

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_ddpo import create_reward_fn, create_mattersim_reward_fn, wrap_reward_top_k


def main():
    parser = argparse.ArgumentParser(
        description="Train MatterGen with standard DDPO loss (ablation)"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mattergen_base")
    parser.add_argument(
        "--critic-checkpoint",
        type=str,
        default="checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt",
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints/standard_ddpo_bulk_modulus")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.2,
        help="Single PPO clip epsilon applied to the combined cont+disc ratio.",
    )
    parser.add_argument("--kl-coeff", type=float, default=0.1)
    parser.add_argument("--entropy-coeff", type=float, default=0.01)
    parser.add_argument(
        "--reward-type",
        type=str,
        default="bulk_modulus",
        choices=["bulk_modulus", "mattersim"],
    )
    parser.add_argument("--mattersim-n-points", type=int, default=5)
    parser.add_argument("--mattersim-strain", type=float, default=0.03)
    parser.add_argument("--mattersim-no-relax", action="store_true")
    parser.add_argument("--reward-top-k", type=int, default=None)
    parser.add_argument("--num-rollout-batches", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_dict = {**vars(args), "timestamp": datetime.now().isoformat()}
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    device = get_device()
    print(f"Using device: {device}")
    print("=" * 60)

    checkpoint_path = Path(args.checkpoint).resolve()
    print(f"Loading MatterGen from: {checkpoint_path}")
    checkpoint_info = MatterGenCheckpointInfo(model_path=checkpoint_path)
    model = load_model_diffusion(checkpoint_info).to(device)

    diffusion_module = model.diffusion_module
    denoiser = diffusion_module.model

    print("Creating MatterGenActor...")
    actor = MatterGenActor(denoiser=denoiser, diffusion_module=diffusion_module)

    critic_path = Path(args.critic_checkpoint)
    if critic_path.exists():
        print(f"Loading critic from: {critic_path}")
        critic = ValueNetwork.from_classifier_checkpoint(critic_path, device=device)
    else:
        print("Critic checkpoint not found, creating from scratch")
        critic = ValueNetwork(hidden_dim=512, mlp_hidden_dim=256)

    if args.reward_type != "mattersim":
        if critic_path.exists():
            print(f"Loading frozen reward network from: {critic_path}")
            reward_net = BulkModulusLoRAMLPTimePredictor.from_checkpoint(critic_path, device=device)
        else:
            reward_net = BulkModulusLoRAMLPTimePredictor(hidden_dim=512, mlp_hidden_dim=256).to(device)
        reward_net.eval()
        for param in reward_net.parameters():
            param.requires_grad = False

    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=args.batch_size,
        num_batches=1,
    )
    sampling_config = generator.load_sampling_config(batch_size=args.batch_size, num_batches=1)
    condition_loader = generator.get_condition_loader(sampling_config)
    sampler = instantiate(sampling_config.sampler_partial)(pl_module=model)

    if args.reward_type == "mattersim":
        print(f"Using MatterSim reward (n_points={args.mattersim_n_points}, strain={args.mattersim_strain})")
        reward_fn = create_mattersim_reward_fn(
            device=device,
            n_points=args.mattersim_n_points,
            strain=args.mattersim_strain,
            relax=not args.mattersim_no_relax,
        )
    else:
        reward_fn = create_reward_fn(reward_net, device=device)

    if args.reward_top_k is not None:
        print(f"Using top-{args.reward_top_k} reward aggregation")
        reward_fn = wrap_reward_top_k(reward_fn, k=args.reward_top_k)

    standard_config = StandardDDPOConfig(
        clip_eps=args.clip_eps,
        ppo_epochs=args.ppo_epochs,
        kl_coeff=args.kl_coeff,
        entropy_coeff=args.entropy_coeff,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )

    print("Creating StandardDDPOTrainer...")
    trainer = StandardDDPOTrainer(
        actor=actor,
        critic=critic,
        reward_fn=reward_fn,
        config=standard_config,
        device=device,
    )

    start_epoch = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"Resuming from: {resume_path}")
            start_epoch = trainer.load_checkpoint(resume_path)

    print("=" * 60)
    print(f"Starting standard DDPO training for {args.num_epochs} epochs...")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  PPO epochs:     {args.ppo_epochs}")
    print(f"  Clip eps:       {args.clip_eps}  (single combined ratio)")
    print(f"  KL coeff:       {args.kl_coeff}")
    print(f"  Entropy coeff:  {args.entropy_coeff}")
    print("=" * 60)

    metrics_history = trainer.train(
        sampler=sampler,
        condition_loader=condition_loader,
        num_epochs=args.num_epochs,
        num_diffusion_batches=args.num_rollout_batches,
        save_path=output_path,
        save_every=args.save_every,
    )

    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    print("=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_path}")


if __name__ == "__main__":
    main()
