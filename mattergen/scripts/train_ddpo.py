#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Train MatterGen with DDPO (Denoising Diffusion Policy Optimization).

This script implements stable DDPO training with a decoupled hybrid PPO strategy
that handles the mixed continuous (coordinates/lattice) and discrete (atom types)
action space.

Usage:
    python -m mattergen.scripts.train_ddpo \
        --checkpoint checkpoints/mattergen_base \
        --critic-checkpoint checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt \
        --output-dir checkpoints/ddpo_bulk_modulus \
        --num-epochs 100

Key features:
- Separate PPO ratios for continuous and discrete actions
- Stricter clipping for discrete actions (ε=0.1 vs ε=0.2)
- KL divergence anchor against frozen reference model
- Entropy regularization for exploration
- NaN protection via probability clamping
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from hydra.utils import instantiate
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.eval_utils import load_model_diffusion
from mattergen.common.utils.globals import get_device
from mattergen.generator import CrystalGenerator
from mattergen.rl import DDPOConfig, DDPOTrainer, MatterGenActor, ValueNetwork

# Import bulk modulus calculator
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from calc_bulk_modulus_single import calc_bulk_modulus_value


def create_reward_fn(reward_type: str = "bulk_modulus", device: torch.device = None):
    """
    Create a reward function for training.
    
    Args:
        reward_type: Type of reward ("bulk_modulus", "simple_density")
        device: Device for computations
        
    Returns:
        Callable that takes a ChemGraph and returns a float reward
    """
    if reward_type == "bulk_modulus":
        def reward_fn(x_0: ChemGraph) -> float:
            """
            Compute bulk modulus reward using MatterSim-based E-V fitting.
            """
            try:
                # Convert ChemGraph to ASE Atoms - ensure no grad tracking
                with torch.no_grad():
                    batch_size = x_0.get_batch_size()
                    total_reward = 0.0
                    
                    for i in range(batch_size):
                        # Extract single crystal from batch
                        batch_mask = x_0.get_batch_idx("pos") == i
                        
                        # Handle positions (may have gradients)
                        pos_tensor = x_0["pos"][batch_mask]
                        if pos_tensor.requires_grad:
                            positions = pos_tensor.detach().cpu().numpy()
                        else:
                            positions = pos_tensor.cpu().numpy()
                        
                        # Handle atomic numbers (usually LongTensor, no grad)
                        atom_tensor = x_0["atomic_numbers"][batch_mask]
                        atomic_numbers = atom_tensor.cpu().numpy()
                        
                        # Handle cell (may have gradients)
                        cell_tensor = x_0["cell"][i]
                        if cell_tensor.requires_grad:
                            cell = cell_tensor.detach().cpu().numpy()
                        else:
                            cell = cell_tensor.cpu().numpy()
                        
                        # Create ASE Atoms
                        # MatterGen outputs fractional coordinates, ASE expects Cartesian
                        # Convert fractional -> Cartesian: pos_cart = pos_frac @ cell
                        positions_cart = positions @ cell
                        
                        from ase import Atoms
                        atoms = Atoms(
                            numbers=atomic_numbers.astype(int),
                            positions=positions_cart,
                            cell=cell,
                            pbc=True,
                        )
                        
                        # Calculate bulk modulus (fast E-V fitting method)
                        # MatterSim needs autograd to compute forces/stress internally
                        # We use enable_grad() but since inputs are numpy (detached), this is safe
                        with torch.enable_grad():
                            bulk_mod = calc_bulk_modulus_value(atoms, n_points=5, strain=0.03)
                        
                        # Clip to reasonable range and normalize
                        bulk_mod = max(0.0, min(float(bulk_mod), 500.0))  # Clip to [0, 500] GPa
                        total_reward += bulk_mod
                    
                    return total_reward / batch_size
                
            except Exception as e:
                import traceback
                print(f"Reward computation failed: {e}")
                traceback.print_exc()
                return 0.0
        
        return reward_fn
    
    elif reward_type == "simple_density":
        def reward_fn(x_0: ChemGraph) -> float:
            """Heuristic reward favoring compact structures."""
            batch_size = x_0.get_batch_size()
            total_reward = 0.0
            
            for i in range(batch_size):
                batch_mask = x_0.get_batch_idx("pos") == i
                positions = x_0["pos"][batch_mask]
                cell = x_0["cell"][i]
                
                volume = torch.abs(torch.det(cell)).item()
                n_atoms = positions.shape[0]
                density = n_atoms / (volume + 1e-6)
                
                reward = density * 50.0
                total_reward += min(reward, 200.0)
            
            return total_reward / batch_size
        
        return reward_fn
    
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def main():
    parser = argparse.ArgumentParser(description="Train MatterGen with DDPO")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/mattergen_base",
        help="Path to pretrained MatterGen checkpoint",
    )
    parser.add_argument(
        "--critic-checkpoint",
        type=str,
        default="checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt",
        help="Path to bulk modulus classifier checkpoint for critic initialization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/ddpo_bulk_modulus",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (trajectories per epoch)",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=3,
        help="Number of PPO update epochs per rollout",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=1e-5,
        help="Actor learning rate",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=1e-4,
        help="Critic learning rate",
    )
    parser.add_argument(
        "--clip-eps-cont",
        type=float,
        default=0.2,
        help="PPO clipping for continuous actions",
    )
    parser.add_argument(
        "--clip-eps-disc",
        type=float,
        default=0.1,
        help="PPO clipping for discrete actions (stricter)",
    )
    parser.add_argument(
        "--kl-coeff",
        type=float,
        default=0.1,
        help="KL divergence coefficient (anchor strength)",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="Entropy regularization coefficient",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="bulk_modulus",
        choices=["bulk_modulus", "simple_density"],
        help="Reward function type",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = vars(args)
    config_dict["timestamp"] = datetime.now().isoformat()
    with open(output_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    device = get_device()
    print(f"Using device: {device}")
    print("=" * 60)

    # Load MatterGen model
    checkpoint_path = Path(args.checkpoint).resolve()
    print(f"Loading MatterGen from: {checkpoint_path}")
    checkpoint_info = MatterGenCheckpointInfo(model_path=checkpoint_path)
    model = load_model_diffusion(checkpoint_info)
    model = model.to(device)
    
    diffusion_module = model.diffusion_module
    denoiser = diffusion_module.model
    
    # Create MatterGenActor wrapper
    print("Creating MatterGenActor...")
    actor = MatterGenActor(denoiser=denoiser, diffusion_module=diffusion_module)
    
    # Create ValueNetwork (from classifier checkpoint)
    critic_path = Path(args.critic_checkpoint)
    if critic_path.exists():
        print(f"Loading critic from: {critic_path}")
        critic = ValueNetwork.from_classifier_checkpoint(critic_path, device=device)
    else:
        print("Critic checkpoint not found, creating from scratch")
        critic = ValueNetwork(hidden_dim=512, mlp_hidden_dim=256)
    
    # Create sampler for trajectory collection
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        batch_size=args.batch_size,
        num_batches=1,
    )
    sampling_config = generator.load_sampling_config(
        batch_size=args.batch_size,
        num_batches=1,
    )
    condition_loader = generator.get_condition_loader(sampling_config)
    sampler_partial = instantiate(sampling_config.sampler_partial)
    sampler = sampler_partial(pl_module=model)
    
    # Create reward function
    reward_fn = create_reward_fn(args.reward_type, device=device)
    
    # Create DDPO config
    ddpo_config = DDPOConfig(
        clip_eps_cont=args.clip_eps_cont,
        clip_eps_disc=args.clip_eps_disc,
        ppo_epochs=args.ppo_epochs,
        kl_coeff=args.kl_coeff,
        entropy_coeff=args.entropy_coeff,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )
    
    # Create trainer
    print("Creating DDPOTrainer...")
    trainer = DDPOTrainer(
        actor=actor,
        critic=critic,
        reward_fn=reward_fn,
        config=ddpo_config,
        device=device,
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"Resuming from: {resume_path}")
            start_epoch = trainer.load_checkpoint(resume_path)
    
    print("=" * 60)
    print(f"Starting DDPO training for {args.num_epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  PPO epochs: {args.ppo_epochs}")
    print(f"  Clip (cont/disc): {args.clip_eps_cont}/{args.clip_eps_disc}")
    print(f"  KL coeff: {args.kl_coeff}")
    print(f"  Entropy coeff: {args.entropy_coeff}")
    print("=" * 60)
    
    # Train
    metrics_history = trainer.train(
        sampler=sampler,
        condition_loader=condition_loader,
        num_epochs=args.num_epochs,
        trajectories_per_epoch=args.batch_size,
        save_path=output_path,
        save_every=args.save_every,
    )
    
    # Save final metrics
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    print("=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_path}")
    
    # Summary
    if metrics_history:
        rewards = [m.get("mean_reward", 0) for m in metrics_history]
        print(f"Final mean reward: {rewards[-1]:.2f}")
        print(f"Best mean reward: {max(rewards):.2f}")


if __name__ == "__main__":
    main()
