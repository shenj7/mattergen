"""
Reinforcement Learning module for MatterGen.

This package provides DDPO (Denoising Diffusion Policy Optimization) training
with decoupled handling of discrete and continuous action spaces.
"""

from mattergen.rl.ddpo_trainer import (
    DDPOConfig,
    DDPOTrainer,
    MatterGenActor,
    Trajectory,
    TrajectoryStep,
    ValueNetwork,
)

__all__ = [
    "DDPOConfig",
    "DDPOTrainer",
    "MatterGenActor",
    "Trajectory",
    "TrajectoryStep",
    "ValueNetwork",
]
