
"""
DDPO Implementation for MatterGen.
Adapted from https://github.com/kvablack/ddpo-pytorch (PyTorch)
and https://github.com/jannerm/ddpo (JAX)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.sampling.predictors_correctors import Sampler
from mattergen.diffusion.corruption.sde_lib import SDE, VPSDE
from mattergen.diffusion.corruption.corruption import maybe_expand

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DDPOConfig:
    """Configuration for DDPO training."""
    # PPO hyperparameters
    clip_eps: float = 0.1  # PPO clip range (matches reference repo default)
    ppo_epochs: int = 4    # Number of PPO epochs per rollout
    num_batches_per_epoch: int = 1 # Number of batches to sample per epoch
    
    # Regularization
    kl_coeff: float = 0.0  # KL divergence coefficient (optional)
    entropy_coeff: float = 0.0  # Entropy bonus coefficient
    value_coeff: float = 0.5  # Value loss coefficient
    
    # Numerical stability
    prob_clamp_min: float = 1e-6
    prob_clamp_max: float = 1.0 - 1e-6
    grad_clip_norm: float = 1.0
    
    # Learning rates
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4

    # Sampling
    num_inference_steps: int = 100
    resolution: int = 64 # Not used for mattergen, but kept for compat
    guidance_scale: float = 1.0
    eta: float = 1.0


# =============================================================================
# Trajectory Storage
# =============================================================================

class TrajectoryStep(NamedTuple):
    """Single step in a diffusion trajectory."""
    latents: ChemGraph  # x_t
    next_latents: ChemGraph # x_{t-1}
    log_probs: torch.Tensor # log pi(x_{t-1} | x_t)
    timestep: torch.Tensor  # t
    prompt_embeds: torch.Tensor | None = None # Conditioning


@dataclass
class Trajectory:
    """Full diffusion trajectory with terminal reward."""
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_sample: ChemGraph | None = None
    reward: float = 0.0
    advantage: float = 0.0


# =============================================================================
# DDPOSampler: Custom Sampler with Log Probabilities
# =============================================================================

class DDPOSampler(Sampler):
    """
    Sampler that records log probabilities of the transition steps.
    Implements Ancestral Sampling (Euler-Maruyama) for the reverse SDE.
    """
    def __init__(
        self,
        corruption: SDE,
        score_fn: Callable,
        device: torch.device,
        N: int,
        eps_t: float = 1e-3,
        max_t: float | None = None,
    ):
        super().__init__(corruption, score_fn)
        self.device = device
        self.N = N
        self.eps_t = eps_t
        self.max_t = max_t if max_t is not None else corruption.T

    def _get_ancestral_step(self, x, t, dt, batch_idx, batch):
        """
        Compute mean and std for the transition p(x_{t-1} | x_t, x_0_pred).
        """
        sde = self.corruption
        # Current and next timestep
        s = t + dt
        
        # 1. Get coefficients for the forward diffusion q(x_t | x_0) and q(x_s | x_0)
        alpha_t, sigma_t = sde.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx, batch=batch)
        alpha_s, sigma_s = sde.mean_coeff_and_std(x=x, t=s, batch_idx=batch_idx, batch=batch)
        
        # Handle time zero special case
        if batch_idx is None:
            is_time_zero = s <= 0
        else:
            is_time_zero = s[batch_idx] <= 0
        sigma_s[is_time_zero] = 0

        # 2. Compute posterior mean and variance for q(x_s | x_t, x_0)
        sigma2_t_given_s = sigma_t**2 - sigma_s**2 * (alpha_t / alpha_s)**2
        sigma2_t_given_s = torch.clamp(sigma2_t_given_s, min=0.0) 
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        
        std = sigma_t_given_s * sigma_s / sigma_t
        
        alpha_t_given_s = alpha_t / alpha_s
        # Avoid division by zero
        alpha_t_given_s = torch.clamp(alpha_t_given_s, min=1e-4)
        
        score_coeff = sigma2_t_given_s / alpha_t_given_s
        x_coeff = 1.0 / alpha_t_given_s

        std[is_time_zero] = 0
        
        return x_coeff, score_coeff, std

    def _get_d3pm_step(self, x, t, dt, batch_idx, score, corruption):
        """
        Compute transition probabilities for D3PM: p(x_{t-1} | x_t, x_0_pred).
        Returns logits for the categorical distribution.
        """
        from mattergen.diffusion.discrete_time import to_discrete_time
        
        # Convert continuous t to discrete index
        t_idx = to_discrete_time(t=t, N=corruption.N, T=corruption.T)
        
        # Score is the logits for p(x_0 | x_t)
        # We need to compute p(x_{t-1} | x_t) using the posterior formula
        # q(x_{t-1} | x_t, x_0)
        
        class_probs = torch.softmax(score, dim=-1)
        
        # Use D3PM's internal logic to get posterior q(x_{t-1} | x_t, x_0)
        # We assume `score` predicts x_0 logits (predict_x0=True is standard)
        
        # To avoid external dependency on private methods, we rely on public APIs if possible,
        # or replicate the logic. mattergen's D3PMAncestralSamplingPredictor uses `sample_and_compute_posterior_q`.
        
        logits, _ = corruption.d3pm.sample_and_compute_posterior_q(
            x_0=class_probs,
            t=t_idx[batch_idx].to(torch.long),
            make_one_hot=False,
            samples=corruption._to_zero_based(x),
            return_logits=True
        )
        
        return logits


    def sample_step(self, batch, t, dt):
        """
        Perform one sampling step for all fields and return next_batch and log_prob.
        """
        # Joint score: dict mapping field_name -> score_tensor
        score = self.score_fn(batch, t)
        
        next_batch_data = {}
        total_log_prob = torch.zeros(batch.get_batch_size(), device=self.device)
        
        # Iterate over all corruptions (fields) in the MultiCorruption
        # We assume self.corruption is a MultiCorruption object
        for field_name, field_corruption in self.corruption.corruptions.items():
            if field_name not in batch:
                continue
                
            x = batch[field_name]
            field_score = score[field_name]
            batch_idx = batch.get_batch_idx(field_name)
            
            # Check if discrete (D3PM) or continuous (SDE)
            # This check depends on the type of `field_corruption`
            
            if hasattr(field_corruption, 'd3pm'): # D3PMCorruption
                # Discrete step
                logits = self._get_d3pm_step(x, t, dt, batch_idx, field_score, field_corruption)
                dist = Categorical(logits=logits)
                
                # Sample next state
                # Note: logits are for x_{t-1} (zero-based). Need to convert back if needed.
                next_x_idx = dist.sample()
                next_x = field_corruption._to_non_zero_based(next_x_idx)
                
                # Compute log prob
                log_prob = dist.log_prob(next_x_idx)
                
            else: # Standard SDE (VPSDE/VESDE)
                # Continuous step (Ancestral)
                # Temporarily attach the single corruption to 'self' or pass it explicitly
                # We'll pass `field_corruption` as `sde` to a helper
                # But _get_ancestral_step relies on `self.corruption` being the sde.
                # Let's refactor _get_ancestral_step to accept sde.
                
                # Helper to bind SDE methods
                alpha_t, sigma_t = field_corruption.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx, batch=batch)
                s = t + dt
                alpha_s, sigma_s = field_corruption.mean_coeff_and_std(x=x, t=s, batch_idx=batch_idx, batch=batch)
                 
                if batch_idx is None:
                    is_time_zero = s <= 0
                else:
                    is_time_zero = s[batch_idx] <= 0
                sigma_s[is_time_zero] = 0

                sigma2_t_given_s = sigma_t**2 - sigma_s**2 * (alpha_t / alpha_s)**2
                sigma2_t_given_s = torch.clamp(sigma2_t_given_s, min=0.0) 
                
                # Correct coefficients
                alpha_t_given_s = alpha_t / alpha_s
                alpha_t_given_s = torch.clamp(alpha_t_given_s, min=1e-4) # Avoid zero div
                
                sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
                std = sigma_t_given_s * sigma_s / sigma_t
                std[is_time_zero] = 0
                
                score_coeff = sigma2_t_given_s / alpha_t_given_s
                x_coeff = 1.0 / alpha_t_given_s
                
                # Update
                mean = x_coeff * x + score_coeff * field_score
                z = torch.randn_like(x)
                next_x = mean + std * z
                
                # Log Prob
                # Sum over all dimensions except batch
                log_prob = Normal(mean, std.clamp(min=1e-6)).log_prob(next_x)
                if log_prob.dim() > 1:
                    log_prob = log_prob.sum(dim=list(range(1, log_prob.dim())))
            
            # Aggregate log probs per graph
            if batch_idx is not None:
                # If x is per-atom, sum to per-graph
                # log_prob shape: [num_atoms] -> scatter_add -> [num_graphs]
                total_log_prob.index_add_(0, batch_idx, log_prob)
            else:
                 # If x is per-graph (like cell), it's already aligned
                 total_log_prob += log_prob
            
            next_batch_data[field_name] = next_x
            
            
        return batch.replace(**next_batch_data), total_log_prob

    @torch.no_grad()
    def sample_trajectory(self, conditioning_data: BatchedData) -> Tuple[ChemGraph, list[TrajectoryStep]]:
        """Sample a Full Trajectory and return steps."""
        conditioning_data = conditioning_data.to(self.device)
        
        # Prior Sampling
        # MultiCorruption doesn't have prior_sampling, but its components do
        samples = {}
        for k, corruption in self.corruption.corruptions.items():
            samples[k] = corruption.prior_sampling(
                shape=conditioning_data[k].shape,
                conditioning_data=conditioning_data,
                batch_idx=conditioning_data.get_batch_idx(field_name=k),
            ).to(self.device)
            
        # Replace data in conditioning_data with samples
        batch = conditioning_data.replace(**samples)
        
        timesteps = torch.linspace(self.max_t, self.eps_t, self.N, device=self.device)
        dt = -torch.tensor((self.max_t - self.eps_t) / (self.N - 1)).to(self.device)
        
        trajectory_steps = []

        # Denoising Loop
        for i in range(self.N):
            t = torch.full((batch.get_batch_size(),), timesteps[i], device=self.device)
            
            # Record current state
            current_latents = batch
            
            # Step
            next_batch, log_prob = self.sample_step(batch, t, dt)
            
            # Record step (x_t, x_{t-1}, log_prob)
            step = TrajectoryStep(
                latents=current_latents.clone(),
                next_latents=next_batch.clone(),
                log_probs=log_prob,
                timestep=t,
                prompt_embeds=None 
            )
            trajectory_steps.append(step)
            
            batch = next_batch
            
        return batch, trajectory_steps
    
    def evaluate_log_prob(self, batch, next_batch, t, dt):
        """
        Evaluate log p(x_{t-1} | x_t) for a given transition (batch -> next_batch).
        Used during PPO update to re-compute log probs with updated score model.
        """
        # Joint score with CURRENT model
        score = self.score_fn(batch, t)
        
        total_log_prob = torch.zeros(batch.get_batch_size(), device=self.device)
        
        for field_name, field_corruption in self.corruption.corruptions.items():
            if field_name not in batch: continue
                
            x = batch[field_name]
            next_x = next_batch[field_name] # The ACTION taken
            field_score = score[field_name]
            batch_idx = batch.get_batch_idx(field_name)
            
            if hasattr(field_corruption, 'd3pm'): 
                # Discrete
                logits = self._get_d3pm_step(x, t, dt, batch_idx, field_score, field_corruption)
                dist = Categorical(logits=logits)
                
                # We need the index of next_x
                next_x_idx = field_corruption._to_zero_based(next_x) 
                log_prob = dist.log_prob(next_x_idx)
                
            else: 
                # Continuous (Ancestral)
                # Re-compute parameters
                alpha_t, sigma_t = field_corruption.mean_coeff_and_std(x=x, t=t, batch_idx=batch_idx, batch=batch)
                s = t + dt
                alpha_s, sigma_s = field_corruption.mean_coeff_and_std(x=x, t=s, batch_idx=batch_idx, batch=batch)
                 
                if batch_idx is None:
                    is_time_zero = s <= 0
                else:
                    is_time_zero = s[batch_idx] <= 0
                sigma_s[is_time_zero] = 0

                sigma2_t_given_s = sigma_t**2 - sigma_s**2 * (alpha_t / alpha_s)**2
                sigma2_t_given_s = torch.clamp(sigma2_t_given_s, min=0.0) 
                
                alpha_t_given_s = torch.clamp(alpha_t / alpha_s, min=1e-4)
                
                sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
                std = sigma_t_given_s * sigma_s / sigma_t
                std[is_time_zero] = 0
                
                score_coeff = sigma2_t_given_s / alpha_t_given_s
                x_coeff = 1.0 / alpha_t_given_s
                
                mean = x_coeff * x + score_coeff * field_score
                
                # Log Prob of next_x under N(mean, std)
                log_prob = Normal(mean, std.clamp(min=1e-6)).log_prob(next_x)
                if log_prob.dim() > 1:
                    log_prob = log_prob.sum(dim=list(range(1, log_prob.dim())))
            
            if batch_idx is not None:
                total_log_prob.index_add_(0, batch_idx, log_prob)
            else:
                 total_log_prob += log_prob

        return total_log_prob

 

# =============================================================================
# DDPOTrainer: Main Training Loop
# =============================================================================

class DDPOTrainer:
    """
    DDPO Trainer adapted for MatterGen.
    """
    def __init__(
        self,
        diffusion_module: DiffusionModule,
        reward_model: nn.Module,
        config: DDPOConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.diffusion_module = diffusion_module.to(device)
        self.reward_model = reward_model.to(device)
        # Assuming reward_model can act as both Critic (V) and Reward (R)
        # If frozen, we might need a trainable copy for V??
        # The prompt said: "The critic should still be from checkpoints/... and the reward ... from the same checkpoint"
        # Usually Critic is TRAINABLE to adapt to policy changes (Value learning).
        # Reward function is FIXED.
        # So we should make a copy of reward_model to be the Critic.
        
        self.critic = copy.deepcopy(reward_model).to(device)
        self.critic.train()
        
        # Freezing the Reward Model (it's an oracle/classifier)
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()
            
        self.optimizer = torch.optim.AdamW(
            self.diffusion_module.parameters(),
            lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr
        )
        
    def _compute_advantages(self, trajectories: list[Trajectory]):
        """
        Compute advantages: A_t = R - V(s_t).
        For terminal reward setting, R is fixed for the whole trajectory.
        """
        all_advantages = []
        all_returns = []
        
        for traj in trajectories:
            reward = traj.reward # [batch_size] tensor
            
            #Stack latent states x_t for this trajectory to batch-compute values
            # traj.steps is list of TrajectoryStep
            # Each step has latents: ChemGraph of batch_size
            
            # We iterate steps to compute V(s_t)
            # Since simpler to loop:
            for step in traj.steps:
                x_t = step.latents
                t = step.timestep
                
                with torch.no_grad():
                    # Critic Value Estimate
                    values, _ = self.critic(x_t, t)
                    # values is [batch_size]
                
                # Advantage = Reward - Value
                adv = reward - values
                all_advantages.append(adv)
                all_returns.append(reward)
                
        return all_advantages, all_returns
    
    def calculate_loss(self, latents: list[ChemGraph], timesteps: list[torch.Tensor], 
                      next_latents: list[ChemGraph], old_log_probs: list[torch.Tensor], 
                      advantages: torch.Tensor, returns: torch.Tensor, sampler: DDPOSampler):
        """
        Compute PPO Loss.
        """
        new_log_probs_list = []
        
        # 1. Re-evaluate Log Probs
        for i in range(len(latents)):
             x = latents[i]
             next_x = next_latents[i]
             t = timesteps[i]
             
             # Calculate timestep step size dt (assumed constant from sampler)
             dt = -torch.tensor((sampler.max_t - sampler.eps_t) / (sampler.N - 1)).to(self.device)
             
             log_prob = sampler.evaluate_log_prob(x, next_x, t, dt)
             new_log_probs_list.append(log_prob)
             
        new_log_probs = torch.cat(new_log_probs_list)
        old_log_probs_tensor = torch.cat(old_log_probs)
        
        # 2. Ratio
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        
        # 3. Advantages
        # Normalize advantages
        if advantages.std() > 1e-6:
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
             
        # 4. Surrogate Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss, new_log_probs

    def update_critic(self, latents, timesteps, returns):
        """Update value function."""
        loss_sum = 0
        for i in range(len(latents)):
            x = latents[i]
            t = timesteps[i]
            target = returns[i] # [batch_size]
            
            pred_value, _ = self.critic(x, t)
            loss = F.mse_loss(pred_value, target)
            
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()
            loss_sum += loss.item()
            
        return loss_sum / len(latents)

    def step(self, trajectories: list[Trajectory], sampler: DDPOSampler):
        """
        Perform one PPO update step using collected trajectories.
        """
        # Flatten trajectories 
        flat_latents = []
        flat_next_latents = [] 
        flat_timesteps = []
        flat_old_log_probs = []
        
        # 1. Compute Advantages/Returns (GAE / Fixed)
        # List of tensors [batch_size]
        adv_list, ret_list = self._compute_advantages(trajectories)
        
        # Flatten Trajectory Steps
        for traj in trajectories:
            for step in traj.steps:
                flat_latents.append(step.latents)
                flat_next_latents.append(step.next_latents)
                flat_timesteps.append(step.timestep)
                flat_old_log_probs.append(step.log_probs)
        
        # Concatenate Adv/Ret
        advantages = torch.cat(adv_list)
        returns = torch.cat(ret_list)
        
        # PPO Epochs
        for _ in range(self.config.ppo_epochs):
             loss, _ = self.calculate_loss(flat_latents, flat_timesteps, flat_next_latents, 
                                           flat_old_log_probs, advantages, returns, sampler)
             
             self.optimizer.zero_grad()
             loss.backward()
             torch.nn.utils.clip_grad_norm_(self.diffusion_module.parameters(), self.config.grad_clip_norm)
             self.optimizer.step()
        
        # Update Critic
        self.update_critic(flat_latents, flat_timesteps, ret_list)
        
        return loss.item()

    def save_checkpoint(self, path: "Path | str") -> None:
        """Save the trained diffusion_module in mattergen-generate compatible format."""
        import torch
        torch.save({"state_dict": self.diffusion_module.state_dict()}, path)

    def train_loop(
        self,
        sampler: DDPOSampler,
        dataloader,
        num_epochs: int = 10,
        save_path: "Path | str | None" = None,
        save_every: int = 1,
    ):
        """
        Main training loop.

        Args:
            save_path: If provided, saves last.ckpt (and best_model.ckpt) here each
                       ``save_every`` epochs. Use --model_path=<save_path> with
                       mattergen-generate to run generation with the trained weights.
            save_every: Save a checkpoint every this many epochs.
        """
        from pathlib import Path as _Path
        import os as _os

        if save_path is not None:
            save_path = _Path(save_path)
            _os.makedirs(save_path, exist_ok=True)

        best_mean_reward = float("-inf")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            trajectories = []

            # 1. Collect Rollouts
            data_iter = iter(dataloader)

            for i in range(self.config.num_batches_per_epoch):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    conditioning_data, _ = batch
                else:
                    conditioning_data = batch

                try:
                    final_sample, steps = sampler.sample_trajectory(conditioning_data)

                    t_zero = torch.zeros(final_sample.get_batch_size(), device=self.device)
                    with torch.no_grad():
                        mu, _ = self.reward_model(final_sample, t_zero)
                        rewards = mu

                except (IndexError, RuntimeError) as e:
                    print(f"Skipping batch due to error: {e}")
                    continue

                traj = Trajectory(
                    steps=steps,
                    final_sample=final_sample,
                    reward=rewards
                )
                trajectories.append(traj)

            # 2. Update
            if len(trajectories) > 0:
                loss = self.step(trajectories, sampler)
                all_rewards = torch.cat([t.reward for t in trajectories])
                mean_reward = all_rewards.mean().item()
                print(f"  Loss: {loss:.4f}  Mean reward: {mean_reward:.4f}")

                if save_path is not None and (epoch + 1) % save_every == 0:
                    self.save_checkpoint(save_path / "last.ckpt")

                if save_path is not None and mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    self.save_checkpoint(save_path / "best_model.ckpt")
                    print(f"  New best reward {best_mean_reward:.4f} — saved best_model.ckpt")
            else:
                print("  No trajectories collected. Skipping step.")

