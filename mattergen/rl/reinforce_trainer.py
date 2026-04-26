"""
REINFORCE Trainer for MatterGen — ablation without actor-critic.

Key differences from DDPOTrainer:
- No learned value network (no critic)
- Advantage = R - baseline, where baseline is an EMA of past rewards
- Combined (cont + disc) log-prob — no decoupling
- No PPO clipping
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.rl.ddpo_trainer import (
    DDPOConfig,
    DDPOTrainer,
    MatterGenActor,
    Trajectory,
)


@dataclass
class REINFORCEConfig:
    lr: float = 1e-4
    entropy_coeff: float = 0.01
    kl_coeff: float = 0.1
    grad_clip_norm: float = 1.0
    timestep_subsample_frac: float = 0.2
    ppo_mb_size: int = 4
    prob_clamp_min: float = 1e-6
    baseline_ema: float = 0.99  # EMA decay for running baseline


class REINFORCETrainer(DDPOTrainer):
    """
    Pure policy gradient without a learned value critic.
    Inherits trajectory collection and the training loop from DDPOTrainer;
    overrides update_step with a REINFORCE objective.
    """

    def __init__(
        self,
        actor: MatterGenActor,
        reward_fn: Callable[[ChemGraph], torch.Tensor],
        config: REINFORCEConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        # Bypass DDPOTrainer.__init__ — no critic needed.
        self.actor = actor.to(device)
        self.critic = None
        self.reward_fn = reward_fn
        self.config = config or REINFORCEConfig()
        self.device = torch.device(device)

        self.ref_actor = copy.deepcopy(actor)
        self.ref_actor.eval()
        for p in self.ref_actor.parameters():
            p.requires_grad = False
        self.ref_actor = self.ref_actor.to(device)

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=self.config.lr
        )
        self.critic_optimizer = None

        self.metrics_history: list[dict] = []
        self._baseline: float = 0.0

    def _update_baseline(self, trajectories: list[Trajectory]) -> None:
        all_rewards: list[float] = []
        for traj in trajectories:
            if traj.final_sample is None:
                continue
            r = traj.reward
            if isinstance(r, torch.Tensor):
                all_rewards.extend(r.tolist())
            else:
                all_rewards.append(float(r))
        if all_rewards:
            batch_mean = sum(all_rewards) / len(all_rewards)
            d = self.config.baseline_ema
            self._baseline = d * self._baseline + (1.0 - d) * batch_mean

    def compute_advantages(
        self, trajectories: list[Trajectory]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Advantage = R - EMA baseline (no critic)."""
        advantages_list, returns_list = [], []
        for traj in trajectories:
            if traj.final_sample is None:
                continue
            reward = torch.as_tensor(traj.reward, dtype=torch.float32, device=self.device)
            for _ in traj.steps:
                advantages_list.append(reward - self._baseline)
                returns_list.append(reward)
        return advantages_list, returns_list

    def _compute_reinforce_loss(
        self,
        states: list[ChemGraph],
        timesteps: torch.Tensor,
        dts: torch.Tensor,
        actions_pos: torch.Tensor,
        actions_cell: torch.Tensor,
        actions_atoms: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        eval_cfg = DDPOConfig(prob_clamp_min=cfg.prob_clamp_min)

        new_lp_cont, new_lp_disc, entropy_disc = self.actor.evaluate_actions(
            states, timesteps, dts, actions_pos, actions_cell, actions_atoms, eval_cfg
        )

        # Combined log-prob (no decoupling)
        log_prob = new_lp_cont + new_lp_disc
        policy_loss = -(log_prob * advantages).mean()

        entropy_bonus = -cfg.entropy_coeff * entropy_disc.mean()

        # Optional KL anchor against frozen reference
        if cfg.kl_coeff > 0:
            with torch.no_grad():
                ref_lp_cont, ref_lp_disc, _ = self.ref_actor.evaluate_actions(
                    states, timesteps, dts, actions_pos, actions_cell, actions_atoms, eval_cfg
                )
            kl = ((ref_lp_cont - new_lp_cont) + (ref_lp_disc - new_lp_disc)).mean().clamp(min=0)
        else:
            kl = torch.tensor(0.0, device=self.device)

        total_loss = policy_loss + entropy_bonus + cfg.kl_coeff * kl
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "entropy": entropy_disc.mean(),
            "kl": kl,
        }

    def update_step(self, trajectories: list[Trajectory]) -> dict[str, float]:
        self._update_baseline(trajectories)
        self.actor.train()

        states = [step.state for traj in trajectories for step in traj.steps]
        timesteps = torch.stack([step.timestep for traj in trajectories for step in traj.steps]).detach()
        dts = torch.stack([step.dt for traj in trajectories for step in traj.steps]).detach()
        actions_pos = torch.stack([step.action_pos for traj in trajectories for step in traj.steps]).detach()
        actions_cell = torch.stack([step.action_cell for traj in trajectories for step in traj.steps]).detach()
        actions_atoms = torch.stack([step.action_atoms for traj in trajectories for step in traj.steps]).detach()

        if not states:
            return {"error": "No valid trajectory steps"}

        with torch.no_grad():
            advantages_list, _ = self.compute_advantages(trajectories)
            if not advantages_list:
                return {"error": "No advantages computed"}
            advantages = torch.stack(advantages_list).to(self.device).detach()

        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = len(states)
        subsample_n = max(self.config.ppo_mb_size, int(num_samples * self.config.timestep_subsample_frac))
        if subsample_n < num_samples:
            idx = torch.randperm(num_samples, device=self.device)[:subsample_n].tolist()
            states = [states[j] for j in idx]
            timesteps = timesteps[idx]
            dts = dts[idx]
            actions_pos = actions_pos[idx]
            actions_cell = actions_cell[idx]
            actions_atoms = actions_atoms[idx]
            advantages = advantages[idx]
            num_samples = len(states)

        minibatch_size = self.config.ppo_mb_size
        batch_losses = []

        indices = torch.randperm(num_samples)
        for start in range(0, num_samples, minibatch_size):
            mb = indices[start:start + minibatch_size].tolist()

            losses = self._compute_reinforce_loss(
                [states[j] for j in mb],
                timesteps[mb], dts[mb],
                actions_pos[mb], actions_cell[mb], actions_atoms[mb],
                advantages[mb],
            )

            if torch.isnan(losses["total_loss"]) or torch.isinf(losses["total_loss"]):
                print("Warning: NaN/Inf loss, skipping")
                continue

            self.actor_optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_optimizer.step()

            batch_losses.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()})

        if not batch_losses:
            return {"error": "All batches skipped due to NaN"}
        return {k: sum(m[k] for m in batch_losses) / len(batch_losses) for k in batch_losses[0]}

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
        torch.save({
            "epoch": epoch,
            "actor_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "baseline": self._baseline,
            "config": self.config.__dict__,
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self._baseline = ckpt.get("baseline", 0.0)
        return ckpt["epoch"]
