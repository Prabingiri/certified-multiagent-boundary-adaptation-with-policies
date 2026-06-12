"""Shared PPO primitives for masked CPAC training."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Transition:
    """One masked active-interface transition."""

    state_actor: np.ndarray
    state_critic: np.ndarray
    action: int
    mask: np.ndarray
    log_prob: float
    reward: float
    value: float
    done: bool
    prior_action: int = 0


@dataclass
class RolloutBuffer:
    """Append-only list of transitions, flushed every `rollout_length` epochs."""

    transitions: list[Transition] = field(default_factory=list)

    def add(self, tr: Transition) -> None:
        self.transitions.append(tr)

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()

    def as_tensors(self, device: torch.device) -> dict:
        """Stack transitions and map signed actions to tensor indices."""
        t = self.transitions
        actions_signed = np.array([tr.action for tr in t], dtype=np.int64)
        actions_idx = actions_signed + 1  # {-1,0,+1} -> {0,1,2}
        prior_signed = np.array(
            [getattr(tr, "prior_action", tr.action) for tr in t],
            dtype=np.int64,
        )
        prior_idx = prior_signed + 1
        return dict(
            states_actor=torch.as_tensor(
                np.stack([tr.state_actor for tr in t]),
                dtype=torch.float32, device=device,
            ),
            states_critic=torch.as_tensor(
                np.stack([tr.state_critic for tr in t]),
                dtype=torch.float32, device=device,
            ),
            actions=torch.as_tensor(actions_idx, dtype=torch.long, device=device),
            prior_actions=torch.as_tensor(
                prior_idx, dtype=torch.long, device=device,
            ),
            masks=torch.as_tensor(
                np.stack([tr.mask for tr in t]),
                dtype=torch.float32, device=device,
            ),
            old_log_probs=torch.as_tensor(
                [tr.log_prob for tr in t],
                dtype=torch.float32, device=device,
            ),
            rewards=torch.as_tensor(
                [tr.reward for tr in t],
                dtype=torch.float32, device=device,
            ),
            values=torch.as_tensor(
                [tr.value for tr in t],
                dtype=torch.float32, device=device,
            ),
            dones=torch.as_tensor(
                [tr.done for tr in t],
                dtype=torch.float32, device=device,
            ),
        )


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float = 0.0,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates and bootstrapped returns."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t].item()
        delta = (
            rewards[t].item()
            + gamma * next_value * non_terminal
            - values[t].item()
        )
        gae = delta + gamma * lam * non_terminal * gae
        adv[t] = gae
        next_value = values[t].item()
    returns = adv + values
    return adv, returns


def masked_log_probs_and_entropy(
    logits: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return masked-policy log probabilities and feasible-support entropy."""
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(
        masks > 0, logits, torch.full_like(logits, neg_inf),
    )
    log_probs_all = torch.log_softmax(masked_logits, dim=-1)
    log_prob_taken = log_probs_all.gather(
        1, actions.unsqueeze(-1),
    ).squeeze(-1)
    probs = torch.softmax(masked_logits, dim=-1)
    entropy = -(probs * torch.where(
        masks > 0, log_probs_all, torch.zeros_like(log_probs_all),
    )).sum(-1)
    return log_prob_taken, entropy


def clipped_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Negative PPO-Clip surrogate for gradient-descent optimizers."""
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss
