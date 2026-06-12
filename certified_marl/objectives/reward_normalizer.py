r"""Reward normalizer used by CPAC training.

CPAC training uses finalized pairwise rewards divided by a running
standard-deviation estimate and clipped. This implementation uses an EMA
standard deviation with alpha=0.99, clip=5.0, and warmup=100, matching the
reported CPAC settings. It divides by the running standard deviation and does
not subtract a running mean.

Behavior:
  - EMA of mean and variance for a stable running standard deviation.
  - Sign-preserving: divide by std only; do not center rewards.
  - Clip normalized values at +-clip_std.
  - Warmup: first warmup_steps samples update stats and are returned unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class RewardNormalizerConfig:
    """Configuration for the running EMA-based reward normalizer."""

    # EMA decay (closer to 1 = slower smoothing, more stable).
    # 0.99 means roughly the last 100 samples dominate.
    alpha: float = 0.99

    # Clip normalized rewards to +- clip_std after dividing by running std.
    # Standard PPO clip is 5 (5-sigma).
    clip_std: float = 5.0

    # Minimum running std to avoid division by zero during warmup.
    epsilon: float = 1.0e-6

    # Number of samples to accumulate stats before applying normalization.
    # First `warmup_steps` rewards are passed through unchanged. After
    # that, normalization kicks in.
    warmup_steps: int = 100


class RewardNormalizer:
    """EMA-based running reward normalizer (sign-preserving, clipped).

    Usage
    -----
    >>> rn = RewardNormalizer()
    >>> for r in rewards:
    ...     rn.update(r)
    ...     normalized = rn.normalize(r)

    Or in a single call:
    >>> normalized = rn.update_and_normalize(r)
    """

    def __init__(self, cfg: RewardNormalizerConfig | None = None) -> None:
        self.cfg = cfg or RewardNormalizerConfig()
        self.count: int = 0
        self.running_mean: float = 0.0
        # Initialize running variance to 1.0 so that during warmup the
        # normalizer (when active) returns the raw reward (since
        # x / sqrt(1) = x).
        self.running_var: float = 1.0

    @property
    def std(self) -> float:
        """Running standard deviation (clamped at epsilon)."""
        return max(math.sqrt(max(self.running_var, 0.0)), self.cfg.epsilon)

    def update(self, x: float) -> None:
        """EMA update of running mean and variance."""
        x = float(x)
        self.count += 1
        if self.count == 1:
            self.running_mean = x
            # Don't change running_var on first sample (no variance estimable).
            return
        alpha = self.cfg.alpha
        # Mean update
        delta = x - self.running_mean
        self.running_mean += (1.0 - alpha) * delta
        # Variance update (using post-update mean - same as Welford EMA)
        delta_post = x - self.running_mean
        self.running_var = (
            alpha * self.running_var
            + (1.0 - alpha) * (delta_post ** 2)
        )

    def normalize(self, x: float) -> float:
        """Divide by running std and clip. Sign preserved.

        During warmup (count < warmup_steps), returns raw x.
        """
        x = float(x)
        if self.count < self.cfg.warmup_steps:
            return x
        std = self.std
        out = x / std
        # Clip at +- clip_std.
        clip = self.cfg.clip_std
        if out > clip:
            return clip
        if out < -clip:
            return -clip
        return out

    def update_and_normalize(self, x: float) -> float:
        """Update stats then normalize. Convenience method."""
        self.update(x)
        return self.normalize(x)

