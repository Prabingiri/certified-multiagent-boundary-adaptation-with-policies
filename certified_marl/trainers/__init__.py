"""Masked PPO trainer exports."""

__all__ = [
    "RolloutBuffer",
    "Transition",
    "compute_gae",
    "clipped_ppo_loss",
    "masked_log_probs_and_entropy",
    "MaskedPPOConfig",
    "MaskedPPOTrainer",
    "actor_state_vec",
    "critic_state_vec",
    "mask_from_safe_actions",
]


def __getattr__(name: str):
    if name in {"RolloutBuffer", "Transition", "compute_gae",
                "clipped_ppo_loss", "masked_log_probs_and_entropy"}:
        from certified_marl.trainers import _common as _c
        return getattr(_c, name)
    if name in {"MaskedPPOConfig", "MaskedPPOTrainer", "actor_state_vec",
                "critic_state_vec", "mask_from_safe_actions"}:
        from certified_marl.trainers import masked_ppo as _mp
        return getattr(_mp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")