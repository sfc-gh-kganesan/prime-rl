"""Build the config dict sent to Arctic RL's grpo_loss on each /fwd-bwd call."""

from __future__ import annotations

from loguru import logger

_CISPO_LOGGED = False


def build_grpo_loss_config(
    current_version: int,
    eps_clip: float = 0.2,
    use_cispo: bool = False,
    loss_agg_mode: str = "token-mean",
) -> dict:
    """Build the ``processing.config`` dict for Arctic RL's grpo_loss."""
    global _CISPO_LOGGED
    if use_cispo:
        if not _CISPO_LOGGED:
            logger.info(f"LOSS CONFIG: CISPO (eps=0.2/0.28, agg={loss_agg_mode})")
            _CISPO_LOGGED = True
        return {
            "use_cispo_loss": True,
            "eps_clip": eps_clip,
            "eps_clip_higher": 0.28,
            "loss_agg_mode": loss_agg_mode,
            "prox_logp_method": "recompute",
            "importance_sampling_level": "token",
            "current_version": current_version,
        }

    if not _CISPO_LOGGED:
        logger.info(f"LOSS CONFIG: vanilla PPO (agg={loss_agg_mode})")
        _CISPO_LOGGED = True
    return {
        "eps_clip": eps_clip,
        "loss_agg_mode": loss_agg_mode,
        "prox_logp_method": "recompute",
        "importance_sampling_level": "token",
        "current_version": current_version,
    }
