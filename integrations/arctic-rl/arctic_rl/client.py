"""Build the ArcticRLClient that the trainer uses to talk to the Arctic RL server.

Mirrors the convergence-validated pattern from arctic-skyrl PR #17 / commit
f5e07dd: pass a full DeepSpeed config (optimizer, gradient_clipping=1.0,
gradient_accumulation_steps, train_micro_batch_size_per_gpu=1, bf16) plus a
small training_config={"lr": lr}. The server uses the ds_config to drive
DeepSpeed's optimizer/grad-accumulation; the training_config carries the LR
that the orchestrator's polling loop reads each step.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import tomli
from loguru import logger

from arctic_rl.config import ArcticConfig

if TYPE_CHECKING:
    from prime_rl.configs.trainer import TrainerConfig


def _read_orchestrator_toml() -> dict | None:
    arctic_toml = os.environ.get("ARCTIC_CONFIG_TOML")
    if not arctic_toml:
        return None
    orch_toml = Path(arctic_toml).parent / "orchestrator.toml"
    if not orch_toml.exists():
        return None
    with open(orch_toml, "rb") as f:
        return tomli.load(f)


def _compute_grad_accum_steps(orch: dict) -> int:
    """Compute gradient_accumulation_steps from orchestrator config.

    Mirrors arctic-skyrl's `grad_accum_steps = train_batch / mini_batch`. In
    prime-rl the analogue is `batch_size * rollouts_per_example` for the train
    batch and the trainer's microbatch size for the inner loop. With a single
    consolidated [B_total, max_S] fwd-bwd per orchestrator step, the natural
    grad_accum_steps is 1 — the server already accumulates over the packed
    batch. We expose a knob so larger setups can split.
    """
    return max(1, int(orch.get("micro_batches_per_step", 1)))


def _build_ds_config(arctic_cfg: ArcticConfig, trainer_cfg: TrainerConfig) -> dict:
    """Build the DeepSpeed config the server passes to deepspeed.initialize.

    The convergence-fix commit (arctic-skyrl f5e07dd) calls out three knobs as
    load-bearing for GRPO convergence: optimizer.AdamW with the user's LR,
    gradient_clipping=1.0, gradient_accumulation_steps matching the train/mini
    ratio. Without these the server defaulted to lr=1e-5 (10x too high) and
    updated weights every micro-batch.
    """
    optim = trainer_cfg.optim
    lr = getattr(optim, "lr", 1e-6)
    betas = getattr(optim, "betas", (0.9, 0.999))
    eps = getattr(optim, "eps", 1e-8)
    weight_decay = getattr(optim, "weight_decay", 0.0)

    orch = _read_orchestrator_toml() or {}
    grad_accum = _compute_grad_accum_steps(orch)

    return {
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [float(betas[0]), float(betas[1])],
                "eps": float(eps),
                "weight_decay": float(weight_decay),
            },
        },
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": grad_accum,
        "bf16": {"enabled": True},
    }


def _build_vllm_config(arctic_cfg: ArcticConfig, trainer_cfg: TrainerConfig) -> dict:
    vllm_config = dict(arctic_cfg.vllm_config or {})
    vllm_config.setdefault("max_model_len", trainer_cfg.model.seq_len)
    vllm_config.setdefault("tensor_parallel_size", arctic_cfg.sampling_tensor_parallel_size)
    return vllm_config


def _build_ds_worker_config(arctic_cfg: ArcticConfig) -> dict | None:
    """Build the ds_worker_config dict that activates ZoRRO on the server.

    Mirrors arctic-skyrl's ZoRRO wiring — the server reads ``use_zorro`` and
    forwards ``response_len``, ``rollout_n``, ``temperature``,
    ``tiled_logits_compute``, ``use_unpad``, and ``max_token_len`` to
    ``Qwen3ModelOncePatcher.__init__``.
    """
    if not arctic_cfg.use_zorro:
        return None

    orch = _read_orchestrator_toml() or {}

    response_len = arctic_cfg.zorro_response_len
    rollout_n = arctic_cfg.zorro_rollout_n
    temperature = arctic_cfg.zorro_temperature

    if response_len is None:
        response_len = (
            orch.get("train", {}).get("sampling", {}).get("max_completion_tokens")
            or orch.get("eval", {}).get("sampling", {}).get("max_completion_tokens")
        )
    if rollout_n is None:
        rollout_n = orch.get("rollouts_per_example", 1)
    if temperature is None:
        temperature = orch.get("train", {}).get("sampling", {}).get("temperature", 1.0)

    if response_len is None:
        raise ValueError(
            "use_zorro=True requires zorro_response_len (or orchestrator.train.sampling."
            "max_completion_tokens) to be set."
        )

    return {
        "use_zorro": True,
        "response_len": int(response_len),
        "rollout_n": int(rollout_n),
        "temperature": float(temperature),
        "tiled_logits_compute": arctic_cfg.zorro_tiled_logits_compute,
        "use_unpad": arctic_cfg.zorro_use_unpad,
        "use_autocast": arctic_cfg.zorro_use_autocast,
        "max_token_len": int(response_len) * int(rollout_n),
    }


def build_arctic_client(arctic_cfg: ArcticConfig, trainer_cfg: TrainerConfig):
    """Build and return an Arctic RL client. Blocks until all jobs are RUNNING.

    Targets the ``arctic_training.arctic_rl`` factory API (ArcticTraining-dss
    tunji/verl_integration), which carries ``ds_worker_config`` for ZoRRO and,
    on that branch, the Qwen3 patcher emits ``logprobs`` directly — so no
    server-side patch is needed. ``comm_protocol="http"`` selects the HTTP client
    (async ``fwd_bwd``/``step``/``sync_weights``; the trainer wraps them).
    """
    from arctic_training.arctic_rl.client import create_arctic_rl_client
    from arctic_training.arctic_rl.config import ArcticRLClientConfig

    optim_lr = getattr(trainer_cfg.optim, "lr", 1e-6)
    ds_config = _build_ds_config(arctic_cfg, trainer_cfg)
    training_config = {"lr": optim_lr}
    vllm_config = _build_vllm_config(arctic_cfg, trainer_cfg)
    ds_worker_config = _build_ds_worker_config(arctic_cfg)

    logger.info(
        "Arctic ds_config: lr={} grad_accum={} grad_clip={} bf16={} use_zorro={}",
        ds_config["optimizer"]["params"]["lr"],
        ds_config["gradient_accumulation_steps"],
        ds_config["gradient_clipping"],
        ds_config["bf16"]["enabled"],
        bool(ds_worker_config and ds_worker_config.get("use_zorro")),
    )
    if ds_worker_config is not None:
        logger.info("ZoRRO enabled — ds_worker_config={}", ds_worker_config)

    client_config = ArcticRLClientConfig(
        backend="local",
        comm_protocol="http",
        model_name=trainer_cfg.model.name,
        ds_config=ds_config,
        training_config=training_config,
        vllm_config=vllm_config,
        ds_worker_config=ds_worker_config,
        training_gpus=arctic_cfg.training_gpus,
        sampling_gpus=arctic_cfg.sampling_tensor_parallel_size,
        log_prob_gpus=arctic_cfg.log_prob_gpus,
        log_prob_engine="vllm",
        startup_timeout=float(os.environ.get("ARCTIC_STARTUP_TIMEOUT", 900)),
        server_logs=os.environ.get("ARCTIC_SERVER_LOGS", "1") == "1",
    )

    logger.info("Initializing Arctic RL client (model={}, training_gpus={}, sampling_gpus={})",
                client_config.model_name, client_config.training_gpus, client_config.sampling_gpus)
    client = create_arctic_rl_client(client_config)
    logger.info(
        "Arctic RL client ready: training={} sampling={} log_prob={}",
        client.training_job_id,
        client.sampling_job_id,
        client.log_prob_job_id,
    )
    return client
