"""Arctic trainer adapter.

Replaces PRIME-RL's torchrun+FSDP2 trainer with a single CPU process that
issues ``/fwd-bwd``, ``/step``, and ``/sync-weights`` over HTTP to the
Arctic RL server. Reuses PRIME-RL's DataLoader and writes the STABLE
marker after each weight sync so the orchestrator's polling loop is
unchanged.
"""

from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from arctic_rl.client import build_arctic_client
from arctic_rl.config import ArcticConfig
from arctic_rl.context import microbatches_to_arctic_context
from arctic_rl.loss_config import build_grpo_loss_config
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.scheduler import setup_scheduler


def _run_async(coro):
    """Run an async ArcticRL HTTP-client coroutine to completion from sync code."""
    return asyncio.run(coro)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_single_process_dist() -> None:
    """world_size=1 gloo group so PRIME-RL's DataLoader can call get_world()."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_get_free_port()))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


def _write_stable_marker(broadcast_dir: Path, step: int) -> None:
    """Write the zero-byte STABLE file the orchestrator polls before each step."""
    stable_path = broadcast_dir / f"step_{step}" / "STABLE"
    stable_path.parent.mkdir(parents=True, exist_ok=True)
    stable_path.touch()


def _concat_microbatches(mbs: list[dict]) -> dict:
    """Concatenate the DataLoader's per-DP microbatches into one consolidated batch."""
    if len(mbs) == 1:
        return dict(mbs[0])

    out: dict = {}
    for key, value in mbs[0].items():
        if value is None:
            out[key] = None
        elif isinstance(value, torch.Tensor):
            out[key] = torch.cat([mb[key] for mb in mbs], dim=0)
        else:
            out[key] = value
    return out


class ArcticTrainerAdapter:
    """Single-process trainer that delegates fwd/bwd/optimizer to Arctic RL.

    Reuses PRIME-RL's DataLoader and scheduler unchanged; writes the
    STABLE marker after each ``sync_weights`` so the orchestrator's
    polling loop is unaffected.
    """

    def __init__(self, trainer_cfg: TrainerConfig, arctic_cfg: ArcticConfig):
        self.trainer_cfg = trainer_cfg
        self.arctic_cfg = arctic_cfg

        _init_single_process_dist()

        from transformers import AutoTokenizer

        from prime_rl.trainer.rl.data import DataLoader, FakeDataLoader
        from prime_rl.trainer.runs import Progress, setup_multi_run_manager

        self.progress = Progress()

        # MultiRunManager is a singleton DataLoader depends on.
        setup_multi_run_manager(
            output_dir=trainer_cfg.output_dir,
            max_runs=trainer_cfg.max_concurrent_runs,
            device=torch.device("cpu"),
        )

        if trainer_cfg.data.fake is not None:
            self.loader = FakeDataLoader(
                config=trainer_cfg.data.fake,
                seq_len=trainer_cfg.data.fake.seq_len if hasattr(trainer_cfg.data.fake, "seq_len") else 512,
                dp_world_size=1,
            )
            self._using_fake = True
        else:
            tokenizer = AutoTokenizer.from_pretrained(trainer_cfg.tokenizer.name)
            self.loader = DataLoader(
                output_dir=trainer_cfg.output_dir,
                start_step=self.progress.step,
                dp_world_size=1,
                seq_len=trainer_cfg.data.seq_len if hasattr(trainer_cfg.data, "seq_len") else 2048,
                pad_to_multiple_of=trainer_cfg.data.pad_to_multiple_of
                if hasattr(trainer_cfg.data, "pad_to_multiple_of")
                else 64,
                tokenizer=tokenizer,
                config=trainer_cfg.rollout_transport,
            )
            self._using_fake = False

        self.client = build_arctic_client(arctic_cfg, trainer_cfg)

        # Resolve the ZoRRO response_len (the response-region width the server's
        # patcher splits on) the same way the client builds ds_worker_config:
        # explicit config, else orchestrator.train.sampling.max_completion_tokens.
        self._zorro_response_len = None
        self._zorro_meta: dict = {}
        if arctic_cfg.use_zorro:
            from arctic_rl.client import _read_orchestrator_toml

            orch = _read_orchestrator_toml() or {}
            self._zorro_response_len = arctic_cfg.zorro_response_len or (
                orch.get("train", {}).get("sampling", {}).get("max_completion_tokens")
                or orch.get("eval", {}).get("sampling", {}).get("max_completion_tokens")
            )
            assert self._zorro_response_len, "use_zorro requires a resolvable response_len"
            self._zorro_response_len = int(self._zorro_response_len)

            rollout_n = int(arctic_cfg.zorro_rollout_n or orch.get("rollouts_per_example", 1))
            temperature = float(
                arctic_cfg.zorro_temperature
                if arctic_cfg.zorro_temperature is not None
                else orch.get("train", {}).get("sampling", {}).get("temperature", 1.0)
            )
            from transformers import AutoTokenizer as _AT

            _tok = _AT.from_pretrained(trainer_cfg.tokenizer.name)
            pad_id = _tok.pad_token_id if _tok.pad_token_id is not None else (_tok.eos_token_id or 0)
            # Static meta the AT-dss server reads for the ZoRRO + grpo path.
            self._zorro_meta = {
                "pad_token_id": int(pad_id),
                "temperature": temperature,
                "use_zorro": True,
                "rollout_n": rollout_n,
                "zorro_max_rollouts": rollout_n,
                "max_response_len": self._zorro_response_len,
                "calculate_entropy": False,
                "rollout_is_weights": None,
            }
            logger.info("ZoRRO meta: {}", self._zorro_meta)

        # Dummy single-parameter optimizer used only to drive the LR schedule.
        # The actual optimizer lives server-side; we pass the computed LR via
        # adam_params on each /step call, which overrides the server's own LR.
        _dummy = torch.nn.Parameter(torch.zeros(1))
        _optim = torch.optim.AdamW([_dummy], lr=trainer_cfg.optim.lr)
        self._lr_scheduler = setup_scheduler(
            _optim, trainer_cfg.scheduler, trainer_cfg.max_steps or 1, trainer_cfg.optim.lr
        )
        self._optim = _optim

        # <output_dir>/run_default/broadcasts matches PRIME-RL's native
        # broadcast directory layout.
        self.broadcast_dir = Path(trainer_cfg.output_dir) / "run_default" / "broadcasts"

        # Old-API ArcticRLClient owns the server subprocess and exposes job IDs
        # directly. The verifiers backend reads these from a small reconnect.json
        # written here so the orchestrator process can attach.
        reconnect_path = Path(trainer_cfg.output_dir) / "configs" / "reconnect.json"
        reconnect_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        reconnect_path.write_text(
            _json.dumps(
                {
                    "host": self.client.config.host,
                    "port": self.client.config.port,
                    "backend": self.client.config.backend,
                    "model_name": self.client.config.model_name,
                    "training_job_id": self.client.training_job_id,
                    "sampling_job_id": self.client.sampling_job_id,
                    "log_prob_job_id": self.client.log_prob_job_id,
                }
            )
        )
        logger.info("Wrote reconnect config → {}", reconnect_path)

    def run(self) -> None:
        max_steps = self.trainer_cfg.max_steps or 100
        while self.progress.step < max_steps:
            step = self.progress.step
            logger.info("Step {} — waiting for batch", step)
            self.loader.wait_for_batch()
            mbs = self.loader.get_batch()

            processing_config = build_grpo_loss_config(
                current_version=step,
                use_cispo=self.arctic_cfg.use_cispo_loss,
                loss_agg_mode=self.arctic_cfg.loss_agg_mode,
            )

            # Send all rollouts in one consolidated [B_total, max_S] batch
            # so the server's torch.chunk(dim=0, world_size) splits cleanly.
            # The server repacks per DP shard before the model forward, so
            # activation memory matches a per-microbatch packed call.
            kwargs = microbatches_to_arctic_context(
                mbs, use_zorro=self.arctic_cfg.use_zorro, response_len=self._zorro_response_len
            )
            logger.info(
                "Step {} — /fwd-bwd (B={}, S={}, across {} source microbatch(es))",
                step,
                kwargs["input_ids"].shape[0],
                kwargs["input_ids"].shape[1],
                len(mbs),
            )
            # Write the scheduled LR before fwd-bwd so the orchestrator can
            # read it during rollout generation for this same step.
            self._lr_scheduler.step()
            current_lr = self._optim.param_groups[0]["lr"]
            (self.broadcast_dir.parent / "last_lr").write_text(str(current_lr))

            # arctic_rl (tunji/skyrl_integration) wire format: the server's
            # unpack_batch maps batch["batch"] -> model-forward kwargs and
            # batch["meta"] -> context (loss tensors, NOT sent to the model).
            # grpo self-computes logprobs from logits, so no compute_logprobs post.
            # position_ids is required by ZoRRO's Qwen3ModelOncePatcher.
            # verl_integration contract: the server forwards `engine(**batch, **meta)`
            # and calls `verl_grpo_loss(model_outputs, batch, meta, config, device)`.
            # RL tensors go in `batch` with verl names (response_mask/old_log_probs/
            # advantages); the ZoRRO patched forward ignores the extras and supplies
            # `logprobs`. `meta` carries config only. batch/meta must not share keys.
            model_kwargs = {
                "input_ids": kwargs["input_ids"],
                "attention_mask": kwargs["attention_mask"],
                "response_mask": kwargs["loss_mask"],
                "advantages": kwargs["advantages"],
            }
            if "position_ids" in kwargs:
                model_kwargs["position_ids"] = kwargs["position_ids"]
            if "prompts" in kwargs:
                model_kwargs["prompts"] = kwargs["prompts"]

            seq_len = kwargs["input_ids"].shape[1]
            b = kwargs["input_ids"].shape[0]
            context = dict(self._zorro_meta)
            context.setdefault("actor_config", {})
            context.setdefault("policy_loss_config", {})
            context["batch_num_tokens"] = int(kwargs["loss_mask"].sum().item())
            context["global_batch_size"] = b
            if self._zorro_meta:
                context["max_prompt_len"] = seq_len - self._zorro_response_len
                context["max_token_len_per_gpu"] = seq_len * b

            # Compute old_log_probs from the CURRENT training policy (a no-grad
            # forward), NOT from the sampler's rollout logprobs. This makes the PPO
            # importance ratio exp(new-old) ≈ 1 at the gradient step, eliminating the
            # vLLM↔DeepSpeed logprob gap that otherwise biases verl_grpo and causes
            # the policy to drift/collapse after the initial reward rise.
            nograd_kwargs = {k: v for k, v in model_kwargs.items() if k != "old_log_probs"}
            nograd_resp = _run_async(
                self.client.fwd_no_grad(
                    {"batch": nograd_kwargs, "meta": context, "processing": {"post": [], "loss_fn": None}},
                    reference_model=False,
                )
            )
            old_lp = nograd_resp.get("batch", nograd_resp).get("logprobs")
            if old_lp is not None:
                model_kwargs["old_log_probs"] = old_lp.to(kwargs["advantages"].device)
            else:
                model_kwargs["old_log_probs"] = torch.zeros_like(kwargs["advantages"])

            result = _run_async(
                self.client.fwd_bwd(
                    {"batch": model_kwargs, "meta": context},
                    processing={"loss_fn": "verl_grpo", "post": [], "config": processing_config},
                )
            )
            avg_loss = result.get("avg_loss") or result.get("loss") or float("nan")
            logger.info("Step {} — avg_loss={:.4f}", step, avg_loss)

            logger.info("Step {} — /step", step)
            step_result = _run_async(self.client.step()) or {}

            if step > 0:
                logger.info("Step {} — /sync-weights", step)
                _run_async(self.client.sync_weights())
                _write_stable_marker(self.broadcast_dir, step)

            # Clear ready_to_update for the next step. Native PRIME-RL does
            # this via FileSystemWeightBroadcast.broadcast_weights() which we
            # skip (Arctic owns the weights).
            if not self._using_fake:
                mgr = get_multi_run_manager()
                for idx in mgr.used_idxs:
                    mgr.ready_to_update[idx] = False

            self.progress.step += 1
            logger.info(
                "Step {} done — last_lr={} grad_norm={}",
                step,
                step_result.get("last_lr"),
                step_result.get("grad_norm"),
            )

        logger.success("Training finished after {} steps", self.progress.step)
