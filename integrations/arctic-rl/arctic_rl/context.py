"""Translate PRIME-RL packed microbatches to Arctic RL's batch format.

PRIME-RL packs multiple rollouts into ``[1, T]`` microbatches; Arctic RL
expects ``[B, max_S]`` with one rollout per row. We unpack, apply a
per-rollout left-shift (so labels match Arctic's shifted-index convention)
and zero the loss mask at the wrap-around position.
"""

from __future__ import annotations

import os

import torch

from arctic_rl.unpack import iter_rollout_slices, unpack_packed_microbatch
from prime_rl.trainer.rl.data import TensorMicroBatch

# Pad values mirror PRIME-RL's pad_micro_batch so the padded batch is
# numerically indistinguishable from a natively-padded one.
_PAD_VALUES: dict[str, float | int | bool] = {
    "input_ids": 1,
    "position_ids": 0,
    "old_log_probs_shifted": 0.0,
    "advantages": 0.0,
    "teacher_log_probs_shifted": 0.0,
    "loss_mask": False,
}


def _extract_and_roll_rollout(mb: TensorMicroBatch, start: int, end: int) -> dict:
    """Extract one rollout's slice from a packed mb, applying per-rollout `torch.roll(-1)`.

    Returns 1-D tensors (no batch dim). The roll is scoped to this rollout so
    cross-rollout contamination at boundary positions is impossible.
    """
    out: dict = {
        "input_ids": mb["input_ids"][0, start:end].clone(),
        "position_ids": mb["position_ids"][0, start:end].clone(),
        "old_log_probs_shifted": torch.roll(mb["inference_logprobs"][0, start:end], shifts=-1, dims=-1),
        "advantages": torch.roll(mb["advantages"][0, start:end], shifts=-1, dims=-1),
    }
    loss_mask = torch.roll(mb["loss_mask"][0, start:end], shifts=-1, dims=-1).clone()
    # The last position wraps around within this rollout; not a valid target.
    loss_mask[-1] = False
    out["loss_mask"] = loss_mask

    teacher = mb.get("teacher_logprobs")
    if teacher is not None:
        out["teacher_log_probs_shifted"] = torch.roll(teacher[0, start:end], shifts=-1, dims=-1)
    return out


def _zorro_layout(mbs: list[TensorMicroBatch], response_len: int) -> dict:
    """Reformat prime-rl's variable [prompt|response] rollouts into the rigid
    ``[left-pad prompt | right-pad response]`` layout ZoRRO's Qwen3ModelOncePatcher
    requires (prompt_len = seq_len - response_len; response = last response_len tokens).

    The patcher's ``find_prompt_groups`` dedups rollouts whose ``input_ids[:, :prompt_len]``
    are byte-identical, so prompts must be LEFT-padded to a uniform max_prompt_len
    (right-aligned to the prompt/response boundary). RL tensors (old_log_probs,
    advantages, loss_mask) are response-only, placed in the response region.

    ARCTIC_ZORRO_SHIFT env toggles the response-region roll convention (0 = none,
    matching verl's response-only tensors; -1 = next-token roll). Settled empirically.
    """
    shift = int(os.environ.get("ARCTIC_ZORRO_SHIFT", "0"))

    rollouts: list[dict] = []
    example_ids: list[int] = []
    for mb in mbs:
        mb_ids = mb.get("example_ids") or []
        for i, (start, end) in enumerate(iter_rollout_slices(mb)):
            ids = mb["input_ids"][0, start:end]
            lm = mb["loss_mask"][0, start:end].bool()
            resp_idx = lm.nonzero(as_tuple=True)[0]
            if resp_idx.numel() == 0:
                continue
            r0, r1 = int(resp_idx[0]), int(resp_idx[-1]) + 1
            rollouts.append(
                {
                    "prompt_ids": ids[:r0].clone(),
                    "resp_ids": ids[r0:r1].clone(),
                    "resp_lp": mb["inference_logprobs"][0, start:end][r0:r1].clone(),
                    "resp_adv": mb["advantages"][0, start:end][r0:r1].clone(),
                }
            )
            if i < len(mb_ids):
                example_ids.append(mb_ids[i])

    assert rollouts, "Expected at least one rollout with response tokens"
    max_plen = max(r["prompt_ids"].shape[0] for r in rollouts)
    seq = max_plen + response_len
    b = len(rollouts)
    dev = rollouts[0]["prompt_ids"].device

    input_ids = torch.full((b, seq), 1, dtype=torch.long, device=dev)
    attn = torch.zeros((b, seq), dtype=torch.long, device=dev)
    old_lp = torch.zeros((b, seq), dtype=torch.float32, device=dev)
    adv = torch.zeros((b, seq), dtype=torch.float32, device=dev)
    loss_mask = torch.zeros((b, seq), dtype=torch.bool, device=dev)

    for i, r in enumerate(rollouts):
        pl = r["prompt_ids"].shape[0]
        rl = min(r["resp_ids"].shape[0], response_len)
        # prompt left-padded (right-aligned, ending at the prompt/response boundary)
        input_ids[i, max_plen - pl : max_plen] = r["prompt_ids"]
        attn[i, max_plen - pl : max_plen] = 1
        # response left-aligned in the response region (right-padded)
        input_ids[i, max_plen : max_plen + rl] = r["resp_ids"][:rl]
        attn[i, max_plen : max_plen + rl] = 1
        rlp = r["resp_lp"][:rl]
        rad = r["resp_adv"][:rl]
        rmask = torch.ones(rl, dtype=torch.bool, device=dev)
        if shift:
            rlp = torch.roll(rlp, shifts=shift, dims=-1)
            rad = torch.roll(rad, shifts=shift, dims=-1)
            if shift < 0:
                rmask[shift:] = False  # wrapped tail is not a valid target
        old_lp[i, max_plen : max_plen + rl] = rlp
        adv[i, max_plen : max_plen + rl] = rad
        loss_mask[i, max_plen : max_plen + rl] = rmask

    position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
    out = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "position_ids": position_ids,
        # The server reads prompts.shape[1] as the prompt width to derive
        # response_len (= seq_len - prompt_width) for its packing/response math.
        "prompts": input_ids[:, :max_plen].clone(),
        "old_log_probs_shifted": old_lp,
        "advantages": adv,
        "loss_mask": loss_mask,
    }
    if len(example_ids) == b:
        out["prompt_group_ids"] = torch.tensor(example_ids, dtype=torch.long)
    return out


def microbatches_to_arctic_context(
    mbs: list[TensorMicroBatch], use_zorro: bool = False, response_len: int | None = None
) -> dict:
    """Consolidate all rollouts from a list of packed microbatches into one `[B, max_S]` batch.

    When ``use_zorro`` is True, routes to :func:`_zorro_layout`, which produces the
    rigid ``[left-pad prompt | right-pad response]`` layout the server's
    Qwen3ModelOncePatcher requires (response = last ``response_len`` tokens). Otherwise
    produces the standard per-rollout ``[B, max_S]`` rolled layout.
    """
    assert mbs, "Expected at least one microbatch"

    if use_zorro:
        assert response_len is not None, "use_zorro requires response_len"
        return _zorro_layout(mbs, response_len)

    rollouts: list[dict] = []
    raw_example_ids: list[int] = []
    for mb in mbs:
        slices = iter_rollout_slices(mb)
        mb_ids = mb.get("example_ids") or []
        for i, (start, end) in enumerate(slices):
            rollouts.append(_extract_and_roll_rollout(mb, start, end))
            if i < len(mb_ids):
                raw_example_ids.append(mb_ids[i])

    b = len(rollouts)
    max_s = max(r["input_ids"].shape[0] for r in rollouts)
    ref = rollouts[0]

    out: dict = {}
    for key, ref_tensor in ref.items():
        pad_val = _PAD_VALUES.get(key, 0)
        tensor = torch.full((b, max_s), pad_val, dtype=ref_tensor.dtype, device=ref_tensor.device)
        for i, r in enumerate(rollouts):
            length = r[key].shape[0]
            tensor[i, :length] = r[key]
        out[key] = tensor

    attention_mask = torch.zeros((b, max_s), dtype=ref["input_ids"].dtype, device=ref["input_ids"].device)
    for i, r in enumerate(rollouts):
        attention_mask[i, : r["input_ids"].shape[0]] = 1
    out["attention_mask"] = attention_mask

    # Build prompt_group_ids for prompt-mean loss aggregation. Maps each rollout row to its
    # example group so agg_loss("prompt-mean") averages token losses per rollout before
    # averaging across rollouts. Only present when all rollouts carried an example_id.
    if len(raw_example_ids) == b:
        out["prompt_group_ids"] = torch.tensor(raw_example_ids, dtype=torch.long)

    if use_zorro and len(raw_example_ids) == b:
        # Group rollout-row indices by shared example_id. Useful for client-side
        # diagnostics and as the pre-computed groups an alternative server path
        # (DedupActorWrapper) could consume. The Qwen3ModelOncePatcher path
        # currently ignores this and rebuilds groups from input_ids itself.
        groups: dict[int, list[int]] = {}
        for row, eid in enumerate(raw_example_ids):
            groups.setdefault(int(eid), []).append(row)
        out["zorro_prompt_groups"] = list(groups.values())

    return out


def microbatch_to_arctic_context(mb: TensorMicroBatch) -> dict:
    """Legacy single-microbatch path kept for unit tests.

    Prefer `microbatches_to_arctic_context([mb, ...])` in the training loop
    so that a single-rollout bin doesn't trip the server's DP-chunk assertion.
    """
    loss_mask = torch.roll(mb["loss_mask"], shifts=-1, dims=-1).clone()
    loss_mask[..., -1] = False

    rolled: dict = {
        "input_ids": mb["input_ids"],
        "position_ids": mb["position_ids"],
        "old_log_probs_shifted": torch.roll(mb["inference_logprobs"], shifts=-1, dims=-1),
        "advantages": torch.roll(mb["advantages"], shifts=-1, dims=-1),
        "loss_mask": loss_mask,
    }
    teacher = mb.get("teacher_logprobs")
    if teacher is not None:
        rolled["teacher_log_probs_shifted"] = torch.roll(teacher, shifts=-1, dims=-1)

    return unpack_packed_microbatch(rolled)
