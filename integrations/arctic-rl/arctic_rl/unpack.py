"""Unpack PRIME-RL packed ``[1, T]`` microbatches to ``[B, S]`` padded form.

Required because Arctic RL's per-replica chunk along dim 0 fails when
``world_size > 1`` and the input batch is ``[1, T]``.
"""

from __future__ import annotations

import torch

# Pad values mirror PRIME-RL's pad_micro_batch. loss_mask=False is the
# authoritative guard at padded positions.
_PAD_VALUES: dict[str, float | int | bool] = {
    "input_ids": 1,
    "position_ids": 0,
    "advantages": 0.0,
    "inference_logprobs": 0.0,
    "old_log_probs_shifted": 0.0,
    "teacher_logprobs": 0.0,
    "teacher_log_probs_shifted": 0.0,
    "loss_mask": False,
    "temperatures": 1.0,
}


def _detect_rollout_starts(position_ids: torch.Tensor) -> list[int]:
    """Indices where a new rollout begins in a packed ``[1, T]`` tensor.

    A boundary is any position where position_ids decreases (or doesn't
    increment by 1), plus position 0.
    """
    assert position_ids.dim() == 2 and position_ids.shape[0] == 1, (
        f"Expected position_ids of shape [1, T], got {tuple(position_ids.shape)}"
    )
    pos_flat = position_ids.squeeze(0)
    t = pos_flat.shape[0]
    is_start = torch.zeros(t, dtype=torch.bool, device=pos_flat.device)
    is_start[0] = True
    is_start[1:] = pos_flat[1:] - pos_flat[:-1] != 1
    return is_start.nonzero(as_tuple=True)[0].tolist()


def iter_rollout_slices(mb: dict) -> list[tuple[int, int]]:
    """Per-rollout ``(start, end)`` slices, dropping the trailing pad segment.

    Guarded so a fully-masked real rollout in a non-trailing position
    isn't silently discarded.
    """
    position_ids = mb["position_ids"]
    loss_mask = mb["loss_mask"]

    t = position_ids.shape[1]
    starts = _detect_rollout_starts(position_ids)
    ends = starts[1:] + [t]
    slices = list(zip(starts, ends))

    loss_mask_flat = loss_mask.squeeze(0)
    if len(slices) > 1 and not loss_mask_flat[slices[-1][0] : slices[-1][1]].any():
        slices = slices[:-1]

    assert len(slices) >= 1, "Expected at least one rollout after dropping trailing pad"
    return slices


def unpack_packed_microbatch(rolled: dict) -> dict:
    """Convert a ``[1, T]`` packed microbatch to ``[B, S]`` padded form.

    Caller must have already applied any per-token alignment shift.
    Adds an ``attention_mask`` to the output.
    """
    position_ids = rolled["position_ids"]
    loss_mask = rolled["loss_mask"]

    t = position_ids.shape[1]
    starts = _detect_rollout_starts(position_ids)
    n_segs = len(starts)
    ends = starts[1:] + [t]
    lengths = [e - s for s, e in zip(starts, ends)]

    loss_mask_flat = loss_mask.squeeze(0)
    if n_segs > 1 and not loss_mask_flat[starts[-1] : ends[-1]].any():
        starts = starts[:-1]
        ends = ends[:-1]
        lengths = lengths[:-1]

    assert len(lengths) >= 1, "Expected at least one rollout after dropping trailing pad"

    b = len(lengths)
    s = max(lengths)

    out: dict = {}
    for key, value in rolled.items():
        if not isinstance(value, torch.Tensor) or value.dim() != 2 or value.shape[0] != 1:
            out[key] = value
            continue
        pad_val = _PAD_VALUES.get(key, 0)
        flat = value.squeeze(0)
        padded = torch.full((b, s), pad_val, dtype=value.dtype, device=value.device)
        for i, (seg_start, seg_len) in enumerate(zip(starts, lengths)):
            padded[i, :seg_len] = flat[seg_start : seg_start + seg_len]
        out[key] = padded

    attention_mask = torch.zeros((b, s), dtype=position_ids.dtype, device=position_ids.device)
    for i, seg_len in enumerate(lengths):
        attention_mask[i, :seg_len] = 1
    out["attention_mask"] = attention_mask

    return out
