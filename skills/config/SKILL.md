---
name: config
description: How the prime-rl config system works — TOML files, CLI, config composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Config

prime-rl uses `pydantic_config` (combines `tyro` and `pydantic`) for configuration. 

## Use configs

Every entrypoint accepts TOML files via `@` syntax and CLI overrides to configure it.

```bash
# Configure RL training with a TOML file
uv run rl @ examples/reverse_text/rl.toml

# Override specific fields via CLI
uv run rl @ examples/reverse_text/rl.toml --max-steps 50
```

Config resolve in the following order:

1. CLI arguments
2. Config files (merged left-to-right)
3. Class defaults (lowest)

## Compose configs

Multiple config files are merged left-to-right (later files override earlier ones):

```bash
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml
```

Nested configs can be loaded for specific sections:

```bash
uv run rl --model @ model.toml --data @ data.toml
```

Mixed composition works too:

```bash
uv run rl @ base.toml --trainer @ trainer_override.toml --trainer.lr 1e-3
```

Merging is deep — unset fields in the override are preserved from the base config.

## Inspect & validate configs

Use `--help` to see all available fields and their defaults. When combined with a config file, defaults reflect the TOML values:

```bash
uv run rl --help                                  # shows class defaults
uv run rl @ examples/reverse_text/rl.toml --help  # shows defaults from TOML
```

Use `--dry-run` to validate and dump the fully resolved config:

```bash
uv run rl @ examples/reverse_text/rl.toml --dry-run --output-dir /tmp/test
# Writes resolved TOML to /tmp/test/configs
```

## Naming

CLI uses kebab-case (`--model.max-model-len`), TOML uses snake_case (`max_model_len`). Both refer to the same field.

## General rules

- **Fail early**: incompatible option combinations (e.g. CP requires flash attention, NCCL broadcast requires async level 1) should raise in `model_validator` at config resolution time, not at runtime. When adding new constraints, add a validator to the config class.
- **Deprecation**: when renaming or removing config fields, emit a deprecation warning with a clear migration path (e.g. "field X is deprecated, use Y instead"). Do not silently drop fields — help users update their configs.

## Important patterns

### Boolean fields

```bash
uv run inference --model.enforce-eager          # sets to true
uv run inference --model.no-enforce-eager       # sets to false
```

In TOML, booleans must be explicit:

```toml
[model]
enforce_eager = true
```

### None fields

TOML has no null type. Use the string `"None"`:

```toml
max_model_len = "None"
```

On the CLI, pass `None` as a plain string:

```bash
uv run inference --model.max-model-len None
```

### List fields

In TOML, use `[[double brackets]]` (array of tables) for lists of objects:

```toml
[[orchestrator.env]]
id = "reverse-text"

[[orchestrator.env]]
id = "math-env"
```

On the CLI, list items are indexed: `--env.0.id reverse-text --env.1.id math-env`.

### Dict fields

In TOML, use a section:

```toml
[vllm_extra]
key1 = "value1"
key2 = 123
```

On the CLI, pass as a JSON string:

```bash
uv run inference --vllm-extra '{"key1": "value1", "key2": 123}'
```

### Discriminated unions

Some config fields use discriminated unions (e.g. loss type, data type). Set the `type` field to select the variant:

```toml
[trainer.loss]
type = "sft"

[data]
type = "fake"
batch_size = 2
```

On the CLI:

```bash
uv run sft --data.type fake --data.batch-size 4
```

If you wish to configure values of the default variant, you don't need to set the `type` field.

### SFT with images (multimodal)

SFT supports VLMs through the `renderers` package. Set `[model.vlm]` to opt in:

```toml
[model]
name = "Qwen/Qwen3.5-0.8B"

[model.vlm]
freeze_vision_encoder = true  # required when combined with LoRA
```

The trainer uses `create_renderer(tokenizer, "auto")` to pick the right renderer from the tokenizer's model name (Qwen3.5, Qwen3-VL, GLM, etc.). Image content blocks in `messages` are auto-resolved to pixel features by the renderer's processor; no prime-rl-side code knows about specific VLMs.

Constraints (enforced by validators in `SFTConfig.validate_vlm_constraints`):
- `data.micro_batch_size = 1` and `val.data.micro_batch_size = 1` (image samples can't be packed across samples — pixel buffers are variable per image).
- `model.cp = 1` (sequence sharding would split images across ranks).
- LoRA requires `model.vlm.freeze_vision_encoder = true` (a separate validator in `trainer.py`).

Loading local data files (e.g. JSONL or JSONL.zst):

```toml
[data]
type = "sft"
name = "json"
data_files = ["/path/to/file.jsonl.zst"]
```

`.zst` files are transparently decompressed to `$TMPDIR` before `load_dataset("json", data_files=...)`.

The renderer also handles tool calls and tool messages — OpenAI-format `tool_calls[].function.arguments` may be either a JSON string or a dict; the renderer accepts both.

### MoE + activation checkpointing trap

`[model.ac] mode = "full"` (the default) is **not safe** for MoE models. `torch.utils.checkpoint` raises mid-training:

```
torch.utils.checkpoint.CheckpointError:
  Recomputed values for the following tensors have different metadata than during the forward pass.
```

Root cause: the token-choice router does `topk` over near-tied bf16 scores. GPU bf16 matmul reduction order is non-deterministic, so the same input produces slightly different gate logits on the second call (forward vs activation-checkpoint recompute). Different winning experts → different `num_tokens_per_expert` → per-expert tensor shapes don't match between forward-saved and backward-recomputed → backward dies. The bug is in the MoE side, not the checkpoint wrapper. See [PyTorch #171355](https://github.com/pytorch/pytorch/issues/171355).

The fix is to **not** activation-checkpoint the MoE block. Use selective AC and exclude `routed_experts`:

```toml
[model.ac]
mode = "selective"
freq = 1
# Skip routed_experts: MoE recompute drifts between forward and backward.
# Keep norm/attn_proj/linear_attn to recover most of the memory savings.
targets = ["norm", "attn_proj", "linear_attn"]
```

Trade-off: MoE outputs and per-expert intermediates stay in memory across forward → backward, so peak usage rises. For Qwen3.5-35B-A3B at seq_len=32k, 8x B300 (275 GiB each), we measured ~126 GiB peak with full AC vs ~249 GiB with selective AC excluding `routed_experts`. Plan accordingly.

This is **not Blackwell-specific** and **not VLM-specific** — anyone training any MoE model with full AC will hit it eventually (the per-step crash probability is roughly proportional to the number of router invocations).

### SFT hard distill override

For hosted multi-tenant runs where the trainer image's `trainer.loss.type` is fixed, the orchestrator exposes a per-run override that forces SFT loss on every micro-batch without rebuilding the trainer. Set `orchestrator.use_sft_loss = true` alongside `orchestrator.teacher_rollout_model`; both must be configured together (the orchestrator validator enforces this). The orchestrator stamps each `TrainingSample.sft_loss = True`, which the trainer's `compute_loss` honors by dispatching to `sft_loss_fn` per batch — independent of the trainer's configured default loss.

### Model fields

For `BaseModel | None` fields (like `[ckpt]`, `[wandb]`, `[compile]`), a bare flag enables them with defaults:

```bash
uv run rl @ config.toml --model.compile              # enables compilation with defaults (fullgraph = false)
uv run rl @ config.toml --model.compile.fullgraph    # enables compilation and sets nested field (fullgraph = true)
```

In TOML, an empty section header does the same:

```toml
[ckpt]  # enables checkpointing with defaults
```

## Key files

- `src/prime_rl/utils/config.py` — re-exports `BaseConfig` and `cli` from pydantic_config
- `src/prime_rl/configs/` — all domain-specific config classes
- `configs/debug/` — minimal debug configs for testing
- `examples/` — full example configs for various tasks
