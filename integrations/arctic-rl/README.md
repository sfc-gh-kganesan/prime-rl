# Arctic RL backend for PRIME-RL

Opt-in integration that delegates training (forward/backward/optimizer/weight-sync)
and rollout generation to a remote Arctic RL server. Lives entirely under
`integrations/arctic-rl/` so core PRIME-RL has zero Arctic-specific code.

## Enabling

Add this to any PRIME-RL `rl.toml`:

```toml
[trainer]
backend = "arctic_rl"

[arctic]
backend = "remote"
url = "http://your-arctic-server:7000"
```

The PRIME-RL launcher (`prime_rl.entrypoints.rl:main`) peeks at
`trainer.backend` before strict config parse and dynamically dispatches to
`arctic_rl.entrypoint:main` when it's set to `"arctic_rl"`. Otherwise the
native PRIME-RL path runs unchanged and `arctic_rl` is never imported.

## Install

Install this integration alongside PRIME-RL:

```
uv pip install -e ./integrations/arctic-rl
```

Plus the Arctic RL client (private until release — see Arctic RL release
notes for the install command).

## What gets replaced

In Arctic mode the launcher swaps two of PRIME-RL's three subprocesses:

| Slot | Native PRIME-RL | Arctic mode |
|---|---|---|
| Trainer | `torchrun` + FSDP2 | `arctic-trainer` (single CPU process, HTTP client) |
| Inference | vLLM server | `arctic-shim` (FastAPI OpenAI-compat proxy) |
| Orchestrator | verifiers + rollouts | unchanged |

The orchestrator points at the shim's `base_url` (rewritten by the
launcher); rollout files / STABLE markers / weight-sync semantics are
all preserved.

## Out of scope (rejected by validator)

- Multi-node deployment
- LoRA, multi-run, `teacher_inference`, `[inference]` (mutually exclusive)
- Multimodal
