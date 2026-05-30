"""arctic-trainer entrypoint.

Mirror of prime_rl.trainer.rl.train:main, but single-process (no torchrun)
and running the Arctic HTTP adapter instead of a local FSDP2 model.

The launcher writes trainer.toml and arctic.toml side-by-side. We load the
trainer config via the standard `cli(TrainerConfig)` mechanism and pick up
arctic.toml via the ARCTIC_CONFIG_TOML env var (populated by the launcher
because pydantic-config's cli() takes one positional @-path arg at a time).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import tomli
from loguru import logger

from arctic_rl.config import ArcticConfig
from arctic_rl.trainer import ArcticTrainerAdapter
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def _load_arctic_config() -> ArcticConfig:
    toml_path = os.environ.get("ARCTIC_CONFIG_TOML")
    if not toml_path:
        raise RuntimeError(
            "ARCTIC_CONFIG_TOML env var not set. The arctic-trainer is meant to be "
            "launched by rl_arctic_local which sets this variable pointing to arctic.toml."
        )
    path = Path(toml_path)
    if not path.exists():
        raise RuntimeError(f"ARCTIC_CONFIG_TOML points at non-existent path: {path}")
    with open(path, "rb") as f:
        data = tomli.load(f)
    return ArcticConfig(**data)


def main():
    set_proc_title("ArcticTrainer")
    trainer_cfg: TrainerConfig = cli(TrainerConfig)
    arctic_cfg = _load_arctic_config()
    logger.info("ArcticTrainer starting (backend={})", arctic_cfg.backend)
    ArcticTrainerAdapter(trainer_cfg, arctic_cfg).run()


if __name__ == "__main__":
    sys.exit(main())
