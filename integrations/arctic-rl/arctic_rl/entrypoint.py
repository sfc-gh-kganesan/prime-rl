"""Arctic RL backend entrypoint.

Selected via `[trainer] backend = "arctic_rl"` in `rl.toml`. Dispatched
generically by `prime_rl.entrypoints.rl:main`, which dynamically imports
`arctic_rl.entrypoint:main` — no Arctic-specific code lives in core
prime-rl.

The integration parses its own extended config (RLConfig + an `arctic`
block), then spawns:
  - `arctic-trainer` (single-process HTTP client to the Arctic RL server)
  - `arctic-shim`    (FastAPI OAI-compat proxy to Arctic RL `/generate`)
  - the prime-rl orchestrator (unchanged), pointed at the shim's base_url.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread
from typing import Annotated

import tomli_w
from pydantic import Field, model_validator

from prime_rl.configs.rl import RLConfig
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import (

    get_log_dir,

)
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process, set_proc_title

from arctic_rl.config import ArcticConfig

ARCTIC_TOML = "arctic.toml"
TRAINER_TOML = "trainer.toml"
ORCHESTRATOR_TOML = "orchestrator.toml"


class ArcticRLConfig(RLConfig):
    """Extended RLConfig with the integration-specific `arctic` block.

    Only loaded when the launcher dispatches to `arctic_rl.entrypoint:main`
    (i.e. when `trainer.backend = "arctic_rl"`). Core prime-rl never sees
    this schema.
    """

    arctic: Annotated[
        ArcticConfig,
        Field(description="Arctic RL backend config. Required when trainer.backend = 'arctic_rl'."),
    ]

    @model_validator(mode="after")
    def _validate_arctic_exclusivity(self) -> "ArcticRLConfig":
        # `[inference]` is a no-op in Arctic mode — Arctic owns sampling via
        # its own server, the native vLLM block is silently ignored. We don't
        # reject it so a native recipe can be flipped to Arctic with just an
        # overlay snippet (no edits to the original file).
        if self.inference is not None:
            self.inference = None
        if getattr(self, "teacher_inference", None) is not None:
            raise ValueError("Arctic mode does not support teacher_inference.")
        if self.deployment.type == "multi_node":
            raise ValueError("Arctic is single-node only in the initial integration.")
        if self.trainer.model.lora is not None:
            raise ValueError("Arctic does not support LoRA.")
        if self.trainer.max_concurrent_runs > 1:
            raise ValueError("Arctic does not support multi-run.")
        return self


def write_arctic_subconfigs(config: ArcticRLConfig, output_dir: Path) -> None:
    """Write trainer/orchestrator/arctic subconfigs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / TRAINER_TOML, "wb") as f:
        tomli_w.dump(config.trainer.model_dump(exclude_none=True, mode="json"), f)
    with open(output_dir / ORCHESTRATOR_TOML, "wb") as f:
        tomli_w.dump(config.orchestrator.model_dump(exclude_none=True, mode="json"), f)
    with open(output_dir / ARCTIC_TOML, "wb") as f:
        tomli_w.dump(config.arctic.model_dump(exclude_none=True, mode="json"), f)


def _run_arctic(config: ArcticRLConfig) -> None:
    """Launch arctic-trainer + orchestrator (no shim).

    The orchestrator's client is rewired to load
    ``arctic_rl.verifiers_backend.ArcticClient`` via verifiers'
    ``client_type="custom"`` extension; the ArcticClient calls the Arctic
    RL server directly in-process, replacing the FastAPI shim subprocess
    we used in the earlier version of this integration.
    """
    arctic_cfg = config.arctic
    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    config_dir = config.output_dir / "configs"
    write_arctic_subconfigs(config, config_dir)
    arctic_toml_path = config_dir / ARCTIC_TOML
    logger.info(f"Wrote subconfigs (including arctic.toml) to {config_dir}")

    if config.dry_run:
        logger.success("Dry run complete (arctic mode).")
        return

    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating arctic processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        # 1. Start arctic-trainer. Trainer signals readiness by writing
        # reconnect.json; clear any stale copy first.
        (config_dir / "reconnect.json").unlink(missing_ok=True)

        trainer_log_path = log_dir / "arctic_trainer.log"
        trainer_cmd = ["arctic-trainer", "@", (config_dir / TRAINER_TOML).as_posix()]
        logger.info("Starting arctic-trainer (%s)", " ".join(trainer_cmd))
        trainer_env = {
            **os.environ,
            "ARCTIC_CONFIG_TOML": arctic_toml_path.as_posix(),
            "PYTHONUNBUFFERED": "1",
            "LOGURU_FORCE_COLORS": "1",
        }
        # Ray 2.40+ auto-detects uv from env vars and re-launches its raylet
        # workers via `uv run python`, which lands them in a freshly-created
        # empty venv without ray. Strip uv-specific vars and disable Ray's uv
        # hook so workers run under sys.executable (the prime-rl venv python).
        for _k in (
            "VIRTUAL_ENV",
            "UV_PROJECT_ENVIRONMENT",
            "UV_ACTIVE",
            "UV_PROJECT",
            "UV_RUN_RECURSION_DEPTH",
        ):
            trainer_env.pop(_k, None)
        trainer_env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
        with open(trainer_log_path, "w") as f:
            trainer_process = Popen(trainer_cmd, env=trainer_env, stdout=f, stderr=f)
        processes.append(trainer_process)

        # Wait for reconnect.json (no timeout; trainer-side init can take 10+ min).
        reconnect_path = config_dir / "reconnect.json"
        logger.info("Waiting for arctic-trainer to initialize Arctic RL jobs…")
        while not reconnect_path.exists():
            if trainer_process.poll() is not None:
                raise RuntimeError(
                    f"arctic-trainer exited (code {trainer_process.returncode}) "
                    f"before writing reconnect.json. See {trainer_log_path}."
                )
            time.sleep(1)
        logger.success("arctic-trainer ready (reconnect.json written)")

        stop_event = Event()
        stop_events["arctic_trainer"] = stop_event
        t = Thread(
            target=monitor_process,
            args=(trainer_process, stop_event, error_queue, "arctic_trainer"),
            daemon=True,
        )
        t.start()
        monitor_threads.append(t)

        # 2. Rewrite orchestrator config to use the in-process ArcticClient
        # via verifiers' custom client_type extension. Arctic RL connection
        # info is passed via env var read by the ArcticClient at first call.
        config.orchestrator.student.client.client_type = "custom"
        config.orchestrator.student.client.class_path = (
            "arctic_rl.verifiers_backend.ArcticClient"
        )
        # The Arctic RL coordinator (spawned in-process by ArcticRLClient with
        # backend="local") listens on localhost:7000.
        arctic_url = "http://localhost:7000"
        config.orchestrator.student.client.base_url = [arctic_url]
        config.orchestrator.student.client.skip_model_check = True
        write_arctic_subconfigs(config, config_dir)

        orch_env = {
            **os.environ,
            "LOGURU_FORCE_COLORS": "1",
            "WANDB_PROGRAM": "uv run rl (arctic)",
            "WANDB_ARGS": json.dumps(sys.argv),
            "ARCTIC_RL_RECONNECT_CONFIG": reconnect_path.as_posix(),
            "PRIME_RL_DIRECT_ARCTIC_RECONNECT_CONFIG": reconnect_path.as_posix(),
            "PRIME_RL_ARCTIC_TOKENIZER_NAME": config.trainer.model.name,
            "PRIME_RL_ARCTIC_ENABLE_THINKING": "1" if arctic_cfg.enable_thinking else "0",
        }

        # 3. Start orchestrator.
        orch_cmd = ["orchestrator", "@", (config_dir / ORCHESTRATOR_TOML).as_posix()]
        logger.info("Starting orchestrator")
        with open(log_dir / "orchestrator.log", "w") as f:
            orch_process = Popen(orch_cmd, stdout=f, stderr=f, env=orch_env)
        processes.append(orch_process)

        stop_event = Event()
        stop_events["orchestrator"] = stop_event
        t = Thread(
            target=monitor_process,
            args=(orch_process, stop_event, error_queue, "orchestrator"),
            daemon=True,
        )
        t.start()
        monitor_threads.append(t)

        logger.success("Arctic startup complete. Tailing trainer log...")
        tail_process = Popen(f"tail -F '{trainer_log_path}'", shell=True)
        processes.append(tail_process)

        while not (stop_events["orchestrator"].is_set() and stop_events["arctic_trainer"].is_set()):
            if error_queue:
                logger.error(f"Error: {error_queue[0]}")
                logger.error("Terminating all arctic processes...")
                cleanup_threads(monitor_threads)
                cleanup_processes(processes)
                sys.exit(1)
            time.sleep(1)

        if orch_process.returncode != 0:
            logger.error(f"Orchestrator failed with exit code {orch_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)
        if trainer_process.returncode != 0:
            logger.error(f"Arctic trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(monitor_threads)
            cleanup_processes(processes)
            sys.exit(1)

        logger.success("Arctic RL training finished.")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt, terminating arctic processes...")
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        sys.exit(1)
    except Exception:
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)
        raise



def main() -> None:
    """Console-script entrypoint for the `arctic_rl` backend.

    Dispatched generically by `prime_rl.entrypoints.rl:main` when
    `trainer.backend = "arctic_rl"`. Parses the integration's extended
    schema (`ArcticRLConfig`) and runs the launcher.
    """
    set_proc_title("Launcher (arctic_rl)")
    config = cli(ArcticRLConfig)
    _run_arctic(config)


if __name__ == "__main__":
    main()
