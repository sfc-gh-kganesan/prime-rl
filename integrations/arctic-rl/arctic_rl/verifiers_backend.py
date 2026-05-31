from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import requests
from openai.types.chat import ChatCompletion

from arctic_rl.oai_translation import (
    _arctic_results_to_oai_response,
    extract_routing_metadata,
    oai_sampling_params,
)
from loguru import logger as _logger

def get_logger():
    return _logger

_DISABLED_ENV_VALUES = {"0", "false", "no", "off"}
_DEFAULT_MAX_BATCH = 64
_DEFAULT_MAX_INFLIGHT_BATCHES = 12
_DEFAULT_FLUSH_INTERVAL_S = 0.05
_DEFAULT_MAX_QUEUE = 4096
_DEFAULT_LOG_EVERY_N = 256
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_RETRY_BASE_DELAY_S = 0.2
_TOKENIZER: Any | None = None
_TOKENIZER_ENABLE_THINKING = False


class _ArcticStreamUnavailable(RuntimeError):
    pass


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in _DISABLED_ENV_VALUES


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        get_logger().warning("{}={} is not an integer; using {}", name, value, default)
        return default
    return max(minimum, parsed)


def _ensure_verifiers_tokenizer() -> Any:
    global _TOKENIZER, _TOKENIZER_ENABLE_THINKING
    if _TOKENIZER is not None:
        return _TOKENIZER

    tokenizer_name = os.environ.get("PRIME_RL_ARCTIC_TOKENIZER_NAME")
    if not tokenizer_name:
        raise RuntimeError("Arctic verifiers tokenizer is not registered and PRIME_RL_ARCTIC_TOKENIZER_NAME is not set")

    from transformers import AutoTokenizer

    trust_remote_code_env = os.environ.get("PRIME_RL_ARCTIC_TOKENIZER_TRUST_REMOTE_CODE")
    trust_remote_code = (
        None if trust_remote_code_env is None else trust_remote_code_env.lower() not in _DISABLED_ENV_VALUES
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
    if chat_template := os.environ.get("PRIME_RL_ARCTIC_TOKENIZER_CHAT_TEMPLATE"):
        template_path = Path(chat_template)
        tokenizer.chat_template = template_path.read_text() if template_path.is_file() else chat_template
    tokenizer.pad_token_id = tokenizer.eos_token_id

    _TOKENIZER = tokenizer
    _TOKENIZER_ENABLE_THINKING = _env_flag("PRIME_RL_ARCTIC_ENABLE_THINKING", default=False)
    get_logger().info(
        "Loaded Arctic verifiers tokenizer from env (name={}, enable_thinking={})",
        tokenizer_name,
        _TOKENIZER_ENABLE_THINKING,
    )
    return _TOKENIZER


    guard = getattr(native_response.choices[0], _REASONING_GUARD_KEY, None)
    if isinstance(guard, dict):
        return guard
    return None


    tokens = getattr(response.message, "tokens", None)
    if tokens is None:
        return response

    completion_ids = list(tokens.completion_ids)
    completion_mask = [int(value) for value in tokens.completion_mask]
    completion_logprobs = list(tokens.completion_logprobs)
    if len(completion_mask) != len(completion_ids) or len(completion_logprobs) != len(completion_ids):
        raise RuntimeError(
            "reasoning guard received misaligned completion token fields "
            f"(ids={len(completion_ids)}, mask={len(completion_mask)}, logprobs={len(completion_logprobs)})"
        )

    for span in guard.get("masked_token_spans") or []:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        start = max(0, min(start, len(completion_mask)))
        end = max(start, min(end, len(completion_mask)))
        for idx in range(start, end):
            completion_mask[idx] = 0
            completion_logprobs[idx] = 0.0

    tokens.completion_mask = completion_mask
    tokens.completion_logprobs = completion_logprobs
    setattr(tokens, _REASONING_GUARD_KEY, guard)
    response.message.tokens = tokens
    setattr(response.message, _REASONING_GUARD_KEY, guard)
    return response


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(exclude_none=True)
        except TypeError:
            value = value.model_dump()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _coerce_token_ids(tokens: Any) -> list[int]:
    if isinstance(tokens, Mapping):
        tokens = tokens.get("input_ids")
    elif hasattr(tokens, "data") and isinstance(tokens.data, Mapping):
        tokens = tokens.data.get("input_ids")
    elif not isinstance(tokens, list):
        try:
            tokens = tokens["input_ids"]
        except (KeyError, TypeError, AttributeError):
            pass
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    if isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], list):
        tokens = tokens[0]
    if not isinstance(tokens, list):
        raise TypeError(f"Expected tokenizer output to be a token-id list, got {type(tokens).__name__}")
    return [int(token_id) for token_id in tokens]


def _local_tokenize(
    *,
    messages: str | list[Any],
    tools: list[Any] | None,
    extra_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    tokenizer = _ensure_verifiers_tokenizer()

    extra_kwargs = dict(extra_kwargs or {})
    if isinstance(messages, str):
        return _coerce_token_ids(
            tokenizer.encode(
                messages,
                add_special_tokens=bool(extra_kwargs.pop("add_special_tokens", False)),
            )
        )

    template_kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": bool(extra_kwargs.pop("add_generation_prompt", True)),
        "enable_thinking": extra_kwargs.pop("enable_thinking", _TOKENIZER_ENABLE_THINKING),
    }
    template_kwargs.update(extra_kwargs)
    if tools is not None:
        template_kwargs["tools"] = _jsonable(tools)
    return _coerce_token_ids(tokenizer.apply_chat_template(_jsonable(messages), **template_kwargs))


class _ArcticGenerateBatcher:
    def __init__(self, reconnect_config: Path):
        reconnect = json.loads(reconnect_config.read_text())
        host = reconnect["host"]
        port = int(reconnect["port"])
        self.base_url = f"http://{host}:{port}"
        self.sampling_job_id = int(reconnect["sampling_job_id"])
        self.max_batch = max(1, int(os.environ.get("PRIME_RL_ARCTIC_MAX_BATCH", str(_DEFAULT_MAX_BATCH))))
        self.max_inflight_batches = max(
            1, int(os.environ.get("PRIME_RL_ARCTIC_MAX_INFLIGHT_BATCHES", str(_DEFAULT_MAX_INFLIGHT_BATCHES)))
        )
        self.flush_interval_s = max(
            0.0, float(os.environ.get("PRIME_RL_ARCTIC_FLUSH_INTERVAL_S", str(_DEFAULT_FLUSH_INTERVAL_S)))
        )
        self.max_queue = max(1, int(os.environ.get("PRIME_RL_ARCTIC_MAX_QUEUE", str(_DEFAULT_MAX_QUEUE))))
        self.log_every_n = max(0, int(os.environ.get("PRIME_RL_ARCTIC_LOG_EVERY_N", str(_DEFAULT_LOG_EVERY_N))))
        self.max_retries = max(0, int(os.environ.get("PRIME_RL_ARCTIC_MAX_RETRIES", str(_DEFAULT_MAX_RETRIES))))
        self.retry_base_delay_s = max(
            0.0,
            float(os.environ.get("PRIME_RL_ARCTIC_RETRY_BASE_DELAY_S", str(_DEFAULT_RETRY_BASE_DELAY_S))),
        )
        self.split_on_retryable_error = os.environ.get("PRIME_RL_ARCTIC_SPLIT_ON_5XX", "1") != "0"
        self.stream_results = os.environ.get("PRIME_RL_ARCTIC_STREAM", "1").lower() not in _DISABLED_ENV_VALUES
        self.stream_fallback_logged = False
        self.queue: list[dict[str, Any]] = []
        self.condition = asyncio.Condition()
        self.semaphore = asyncio.Semaphore(self.max_inflight_batches)
        self.tasks: set[asyncio.Task] = set()
        self.loop_task: asyncio.Task | None = None
        self.thread_local = threading.local()
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_inflight_batches,
            thread_name_prefix="prime-rl-arctic-generate",
        )
        self.completed = 0
        self.failed = 0
        self.backend_calls = 0
        self.backend_prompts = 0
        self.backend_elapsed_total = 0.0
        self.backend_elapsed_max = 0.0
        self.queue_wait_total = 0.0
        self.queue_wait_max = 0.0
        self.backend_failures = 0
        self.backend_retries = 0
        self.backend_splits = 0
        get_logger().info(
            "Arctic generate batcher ready "
            f"(url={self.base_url} job_id={self.sampling_job_id} max_batch={self.max_batch} "
            f"max_inflight_batches={self.max_inflight_batches} flush_s={self.flush_interval_s:.3f} "
            f"max_retries={self.max_retries} split_on_5xx={self.split_on_retryable_error} "
            f"stream={self.stream_results} backend=ArcticHTTP)"
        )

    def ensure_started(self) -> None:
        if self.loop_task is None:
            self.loop_task = asyncio.create_task(self._batch_loop(), name="arctic-generate-batch-loop")

    async def generate(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        routing_key: str | None,
        strict: bool,
    ) -> dict[str, Any]:
        self.ensure_started()
        future = asyncio.get_running_loop().create_future()
        async with self.condition:
            if len(self.queue) >= self.max_queue:
                raise RuntimeError(f"Arctic generate queue exceeded max_queue={self.max_queue}")
            self.queue.append(
                {
                    "prompt_ids": prompt_ids,
                    "sampling_params": sampling_params,
                    "routing_key": routing_key,
                    "strict": strict,
                    "future": future,
                    "enqueued_at": time.perf_counter(),
                }
            )
            self.condition.notify()
        return await future

    async def _batch_loop(self) -> None:
        while True:
            await self.semaphore.acquire()
            try:
                batch = await self._pop_batch()
            except BaseException:
                self.semaphore.release()
                raise
            task = asyncio.create_task(self._run_batch(batch))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

    async def _pop_batch(self) -> list[dict[str, Any]]:
        async with self.condition:
            while not self.queue:
                await self.condition.wait()
            strict = bool(self.queue[0]["strict"])
            deadline = asyncio.get_running_loop().time() + self.flush_interval_s
            while self._matching_queue_size(strict) < self.max_batch:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

            selected: list[dict[str, Any]] = []
            remaining_items: list[dict[str, Any]] = []
            for item in self.queue:
                if len(selected) < self.max_batch and bool(item["strict"]) == strict:
                    selected.append(item)
                else:
                    remaining_items.append(item)
            self.queue = remaining_items
            return selected

    def _matching_queue_size(self, strict: bool) -> int:
        return sum(1 for item in self.queue if bool(item["strict"]) == strict)

    def _thread_session(self) -> requests.Session:
        session = getattr(self.thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self.thread_local.session = session
        return session

    def _generate_http(
        self,
        *,
        prompts: list[list[int]],
        sampling_params: list[dict[str, Any]],
        routing_key: list[str | None],
        strict: bool,
    ) -> list[dict[str, Any]]:
        # ReplicaPool.generate expects a single sampling_params dict broadcast
        # across the batch, not a per-prompt list. The orchestrator builds the
        # batch from one config so all entries are identical in practice.
        if any(p != sampling_params[0] for p in sampling_params[1:]):
            raise RuntimeError(
                "Arctic /generate does not support per-prompt sampling_params; "
                "all entries in the batch must match."
            )
        # verl_integration's /generate schema is `prompts: List[str]`, so decode
        # token-id prompts to text. Safe for RL: the loss masks prompt tokens, and
        # ZoRRO dedups on (consistent) prompt text; only response tokens matter.
        tok = _ensure_verifiers_tokenizer()
        text_prompts = [tok.decode(p, skip_special_tokens=False) for p in prompts]
        payload: dict[str, Any] = {
            "prompts": text_prompts,
            "sampling_params": sampling_params[0] if sampling_params else {},
        }
        if any(key is not None for key in routing_key):
            payload["routing_key"] = routing_key
        if strict:
            payload["strict"] = True
        response = self._thread_session().post(
            f"{self.base_url}/generate",
            params={"job_id": self.sampling_job_id},
            json=payload,
        )
        response.raise_for_status()
        return response.json()["results"]

    def _resolve_stream_result(self, item: dict[str, Any], result: dict[str, Any]) -> None:
        if not item["future"].done():
            item["future"].set_result(result)
            self.completed += 1

    def _resolve_stream_error(self, item: dict[str, Any], exc: Exception) -> None:
        if not item["future"].done():
            item["future"].set_exception(exc)
            self.failed += 1

    def _event_index(self, event: dict[str, Any], batch_size: int) -> int:
        try:
            index = int(event["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Arctic stream event missing valid index: {event!r}") from exc
        if index < 0 or index >= batch_size:
            raise RuntimeError(f"Arctic stream event index {index} outside batch size {batch_size}")
        return index

    def _generate_http_stream(
        self,
        *,
        batch: list[dict[str, Any]],
        prompts: list[list[int]],
        sampling_params: list[dict[str, Any]],
        routing_key: list[str | None],
        strict: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        if any(p != sampling_params[0] for p in sampling_params[1:]):
            raise RuntimeError(
                "Arctic /generate-stream does not support per-prompt sampling_params; "
                "all entries in the batch must match."
            )
        tok = _ensure_verifiers_tokenizer()
        text_prompts = [tok.decode(p, skip_special_tokens=False) for p in prompts]
        payload: dict[str, Any] = {
            "prompts": text_prompts,
            "sampling_params": sampling_params[0] if sampling_params else {},
        }
        if any(key is not None for key in routing_key):
            payload["routing_key"] = routing_key
        if strict:
            payload["strict"] = True

        response = self._thread_session().post(
            f"{self.base_url}/generate-stream",
            params={"job_id": self.sampling_job_id},
            json=payload,
            stream=True,
        )
        if response.status_code == 404:
            response.close()
            raise _ArcticStreamUnavailable("/generate-stream endpoint is unavailable")
        try:
            response.raise_for_status()
        except Exception:
            response.close()
            raise

        seen: set[int] = set()
        done = False
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                event = json.loads(raw_line)
                event_type = event.get("type")
                if event_type == "result":
                    index = self._event_index(event, len(batch))
                    seen.add(index)
                    loop.call_soon_threadsafe(
                        self._resolve_stream_result,
                        batch[index],
                        event["result"],
                    )
                elif event_type == "error":
                    index = self._event_index(event, len(batch))
                    seen.add(index)
                    loop.call_soon_threadsafe(
                        self._resolve_stream_error,
                        batch[index],
                        RuntimeError(str(event.get("error", "Arctic stream prompt failed"))),
                    )
                elif event_type == "stream_error":
                    raise RuntimeError(str(event.get("error", "Arctic stream failed")))
                elif event_type == "done":
                    done = True
                    break
                else:
                    raise RuntimeError(f"Unknown Arctic stream event type: {event!r}")
        finally:
            response.close()

        if not done:
            raise RuntimeError("Arctic stream ended before a done event")
        missing = sorted(set(range(len(batch))) - seen)
        if missing:
            raise RuntimeError(f"Arctic stream ended without results for indexes {missing[:16]}")

    async def _run_batch(self, batch: list[dict[str, Any]]) -> None:
        now = time.perf_counter()
        for item in batch:
            queue_wait = now - float(item["enqueued_at"])
            self.queue_wait_total += queue_wait
            self.queue_wait_max = max(self.queue_wait_max, queue_wait)
        try:
            await self._resolve_batch(batch)
        finally:
            self.semaphore.release()
            self._maybe_log_stats()

    async def _resolve_batch(self, batch: list[dict[str, Any]], *, attempt: int = 0) -> None:
        if self.stream_results:
            await self._resolve_batch_stream(batch, attempt=attempt)
            return

        try:
            results = await self._post_batch(batch)
            if len(results) != len(batch):
                raise RuntimeError(f"Arctic returned {len(results)} results for {len(batch)} prompts")
        except Exception as exc:
            if self._is_retryable(exc):
                if self.split_on_retryable_error and len(batch) > 1:
                    midpoint = max(1, len(batch) // 2)
                    self.backend_splits += 1
                    get_logger().warning(
                        "arctic-generate batch failed; splitting batch size {} -> {}/{}: {}",
                        len(batch),
                        midpoint,
                        len(batch) - midpoint,
                        self._short_error(exc),
                    )
                    await self._resolve_batch(batch[:midpoint])
                    await self._resolve_batch(batch[midpoint:])
                    return
                if attempt < self.max_retries:
                    self.backend_retries += 1
                    delay_s = self.retry_base_delay_s * (2**attempt)
                    if delay_s > 0:
                        await asyncio.sleep(delay_s)
                    get_logger().warning(
                        "arctic-generate batch failed; retrying batch size {} attempt {}/{}: {}",
                        len(batch),
                        attempt + 1,
                        self.max_retries,
                        self._short_error(exc),
                    )
                    await self._resolve_batch(batch, attempt=attempt + 1)
                    return

            self.failed += len(batch)
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(exc)
            return

        for item, result in zip(batch, results):
            if not item["future"].done():
                item["future"].set_result(result)
        self.completed += len(batch)

    async def _resolve_batch_stream(self, batch: list[dict[str, Any]], *, attempt: int = 0) -> None:
        try:
            await self._post_batch_stream(batch)
            await asyncio.sleep(0)
            unresolved = [item for item in batch if not item["future"].done()]
            if unresolved:
                raise RuntimeError(f"Arctic stream left {len(unresolved)} prompts unresolved")
            return
        except _ArcticStreamUnavailable:
            self.stream_results = False
            if not self.stream_fallback_logged:
                self.stream_fallback_logged = True
                get_logger().warning("/generate-stream unavailable; falling back to batch /generate")
            unresolved = [item for item in batch if not item["future"].done()]
            if unresolved:
                await self._resolve_batch(unresolved, attempt=attempt)
            return
        except Exception as exc:
            unresolved = [item for item in batch if not item["future"].done()]
            if not unresolved:
                return
            if self._is_retryable(exc):
                if self.split_on_retryable_error and len(unresolved) > 1:
                    midpoint = max(1, len(unresolved) // 2)
                    self.backend_splits += 1
                    get_logger().warning(
                        "arctic-generate stream failed; splitting unresolved batch size {} -> {}/{}: {}",
                        len(unresolved),
                        midpoint,
                        len(unresolved) - midpoint,
                        self._short_error(exc),
                    )
                    await self._resolve_batch_stream(unresolved[:midpoint])
                    await self._resolve_batch_stream(unresolved[midpoint:])
                    return
                if attempt < self.max_retries:
                    self.backend_retries += 1
                    delay_s = self.retry_base_delay_s * (2**attempt)
                    if delay_s > 0:
                        await asyncio.sleep(delay_s)
                    get_logger().warning(
                        "arctic-generate stream failed; retrying unresolved batch size {} attempt {}/{}: {}",
                        len(unresolved),
                        attempt + 1,
                        self.max_retries,
                        self._short_error(exc),
                    )
                    await self._resolve_batch_stream(unresolved, attempt=attempt + 1)
                    return

            self.failed += len(unresolved)
            for item in unresolved:
                if not item["future"].done():
                    item["future"].set_exception(exc)

    async def _post_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [item["prompt_ids"] for item in batch]
        sampling_params = [item["sampling_params"] for item in batch]
        routing_keys = [item["routing_key"] for item in batch]
        strict = bool(batch[0]["strict"])
        started = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.executor,
                lambda: self._generate_http(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    routing_key=routing_keys,
                    strict=strict,
                ),
            )
        except Exception:
            self.backend_failures += 1
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.backend_calls += 1
            self.backend_prompts += len(batch)
            self.backend_elapsed_total += elapsed
            self.backend_elapsed_max = max(self.backend_elapsed_max, elapsed)

    async def _post_batch_stream(self, batch: list[dict[str, Any]]) -> None:
        prompts = [item["prompt_ids"] for item in batch]
        sampling_params = [item["sampling_params"] for item in batch]
        routing_keys = [item["routing_key"] for item in batch]
        strict = bool(batch[0]["strict"])
        started = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self._generate_http_stream(
                    batch=batch,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    routing_key=routing_keys,
                    strict=strict,
                    loop=loop,
                ),
            )
        except Exception:
            self.backend_failures += 1
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.backend_calls += 1
            self.backend_prompts += len(batch)
            self.backend_elapsed_total += elapsed
            self.backend_elapsed_max = max(self.backend_elapsed_max, elapsed)

    def _is_retryable(self, exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code == 429 or status_code >= 500
        return isinstance(
            exc,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ),
        )

    def _short_error(self, exc: Exception) -> str:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        detail = getattr(response, "text", "") or ""
        if isinstance(status_code, int):
            return f"HTTP {status_code}: {detail.strip().replace(chr(10), ' ')[:300]}"
        return repr(exc)[:300]

    def _maybe_log_stats(self) -> None:
        if self.log_every_n <= 0 or self.backend_prompts <= 0:
            return
        total = self.completed + self.failed
        if total <= 0 or total % self.log_every_n != 0:
            return
        avg_backend_s = self.backend_elapsed_total / max(self.backend_calls, 1)
        avg_queue_s = self.queue_wait_total / max(self.backend_prompts, 1)
        get_logger().info(
            "arctic-generate stats: "
            f"completed={self.completed} failed={self.failed} queue_len={len(self.queue)} "
            f"backend_calls={self.backend_calls} backend_prompts={self.backend_prompts} "
            f"backend_failures={self.backend_failures} retries={self.backend_retries} splits={self.backend_splits} "
            f"avg_backend_s={avg_backend_s:.2f} max_backend_s={self.backend_elapsed_max:.2f} "
            f"avg_queue_s={avg_queue_s:.3f} max_queue_s={self.queue_wait_max:.3f}"
        )


_BATCHER: _ArcticGenerateBatcher | None = None
_PATCHED = False


def _get_batcher() -> _ArcticGenerateBatcher:
    global _BATCHER
    if _BATCHER is None:
        reconnect = os.environ.get("PRIME_RL_DIRECT_ARCTIC_RECONNECT_CONFIG")
        if not reconnect:
            raise RuntimeError("PRIME_RL_DIRECT_ARCTIC_RECONNECT_CONFIG is required for Arctic generation")
        _BATCHER = _ArcticGenerateBatcher(Path(reconnect))
    return _BATCHER


def _normalize_sampling_args(raw_sampling_args: Any) -> dict[str, Any]:
    normalized = dict(raw_sampling_args)
    if "max_tokens" in normalized:
        normalized["max_completion_tokens"] = normalized.pop("max_tokens")
    normalized["logprobs"] = True
    extra_body = dict(return_token_ids=True)
    if "extra_body" in normalized:
        normalized["extra_body"] = {
            **normalized["extra_body"],
            **extra_body,
        }
    else:
        normalized["extra_body"] = extra_body
    return {key: value for key, value in normalized.items() if value is not None}


async def _prompt_ids_for_request(
    client: Any,
    state: dict[str, Any],
    prompt: Any,
    tools: list[Any] | None,
    model: str,
) -> list[int]:
    if len(state["trajectory"]) == 0:
        return await client.tokenize(messages=prompt, tools=tools, model=model)

    prompt_ids = await client.get_prompt_ids(state, prompt, tools)
    if prompt_ids is None:
        get_logger().debug(
            "Arctic token-id stitching failed for a multi-turn rollout; falling back to local full-prompt tokenization."
        )
        return await client.tokenize(messages=prompt, tools=tools, model=model)
    return prompt_ids


# ============================================================================
# ArcticClient — verifiers.clients.Client subclass
# Replaces the resolve-poc monkey-patch with a proper Client subclass that
# integrators can plug in via verifiers' ClientConfig(client_type="custom",
# class_path="arctic_rl.verifiers_backend.ArcticClient").
# ============================================================================

from openai.types.chat import ChatCompletion as _ChatCompletion
from verifiers.clients.openai_chat_completions_token_client import (
    OpenAIChatCompletionsTokenClient as _OpenAIChatCompletionsTokenClient,
    _has_multimodal_content as _has_multimodal_content,
)


class ArcticClient(_OpenAIChatCompletionsTokenClient):
    """Verifiers Client that calls an Arctic RL server directly (no shim).

    Overrides three methods of ``OpenAIChatCompletionsTokenClient``:

    - ``tokenize``: tokenize locally with the registered tokenizer.
    - ``get_native_response``: build the request and route through the
      shared :class:`_ArcticGenerateBatcher` instead of calling OpenAI.
    - ``from_native_response``: apply the optional reasoning-guard mask.
    """

    async def tokenize(
        self,
        messages,
        tools,
        model,
        extra_kwargs=None,
        **kwargs,
    ):
        return _local_tokenize(messages=messages, tools=tools, extra_kwargs=extra_kwargs)

    async def get_native_response(
        self,
        prompt,
        model,
        sampling_args,
        tools=None,
        **kwargs,
    ):
        from typing import cast as _cast

        state = _cast(dict, kwargs.get("state"))
        if state is None:
            raise RuntimeError("ArcticClient requires verifiers state in get_native_response")

        has_multimodal = _has_multimodal_content(prompt) or any(
            _has_multimodal_content(step["prompt"]) for step in state["trajectory"]
        )
        if has_multimodal:
            raise RuntimeError("ArcticClient does not support multimodal prompts")

        normalized_sampling_args = _normalize_sampling_args(sampling_args)
        prompt_ids = await _prompt_ids_for_request(self, state, prompt, tools, model)
        extra_body = normalized_sampling_args.pop("extra_body", {})
        body = dict(
            model=model,
            messages=prompt,
            tools=tools,
            tokens=prompt_ids,
            **normalized_sampling_args,
            **extra_body,
        )
        routing_key, strict = extract_routing_metadata(body)
        result = await _get_batcher().generate(
            prompt_ids=prompt_ids,
            sampling_params=oai_sampling_params(body),
            routing_key=routing_key,
            strict=strict,
        )
        response = _arctic_results_to_oai_response(
            [result],
            model=model,
            request_id=f"chatcmpl-{uuid.uuid4().hex}",
            prompt_token_ids=prompt_ids,
        )
        return _ChatCompletion.model_validate(response)

    async def from_native_response(self, response):
        return await _OpenAIChatCompletionsTokenClient.from_native_response(self, response)
