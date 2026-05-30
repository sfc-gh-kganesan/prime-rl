"""Pure OpenAI-chat <-> Arctic /generate translation functions."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def oai_sampling_params(body: dict) -> dict:
    """Extract Arctic sampling params from an OpenAI chat-completions body."""
    params: dict[str, Any] = {}

    if (temperature := body.get("temperature")) is not None:
        params["temperature"] = float(temperature)
    if (top_p := body.get("top_p")) is not None:
        params["top_p"] = float(top_p)
    if (top_k := body.get("top_k")) is not None:
        params["top_k"] = int(top_k)
    if (max_tokens := body.get("max_tokens") or body.get("max_completion_tokens")) is not None:
        params["max_tokens"] = int(max_tokens)

    nested = body.get("extra_body") or {}
    if (stop := body.get("stop", nested.get("stop"))) is not None:
        params["stop"] = stop
    if (n := body.get("n")) is not None:
        params["n"] = int(n)
    if (seed := body.get("seed")) is not None:
        params["seed"] = int(seed)

    if body.get("logprobs") is True and (top_logprobs := body.get("top_logprobs")) is not None:
        params["logprobs"] = int(top_logprobs)
    elif body.get("logprobs") is True:
        params["logprobs"] = 1

    for key in (
        "min_tokens",
        "repetition_penalty",
        "include_stop_str_in_output",
        "skip_special_tokens",
    ):
        if key in body:
            params[key] = body[key]
        elif key in nested:
            params[key] = nested[key]

    return params


def extract_routing_metadata(body: dict) -> tuple[str | None, bool]:
    """Pull the optional Arctic routing annotation from an OpenAI request body."""
    nested = body.get("extra_body") or {}
    routing_key = body.get("routing_key")
    if routing_key is None:
        routing_key = nested.get("routing_key")
    if routing_key is not None and not isinstance(routing_key, str):
        routing_key = str(routing_key)
    strict = bool(body.get("routing_strict") or nested.get("routing_strict"))
    return routing_key, strict


def _serialize_tool_arguments(arguments: Any) -> str | None:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        return json.dumps(arguments)
    return None


def _tool_call_from_payload(payload: dict[str, Any], index: int) -> dict[str, Any] | None:
    function = payload.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = payload.get("name")
        arguments = payload.get("arguments")

    if not isinstance(name, str) or not name:
        return None

    serialized_arguments = _serialize_tool_arguments(arguments)
    if serialized_arguments is None:
        return None

    tool_call_id = payload.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        tool_call_id = f"call_{index}"

    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": serialized_arguments,
        },
    }


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        return None

    tool_calls: list[dict[str, Any]] = []
    for raw_payload in matches:
        try:
            parsed_payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None

        payloads = parsed_payload if isinstance(parsed_payload, list) else [parsed_payload]
        for payload in payloads:
            if not isinstance(payload, dict):
                return None
            tool_call = _tool_call_from_payload(payload, len(tool_calls))
            if tool_call is None:
                return None
            tool_calls.append(tool_call)

    return tool_calls or None


def _tool_content_from_text(text: str) -> str | None:
    content = TOOL_CALL_RE.sub("", text).strip()
    return content or None


def _logprob_value(position: Any, token_id: int) -> float:
    if isinstance(position, (int, float)):
        return float(position)
    if not isinstance(position, dict) or not position:
        raise ValueError(f"Missing logprob for sampled token_id={token_id}")

    if "logprob" in position:
        return float(position["logprob"])

    entry = position.get(token_id)
    if entry is None:
        entry = position.get(str(token_id))
    if entry is None:
        raise ValueError(f"Missing logprob for sampled token_id={token_id}")

    if isinstance(entry, dict):
        if "logprob" not in entry:
            raise ValueError(f"Missing logprob for sampled token_id={token_id}")
        return float(entry["logprob"])
    return float(entry)


def _oai_logprobs_content(token_ids: list[int], logprobs: Any) -> list[dict[str, Any]]:
    if not isinstance(logprobs, list):
        return []
    return [
        {
            "token": "",
            "bytes": [],
            "logprob": _logprob_value(logprobs[index] if index < len(logprobs) else None, token_id),
            "top_logprobs": [],
        }
        for index, token_id in enumerate(token_ids)
    ]


def _arctic_result_to_oai_choice(result: dict, index: int) -> dict:
    """Convert one Arctic /generate result to one OpenAI chat choice."""
    text = result.get("text", "")
    tool_calls = parse_tool_calls(text) if isinstance(text, str) else None
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
    }
    finish_reason = result.get("finish_reason", "stop")
    if tool_calls is not None:
        message = {
            "role": "assistant",
            "content": _tool_content_from_text(text),
            "tool_calls": tool_calls,
        }
        finish_reason = "tool_calls"

    choice: dict[str, Any] = {
        "index": index,
        "message": message,
        "finish_reason": finish_reason,
    }
    token_ids = result.get("token_ids")
    if isinstance(token_ids, list):
        choice["token_ids"] = token_ids
        if (logprobs := result.get("logprobs")) is not None:
            choice["logprobs"] = {"content": _oai_logprobs_content(token_ids, logprobs)}
    return choice


def _arctic_results_to_oai_response(
    results: list[dict],
    model: str,
    request_id: str | None = None,
    prompt_token_ids: list[int] | None = None,
) -> dict:
    """Shape a full OpenAI chat.completion response from Arctic results."""
    completion_tokens = sum(
        len(token_ids) for result in results if isinstance((token_ids := result.get("token_ids")), list)
    )
    prompt_tokens = len(prompt_token_ids or [])
    response = {
        "id": request_id or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [_arctic_result_to_oai_choice(result, index) for index, result in enumerate(results)],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if prompt_token_ids is not None:
        response["prompt_token_ids"] = prompt_token_ids
    return response
