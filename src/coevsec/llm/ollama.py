"""Ollama HTTP backend targeting the L40 host.

The endpoint URL is supplied via config/env (``COEVSEC_OLLAMA_BASE_URL``) so no
local Ollama install is assumed. Uses the ``/api/chat`` endpoint with
``format: json`` for structured tool-calling and repairs/retries malformed JSON.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from coevsec.llm.base import LLMBackend, LLMResponse, ToolCall


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        num_ctx: int | None = None,
    ) -> None:
        self.num_ctx = num_ctx
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def health_check(self) -> bool:  # pragma: no cover - network dependent
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = int(seed)
        if self.num_ctx:
            payload["options"]["num_ctx"] = int(self.num_ctx)
        num_gpu = os.environ.get("COEVSEC_OLLAMA_NUM_GPU")
        if num_gpu:
            payload["options"]["num_gpu"] = int(num_gpu)
        if json_schema is not None:
            # Ollama accepts a JSON schema object for constrained decoding, and
            # also the literal "json" mode; we pass the schema for stronger
            # structure and fall back to json mode on older servers.
            payload["format"] = json_schema

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s
                )
                resp.raise_for_status()
                body = resp.json()
            except (requests.RequestException, ValueError) as exc:
                last_err = exc
                if attempt == self.max_retries:
                    raise RuntimeError(f"ollama request failed: {exc}") from exc
                continue

            content = body.get("message", {}).get("content", "")
            prompt_tokens = int(body.get("prompt_eval_count", 0))
            completion_tokens = int(body.get("eval_count", 0))

            tool_call = None
            if json_schema is not None:
                tool_call = _parse_tool_call(content)
                if tool_call is None and attempt < self.max_retries:
                    # Older Ollama servers reject a schema object; fall back to json mode.
                    if not isinstance(payload.get("format"), str):
                        payload["format"] = "json"
                    payload["messages"].append({"role": "assistant", "content": content})
                    wants_reasoning = (
                        isinstance(json_schema, dict)
                        and "reasoning" in json_schema.get("properties", {})
                    )
                    shape = (
                        '{"reasoning": "", "tool": <name>, "params": {}, "rationale": "", "strategy": ""}'
                        if wants_reasoning
                        else '{"tool": <name>, "params": {}, "rationale": "", "strategy": ""}'
                    )
                    payload["messages"].append(
                        {
                            "role": "user",
                            "content": "Your previous reply was not valid JSON. "
                            f"Reply with ONLY the JSON object matching the schema: {shape}.",
                        }
                    )
                    continue

            return LLMResponse(
                text=content,
                tool_call=tool_call,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        raise RuntimeError(f"ollama request failed after retries: {last_err}")


def _parse_tool_call(content: str) -> ToolCall | None:
    """Best-effort extraction of a tool call from model output."""
    text = content.strip()
    # Strip common code fences.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to locate the first balanced JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        return None
    return ToolCall(
        tool=str(obj["tool"]),
        params=params,
        rationale=str(obj.get("rationale", "")),
        strategy=str(obj.get("strategy", "")),
        reasoning=str(obj.get("reasoning", "")),
    )
