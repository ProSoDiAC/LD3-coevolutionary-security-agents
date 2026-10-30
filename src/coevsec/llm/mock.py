"""Deterministic mock LLM backend for offline development and CI.

Lets the LLM *code path* (prompt assembly, JSON parsing, budget accounting) be
exercised end-to-end without a network or GPU. It selects an allowed tool
pseudo-randomly (seeded by the prompt) and emits schema-shaped JSON.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from coevsec.llm.base import LLMBackend, LLMResponse, ToolCall


class MockBackend(LLMBackend):
    def __init__(self, model: str = "mock", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def health_check(self) -> bool:
        return True

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
        seed = int(hashlib.sha256((system + user).encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        tool_call = None
        if json_schema is not None:
            allowed = _allowed_tools(json_schema)
            tool = rng.choice(allowed) if allowed else "wait"
            tool_call = ToolCall(
                tool=tool,
                params={},
                rationale="mock decision",
                strategy="mock",
            )
        text = "" if tool_call is None else '{"tool": "%s", "params": {}}' % tool_call.tool
        return LLMResponse(
            text=text,
            tool_call=tool_call,
            prompt_tokens=len(system.split()) + len(user.split()),
            completion_tokens=8,
        )


def _allowed_tools(schema: dict[str, Any]) -> list[str]:
    props = schema.get("properties", {})
    tool_prop = props.get("tool", {})
    enum = tool_prop.get("enum")
    if isinstance(enum, list):
        return [str(x) for x in enum]
    return []
