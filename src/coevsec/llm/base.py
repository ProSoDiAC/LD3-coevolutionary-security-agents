"""LLM backend interface and structured tool-calling contract."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A parsed tool selection produced by the model."""

    tool: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    strategy: str = ""
    reasoning: str = ""


@dataclass
class LLMResponse:
    """A single completion, including token accounting for budget enforcement."""

    text: str
    tool_call: ToolCall | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMBackend(abc.ABC):
    """Abstract chat completion backend with optional structured output."""

    model: str

    @abc.abstractmethod
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
        """Return a completion. If ``json_schema`` is given, request JSON output.

        ``seed`` fixes the sampling seed where the backend supports it.
        """

    def health_check(self) -> bool:  # pragma: no cover - network dependent
        """Best-effort connectivity probe; subclasses may override."""
        return True
