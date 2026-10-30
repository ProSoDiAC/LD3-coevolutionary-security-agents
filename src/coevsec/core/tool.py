"""Structured tool specifications.

Tools are the *capabilities* the environment exposes. An
agent selects a tool and supplies parameters; the LLM never executes raw code.
Each tool carries a JSON schema so parameters can be validated (and so the LLM
can be prompted with an exact contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jsonschema

from coevsec.core.types import Role


@dataclass
class ToolSpec:
    """A single capability exposed by the environment.

    Parameters
    ----------
    name        Unique tool identifier used in :class:`~coevsec.core.types.Action`.
    description Human/LLM-readable summary of what the tool does.
    roles       Which agent roles may use it.
    parameters  JSON schema (draft 2020-12) for the ``params`` object.
    cost        Base resource cost charged whenever the tool is used.
    category    Taxonomy category for post-hoc analysis (see taxonomy.py).
    """

    name: str
    description: str
    roles: tuple[Role, ...]
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    cost: float = 1.0
    category: str = "misc"

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Return a list of human-readable validation errors (empty if valid)."""
        try:
            jsonschema.validate(instance=params, schema=self.parameters)
            return []
        except jsonschema.ValidationError as exc:  # pragma: no cover - message formatting
            return [exc.message]

    def to_prompt_dict(self) -> dict[str, Any]:
        """Compact representation used when prompting an LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Holds all tools and answers "what can this role do?" queries."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool registration: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def for_role(self, role: Role) -> list[ToolSpec]:
        return [t for t in self._tools.values() if role in t.roles]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())
