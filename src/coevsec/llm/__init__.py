"""Pluggable LLM backends.

The agent layer depends only on :class:`LLMBackend`, so switching from local
Ollama models (on the L40) to a hosted API later requires no agent changes.
"""

from coevsec.llm.base import LLMBackend, LLMResponse, ToolCall

__all__ = ["LLMBackend", "LLMResponse", "ToolCall", "make_backend"]


def make_backend(llm_cfg):
    """Factory building an LLM backend from an :class:`LLMConfig`."""
    from coevsec.core.config import LLMConfig

    assert isinstance(llm_cfg, LLMConfig)
    if llm_cfg.provider == "ollama":
        from coevsec.llm.ollama import OllamaBackend

        return OllamaBackend(
            base_url=llm_cfg.resolved_base_url(),
            model=llm_cfg.resolved_model(),
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            timeout_s=llm_cfg.timeout_s,
            num_ctx=llm_cfg.num_ctx,
        )
    if llm_cfg.provider == "mock":
        from coevsec.llm.mock import MockBackend

        return MockBackend(model=llm_cfg.resolved_model(), max_tokens=llm_cfg.max_tokens)
    raise ValueError(f"unknown llm provider: {llm_cfg.provider}")
