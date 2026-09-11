"""Lazy model loading wrapper.

:class:`LazyModel` wraps a :class:`BaseModel` and defers calling ``load()``
until the model is first used. It also unloads the underlying model when it
has been idle for longer than ``idle_timeout`` seconds. Idle-eviction is
checked lazily on each access — there is no background thread, so wrapping a
model adds zero overhead when it isn't being used.

Wrapping is API-preserving: ``LazyModel`` forwards every ``BaseModel`` method
to the inner model and behaves like one (it inherits from ``BaseModel``).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

from .base import BaseModel, GenerationConfig, GenerationResult, TokenCount


class LazyModel(BaseModel):
    """Defer model loading until first use; unload after idle timeout."""

    def __init__(self, inner: BaseModel, idle_timeout: float | None = 600.0) -> None:
        # Mirror identity from the inner model so callers can introspect it.
        super().__init__(
            model_name=inner.model_name,
            model_type=inner.model_type,
            context_length=inner._context_length,
        )
        self._inner = inner
        self.idle_timeout = idle_timeout
        self._last_used: float = time.time()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ helpers
    def _ensure_loaded(self) -> None:
        with self._lock:
            if not self._inner.is_loaded():
                self._inner.load()
            self._is_loaded = True
            self._last_used = time.time()

    def _maybe_evict(self) -> None:
        if self.idle_timeout is None:
            return
        with self._lock:
            if not self._inner.is_loaded():
                return
            if (time.time() - self._last_used) > self.idle_timeout:
                try:
                    self._inner.unload()
                finally:
                    self._is_loaded = False

    def touch(self) -> None:
        """Mark the model as recently used."""
        self._last_used = time.time()

    @property
    def inner(self) -> BaseModel:
        """The wrapped model instance."""
        return self._inner

    # ------------------------------------------------------------------ BaseModel API
    def load(self) -> None:
        """Load the wrapped model now instead of on first use."""
        self._ensure_loaded()

    def unload(self) -> None:
        """Unload the wrapped model; the next call loads it again."""
        with self._lock:
            if self._inner.is_loaded():
                self._inner.unload()
            self._is_loaded = False

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate via the wrapped model, loading it on demand.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the wrapped model.

        Returns:
            The wrapped model's result.
        """
        self._maybe_evict()
        self._ensure_loaded()
        return self._inner.generate(prompt, config, **kwargs)

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream via the wrapped model, loading it on demand.

        Args:
            prompt: The prompt to send.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the wrapped model.

        Returns:
            The wrapped model's chunk iterator.
        """
        self._maybe_evict()
        self._ensure_loaded()
        return self._inner.generate_stream(prompt, config, **kwargs)

    def count_tokens(self, text: str) -> TokenCount:
        """Count tokens via the wrapped model (loads it: the tokenizer may be needed)."""
        # count_tokens may need the tokenizer; load on demand.
        self._ensure_loaded()
        return self._inner.count_tokens(text)

    def get_context_length(self) -> int:
        """Context length of the wrapped model without forcing a load."""
        if self._inner.is_loaded() or self._inner._context_length is not None:
            return self._inner.get_context_length()
        # Fall back to whatever we were told at construction time.
        return self._context_length or 4096

    def supports_tool_calling(self) -> bool:
        """Delegate to the wrapped model."""
        return self._inner.supports_tool_calling()

    def tool_call_support(self) -> str:
        """Delegate to the wrapped model."""
        return self._inner.tool_call_support()

    def supports_conversation(self) -> bool:
        """Delegate to the wrapped model."""
        return self._inner.supports_conversation()

    def supports_message_protocol(self) -> bool:
        """Delegate to the wrapped model."""
        return self._inner.supports_message_protocol()

    def get_metadata(self) -> dict[str, Any]:
        """Delegate to the wrapped model."""
        return self._inner.get_metadata()

    def __repr__(self) -> str:
        return f"LazyModel({self._inner!r}, idle_timeout={self.idle_timeout})"
