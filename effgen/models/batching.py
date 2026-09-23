"""Continuous batching helper for multi-request inference.

A small queue-based wrapper that coalesces concurrently submitted prompts and
flushes them through ``BaseModel.generate_batch`` (when supported) or via a
sequential fallback. This is useful for serving multiple requests against a
local model with minimal GPU idle time, without forcing callers to think
about batching.

Usage::

    batcher = ContinuousBatcher(model, max_batch_size=8, max_wait_ms=20)
    result = batcher.submit("Hello!")  # blocks until result returned

The batcher runs a single background worker thread. Call ``shutdown()`` (or
use it as a context manager) when done.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from types import TracebackType
from typing import Any

from .base import BaseModel, BatchModel, GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class _Request:
    prompt: str
    config: GenerationConfig | None
    kwargs: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: GenerationResult | None = None
    error: BaseException | None = None


def _settings_key(req: _Request) -> str:
    """A key equal for two requests whose generation settings are equal."""
    config = req.config
    if config is None:
        described: Any = None
    elif dataclasses.is_dataclass(config) and not isinstance(config, type):
        described = dataclasses.asdict(config)
    else:
        described = vars(config) if hasattr(config, "__dict__") else config
    try:
        return json.dumps([described, req.kwargs], sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr((described, sorted(req.kwargs.items(), key=lambda kv: kv[0])))


class ContinuousBatcher:
    """Coalesce concurrent generate() calls into batched inference."""

    def __init__(
        self,
        model: BaseModel,
        max_batch_size: int = 8,
        max_wait_ms: float = 20.0,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self._queue: Queue[_Request] = Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="effgen-batcher", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------ public
    def submit(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Queue *prompt* and block until its result is ready (or *timeout*).

        Args:
            prompt: The prompt to queue.
            config: Sampling and budget settings for the call.
            timeout: Seconds to wait for the result before giving up.
            **kwargs: Extra parameters forwarded to the wrapped model.

        Returns:
            The result for this prompt once its batch has run.
        """
        if self._stop.is_set():
            raise RuntimeError("ContinuousBatcher has been shut down")
        req = _Request(prompt=prompt, config=config, kwargs=kwargs)
        self._queue.put(req)
        if not req.event.wait(timeout=timeout):
            raise TimeoutError("ContinuousBatcher.submit timed out")
        if req.error is not None:
            raise req.error
        assert req.result is not None
        return req.result

    def shutdown(self, wait: bool = True) -> None:
        """Stop the worker thread (optionally joining it)."""
        self._stop.set()
        if wait:
            self._worker.join(timeout=5.0)

    def __enter__(self) -> "ContinuousBatcher":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown()

    # ------------------------------------------------------------------ worker
    def _collect_batch(self) -> list[_Request]:
        try:
            first = self._queue.get(timeout=0.1)
        except Empty:
            return []
        batch = [first]
        deadline = time.time() + self.max_wait
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(self._queue.get(timeout=remaining))
            except Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            try:
                self._dispatch(batch)
            except BaseException as exc:  # noqa: BLE001
                logger.exception("ContinuousBatcher dispatch failed: %s", exc)
                for req in batch:
                    req.error = exc
                    req.event.set()

    def _dispatch(self, batch: list[_Request]) -> None:
        # Group by the generation settings' values — only requests sharing
        # identical settings can be batched in a single forward pass. Two
        # callers that each built an equal config are one group; keying on the
        # config object's identity kept every caller in a group of its own, and
        # a list among the keyword arguments (tool definitions) failed the whole
        # batch on hashing.
        groups: dict[str, list[_Request]] = {}
        for req in batch:
            groups.setdefault(_settings_key(req), []).append(req)

        for group in groups.values():
            prompts = [r.prompt for r in group]
            cfg = group[0].config
            kw = group[0].kwargs
            if isinstance(self.model, BatchModel) and len(prompts) > 1:
                logger.debug(
                    "[batch] %d requests with equal settings share one batched call",
                    len(prompts),
                )
                results = self.model.generate_batch(prompts, cfg, **kw)
            else:
                results = [self.model.generate(p, cfg, **kw) for p in prompts]
            for req, res in zip(group, results):
                req.result = res
                req.event.set()
