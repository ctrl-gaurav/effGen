"""The continuous batcher puts requests with equal settings in one batch.

Concurrent callers each build their own ``GenerationConfig``; two configs with
the same values are the same settings, so their prompts go through one batched
call. Settings that differ still go separately, and keyword arguments that are
lists — tool definitions, stop sequences — are part of the settings rather than
a reason the batch cannot be formed.
"""

from __future__ import annotations

import threading

from effgen.models.base import BatchModel, GenerationConfig, GenerationResult, ModelType, TokenCount
from effgen.models.batching import ContinuousBatcher


class _Counting(BatchModel):
    def __init__(self) -> None:
        super().__init__(model_name="counting", model_type=ModelType.TRANSFORMERS)
        self._is_loaded = True
        self.batches: list[list[str]] = []
        self.singles: list[str] = []
        self._lock = threading.Lock()

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        with self._lock:
            self.singles.append(prompt)
        return GenerationResult(text=f"one:{prompt}", tokens_used=1, finish_reason="stop",
                                model_name=self.model_name)

    def generate_batch(self, prompts, config=None, **kwargs) -> list[GenerationResult]:
        with self._lock:
            self.batches.append(list(prompts))
        return [GenerationResult(text=f"batch:{p}", tokens_used=1, finish_reason="stop",
                                 model_name=self.model_name) for p in prompts]

    def generate_stream(self, prompt, config=None, **kwargs):
        yield self.generate(prompt, config).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096


TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


def _submit_all(batcher: ContinuousBatcher, requests) -> dict[str, str]:
    out: dict[str, str] = {}
    start = threading.Barrier(len(requests))

    def one(prompt, config, kwargs):
        start.wait(5)
        out[prompt] = batcher.submit(prompt, config, timeout=10, **kwargs).text

    threads = [threading.Thread(target=one, args=r) for r in requests]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return out


def test_equal_settings_built_separately_share_one_batch() -> None:
    model = _Counting()
    with ContinuousBatcher(model, max_batch_size=8, max_wait_ms=300) as batcher:
        out = _submit_all(batcher, [
            (f"p{i}", GenerationConfig(temperature=0.0, max_tokens=32), {"tools": TOOLS})
            for i in range(4)
        ])
    assert out == {f"p{i}": f"batch:p{i}" for i in range(4)}
    assert [sorted(b) for b in model.batches] == [["p0", "p1", "p2", "p3"]]
    assert model.singles == []


def test_different_settings_are_not_batched_together() -> None:
    """Over-correction guard: only equal settings share a forward pass."""
    model = _Counting()
    with ContinuousBatcher(model, max_batch_size=8, max_wait_ms=300) as batcher:
        out = _submit_all(batcher, [
            ("cold", GenerationConfig(temperature=0.0), {}),
            ("warm", GenerationConfig(temperature=0.9), {}),
        ])
    assert out == {"cold": "one:cold", "warm": "one:warm"}
    assert model.batches == []
    assert sorted(model.singles) == ["cold", "warm"]
