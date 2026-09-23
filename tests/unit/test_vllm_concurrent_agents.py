"""Agents sharing one in-process vLLM engine take turns on it, in batches.

vLLM's in-process client is not safe to call from two threads at once: a second
call mid-flight corrupts the socket to its engine core, and the process aborts
(``Assertion failed: !_current_out``), leaving the engine core behind holding the
card. So calls take turns on the engine, and the call that takes it next sends
every call that arrived meanwhile as one batch — each prompt with its own
sampling settings — which is how vLLM is meant to be driven.

The engine here is a stand-in that notices overlapping calls and records the
batches it was given, so no GPU is needed; the sampling settings are vLLM's own.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
# The sampling settings are vLLM's own class; the extra is not on the CI unit matrix.
pytest.importorskip("vllm", reason="the vllm extra is not installed")

from effgen.models.base import GenerationConfig  # noqa: E402
from effgen.models.vllm_engine import VLLMEngine  # noqa: E402


class _Engine:
    def __init__(self, pause: float) -> None:
        self.pause = pause
        self.inside = 0
        self.most_inside = 0
        self.batches: list[list[str]] = []
        self.params: list[object] = []
        self._guard = threading.Lock()

    def generate(self, prompts, sampling_params, **kwargs):
        with self._guard:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
        try:
            time.sleep(self.pause)
            self.batches.append(list(prompts))
            self.params.append(sampling_params)
            return [SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[SimpleNamespace(text=f"answer to {p}", token_ids=[7, 8],
                                         finish_reason="stop")],
            ) for p in prompts]
        finally:
            with self._guard:
                self.inside -= 1


class _Tokenizer:
    def encode(self, text):
        return text.split()


@pytest.fixture()
def engine():
    engine = VLLMEngine(model_name="org/model", apply_chat_template=False)
    engine.llm = _Engine(pause=0.05)
    engine.tokenizer = _Tokenizer()
    engine._context_length = 4096
    engine._is_loaded = True
    return engine


def _run_together(engine, n: int, stream_every: int = 0) -> dict[int, str]:
    out: dict[int, str] = {}
    start = threading.Barrier(n)

    def one(i: int) -> None:
        start.wait(5)
        config = GenerationConfig(temperature=0.1 * i, max_tokens=16)
        if stream_every and i % stream_every == 0:
            out[i] = "".join(engine.generate_stream(f"q{i}", config))
        else:
            out[i] = engine.generate(f"q{i}", config).text

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return out


def test_concurrent_calls_never_overlap_on_the_engine(engine) -> None:
    out = _run_together(engine, 8, stream_every=3)
    assert engine.llm.most_inside == 1
    assert out == {i: f"answer to q{i}" for i in range(8)}


def test_calls_that_arrive_together_go_as_one_batch_with_their_own_settings(engine) -> None:
    _run_together(engine, 8)
    assert sum(len(b) for b in engine.llm.batches) == 8
    assert len(engine.llm.batches) < 8
    batched = [p for p in engine.llm.params if isinstance(p, list)]
    assert batched, "no call was batched with another"
    temperatures = sorted(round(sp.temperature, 2) for group in batched for sp in group)
    assert len(set(temperatures)) == len(temperatures)


def test_a_call_on_its_own_is_sent_as_before(engine) -> None:
    """Over-correction guard: one prompt, one ``SamplingParams`` — not a list of one."""
    engine.generate("alone", GenerationConfig(temperature=0.0, max_tokens=8))
    assert engine.llm.batches == [["alone"]]
    assert not isinstance(engine.llm.params[0], list)


def test_a_failing_batch_reaches_every_caller_in_it(engine) -> None:
    """Over-correction guard: an engine error still reaches each caller as its own."""
    def refuse(prompts, sampling_params, **kwargs):
        time.sleep(0.05)
        raise RuntimeError("engine said no")

    engine.llm.generate = refuse
    errors: list[str] = []
    start = threading.Barrier(3)

    def one(i: int) -> None:
        start.wait(5)
        try:
            engine.generate(f"q{i}")
        except Exception as exc:  # noqa: BLE001 - the error is what is asserted
            errors.append(str(exc))

    threads = [threading.Thread(target=one, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(errors) == 3 and all("engine said no" in e for e in errors)
