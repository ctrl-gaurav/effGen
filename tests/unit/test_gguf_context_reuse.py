"""A GGUF engine keeps the prompt it has already evaluated, and takes turns.

llama.cpp holds the previous call's tokens in its KV cache and evaluates only
what follows the longest prefix a new prompt shares with them. An agent's turns
are each the previous prompt plus what happened since, so a turn that keeps the
cache skips recomputing most of its prompt. A call that fixes a seed starts from
a clear context instead, so the same seed on the same prompt draws the same text
every time (``test_gguf_seed_reproducibility.py`` pins that on a real model).

One llama.cpp context is one sequence and is not safe to drive from two threads
at once, so agents sharing one engine take turns on it.

The first cases use a stand-in for ``llama_cpp.Llama`` that behaves the way its
prefix reuse does and notices overlapping calls; the last one drives a real
cached GGUF model on CPU in a subprocess, since a context driven from two
threads at once can take the whole process down.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from effgen.models.base import GenerationConfig
from effgen.models.gguf_engine import GGUFEngine


class _Llama:
    """Tokens are bytes; reuse and overlap behave as llama-cpp-python's do."""

    def __init__(self, pause: float = 0.0) -> None:
        self.n_tokens = 0
        self._input_ids: list[int] = []
        self.resets = 0
        self.evaluated: list[int] = []
        self.inside = 0
        self.most_inside = 0
        self.pause = pause
        self._guard = threading.Lock()

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        return [1] + list(text)

    def reset(self) -> None:
        self.resets += 1
        self.n_tokens = 0

    def _enter(self) -> None:
        with self._guard:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)

    def _leave(self) -> None:
        with self._guard:
            self.inside -= 1

    def _evaluate(self, prompt: str) -> list[int]:
        tokens = self.tokenize(prompt.encode())
        shared = 0
        for a, b in zip(self._input_ids[: self.n_tokens], tokens[:-1]):
            if a != b:
                break
            shared += 1
        self.evaluated.append(len(tokens) - shared)
        time.sleep(self.pause)
        self._input_ids = tokens + [0, 0]
        self.n_tokens = len(self._input_ids)
        return tokens

    def __call__(self, prompt: str, **params):
        self._enter()
        try:
            tokens = self._evaluate(prompt)
            if params.get("stream"):
                return iter([{"choices": [{"text": "o"}]}, {"choices": [{"text": "k"}]}])
            return {"choices": [{"text": "ok", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(tokens), "completion_tokens": 2,
                              "total_tokens": len(tokens) + 2}}
        finally:
            self._leave()


def _engine(llama: _Llama) -> GGUFEngine:
    engine = GGUFEngine(model_name="stand-in.gguf", n_ctx=4096)
    engine._llm = llama
    engine._is_loaded = True
    return engine


GREEDY = GenerationConfig(temperature=0.0, max_tokens=2)


def test_a_turn_that_extends_the_last_prompt_keeps_the_kv_cache() -> None:
    llama = _Llama()
    engine = _engine(llama)
    first = "System: you answer.\nUser: look up the stock of bolts."
    engine.generate(first, GREEDY)
    result = engine.generate(first + "\nObservation: 412 bolts in stock.", GREEDY)
    assert llama.resets == 0
    assert result.metadata["cached_prompt_tokens"] == len(first) + 1
    assert llama.evaluated[1] < llama.evaluated[0]


def test_a_seeded_call_starts_from_a_clear_context() -> None:
    llama = _Llama()
    engine = _engine(llama)
    prompt = "Write one sentence about the sea."
    engine.generate(prompt, GREEDY)
    result = engine.generate(prompt, GenerationConfig(temperature=0.8, max_tokens=2, seed=7))
    assert llama.resets == 1
    assert result.metadata["cached_prompt_tokens"] == 0


def test_a_streamed_turn_keeps_the_kv_cache_too() -> None:
    llama = _Llama()
    engine = _engine(llama)
    engine.generate("User: hello there", GREEDY)
    assert "".join(engine.generate_stream("User: hello there, and more", GREEDY)) == "ok"
    assert llama.resets == 0
    assert engine.last_reused_prompt_tokens > 0


def test_agents_sharing_one_engine_take_turns_on_its_context() -> None:
    llama = _Llama(pause=0.02)
    engine = _engine(llama)

    def call(i: int) -> None:
        if i % 2:
            "".join(engine.generate_stream(f"User: question {i}", GREEDY))
        else:
            engine.generate(f"User: question {i}", GREEDY)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(llama.evaluated) == 6
    assert llama.most_inside == 1


def test_counting_tokens_does_not_wait_for_a_call_in_progress() -> None:
    """Over-correction guard: taking turns on the context does not hold up a count."""
    llama = _Llama(pause=0.5)
    engine = _engine(llama)
    worker = threading.Thread(target=engine.generate, args=("User: a long question", GREEDY))
    worker.start()
    time.sleep(0.05)
    started = time.perf_counter()
    assert engine.count_tokens("four words right here").count > 0
    assert time.perf_counter() - started < 0.25
    worker.join(5)


# --------------------------------------------------------------------------- #
# A real model
# --------------------------------------------------------------------------- #


def _find_gguf() -> str | None:
    for root in (os.environ.get("HF_HUB_CACHE", ""),
                 os.path.expanduser("~/.cache/huggingface/hub"),
                 os.environ.get("HF_HOME", "")):
        if root:
            found = glob.glob(os.path.join(root, "**", "*instruct*.gguf"), recursive=True)
            if found:
                return found[0]
    return None


def test_threads_sharing_a_real_gguf_model_all_finish() -> None:
    pytest.importorskip("llama_cpp")
    path = _find_gguf()
    if path is None:
        pytest.skip("no cached GGUF model found")
    import effgen

    root = os.path.dirname(os.path.dirname(os.path.abspath(effgen.__file__)))
    script = textwrap.dedent(f"""
        import os, sys, threading
        sys.path.insert(0, {root!r})
        import effgen
        if not os.path.abspath(effgen.__file__).startswith({root!r}):
            print("the subprocess imported another effgen:", effgen.__file__)
            sys.exit(3)
        from effgen.models.base import GenerationConfig
        from effgen.models.gguf_engine import GGUFEngine

        engine = GGUFEngine(model_name={path!r}, n_ctx=1024, n_threads=4,
                            n_threads_batch=4)
        engine.load()
        texts = {{}}

        def call(i):
            prompt = f"Question {{i}}: name one colour of the sky."
            texts[i] = engine.generate(prompt, GenerationConfig(temperature=0.0,
                                                                max_tokens=6)).text

        threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(texts) == 4 and all(isinstance(t, str) for t in texts.values())
        print("ALL FINISHED", len(texts))
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "ALL FINISHED 4" in done.stdout
