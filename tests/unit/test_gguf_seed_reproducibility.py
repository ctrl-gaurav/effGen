"""GGUF (llama.cpp) generation is reproducible for a fixed seed.

The sampler is seeded on every completion, so the same seed yields identical
text, a different seed yields different text, and greedy decoding (temperature 0)
is deterministic — including when the second call reuses the first call's KV
cache for the prompt they share. This drives
a real cached GGUF model on CPU — no mock — and skips when llama-cpp-python or a
cached GGUF file is unavailable.
"""
from __future__ import annotations

import glob
import os

import pytest

pytest.importorskip("llama_cpp")

from effgen.models.base import GenerationConfig  # noqa: E402
from effgen.models.gguf_engine import GGUFEngine  # noqa: E402


def _find_gguf() -> str | None:
    roots = [
        os.environ.get("HF_HUB_CACHE", ""),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.environ.get("HF_HOME", ""),
    ]
    for root in roots:
        if not root:
            continue
        matches = glob.glob(os.path.join(root, "**", "*instruct*.gguf"), recursive=True)
        if matches:
            return matches[0]
    return None


_GGUF = _find_gguf()
pytestmark = pytest.mark.skipif(_GGUF is None, reason="no cached GGUF model found")


@pytest.fixture(scope="module")
def engine():
    eng = GGUFEngine(model_name=_GGUF, n_ctx=1024, verbose=False)
    eng.load()
    yield eng
    eng.unload()


def test_same_seed_is_reproducible(engine):
    prompt = "Write one short sentence about the sea."
    cfg = {"temperature": 0.8, "max_tokens": 12}
    a = engine.generate(prompt, GenerationConfig(seed=1234, **cfg)).text
    b = engine.generate(prompt, GenerationConfig(seed=1234, **cfg)).text
    assert a == b


def test_per_call_seed_kwarg_is_reproducible(engine):
    """``seed=`` passed as a bare keyword behaves like the config field.

    The same call shape is pinned for the other local engines — under test for
    vLLM and MLX in ``tests/unit/test_local_engine_call_overrides.py``, live for
    Transformers in ``tests/integration/test_local_seed_reproducible_live.py``.
    """
    prompt = "Write one short sentence about the sea."
    cfg = {"temperature": 0.8, "max_tokens": 12}
    a = engine.generate(prompt, seed=1234, **cfg).text
    b = engine.generate(prompt, seed=1234, **cfg).text
    c = engine.generate(prompt, seed=4321, **cfg).text
    assert a == b
    assert a != c
    # A per-call seed supersedes the config's field for that call.
    assert engine.generate(prompt, GenerationConfig(seed=4321, **cfg), seed=1234).text == a


def test_different_seed_differs(engine):
    prompt = "Write one short sentence about the sea."
    cfg = {"temperature": 0.8, "max_tokens": 12}
    a = engine.generate(prompt, GenerationConfig(seed=1234, **cfg)).text
    c = engine.generate(prompt, GenerationConfig(seed=4321, **cfg)).text
    assert a != c


def test_greedy_is_deterministic(engine):
    prompt = "Write one short sentence about the sea."
    cfg = {"temperature": 0.0, "max_tokens": 12}
    a = engine.generate(prompt, GenerationConfig(**cfg)).text
    b = engine.generate(prompt, GenerationConfig(**cfg)).text
    assert a == b
