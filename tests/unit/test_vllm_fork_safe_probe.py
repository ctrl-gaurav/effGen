"""Loading a model on the vLLM engine does not start CUDA in this process first.

vLLM runs its engine core in a forked process. A parent that has already
started the CUDA driver — ``torch.cuda.is_available()`` does, and
``torch.cuda.mem_get_info()`` creates a whole context — leaves that child
unable to reach the GPU: the load fails with "Cannot re-initialize CUDA in
forked subprocess". So the loader asks whether a device is visible, and how much
memory is free, the way that starts nothing: the device list and the memory
reading come from NVML.

These run without vLLM and without a GPU: a stand-in ``vllm`` module records
what the engine was built with, and the CUDA calls that would start the driver
raise if anything reaches them.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


class _Tokenizer:
    def encode(self, text):
        return list(text)


class _LLM:
    built: list[dict] = []

    def __init__(self, **kwargs) -> None:
        type(self).built.append(kwargs)
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096))

    def get_tokenizer(self):
        return _Tokenizer()


@pytest.fixture()
def no_cuda_start(monkeypatch):
    """A visible device, NVML answers, and any CUDA-starting call fails the test."""
    started: list[str] = []

    def refuse(name):
        def call(*args, **kwargs):
            started.append(name)
            raise AssertionError(f"torch.cuda.{name}() starts CUDA before vLLM forks")
        return call

    monkeypatch.setattr(torch.cuda, "is_available", refuse("is_available"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", refuse("mem_get_info"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", refuse("get_device_properties"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "_get_pynvml_handler", lambda index: ("handle", index),
                        raising=False)
    fake_nvml = types.ModuleType("pynvml")
    fake_nvml.nvmlDeviceGetMemoryInfo = lambda handle: SimpleNamespace(free=40 * 1024**3)
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _LLM
    fake_vllm.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    _LLM.built = []
    return started


def test_free_memory_is_read_from_nvml(no_cuda_start) -> None:
    from effgen.models._vram import free_vram_gb

    assert free_vram_gb() == pytest.approx(40.0)
    assert no_cuda_start == []


def test_the_default_vllm_load_starts_no_cuda_in_this_process(no_cuda_start) -> None:
    from effgen.models import load_model

    model = load_model("org/some-7b-instruct", engine="vllm", apply_chat_template=False)
    assert no_cuda_start == []
    assert len(_LLM.built) == 1
    built = _LLM.built[0]
    assert built["model"] == "org/some-7b-instruct"
    assert built["tensor_parallel_size"] == 1
    # 40 GB free is enough for a 7B, so no quantization is chosen for it.
    assert built.get("quantization") is None
    assert model.count_tokens("four").count == 4


def test_a_process_with_no_visible_device_is_told_so(no_cuda_start, monkeypatch) -> None:
    """Over-correction guard: no device still refuses, with the same message."""
    from effgen.models import load_model

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(Exception, match="(?i)cuda|gpu"):
        load_model("org/some-7b-instruct", engine="vllm", apply_chat_template=False)
    assert _LLM.built == []


def test_transformers_bit_width_follows_where_torch_can_run(monkeypatch) -> None:
    """Over-correction guard: the device list is not the question for Transformers.

    A host whose driver lists a GPU that torch cannot start (a driver older than
    torch's CUDA build) runs a Transformers model on CPU, so no bit width is
    chosen from the GPU's free memory.
    """
    from effgen.models.model_loader import ModelLoader

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ModelLoader, "_free_vram_gb", lambda self: 10.0)
    assert ModelLoader()._auto_select_quantization_bits() is None
