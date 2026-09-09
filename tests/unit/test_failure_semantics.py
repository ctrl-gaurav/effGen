"""Unit tests for failure semantics & error taxonomy.

Covers:
- classify_provider_error() mapping (effGen typed errors + raw SDK shapes).
- run() never returns success=True with empty output; the no-tool direct path
  and the ReAct tool path return the identical failure shape.
- Retry classification: non-retryable errors (auth/not_found) fail fast; only
  retryable/rate-limited errors are retried.
- AgentConfig.raise_on_error raises a typed error.
- metadata["reason"] taxonomy (final_answer / generation_failed).
- AgentConfig.require_model defaults to True and provider/raise_on_error exist.
- VLLMEngine import-error distinction (not-installed vs ABI/CUDA failure).

These use an in-process fake model (no network) — not a mock of live API
behavior, which the project forbids.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.models.errors import (
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    ModelRefusalError,
    ModelTimeoutError,
    ProviderTransientError,
    classify_provider_error,
    simplify_embedded_provider_error,
)


class _FakeModel(BaseModel):
    """Minimal in-process model: either returns fixed text or raises `exc`."""

    def __init__(
        self, *, text: str = "", exc: Exception | None = None, provider: str = "fake",
        usage_metadata: dict | None = None,
    ):
        super().__init__(model_name="fake-model", model_type=ModelType.OPENAI)
        self._text = text
        self._exc = exc
        self._provider = provider
        self._usage_metadata = usage_metadata or {}
        self.calls = 0

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return GenerationResult(
            text=self._text, tokens_used=3, finish_reason="stop", model_name=self.model_name,
            metadata=dict(self._usage_metadata),
        )

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self._text

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 4096

    def unload(self) -> None:  # pragma: no cover
        pass

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config) for p in prompts]

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):  # pragma: no cover
        return self.generate(prompt, config=config)

    def supports_function_calling(self) -> bool:
        return False

    def supports_tool_calling(self) -> bool:
        return False


def _agent(model: _FakeModel, *, tools=None, **cfg_kwargs) -> Agent:
    # These assert the shape of the failure response, so they opt out of the
    # 1.0.0 default of raising — unless a test is checking the raise itself.
    cfg_kwargs.setdefault("raise_on_error", False)
    return Agent(
        AgentConfig(
            name="t",
            model=model,
            tools=tools or [],
            enable_sub_agents=False,
            enable_memory=False,
            **cfg_kwargs,
        )
    )


# ---------------------------------------------------------------------------
# classify_provider_error
# ---------------------------------------------------------------------------


def test_classify_typed_errors():
    assert classify_provider_error(ModelAuthError("openai")).category == "auth"
    assert classify_provider_error(ModelAuthError("openai")).should_retry is False
    assert classify_provider_error(ModelNotFoundError("cerebras", "x")).category == "not_found"
    assert classify_provider_error(ModelNotFoundError("cerebras")).should_retry is False
    assert classify_provider_error(ModelRefusalError("no")).category == "refusal"
    assert classify_provider_error(InvalidRequestError("groq")).category == "invalid_request"
    assert classify_provider_error(ModelTimeoutError("replicate")).should_retry is True
    assert classify_provider_error(ProviderTransientError("groq", status_code=503)).should_retry is True


def test_classify_by_status_code():
    class _E(Exception):
        def __init__(self, status):
            self.status_code = status

    assert classify_provider_error(_E(401)).category == "auth"
    assert classify_provider_error(_E(404)).category == "not_found"
    assert classify_provider_error(_E(429)).category == "rate_limited"
    assert classify_provider_error(_E(429)).should_retry is True
    assert classify_provider_error(_E(400)).category == "invalid_request"
    # 413 payload-too-large is a property of the request, not a rate limit:
    # retrying the same oversized request will not succeed.
    assert classify_provider_error(_E(413)).category == "invalid_request"
    assert classify_provider_error(_E(413)).should_retry is False
    assert classify_provider_error(_E(503)).should_retry is True


def test_classify_trusts_wrapped_error_context_over_reclassification():
    # provider_runtime_error() wraps an SDK exception in a generic RuntimeError
    # and attaches the already-correct classification as .error_context. A
    # wrapped 400 whose *text* doesn't independently trip any of
    # classify_provider_error's own keyword heuristics must still classify as
    # invalid_request/non-retryable — not silently downgrade to
    # unknown/retryable because the status code was lost in wrapping.
    from effgen.models._adapter_utils import provider_runtime_error

    class _FakeBadRequest(Exception):
        status_code = 400

    raw = _FakeBadRequest(
        "This model's maximum context length is 131072 tokens. However, you "
        "requested 133000 tokens. Please reduce the length of the messages "
        "or completion."
    )
    raw_class = classify_provider_error(raw)
    assert raw_class.category == "invalid_request"
    assert raw_class.should_retry is False

    wrapped = provider_runtime_error("groq", "openai/gpt-oss-20b", "generate", raw)
    wrapped_class = classify_provider_error(wrapped)
    assert wrapped_class.category == raw_class.category
    assert wrapped_class.should_retry == raw_class.should_retry


def test_simplify_embedded_provider_error_extracts_inner_message():
    raw = (
        "request too large for openai/gpt-oss-20b: Error code: 413 - "
        "{'error': {'message': 'Limit 6000, Requested 9294.', "
        "'type': 'tokens', 'code': 'rate_limit_exceeded'}} "
        "— reduce the request (fewer/smaller tools or shorter input) or use a "
        "larger-context model."
    )
    out = simplify_embedded_provider_error(raw)
    assert "Error code" not in out
    assert "{" not in out
    assert out == (
        "request too large for openai/gpt-oss-20b: Limit 6000, Requested 9294. "
        "— reduce the request (fewer/smaller tools or shorter input) or use a "
        "larger-context model."
    )


def test_simplify_embedded_provider_error_noop_without_blob():
    assert simplify_embedded_provider_error("plain message, nothing embedded") == (
        "plain message, nothing embedded"
    )


def test_simplify_embedded_provider_error_noop_on_unparseable_blob():
    raw = "Error code: 500 - {not: valid, python: literal}"
    assert simplify_embedded_provider_error(raw) == raw


def test_classify_by_name_and_message():
    assert classify_provider_error(type("RateLimitError", (Exception,), {})()).category == "rate_limited"
    assert classify_provider_error(type("AuthenticationError", (Exception,), {})()).category == "auth"
    assert classify_provider_error(Exception("Invalid API key provided")).category == "auth"
    assert classify_provider_error(Exception("The model does not exist")).category == "not_found"
    # Unknown errors default to retryable so a transient blip is not hard-failed.
    assert classify_provider_error(Exception("something weird")).should_retry is True


def test_classify_huggingface_missing_repo_is_not_found():
    """A missing/inaccessible HF repo id is not_found, so the load is not retried."""
    hf_msg = (
        "totally-not-a-model-xyz is not a local folder and is not a valid model "
        "identifier listed on 'https://huggingface.co/models'\nIf this is a private "
        "repository, make sure to pass a token having permission to this repo either "
        "by logging in with `hf auth login` or by passing `token=<your_token>`"
    )
    cls = classify_provider_error(RuntimeError(hf_msg))
    assert cls.category == "not_found"
    assert cls.not_found is True
    assert cls.should_retry is False
    # Transient/auth/rate-limit wordings keep their own categories.
    assert classify_provider_error(Exception("connection error to host")).category == "transient"
    assert classify_provider_error(Exception("rate limit exceeded")).category == "rate_limited"


def test_classify_quantized_load_that_does_not_fit_is_resource_exhausted():
    """A quantized load with too little VRAM reports an offload instruction, not an OOM."""
    msg = (
        "Some modules are dispatched on the CPU or the disk. Make sure you have "
        "enough GPU RAM to fit the quantized model. If you want to dispatch the "
        "model on the CPU or the disk while keeping these modules in 32-bit, you "
        "need to set `llm_int8_enable_fp32_cpu_offload=True`"
    )
    cls = classify_provider_error(RuntimeError(msg))
    assert cls.category == "resource_exhausted"
    assert cls.should_retry is False
    assert classify_provider_error(RuntimeError("CUDA out of memory")).category == "resource_exhausted"


# ---------------------------------------------------------------------------
# run() failure semantics
# ---------------------------------------------------------------------------


def test_direct_path_never_empty_success():
    a = _agent(_FakeModel(exc=ModelNotFoundError("cerebras", "x")))
    r = a.run("hi")
    assert r.success is False
    assert r.metadata["reason"] == "generation_failed"
    assert isinstance(r.metadata["error"], dict)
    assert r.metadata["error"]["category"] == "not_found"
    assert r.output  # non-empty, human-readable


def test_tool_path_same_shape_as_direct_path():
    from effgen.tools.builtin.calculator import Calculator

    err = ModelNotFoundError("cerebras", "x")
    direct = _agent(_FakeModel(exc=err)).run("hi")
    tooled = _agent(_FakeModel(exc=err), tools=[Calculator()]).run("hi")
    assert direct.success is tooled.success is False
    assert direct.metadata["reason"] == tooled.metadata["reason"] == "generation_failed"
    assert direct.metadata["error"]["category"] == tooled.metadata["error"]["category"]
    assert set(direct.metadata["error"]) >= {"type", "provider", "model", "message", "category"}


def test_no_retry_storm_on_auth():
    m = _FakeModel(exc=ModelAuthError("openai"))
    r = _agent(m).run("hi")
    assert r.success is False
    assert m.calls == 1  # exactly one call — no 3x retry storm


def test_retries_only_retryable(monkeypatch):
    import time

    monkeypatch.setattr(time, "sleep", lambda *_: None)
    m = _FakeModel(exc=ProviderTransientError("groq", status_code=503))
    r = _agent(m).run("hi")
    assert r.success is False
    assert m.calls == 3  # max_retries reached for a retryable error


def test_success_reason_final_answer():
    r = _agent(_FakeModel(text="Canberra")).run("capital of Australia?")
    assert r.success is True
    assert r.output.strip() == "Canberra"
    assert r.metadata["reason"] == "final_answer"


@pytest.mark.parametrize("task", ["", "   ", "\n\t "])
def test_empty_task_rejected_without_model_call(task):
    m = _FakeModel(text="should never be reached")
    r = _agent(m).run(task)
    assert r.success is False
    assert r.metadata["reason"] == "empty_task"
    assert r.metadata["error"]["category"] == "invalid_input"
    assert m.calls == 0  # no model call, no billing


def test_tokens_used_is_total_not_completion_only():
    # GenerationResult.tokens_used (3, per the fake model above) is
    # completion-only; response.tokens_used must report the run's total
    # (prompt + completion), matching its documented meaning and
    # metadata["total_tokens"].
    m = _FakeModel(
        text="Canberra",
        usage_metadata={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    )
    r = _agent(m).run("capital of Australia?")
    assert r.success is True
    assert r.tokens_used == 13
    assert r.tokens_used == r.metadata["total_tokens"]
    assert r.metadata["prompt_tokens"] == 10
    assert r.metadata["completion_tokens"] == 3


def test_run_async_signature_has_output_schema_and_output_model():
    params = inspect.signature(Agent.run_async).parameters
    assert "output_schema" in params
    assert "output_model" in params


def test_run_async_accepts_output_schema_by_keyword():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    m = _FakeModel(text='{"city": "Paris"}')
    agent = _agent(m)

    async def go():
        return await agent.run_async("Which city?", output_schema=schema)

    r = asyncio.run(go())
    assert r.success is True
    assert r.metadata["parsed"] == {"city": "Paris"}


def test_empty_task_with_multimodal_inputs_not_rejected():
    from effgen.core.messages import ImagePart

    m = _FakeModel(text="a red square")
    image = ImagePart(image=b"\x89PNG\r\n", mime="image/png")
    r = _agent(m).run("", inputs=[image])
    assert m.calls == 1
    assert r.metadata.get("reason") != "empty_task"


def test_raise_on_error_raises_typed():
    a = _agent(_FakeModel(exc=ModelNotFoundError("cerebras", "x")), raise_on_error=True)
    with pytest.raises(ModelNotFoundError):
        a.run("hi")

    a2 = _agent(_FakeModel(exc=ModelAuthError("openai")), raise_on_error=True)
    with pytest.raises(ModelAuthError):
        a2.run("hi")


def test_raise_on_error_message_not_double_wrapped():
    """The reconstructed exception must not stack the '<provider> error
    (model=...):' prefix twice (once from the original exception's str(),
    once from re-wrapping it in a fresh ModelNotFoundError)."""
    cause = "Error code: 404 - model not found. Did you mean: gpt-5-nano?"
    original = ModelNotFoundError("openai", "gpt-does-not-exist-999", cause)
    a = _agent(_FakeModel(exc=original), raise_on_error=True)
    with pytest.raises(ModelNotFoundError) as excinfo:
        a.run("hi")
    text = str(excinfo.value)
    assert text.count("openai error") == 1
    assert cause in text


def test_tool_failure_observation_is_not_double_prefixed():
    """A tool that raises is reported once, not with a stacked prefix.

    ``Tool.execute()`` catches the exception and already writes
    ``"Tool execution failed: <cause>"`` into ``ToolResult.error``; the agent
    must pass that through rather than prefixing it a second time, so the
    observation the model reads names the cause once.
    """
    from effgen.tools import tool

    @tool(name="exploding_tool", description="Reports the status of a structure.")
    def exploding_tool(name: str) -> str:
        """Report the status of the named structure."""
        raise RuntimeError("detonator missing")

    a = _agent(_FakeModel(), tools=[exploding_tool])
    observation = a._execute_tool("exploding_tool", '{"name": "bridge"}')

    assert observation.startswith("Error executing tool 'exploding_tool': ")
    assert observation.count("Tool execution failed:") == 1
    assert observation.endswith("detonator missing")


def test_unknown_tool_observation_names_what_is_available():
    a = _agent(_FakeModel())
    observation = a._execute_tool("no_such_tool", "{}")
    assert observation.startswith("Error executing tool 'no_such_tool': ")
    assert "not available" in observation


def test_error_message_is_redacted():
    leaked = "boom sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd failed"
    r = _agent(_FakeModel(exc=ProviderTransientError("openai", message=leaked))).run("hi")
    # monkeypatch-free: the real redactor should scrub the key shape
    assert "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in r.metadata["error"]["message"]


def test_response_metadata_error_trusts_wrapped_classification():
    # End-to-end: an AgentResponse built from a provider_runtime_error()-wrapped
    # exception must report the same category/retryable the original SDK error
    # classified as, not a re-derived "unknown"/retryable=True.
    from effgen.models._adapter_utils import provider_runtime_error

    class _FakeBadRequest(Exception):
        status_code = 400

    raw = _FakeBadRequest(
        "This model's maximum context length is 131072 tokens. However, you "
        "requested 133000 tokens. Please reduce the length of the messages "
        "or completion."
    )
    wrapped = provider_runtime_error("groq", "fake-model", "generate", raw)
    r = _agent(_FakeModel(exc=wrapped)).run("hi")
    assert not r.success
    assert r.metadata["error"]["category"] == "invalid_request"
    assert r.metadata["error"]["retryable"] is False


def test_embedded_sdk_error_body_collapsed_in_response_output():
    """An adapter's message that embeds a raw SDK error body
    ('Error code: 413 - {...}') must surface as prose in response.output,
    not as a dumped Python dict."""
    raw_body = (
        "request too large for openai/gpt-oss-20b: Error code: 413 - "
        "{'error': {'message': 'Request too large for model `openai/gpt-oss-20b` "
        "on tokens per minute (TPM): Limit 6000, Requested 9294.', "
        "'type': 'tokens', 'code': 'rate_limit_exceeded'}} "
        "— reduce the request (fewer/smaller tools or shorter input) or use a "
        "larger-context model."
    )
    r = _agent(
        _FakeModel(exc=InvalidRequestError("groq", "openai/gpt-oss-20b", raw_body))
    ).run("hi")
    assert not r.success
    assert "Error code" not in r.output
    assert "{'error'" not in r.output
    assert "Limit 6000, Requested 9294" in r.output
    assert "reduce the request" in r.output
    assert "Error code" not in r.metadata["error"]["message"]


# ---------------------------------------------------------------------------
# AgentConfig contract
# ---------------------------------------------------------------------------


def test_agentconfig_defaults_and_fields():
    fields = {f.name: f for f in dataclasses.fields(AgentConfig)}
    assert "provider" in fields
    assert "raise_on_error" in fields
    assert fields["require_model"].default is True
    assert fields["raise_on_error"].default is True
    assert fields["provider"].default is None


def test_agentconfig_name_defaults_from_model():
    cfg = AgentConfig(model="openai:gpt-5-nano", require_model=False)
    assert cfg.name == "openai:gpt-5-nano"

    m = _FakeModel(text="x")
    cfg2 = AgentConfig(model=m)
    assert cfg2.name == "agent"

    cfg3 = AgentConfig(name="explicit", model=m)
    assert cfg3.name == "explicit"


def test_bare_tool_name_string_gets_typed_actionable_error():
    # tools= expects Tool instances; a bare name string (the idiom used
    # elsewhere — get_tool_sync(), CLI --allowed-tools) is a natural mistake
    # and must not crash with a raw AttributeError deep in construction.
    with pytest.raises(TypeError, match="not names"):
        _agent(_FakeModel(text="x"), tools=["calculator"])


def test_non_tool_non_string_in_tools_also_gets_typed_error():
    with pytest.raises(TypeError, match="expects Tool instances"):
        _agent(_FakeModel(text="x"), tools=[123])


def test_real_tool_instance_in_tools_still_works():
    from effgen.tools.builtin.calculator import Calculator
    a = _agent(_FakeModel(text="x"), tools=[Calculator()])
    assert "calculator" in a.tools


def test_require_model_true_fails_fast_on_bad_string_model():
    with pytest.raises(RuntimeError):
        Agent(
            AgentConfig(
                name="t",
                model="totally-made-up-model-9000",
                enable_sub_agents=False,
                enable_memory=False,
            )
        )


# ---------------------------------------------------------------------------
# Engine import-error distinction
# ---------------------------------------------------------------------------


def test_vllm_not_installed_vs_abi_error(monkeypatch):
    import importlib.util
    import sys

    pytest.importorskip("torch")  # the engine module imports torch at import time

    from effgen.models.vllm_engine import VLLMEngine

    eng = VLLMEngine(model_name="x", tensor_parallel_size=0)

    # Force `from vllm import ...` to raise ImportError even though vllm may be
    # installed in the test env (setting the module to None does this).
    monkeypatch.setitem(sys.modules, "vllm", None)

    # Case 1: package genuinely absent -> "not installed"
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="not installed"):
        eng.load()

    # Case 2: package present but import failed (ABI/CUDA) -> ABI message
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    with pytest.raises(RuntimeError, match="ABI"):
        eng.load()
