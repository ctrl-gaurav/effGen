"""The failure matrix against the real providers.

The injected matrix in ``tests/unit/test_failure_matrix.py`` produces every
failure class deterministically at the transport layer. This is the other half:
the classes a real endpoint can be made to produce on demand, driven against
the real service with a real credential, so the classification is checked
against what the providers actually send rather than against a reconstruction
of it.

Not every class is summonable politely — a rate limit is the provider's
decision and a connection reset is the network's — so the live table covers a
rejected credential, an absent credential, a request past the context window, a
truncated answer and a deadline the model cannot meet. Each cell asserts what
the injected lane asserts: a typed, classified error with the right retry
verdict, no hang, no credential in what the caller sees, and never a quiet
success.

A provider row is skipped, naming the reason, when its key is absent from the
environment. Anthropic has no key on the machines this runs on. Replicate has
no row here because its failure cells need a prediction the account can afford
to create; the injected lane covers it in full.
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = [pytest.mark.live, pytest.mark.api, pytest.mark.timeout(300)]

#: A syntactically plausible key that no provider will accept.
WRONG_CREDENTIAL = "sk-effgen-live-wrong-9f8e7d6c5b4a39281706f5e4d3c2b1a0"

#: Enough text to pass any current chat model's context window.
OVERSIZED_PROMPT = ("The quick brown fox jumps over the lazy dog. " * 40_000)


def _load_env() -> None:
    """Read the repository credentials at call time, never at import."""
    from pathlib import Path

    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)


#: provider -> (env var, model id, adapter path, shortest deadline the provider
#: accepts). Gemini refuses a deadline under ten seconds outright, so a shorter
#: one measures its request validation rather than its timeout path.
LIVE_PROVIDERS: dict[str, tuple[str, str, str, int]] = {
    "openai": (
        "OPENAI_API_KEY", "gpt-5-nano",
        "effgen.models.openai_adapter.OpenAIAdapter", 1,
    ),
    "groq": (
        "GROQ_API_KEY", "openai/gpt-oss-20b",
        "effgen.models.groq_adapter.GroqAdapter", 1,
    ),
    "gemini": (
        "GOOGLE_API_KEY", "gemini-3.1-flash-lite",
        "effgen.models.gemini_adapter.GeminiAdapter", 10,
    ),
    "together": (
        "TOGETHER_API_KEY", "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "effgen.models.together_adapter.TogetherAdapter", 1,
    ),
    "cerebras": (
        "CEREBRAS_API_KEY", "gpt-oss-120b",
        "effgen.models.cerebras_adapter.CerebrasAdapter", 1,
    ),
    "fireworks": (
        "FIREWORKS_API_KEY", "accounts/fireworks/models/gpt-oss-120b",
        "effgen.models.fireworks_adapter.FireworksAdapter", 1,
    ),
    "hf": (
        "HF_TOKEN", "Qwen/Qwen2.5-7B-Instruct",
        "effgen.models.hf_inference_adapter.HFInferenceAdapter", 1,
    ),
}

PROVIDER_IDS = sorted(LIVE_PROVIDERS)

#: Every other variable an adapter reads a credential from. The absent-credential
#: case has to clear all of them: a second variable still holding a key makes the
#: call succeed, and the case would then be asserting nothing.
ALSO_CARRIES_A_CREDENTIAL: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY",),
    "hf": ("HUGGINGFACE_API_KEY",),
}


def _adapter(provider: str, *, api_key: str | None, **kwargs):
    """Build and load the adapter for *provider*.

    Client-side rate limiting is switched off where the adapter has it: it
    paces calls against a window persisted across runs, and a cell waiting on
    that pacing is measuring the pacing rather than the provider's failure.
    """
    import importlib

    _, model, path, _ = LIVE_PROVIDERS[provider]
    module_name, _, cls_name = path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), cls_name)
    import inspect

    if "enable_rate_limiting" in inspect.signature(cls.__init__).parameters:
        kwargs.setdefault("enable_rate_limiting", False)
    adapter = cls(model_name=model, api_key=api_key, **kwargs)
    adapter.load()
    return adapter


def _require_key(provider: str) -> str:
    _load_env()
    env_var = LIVE_PROVIDERS[provider][0]
    key = os.environ.get(env_var)
    if not key:
        pytest.skip(f"{env_var} is not set")
    return key


def _classify(exc: Exception):
    from effgen.models.errors import classify_provider_error

    return classify_provider_error(exc)


def _skip_if_rate_limited(exc: Exception, provider: str) -> None:
    """Skip when the provider throttled the call instead of answering it.

    A rate limit is the provider's decision and its own row in this matrix; a
    cell that needs a served answer cannot assert anything once the quota is
    spent, so it reports the throttle rather than a failure of the behaviour
    under test.
    """
    if _classify(exc).category == "rate_limited":
        pytest.skip(f"{provider} rate limited this call: {str(exc)[:200]}")


def _no_credential_anywhere(exc: BaseException, secret: str) -> bool:
    """Whether *secret* survives anywhere in the chain the caller can read."""
    seen: set[int] = set()
    node: BaseException | None = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if secret in str(node) or secret in repr(node):
            return False
        for value in vars(node).values():
            if isinstance(value, str | dict | list | tuple) and secret in repr(value):
                return False
        node = node.__cause__ or node.__context__
    return True


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_a_rejected_credential_is_reported_as_auth(provider):
    """A wrong key fails fast and is never quoted back."""
    _require_key(provider)
    started = time.monotonic()
    with pytest.raises(Exception) as caught:  # noqa: B017 - the class is the subject
        adapter = _adapter(provider, api_key=WRONG_CREDENTIAL)
        adapter.generate("Say hello.", max_tokens=16)
    elapsed = time.monotonic() - started
    cls = _classify(caught.value)
    assert cls.category == "auth", f"{provider}: {caught.value}"
    assert cls.retryable is False and cls.should_retry is False
    assert elapsed < 60.0, f"{provider} took {elapsed:.1f}s to reject a bad key"
    assert _no_credential_anywhere(caught.value, WRONG_CREDENTIAL), (
        f"{provider} handed the submitted credential back to the caller"
    )


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_an_absent_credential_names_what_to_set(provider, monkeypatch):
    """With nothing configured, the error says which variable to set."""
    env_var = LIVE_PROVIDERS[provider][0]
    _require_key(provider)
    for name in (env_var, *ALSO_CARRIES_A_CREDENTIAL.get(provider, ())):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(Exception) as caught:  # noqa: B017 - the class is the subject
        adapter = _adapter(provider, api_key=None)
        adapter.generate("Say hello.", max_tokens=16)
    cls = _classify(caught.value)
    assert cls.category == "auth", f"{provider}: {caught.value}"
    assert cls.should_retry is False
    assert env_var in str(caught.value), str(caught.value)


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_an_oversized_request_fails_fast(provider):
    """A prompt past the context window is the request's fault, so no retry."""
    key = _require_key(provider)
    adapter = _adapter(provider, api_key=key)
    started = time.monotonic()
    with pytest.raises(Exception) as caught:  # noqa: B017 - the class is the subject
        adapter.generate(OVERSIZED_PROMPT, max_tokens=16)
    elapsed = time.monotonic() - started
    cls = _classify(caught.value)
    assert cls.category in ("invalid_request", "rate_limited"), (
        f"{provider} classified an oversized prompt as {cls.category}: {caught.value}"
    )
    if cls.category == "invalid_request":
        assert cls.should_retry is False
    assert elapsed < 120.0, f"{provider} took {elapsed:.1f}s"
    assert _no_credential_anywhere(caught.value, key)


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_a_truncated_answer_declares_itself(provider):
    """A cut-off answer comes back flagged, not as a complete one."""
    key = _require_key(provider)
    adapter = _adapter(provider, api_key=key)
    try:
        result = adapter.generate(
            "Write a detailed five paragraph essay about the ocean.", max_tokens=16
        )
    except Exception as exc:  # noqa: BLE001 - a throttle is not a truncation
        _skip_if_rate_limited(exc, provider)
        raise
    assert result.finish_reason == "length", (
        f"{provider} reported {result.finish_reason!r} for a 16-token cap"
    )
    assert (result.metadata or {}).get("truncated") is True


@pytest.mark.parametrize("provider", PROVIDER_IDS)
def test_a_deadline_the_model_cannot_meet_is_a_timeout(provider):
    """The shortest deadline the provider accepts is honoured and reported."""
    key = _require_key(provider)
    deadline = LIVE_PROVIDERS[provider][3]
    adapter = _adapter(provider, api_key=key, timeout=deadline, max_retries=0)
    started = time.monotonic()
    try:
        adapter.generate(
            "Write a detailed twenty paragraph history of the Roman republic.",
            max_tokens=1024,
        )
    except Exception as exc:  # noqa: BLE001 - the classification is the subject
        elapsed = time.monotonic() - started
        cls = _classify(exc)
        assert cls.category in ("timeout", "transient", "rate_limited"), (
            f"{provider} classified a deadline breach as {cls.category}: {exc}"
        )
        assert cls.should_retry is True
        assert elapsed < deadline * 10 + 10, (
            f"{provider} took {elapsed:.1f}s for a {deadline}s deadline"
        )
        assert _no_credential_anywhere(exc, key)
    else:
        pytest.skip(f"{provider} answered inside the {deadline}s deadline")


def test_an_upstream_rate_limit_reaches_the_client_with_a_delay():
    """The server passes on the delay the provider stated.

    Driven against whichever keyed provider answers a 429 first. A run where no
    provider refuses is not a failure — it is the quota holding — so the case
    records that and skips rather than asserting on an event it cannot cause.
    """
    _load_env()
    pytest.importorskip("fastapi")
    from starlette.testclient import TestClient

    from effgen.models.errors import retry_after_seconds
    from effgen.server.app import create_app

    provider = next(
        (p for p in PROVIDER_IDS if os.environ.get(LIVE_PROVIDERS[p][0])), None
    )
    if provider is None:
        pytest.skip("no provider key is set")
    model = LIVE_PROVIDERS[provider][1]

    client = TestClient(
        create_app(api_key="live-matrix-key"), raise_server_exceptions=False
    )
    statuses: list[int] = []
    for _ in range(40):
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer live-matrix-key"},
            json={"model": f"{provider}:{model}", "max_tokens": 8,
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        statuses.append(response.status_code)
        if response.status_code == 429:
            body = response.json()
            assert body["error"]["type"] == "rate_limit_exceeded", body
            stated = retry_after_seconds(RuntimeError(body["error"]["message"]))
            if stated is not None:
                assert "retry-after" in {k.lower() for k in response.headers}, (
                    f"the provider stated {stated}s and the server dropped it"
                )
            return
    pytest.skip(
        f"{provider} served all {len(statuses)} requests without refusing one "
        f"(statuses: {sorted(set(statuses))})"
    )
