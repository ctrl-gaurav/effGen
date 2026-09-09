"""A provider that would not serve the call is a skip; anything else fails.

The rule and the reasoning live in ``tests/_harness/provider_unavailable.py``;
the hook that applies it is in ``tests/conftest.py``. A hook that turns failures
into skips hides real breakage the moment it is too broad, so the negative cases
below matter more than the positive ones — particularly the connection-error
case, which is the shape a local misconfiguration takes.
"""
from __future__ import annotations

import pytest

from tests._harness.provider_unavailable import (
    first_provider_sentence,
    provider_unavailable_reason,
)


@pytest.mark.parametrize(
    "text",
    [
        # A spent free-tier daily quota.
        (
            "RuntimeError: Gemini generation failed [rate_limited]: 429 RESOURCE_EXHAUSTED. "
            "Quota exceeded for metric: generate_content_free_tier_requests, limit: 500"
        ),
        # A full inference queue.
        (
            "AssertionError: Live eval failed: model error: Cerebras rate limit hit for "
            "gpt-oss-120b: Error code: 429 - {'code': 'queue_exceeded'}"
        ),
        # A per-minute token allowance, which Groq reports as 413.
        (
            "groq: Error code: 413 - Request too large ... on tokens per minute (TPM): "
            "Limit 6000, Requested 8212"
        ),
        # An account with no credit.
        "replicate: Error code: 402 - insufficient credit for this prediction",
        # The provider's own infrastructure.
        "openai: Error code: 503 - Service Unavailable",
        "anthropic: overloaded_error: the model is currently overloaded",
    ],
)
def test_a_provider_that_cannot_serve_is_recognised(text):
    assert provider_unavailable_reason(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        # THE case this rule must never absorb. A blank OPENAI_BASE_URL sent 45
        # live cells here on 14 Aug 2026; the wording blames the provider and
        # the cause was local. Converting it to a skip hides the defect.
        (
            "RuntimeError: OpenAI generation failed [will_retry]: Connection error.. "
            "Transient provider error — retry shortly; check the provider status page."
        ),
        "httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol.",
        # A real assertion about the answer.
        "AssertionError: expected 'red' in response; got: 'blue'",
        # effGen built a bad request: a defect, and a 400 is not capacity.
        "openai: Error code: 400 - unknown parameter 'reasoning_effort'",
        # A rejected credential is a configuration problem, not exhaustion.
        "openai: Error code: 401 - Incorrect API key provided",
        # No provider is named, so this is not about a provider call at all.
        "AssertionError: rate limit bucket should refill after 1s",
    ],
)
def test_everything_else_still_fails(text):
    assert provider_unavailable_reason(text) is None


def test_the_reason_quotes_the_provider_s_own_sentence():
    text = (
        "tests/x.py:12: in test_thing\n"
        "E   RuntimeError: Gemini generation failed [rate_limited]: Error code: 429 - quota\n"
    )
    assert "429" in first_provider_sentence(text)


def test_an_empty_failure_is_not_a_provider_problem():
    assert provider_unavailable_reason("") is None
    assert first_provider_sentence("") == ""



# ---------------------------------------------------------------------------
# One refusal per backend, in the words that backend actually uses.
#
# Three consecutive pre-release runs went red on three different providers —
# Groq's tokens-per-day 429 swallowed into a workflow result, Gemini's
# ``503 UNAVAILABLE``, Fireworks' "service overloaded" inside an
# ``invalid_request_error`` envelope — because each was spelled in a way the
# rule had not been taught. The table below is the answer to "why again?": a
# backend is covered here, or it is not covered at all.
# ---------------------------------------------------------------------------

REFUSALS = {
    "groq-tokens-per-day": (
        "RuntimeError: Groq rate limit hit for openai/gpt-oss-20b: Error code: 429 - "
        "{'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in "
        "organization `org_x` service tier `on_demand` on tokens per day (TPD)'}}"
    ),
    "groq-tokens-per-minute-as-413": (
        "groq: Error code: 413 - Request too large ... on tokens per minute (TPM): "
        "Limit 6000, Requested 8212"
    ),
    "groq-inside-a-workflow-result": (
        "AssertionError: assert False is True\n"
        " +  where False = WorkflowResult(success=False, node_results=[{'id': 'src', "
        "'status': 'failed', 'error': \"RateLimitExceeded: Groq rate limit hit for "
        "openai/gpt-oss-20b: Error code: 429\"}]).success"
    ),
    "gemini-quota": (
        "RuntimeError: Gemini generation failed [rate_limited]: 429 RESOURCE_EXHAUSTED. "
        "Quota exceeded for metric: generate_content_free_tier_requests, limit: 500"
    ),
    "gemini-outage": (
        "google.genai.errors.ServerError: 503 UNAVAILABLE. {'error': {'code': 503, "
        "'message': 'The service is currently unavailable.', 'status': 'UNAVAILABLE'}}"
    ),
    "fireworks-overloaded": (
        "fireworks.client.error.ServiceUnavailableError: {\"error\": {\"object\": "
        "\"error\", \"type\": \"invalid_request_error\", \"message\": \"service "
        "overloaded, please try again later\"}}"
    ),
    "cerebras-queue": (
        "AssertionError: Live eval failed: model error: Cerebras rate limit hit for "
        "gpt-oss-120b: Error code: 429 - {'code': 'queue_exceeded'}"
    ),
    "openai-outage": "openai: Error code: 503 - Service Unavailable",
    "openai-quota": "openai.RateLimitError: Error code: 429 - insufficient_quota",
    "anthropic-overloaded": "anthropic: overloaded_error: the model is currently overloaded",
    "anthropic-500": "anthropic.InternalServerError: Error code: 500 - internal error",
    "together-status-kwarg": (
        "together.error.ServiceUnavailableError: status_code=503, "
        "message='Server is overloaded'"
    ),
    "replicate-no-credit": "replicate: Error code: 402 - insufficient credit for this prediction",
    "huggingface-cold-model": (
        "huggingface_hub.errors.HfHubHTTPError: 503 Server Error: Model "
        "meta-llama/Llama-3.1-8B is currently loading"
    ),
    "mistral-http-spelling": "mistralai.models.sdkerror.SDKError: API error occurred: HTTP 429",
    "cohere-too-many": "cohere.errors.TooManyRequestsError: status_code: 429, body: trial limit",
}

NOT_REFUSALS = {
    # THE case this rule must never absorb: the wording blames the provider and
    # the cause is a blank base URL, a proxy or a firewall.
    "connection-error": (
        "RuntimeError: OpenAI generation failed [will_retry]: Connection error.. "
        "Transient provider error — retry shortly; check the provider status page."
    ),
    "will-retry-alone": (
        "RuntimeError: Fireworks generation failed [will_retry]: Connection error.. "
        "Transient provider error — retry shortly."
    ),
    "bad-url": (
        "httpx.UnsupportedProtocol: Request URL is missing an 'http://' or "
        "'https://' protocol."
    ),
    "effgen-built-a-bad-request": "openai: Error code: 400 - unknown parameter 'reasoning_effort'",
    "rejected-credential": "openai: Error code: 401 - Incorrect API key provided",
    "forbidden": "anthropic: Error code: 403 - your account is not permitted this model",
    "model-does-not-exist": (
        "WorkflowResult(success=False, node_results=[{'id': 'bad', 'status': 'failed', "
        "'error': 'AgentError: groq model totally-not-a-real-model-xyz does not exist'}])"
    ),
    "a-real-assertion": "AssertionError: expected 'red' in response; got: 'blue'",
    "no-provider-named": "AssertionError: rate limit bucket should refill after 1s",
    # An offline test asserting effGen's own 503 names no provider on that line,
    # which is what the locality rule protects.
    "our-own-server-status": (
        "tests/server/test_routes.py:88: in test_unavailable\n"
        "    assert response.status_code == 503\n"
        "E   assert 200 == 503\n"
        "  the openai-compatible router is mounted at /v1\n"
    ),
}


@pytest.mark.parametrize("text", REFUSALS.values(), ids=list(REFUSALS))
def test_every_backend_s_refusal_is_recognised(text):
    assert provider_unavailable_reason(text) is not None


@pytest.mark.parametrize("text", NOT_REFUSALS.values(), ids=list(NOT_REFUSALS))
def test_nothing_else_is_absorbed(text):
    assert provider_unavailable_reason(text) is None


def test_the_reason_quotes_what_the_provider_said():
    for name, text in REFUSALS.items():
        sentence = first_provider_sentence(text)
        assert sentence, f"{name} produced no sentence for the skip reason"


def test_a_result_carrying_a_refusal_skips_rather_than_asserting():
    """A framework that reports errors in its return value hides them from the
    report hook, so a live test has to check the result it is about to assert on."""
    from tests._harness.provider_unavailable import skip_if_provider_refused

    with pytest.raises(pytest.skip.Exception) as excinfo:
        skip_if_provider_refused(
            REFUSALS["groq-inside-a-workflow-result"], what="the orchestrated call"
        )
    assert "the orchestrated call" in str(excinfo.value)


def test_a_result_with_an_ordinary_failure_is_left_alone():
    from tests._harness.provider_unavailable import skip_if_provider_refused

    skip_if_provider_refused(NOT_REFUSALS["model-does-not-exist"])
    skip_if_provider_refused(NOT_REFUSALS["a-real-assertion"])
