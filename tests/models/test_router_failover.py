"""Unit tests for RetryPolicy, RouterEvent, and route_and_execute.

Unit tests:
  - RetryPolicy retries on RateLimitExceeded, ProviderTransientError, ModelTimeoutError.
  - RetryPolicy does NOT retry on ModelAuthError, ModelRefusalError, InvalidRequestError.
  - route_and_execute() performs failover on retriable errors.
  - route_and_execute() raises AllCandidatesExhaustedError at hop limit.
  - RouterEvent is emitted with correct fields on each hop.
  - subscribe() callback is invoked for each failover.
  - Auth errors do not trigger failover.
"""

from __future__ import annotations

import pytest

from effgen.models._rate_limit import RateLimitExceeded
from effgen.models.capabilities import Capability
from effgen.models.errors import (
    AllCandidatesExhaustedError,
    InvalidRequestError,
    ModelAuthError,
    ModelRefusalError,
    ModelTimeoutError,
    ProviderTransientError,
)
from effgen.models.registry import ProviderRegistry
from effgen.models.router import (
    ModelRouter,
    NoCandidateError,
    PolicyBasedRouter,
    ProviderModelPair,
    RouterDecision,
    RouterEvent,
    RoutingContext,
    RoutingPolicy,
)
from effgen.models.routing.cost import CostBasedPolicy
from effgen.models.routing.retry import RetryPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _DummyAdapter:
    pass


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    """Each test starts with a fresh ProviderRegistry."""
    snapshot = ProviderRegistry.snapshot()
    ProviderRegistry.clear()
    yield
    ProviderRegistry.restore(snapshot)


def _reg(
    name: str,
    model_id: str = "model-a",
    *,
    key_env: str = "KEY",
    monkeypatch,
    caps=None,
    pricing=None,
    model_info=None,
):
    monkeypatch.setenv(key_env, "dummy")
    ProviderRegistry.register(
        name, _DummyAdapter, {model_id: model_info or {}},
        env_keys=[key_env],
        capabilities=caps or {Capability.chat},
        pricing=pricing or {"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
    )


def _reg_models(name: str, models: dict[str, dict], *, key_env: str, monkeypatch, caps=None):
    monkeypatch.setenv(key_env, "dummy")
    ProviderRegistry.register(
        name,
        _DummyAdapter,
        models,
        env_keys=[key_env],
        capabilities=caps or {Capability.chat},
        pricing={"input_per_1m": 0.0, "output_per_1m": 0.0, "free_tier": True},
    )


class _StaticPolicy(RoutingPolicy):
    """Always returns the first pair from the given order list."""
    def __init__(self, order: list[ProviderModelPair]):
        self._order = list(order)

    @property
    def name(self) -> str:
        return "static"

    def select(self, candidates, context):
        for pair in self._order:
            if pair in candidates:
                eliminated = [(c, "not chosen") for c in candidates if c != pair]
                return RouterDecision(chosen=pair, eliminated=eliminated, policy_name=self.name, score=0.0)
        raise NoCandidateError("No candidate in static order")


# ---------------------------------------------------------------------------
# RetryPolicy unit tests
# ---------------------------------------------------------------------------

class TestRetryPolicyIsRetriable:
    def test_rate_limit_is_retriable(self):
        rp = RetryPolicy()
        assert rp.is_retriable(RateLimitExceeded("burst"))

    def test_transient_error_is_retriable(self):
        rp = RetryPolicy()
        assert rp.is_retriable(ProviderTransientError("groq", status_code=503))

    def test_timeout_is_retriable(self):
        rp = RetryPolicy()
        assert rp.is_retriable(ModelTimeoutError("groq"))

    def test_auth_error_not_retriable(self):
        rp = RetryPolicy()
        assert not rp.is_retriable(ModelAuthError("groq"))

    def test_refusal_not_retriable(self):
        rp = RetryPolicy()
        assert not rp.is_retriable(ModelRefusalError("nope"))

    def test_invalid_request_not_retriable(self):
        rp = RetryPolicy()
        assert not rp.is_retriable(InvalidRequestError("groq"))

    def test_generic_error_not_retriable(self):
        rp = RetryPolicy()
        assert not rp.is_retriable(ValueError("oops"))


class TestRetryPolicyExecute:
    def test_succeeds_on_first_try(self):
        rp = RetryPolicy(max_retries=2, backoff_base=0.0, jitter=0.0)
        call_count = [0]

        def fn():
            call_count[0] += 1
            return "ok"

        result, state = rp.execute_with_retry(fn)
        assert result == "ok"
        assert state.attempt == 0
        assert call_count[0] == 1

    def test_retries_on_rate_limit_then_succeeds(self):
        rp = RetryPolicy(max_retries=2, backoff_base=0.0, jitter=0.0)
        call_count = [0]

        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RateLimitExceeded("throttled")
            return "recovered"

        result, state = rp.execute_with_retry(fn)
        assert result == "recovered"
        assert call_count[0] == 3
        assert len(state.failures) == 2

    def test_raises_immediately_on_auth_error(self):
        rp = RetryPolicy(max_retries=3, backoff_base=0.0, jitter=0.0)
        call_count = [0]

        def fn():
            call_count[0] += 1
            raise ModelAuthError("groq")

        with pytest.raises(ModelAuthError):
            rp.execute_with_retry(fn)

        assert call_count[0] == 1  # no retries

    def test_raises_after_max_retries_exhausted(self):
        rp = RetryPolicy(max_retries=2, backoff_base=0.0, jitter=0.0)

        def fn():
            raise RateLimitExceeded("always throttled")

        with pytest.raises(RateLimitExceeded):
            rp.execute_with_retry(fn)

    def test_sleep_seconds_increases_with_attempts(self):
        rp_det = RetryPolicy(max_retries=3, backoff_base=1.0, jitter=0.0)
        s0 = rp_det.sleep_seconds(0)
        s1 = rp_det.sleep_seconds(1)
        s2 = rp_det.sleep_seconds(2)
        assert (s0, s1, s2) == (1.0, 2.0, 4.0)

    def test_backoff_cap_is_respected(self):
        rp = RetryPolicy(max_retries=10, backoff_base=1.0, jitter=1.0, backoff_cap=2.0)
        import random
        random.seed(42)
        s = rp.sleep_seconds(10)
        assert s <= 2.0


# ---------------------------------------------------------------------------
# route_and_execute + failover unit tests
# ---------------------------------------------------------------------------

class TestRouteAndExecute:
    def test_succeeds_without_failover(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        policy = _StaticPolicy([pair_a])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)

        result = router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            lambda pair: f"called:{pair.provider}",
        )
        assert result == "called:prov_a"

    def test_failover_on_rate_limit(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        policy = _StaticPolicy([pair_a, pair_b])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)

        def fn(pair):
            if pair == pair_a:
                raise RateLimitExceeded("prov_a throttled")
            return f"ok:{pair.provider}"

        result = router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            fn,
        )
        assert result == "ok:prov_b"

    def test_failover_on_transient_error(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        policy = _StaticPolicy([pair_a, pair_b])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)

        def fn(pair):
            if pair == pair_a:
                raise ProviderTransientError("prov_a", status_code=503)
            return f"ok:{pair.provider}"

        result = router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            fn,
        )
        assert result == "ok:prov_b"

    def test_no_retry_on_auth_error(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        policy = _StaticPolicy([pair_a, pair_b])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)
        calls = []

        def fn(pair):
            calls.append(pair)
            raise ModelAuthError("prov_a", "m1", "bad key")

        with pytest.raises(ModelAuthError):
            router.route_and_execute(
                RoutingContext(required_capabilities={Capability.chat}),
                fn,
            )
        assert len(calls) == 1  # no failover

    def test_no_retry_on_refusal(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        policy = _StaticPolicy([pair_a])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)
        calls = []

        def fn(pair):
            calls.append(pair)
            raise ModelRefusalError("content policy")

        with pytest.raises(ModelRefusalError):
            router.route_and_execute(
                RoutingContext(required_capabilities={Capability.chat}),
                fn,
            )
        assert len(calls) == 1

    def test_all_candidates_exhausted_raises_error(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        policy = _StaticPolicy([pair_a, pair_b])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)

        def fn(pair):
            raise RateLimitExceeded(f"{pair.provider} throttled")

        with pytest.raises(AllCandidatesExhaustedError) as exc_info:
            router.route_and_execute(
                RoutingContext(required_capabilities={Capability.chat}),
                fn,
            )
        err = exc_info.value
        assert len(err.failures) >= 2

    def test_hop_limit_enforced(self, monkeypatch):
        for i in range(5):
            _reg(f"prov_{i}", f"m{i}", key_env=f"KEY_{i}", monkeypatch=monkeypatch)

        pairs = [ProviderModelPair(f"prov_{i}", f"m{i}") for i in range(5)]
        policy = _StaticPolicy(pairs)
        router = PolicyBasedRouter(policies=[policy], failover_hops=1)

        call_count = [0]

        def fn(pair):
            call_count[0] += 1
            raise RateLimitExceeded("always throttled")

        with pytest.raises(AllCandidatesExhaustedError) as exc_info:
            router.route_and_execute(
                RoutingContext(required_capabilities={Capability.chat}),
                fn,
            )
        assert exc_info.value.hop_limit == 1
        assert len(exc_info.value.failures) == 2
        assert call_count[0] == 2  # first attempt plus one failover

    def test_failover_blocks_failed_provider_not_just_model(self, monkeypatch):
        _reg_models(
            "prov_a",
            {"m1": {}, "m1b": {}},
            key_env="KEY_A",
            monkeypatch=monkeypatch,
        )
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a1 = ProviderModelPair("prov_a", "m1")
        pair_a2 = ProviderModelPair("prov_a", "m1b")
        pair_b = ProviderModelPair("prov_b", "m2")
        policy = _StaticPolicy([pair_a1, pair_a2, pair_b])
        router = PolicyBasedRouter(policies=[policy], failover_hops=2)
        calls: list[ProviderModelPair] = []

        def fn(pair):
            calls.append(pair)
            if pair.provider == "prov_a":
                raise RateLimitExceeded("provider throttled")
            return f"ok:{pair.provider}"

        result = router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            fn,
        )

        assert result == "ok:prov_b"
        assert calls == [pair_a1, pair_b]

    def test_model_router_exposes_route_and_execute(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        router = ModelRouter(
            policies=[_StaticPolicy([pair_a, pair_b])],
            failover_hops=1,
            retry_policy=RetryPolicy(max_retries=0),
        )
        events: list[RouterEvent] = []
        router.subscribe(events.append)

        def fn(pair):
            if pair == pair_a:
                raise ProviderTransientError("prov_a", status_code=503)
            return pair.provider

        result = router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            fn,
        )

        assert result == "prov_b"
        assert len(events) == 1
        assert events[0].to_provider == "prov_b"

    def test_cost_policy_skips_non_chat_models(self, monkeypatch):
        _reg_models(
            "prov_a",
            {
                "audio-model": {"modality": "stt", "pricing_per_1m_input": 0.0},
                "chat-model": {"modality": "chat", "pricing_per_1m_input": 1.0},
            },
            key_env="KEY_A",
            monkeypatch=monkeypatch,
        )
        router = PolicyBasedRouter(policies=[CostBasedPolicy()])

        decision = router.route(RoutingContext(required_capabilities={Capability.chat}))

        assert decision.chosen == ProviderModelPair("prov_a", "chat-model")
        assert any(pair.model_id == "audio-model" for pair, _reason in decision.eliminated)


# ---------------------------------------------------------------------------
# RouterEvent + subscribe tests
# ---------------------------------------------------------------------------

class TestRouterEvents:
    def test_event_emitted_on_failover(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        policy = _StaticPolicy([pair_a, pair_b])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)

        events: list[RouterEvent] = []
        router.subscribe(events.append)

        def fn(pair):
            if pair == pair_a:
                raise RateLimitExceeded("prov_a throttled")
            return "ok"

        router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            fn,
        )

        assert len(events) == 1
        ev = events[0]
        assert ev.from_provider == "prov_a"
        assert ev.from_model == "m1"
        assert ev.to_provider == "prov_b"
        assert ev.to_model == "m2"
        assert ev.reason == "rate_limited"
        assert ev.hop == 1
        assert isinstance(ev.exception, RateLimitExceeded)

    def test_multiple_events_on_multiple_hops(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        _reg("prov_c", "m3", key_env="KEY_C", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        pair_c = ProviderModelPair("prov_c", "m3")
        policy = _StaticPolicy([pair_a, pair_b, pair_c])
        router = PolicyBasedRouter(policies=[policy], failover_hops=4)

        events: list[RouterEvent] = []
        router.subscribe(events.append)

        def fn(pair):
            if pair in (pair_a, pair_b):
                raise ProviderTransientError(pair.provider, status_code=500)
            return "ok"

        router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            fn,
        )

        assert len(events) == 2
        assert events[0].hop == 1
        assert events[1].hop == 2

    def test_subscriber_exception_does_not_break_routing(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        policy = _StaticPolicy([pair_a, pair_b])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)

        def bad_subscriber(event):
            raise RuntimeError("subscriber crashed")

        router.subscribe(bad_subscriber)

        def fn(pair):
            if pair == pair_a:
                raise RateLimitExceeded("throttled")
            return "ok"

        # Should not raise despite bad subscriber
        result = router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            fn,
        )
        assert result == "ok"

    def test_event_as_dict(self, monkeypatch):
        _reg("prov_a", "m1", key_env="KEY_A", monkeypatch=monkeypatch)
        _reg("prov_b", "m2", key_env="KEY_B", monkeypatch=monkeypatch)
        pair_a = ProviderModelPair("prov_a", "m1")
        pair_b = ProviderModelPair("prov_b", "m2")
        policy = _StaticPolicy([pair_a, pair_b])
        router = PolicyBasedRouter(policies=[policy], failover_hops=3)

        events = []
        router.subscribe(events.append)

        def fn(pair):
            if pair == pair_a:
                raise RateLimitExceeded("throttled")
            return "ok"

        router.route_and_execute(
            RoutingContext(required_capabilities={Capability.chat}),
            fn,
        )

        d = events[0].as_dict()
        assert d["from"] == "prov_a/m1"
        assert d["to"] == "prov_b/m2"
        assert d["reason"] == "rate_limited"
        assert d["hop"] == 1
        assert d["exception_type"] == "RateLimitExceeded"


# ---------------------------------------------------------------------------
# AllCandidatesExhaustedError tests
# ---------------------------------------------------------------------------

class TestAllCandidatesExhaustedError:
    def test_error_message_contains_failures(self):
        failures = [
            ("groq", "openai/gpt-oss-20b", RateLimitExceeded("x")),
            ("cerebras", "llama3.1-8b", ProviderTransientError("cerebras", status_code=503)),
        ]
        err = AllCandidatesExhaustedError(failures, hop_limit=3)
        msg = str(err)
        assert "groq" in msg
        assert "cerebras" in msg
        assert "2 candidate attempts" in msg

    def test_hop_limit_stored(self):
        err = AllCandidatesExhaustedError([], hop_limit=5)
        assert err.hop_limit == 5

    def test_failures_list_stored(self):
        exc = RateLimitExceeded("x")
        err = AllCandidatesExhaustedError([("groq", "m", exc)], hop_limit=3)
        assert err.failures[0][2] is exc
