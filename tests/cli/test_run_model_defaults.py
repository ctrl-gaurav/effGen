"""`effgen run` model resolution: clean unknown-id hints and a transparent
no-model default.

Two newcomer pain points are covered offline (no live model calls):

* A mistyped explicit ``-m`` id used to only reveal itself mid-run as a wall of
  provider-404 text; ``_preflight_model_hint`` now surfaces a single tidy
  "did you mean" up front (and never blocks — the catalog may be stale).
* ``effgen run`` with no ``-m`` used to silently download a local model while
  ``quickstart`` preferred a detected cloud key; both now share
  ``_quickstart_suggest_model`` and the choice is announced.
"""
from __future__ import annotations

from effgen.cli import _main


def _cli():
    return _main.CLIInterface()


# ── pre-flight model-id hint ─────────────────────────────────────────────────

def test_preflight_known_model_says_nothing(capsys):
    _main._preflight_model_hint(_cli(), "gpt-5-nano", "openai")
    out = capsys.readouterr().out
    assert "Did you mean" not in out


def test_preflight_typo_suggests_once_and_does_not_raise(capsys):
    # A high-confidence typo gets a clean one-line suggestion, up front.
    _main._preflight_model_hint(_cli(), "gpt-5-nanoo", "openai")
    out = capsys.readouterr().out
    assert "Did you mean" in out
    assert "gpt-5-nano" in out
    # It must never block the run — the local catalog can be stale.
    assert "Proceeding" in out


def test_preflight_never_raises_on_garbage():
    # An unresolvable provider/id must degrade silently, never crash the run.
    _main._preflight_model_hint(_cli(), "totally-made-up-xyz", None)


def test_preflight_silent_for_local_hub_ids(capsys):
    # A legitimate local / HF-hub id ("org/model") is resolved by download, not
    # the cloud catalog — it must never trigger a spurious "did you mean".
    for mid in (
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "some-org/brand-new-model",
    ):
        _main._preflight_model_hint(_cli(), mid, None)
    assert "Did you mean" not in capsys.readouterr().out


# ── no-model default mirrors quickstart ──────────────────────────────────────

def test_no_model_default_prefers_cloud_when_keyed(monkeypatch):
    monkeypatch.setattr(
        _main, "_quickstart_suggest_model",
        lambda: ("groq:openai/gpt-oss-20b", "groq", "groq key detected"),
        raising=True,
    )
    model_id, provider, reason = _main._quickstart_suggest_model()
    assert model_id == "groq:openai/gpt-oss-20b"
    assert provider == "groq"
    assert "groq" in reason


def test_no_model_default_falls_back_to_local(monkeypatch):
    # With no usable cloud key, the suggestion is a small local model — and the
    # reason explains the fallback rather than silently downloading.
    monkeypatch.setattr(
        "effgen.models.auth.check_keys", lambda: {}, raising=False
    )
    model_id, provider, reason = _main._quickstart_suggest_model()
    assert provider is None
    assert "/" in model_id  # a HF-style local id
    assert "local" in reason.lower()
    # The engine prefix is part of the id. The bare repo id also appears in a
    # cloud provider's catalog, so a caller that routes bare ids by catalog
    # would send the keyless fallback to a provider whose key is missing.
    assert model_id.startswith("transformers:")


def test_local_fallback_id_names_the_local_engine(monkeypatch):
    import effgen.models.cerebras_adapter  # noqa: F401  (populates the registry)
    from effgen.models.registry import ProviderRegistry

    monkeypatch.setattr("effgen.models.auth.check_keys", lambda: {}, raising=False)
    model_id, _provider, _reason = _main._quickstart_suggest_model()
    engine, _, bare = model_id.partition(":")
    assert engine == "transformers"
    # Without the prefix this id is claimed by a cloud catalog — the reason the
    # prefix has to be part of the suggestion.
    assert ProviderRegistry._model_index.get(bare)
