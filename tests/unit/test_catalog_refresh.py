"""Offline tests for the provider catalog refresh / drift / routing contract.

These exercise the catalog-refresh additions on top of the catalog spine without any
network I/O: the live-refresh fetcher registry and record construction, the
once-per-process drift fallback, registry-driven bare-id routing, and the
"did you mean / available now" suggestion helper.
"""

from __future__ import annotations

import datetime

import pytest

from effgen.models import _catalog as C
from effgen.models import _refresh as R

# ---------------------------------------------------------------------------
# Cerebras catalog correctness: dead defaults removed, live default set
# ---------------------------------------------------------------------------


def test_cerebras_catalog_only_live_models():
    from effgen.models import cerebras_models as cm

    assert set(cm.CEREBRAS_MODELS) == {"gpt-oss-120b", "zai-glm-4.7"}
    assert cm.CEREBRAS_DEFAULT_MODEL in cm.CEREBRAS_MODELS
    # The expired ids must be gone (they 404 live).
    assert "llama3.1-8b" not in cm.CEREBRAS_MODELS
    assert "qwen-3-235b-a22b-instruct-2507" not in cm.CEREBRAS_MODELS


def test_committed_snapshots_match_catalog():
    # Invariant preserved: every bundled snapshot still mirrors the
    # in-package catalog the adapters use (no silent drift in the repo).
    for prov in C.known_providers():
        diff = C.check_drift_against_snapshot(prov)
        assert not diff["added"], f"{prov}: snapshot missing {diff['added']}"
        assert not diff["removed"], f"{prov}: snapshot has stale {diff['removed']}"
        assert not diff["changed"], f"{prov}: snapshot changed {diff['changed']}"


def test_cerebras_snapshot_count_is_two():
    snap = C.load_snapshot("cerebras")
    assert snap is not None
    assert snap["count"] == 2


# ---------------------------------------------------------------------------
# providers_for — registry-driven routing input
# ---------------------------------------------------------------------------


def test_providers_for_bare_cloud_slug():
    assert C.providers_for("gpt-oss-120b") == ["cerebras"]
    assert C.providers_for("zai-glm-4.7") == ["cerebras"]
    assert C.providers_for("qwen/qwen3.6-27b") == ["groq"]


def test_a_slug_several_providers_serve_names_them_all():
    """An id more than one provider hosts is not routable on its own.

    Callers naming such a model have to say which provider they mean, so the
    routing input has to report every host rather than picking one.
    """
    assert C.providers_for("openai/gpt-oss-20b") == [
        "groq", "together", "replicate", "hf",
    ]


def test_providers_for_unknown_is_empty():
    assert C.providers_for("totally-made-up-model-xyz") == []


def test_providers_for_honors_prefix():
    assert C.providers_for("cerebras:gpt-oss-120b") == ["cerebras"]
    assert C.providers_for("cerebras:not-a-model") == []


def test_routing_skips_org_slash_ids():
    # An org/model HuggingFace-style id must not be treated as a cloud route by
    # the loader even if some cloud catalog also lists it (local stays local).
    from effgen.models.model_loader import ModelLoader

    mt = ModelLoader()._detect_model_type("Qwen/Qwen2.5-1.5B-Instruct")
    assert mt.value in {"transformers", "vllm"}


# ---------------------------------------------------------------------------
# suggest_for_missing
# ---------------------------------------------------------------------------


def test_suggest_for_missing_points_at_live_models():
    C.reset_stale_warnings()
    hint = C.suggest_for_missing("cerebras", "llama3.1-8b", warn=False)
    assert "gpt-oss-120b" in hint
    assert "Available cerebras models" in hint


def test_suggest_for_missing_unknown_provider_safe():
    assert C.suggest_for_missing("nosuch", "x", warn=False) == ""


# ---------------------------------------------------------------------------
# _refresh contract
# ---------------------------------------------------------------------------


def test_refreshable_providers_cover_all_known():
    assert set(R.refreshable_providers()) == set(C.known_providers())


def test_records_from_live_merges_curated_and_marks_new():
    today = datetime.date.today().isoformat()
    live = [("gpt-oss-120b", {}), ("brand-new-model", {"context": 4096})]
    recs = R._records_from_live("cerebras", live, verified_on=today)
    by_id = {r.id: r for r in recs}
    # Known id keeps curated capabilities + today's verified date.
    assert by_id["gpt-oss-120b"].supports_tools is True
    assert by_id["gpt-oss-120b"].verified_on == today
    # New id is marked as live-sourced (no fabricated price).
    assert by_id["brand-new-model"].price_source == C.PRICE_SOURCE_LIVE
    assert by_id["brand-new-model"].price_in_per_1m is None
    assert isinstance(by_id["brand-new-model"].family, str)  # guessed (may be "")


def test_refresh_models_rejects_unknown_provider():
    with pytest.raises(KeyError):
        R.refresh_models("nosuch")


def test_check_drift_offline_fallback_no_key():
    # Anthropic has no key in this env → check_drift must degrade to the offline
    # catalog-vs-snapshot comparison instead of raising.
    diff = R.check_drift("anthropic", warn=False)
    assert diff["provider"] == "anthropic"
    assert diff["source"] in {"offline-catalog", "live-api"}
    assert "added" in diff and "removed" in diff and "drifted" in diff


def test_has_credentials_unknown_provider_false():
    assert R.has_credentials("nosuch") is False
