"""Per-model coding-suitability notes, and the surfaces that carry them.

A coding turn needs the model to answer with a tool call. Where that has been
measured, the measurement decides; past the table, the loaded model's own report
of its tool calling decides, then the catalog, then the model's size. Nothing
here blocks a run: the answer is a note.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from types import SimpleNamespace

import pytest

from effgen.models import _catalog
from effgen.models._coding import (
    LIMITED,
    SUITABLE,
    UNKNOWN,
    UNSUITABLE,
    VERIFIED_ON,
    CodingSuitability,
    coding_suitability,
    measured_ids,
    parameter_count_b,
    split_model_id,
)

DATA_DIR = pathlib.Path(_catalog.__file__).parent / "_data"


# --------------------------------------------------------------------------
# The matrix the coding surfaces warn from
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_id,provider,verdict",
    [
        ("transformers:Qwen/Qwen2.5-1.5B-Instruct", None, LIMITED),
        ("transformers:google/gemma-2-2b-it", None, UNSUITABLE),
        ("transformers:microsoft/Phi-3.5-mini-instruct", None, UNSUITABLE),
        ("transformers:Qwen/Qwen2.5-7B-Instruct", None, SUITABLE),
        ("transformers:meta-llama/Llama-3.2-3B-Instruct", None, SUITABLE),
        ("gpt-5-nano", "openai", SUITABLE),
        ("openai:gpt-5-nano", None, SUITABLE),
        ("gemini-3.1-flash-lite", "gemini", SUITABLE),
        ("a-model-in-no-catalog-9000", None, UNKNOWN),
    ],
)
def test_the_verdict_matrix(model_id, provider, verdict):
    assert coding_suitability(model_id, provider).verdict == verdict


def test_a_suitable_model_says_nothing():
    assert coding_suitability("gpt-5-nano", "openai").note() == ""


@pytest.mark.parametrize(
    "model_id,provider,fragment",
    [
        ("transformers:Qwen/Qwen2.5-1.5B-Instruct", None, "nothing written"),
        ("transformers:google/gemma-2-2b-it", None, "no tool definitions"),
        ("a-model-in-no-catalog-9000", None, "does not know this id"),
    ],
)
def test_the_note_names_what_happens_and_what_to_do(model_id, provider, fragment):
    note = coding_suitability(model_id, provider).note()
    assert fragment in note
    assert model_id in note


def test_a_measured_note_carries_its_date():
    suitability = coding_suitability("transformers:Qwen/Qwen2.5-1.5B-Instruct")
    assert suitability.evidence == "measured"
    assert suitability.measured_on
    assert f"(measured {suitability.measured_on})" in suitability.note()


# --------------------------------------------------------------------------
# The rules past the measured table
# --------------------------------------------------------------------------

def test_a_model_that_receives_no_tool_definitions_is_unsuitable():
    model = SimpleNamespace(tool_call_support=lambda: "none")
    suitability = coding_suitability("transformers:some/unmeasured-13b", model=model)
    assert suitability.verdict == UNSUITABLE
    assert suitability.evidence == "capability"
    assert "no tool definitions" in suitability.reason


def test_a_catalog_record_without_tools_is_unsuitable(monkeypatch):
    record = SimpleNamespace(supports_tools=False)
    monkeypatch.setattr("effgen.models._catalog.lookup", lambda *a, **k: record)
    suitability = coding_suitability("some-served-model", "openai")
    assert suitability.verdict == UNSUITABLE
    assert suitability.evidence == "catalog"


def test_a_small_local_model_is_limited_by_its_size():
    suitability = coding_suitability("transformers:acme/tiny-3B-instruct")
    assert suitability.verdict == LIMITED
    assert suitability.evidence == "size"
    assert "3B" in suitability.reason


def test_a_large_unmeasured_local_model_is_reported_as_unknown():
    suitability = coding_suitability("transformers:acme/big-70B-instruct")
    assert suitability.verdict == UNKNOWN
    assert "does not know this id" in suitability.reason


def test_a_measurement_wins_over_every_other_rule():
    # A probe that disagrees with the measurement does not overturn it: the
    # measurement is a run on this framework, the probe is a capability report.
    for support in ("none", "template"):
        model = SimpleNamespace(tool_call_support=lambda s=support: s)
        assert coding_suitability(
            "transformers:Qwen/Qwen2.5-7B-Instruct", model=model
        ).verdict == SUITABLE
    # The size rule is likewise not consulted for a measured id.
    assert coding_suitability(
        "transformers:meta-llama/Llama-3.2-3B-Instruct"
    ).verdict == SUITABLE


def test_a_broken_probe_never_raises():
    class _Boom:
        def tool_call_support(self):
            raise RuntimeError("the model is not loaded")

    suitability = coding_suitability("whatever", model=_Boom())
    assert isinstance(suitability, CodingSuitability)
    assert suitability.verdict == UNKNOWN


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
        ("meta-llama/Llama-3.2-3B-Instruct", 3.0),
        ("openai/gpt-oss-120b", 120.0),
        ("gpt-5-nano", None),
        ("gemini-3.1-flash-lite", None),
    ],
)
def test_parameter_count_is_read_not_guessed(model_id, expected):
    assert parameter_count_b(model_id) == expected


@pytest.mark.parametrize(
    "model_id,provider,expected",
    [
        ("transformers:Qwen/Qwen2.5-7B-Instruct", None, ("Qwen/Qwen2.5-7B-Instruct", "transformers")),
        ("openai:gpt-5-nano", None, ("gpt-5-nano", "openai")),
        ("gpt-5-nano", "openai", ("gpt-5-nano", "openai")),
        ("gpt-5-nano", None, ("gpt-5-nano", None)),
    ],
)
def test_a_prefix_is_read_as_the_provider(model_id, provider, expected):
    assert split_model_id(model_id, provider) == expected


# --------------------------------------------------------------------------
# Drift discipline
# --------------------------------------------------------------------------

def test_the_table_is_dated():
    assert len(VERIFIED_ON) == 10 and VERIFIED_ON.count("-") == 2
    for model_id in measured_ids():
        assert coding_suitability(model_id).measured_on, f"{model_id} carries no date"


def test_a_cloud_entry_the_catalog_dropped_is_reported():
    """A measured cloud id must still exist, or the note points at nothing.

    Local checkpoints are resolved by download rather than by the catalog, so
    only the provider-pinned entries are checked here.
    """
    missing = []
    for key in measured_ids():
        if ":" not in key:
            continue
        provider, bare = key.split(":", 1)
        if _catalog.lookup(bare, provider) is None:
            missing.append(key)
    assert not missing, (
        f"the coding table names ids the catalog no longer carries: {missing}"
    )


# --------------------------------------------------------------------------
# The catalog surface
# --------------------------------------------------------------------------

def test_the_record_property_does_not_change_the_serialized_record():
    record = _catalog.lookup("gpt-5-nano", "openai")
    assert record.coding.verdict == SUITABLE
    assert "coding" not in dataclasses.asdict(record)
    assert "coding" not in record.to_dict()
    assert _catalog.ModelRecord.from_dict(record.to_dict()) == record


def test_the_bundled_snapshots_are_byte_identical():
    """The property is derived, so no snapshot may have been rewritten.

    Hashes recorded from the tree before the property existed. A hash moves
    only when a snapshot is deliberately re-cut against the provider's live
    listing — say so here and in the same commit, or the anchor stops meaning
    anything. gemini.json was re-cut on 2026-08-13, dropping three ids Google
    had retired (gemini-2.0-flash, gemini-2.0-flash-lite, gemini-3-pro-preview),
    each confirmed absent from a live GET /v1beta/models. groq.json was re-cut on
    2026-09-08, dropping two ids Groq had retired (llama-3.1-8b-instant,
    llama-3.3-70b-versatile — both answer 404 model_not_found live) and adding
    qwen/qwen3.8-27b, which the live GET /openai/v1/models now lists. Every
    snapshot was rewritten on 2026-09-15 to carry the two cache-price fields a
    record now has (``price_cached_in_per_1m``,
    ``price_cache_write_in_per_1m``): no id, price, capability or date moved,
    only the two keys were inserted after ``price_out_per_1m``.
    """
    expected = {
        "anthropic.json": "c5bdd2ca9f00fdb645dc37d602666badbb731e18707f3ffb986cb8c8b18e23b0",
        "cerebras.json": "6305702115f9d2ea22b45545a92ac1caa8a7d89eb2515eb168606d3537fd688f",
        "fireworks.json": "15f5a2b44b351a66f8a963cd2fd203a219fee12926e3d47a308b12782abd9476",
        "gemini.json": "2d3e864dbb65f3c02dc8d4c091364d0211152405e2538984d75b8f3e4cd82716",
        "groq.json": "7d9d93be586afbecc2153ed00440ab63da39a5399c7ca214d941c0934161e0f0",
        "hf.json": "0024e3525da40f7978037aebc66905c6c83b887a08f5b0d54d1929e9dbd3ce76",
        "hf_inference_catalog.json": "dc47c4080832e649ce1d37aeb77e17f651e20d298f86e55ba37c84355d3c3a1e",
        "openai.json": "1bbd3b24c1586373a3e1ec4463bc4c10f8fbeb523c3d2fdd19c806e3f04e181a",
        "replicate.json": "745b4d0f3e795ed2e6181a72c437f213f88076d11925d7cfb70326e3319e66a2",
        "together.json": "afea97b589a6de3e2b7adc3077324f8bef4c2debeb454cf482fd49b87eaf7515",
    }
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(DATA_DIR.glob("*.json"))
    }
    assert actual == expected
