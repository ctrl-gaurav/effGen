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
    qwen/qwen3.8-27b, which the live GET /openai/v1/models now lists.
    """
    expected = {
        "anthropic.json": "2716848fc0c9a650ac8643326a4c0a34f1476ec0ea7690ff7a67a134b6e6e4ab",
        "cerebras.json": "d45eca795d28d6e6c8c0e188cb9ae3784b7ee49545d0aafdbe1378009923d381",
        "fireworks.json": "a625220d24b0c5644c5c2de009fb2d54f03d15db1903496342e78ee6c97bd235",
        "gemini.json": "2c19277fc5335481251cc5ad8c5badb1d34a97051f58d9e01cf254d872e407a5",
        "groq.json": "8c4e2f4aab160c2a933adfa8528a0db2eb56271f9c59bbec118ecdce96c302e9",
        "hf.json": "1f44e05801fe4ce203ecae0e46805197ebb70a925fa613bd4dc1e00e32a92093",
        "hf_inference_catalog.json": "dc47c4080832e649ce1d37aeb77e17f651e20d298f86e55ba37c84355d3c3a1e",
        "openai.json": "1977347b9a77ebad774e59468fd9c346b48f249f7e9f522c80c05f146a607af0",
        "replicate.json": "4bad0886ef460385ee33a68079dc21044de62d2432e135823a3a39264a947d26",
        "together.json": "91682d878ce9ffbda1322247bbcd4488e13978bd12f09969834aaf5dfcc23d71",
    }
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(DATA_DIR.glob("*.json"))
    }
    assert actual == expected
