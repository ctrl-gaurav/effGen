"""Offline tests for chat-model filtering of the catalog refresh path.

A provider's list-models endpoint returns its entire surface — embeddings,
speech, image/video generators, moderation classifiers, rerankers and, for
OpenAI, the caller's own private ``ft:`` fine-tunes.  Reconciling that raw
listing against the curated chat catalog must filter down to chat /
text-generation models so a refresh never balloons the snapshot, never reports
non-chat models as drift, and — critically — never writes a private fine-tune
id into a shipped catalog file.

These exercise the shared classifier, the live-listing filter and the
persistence guard without any network I/O, plus the data invariants left after
pruning the confirmed-retired ids from the bundled catalogs.
"""

from __future__ import annotations

from effgen.models import _catalog as C
from effgen.models import _refresh as R

# ---------------------------------------------------------------------------
# The shared classifier
# ---------------------------------------------------------------------------

# Realistic ids drawn from live provider listings that are NOT chat models.
_NON_CHAT_SAMPLES = [
    # private fine-tunes (OpenAI) — also caught by is_finetune
    "ft:gpt-3.5-turbo-0125:some-org::abc123",
    "ft:gpt-4o-mini-2024-07-18:acme:proj:XyZ",
    # embeddings
    "text-embedding-3-large",
    "text-embedding-ada-002",
    "BAAI/bge-base-en-v1.5",
    "intfloat/multilingual-e5-large-instruct",
    "gemini-embedding-001",
    # speech / audio / tts / asr
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-tts",
    "gpt-audio",
    "gpt-realtime",
    "canopylabs/orpheus-3b-0.1-ft",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-flash-native-audio-latest",
    # image generation
    "dall-e-3",
    "gpt-image-1",
    "black-forest-labs/FLUX.1-schnell",
    "ByteDance-Seed/Seedream-4.0",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "imagen-4.0-generate-001",
    "gemini-3-pro-image",
    "nano-banana-pro-preview",
    # video generation
    "veo-3.0-generate-001",
    "sora-2",
    "google/veo-3.0",
    "Wan-AI/Wan2.2-T2V-A14B",
    "minimax/video-01-director",
    # moderation / safety classifiers
    "omni-moderation-latest",
    "meta-llama/Llama-Guard-4-12B",
    # rerankers
    "Salesforce/Llama-Rank-V1",
    "mixedbread-ai/mxbai-rerank-large-v2",
    # legacy completion-only / specialized
    "babbage-002",
    "davinci-002",
    "gpt-3.5-turbo-instruct",
    "computer-use-preview",
    "o3-deep-research",
    "aqa",
]

# Genuine chat / text-generation models that must survive filtering.
_CHAT_SAMPLES = [
    "gpt-5.5",
    "gpt-5.4-nano",
    "gpt-4o",
    "o3-mini",
    "gpt-5-codex",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "claude-sonnet-4-6",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mixtral-8x7B-v0.1",
    "google/gemma-3-27b-it",
    "deepseek-ai/DeepSeek-V3.1",
    "zai-org/GLM-4.7-FP8",
    "gpt-oss-120b",
]


def test_is_finetune_detects_private_ids():
    assert C.is_finetune("ft:gpt-3.5-turbo-0125:org::abc")
    assert not C.is_finetune("gpt-4o")
    assert not C.is_finetune("Qwen/Qwen3-32B")


def test_classifier_drops_non_chat_models():
    for mid in _NON_CHAT_SAMPLES:
        assert not C.is_chat_model("openai", mid), f"should drop {mid!r}"


def test_classifier_keeps_chat_models():
    for mid in _CHAT_SAMPLES:
        assert C.is_chat_model("together", mid), f"should keep {mid!r}"


# ---------------------------------------------------------------------------
# The live-listing filter: keep curated + chat, drop ft: and non-chat
# ---------------------------------------------------------------------------


def test_filter_chat_live_filters_listing():
    bundled = {
        "gpt-4o": {},  # already curated
        "whisper-1": {},  # curated non-chat (e.g. a transcription endpoint we ship)
    }
    live = [
        ("gpt-4o", {}),  # curated chat -> keep
        ("whisper-1", {}),  # curated, even though non-chat -> keep (no spurious removed)
        ("gpt-5.5", {}),  # new chat -> keep
        ("text-embedding-3-large", {}),  # new non-chat -> drop
        ("dall-e-3", {}),  # new non-chat -> drop
        ("ft:gpt-4o:org::xyz", {}),  # private fine-tune -> drop
    ]
    kept = {mid for mid, _ in R._filter_chat_live("openai", live, bundled)}
    assert kept == {"gpt-4o", "whisper-1", "gpt-5.5"}
    assert not any(mid.startswith("ft:") for mid in kept)


def test_filter_never_keeps_finetune_even_if_curated():
    # A private id must never survive, even if it somehow appears curated.
    bundled = {"ft:gpt-4o:org::xyz": {}}
    live = [("ft:gpt-4o:org::xyz", {})]
    assert R._filter_chat_live("openai", live, bundled) == []


# ---------------------------------------------------------------------------
# The persistence guard: a private fine-tune can never reach a snapshot file
# ---------------------------------------------------------------------------


def test_save_snapshot_refuses_finetune_ids(tmp_path):
    recs = [
        C.ModelRecord(id="gpt-4o", provider="openai"),
        C.ModelRecord(id="ft:gpt-4o:org::secret", provider="openai"),
    ]
    out = tmp_path / "openai.json"
    C.save_snapshot("openai", recs, path=out)
    data = C.load_snapshot("openai", path=out)
    ids = {m["id"] for m in data["models"]}
    assert ids == {"gpt-4o"}
    assert not any(i.startswith("ft:") for i in ids)


def test_records_from_live_never_yields_finetune():
    live = [
        ("gpt-4o", {}),
        ("ft:gpt-4o:org::secret", {}),
        ("text-embedding-3-large", {}),
    ]
    recs = R._records_from_live("openai", live, verified_on="2026-06-17")
    ids = {r.id for r in recs}
    assert ids == {"gpt-4o"}


# ---------------------------------------------------------------------------
# Data invariants after pruning confirmed-retired ids
# ---------------------------------------------------------------------------

_OPENAI_PRUNED = [
    "gpt-3.5-turbo-16k-0613",
    "gpt-4-32k",
    "gpt-4-turbo-preview",
    "o1-mini",
    "o1-preview",
]
_TOGETHER_PRUNED = [
    "deepseek-ai/DeepSeek-R1",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "meta-llama/Llama-4-Maverick-17B-128E",
    "moonshotai/Kimi-K2.5",
]


def test_retired_openai_ids_pruned_from_catalog_and_snapshot():
    from effgen.models.openai_models import OPENAI_MODELS

    snap_ids = {m["id"] for m in (C.load_snapshot("openai") or {})["models"]}
    for mid in _OPENAI_PRUNED:
        assert mid not in OPENAI_MODELS, f"{mid} still in catalog"
        assert mid not in snap_ids, f"{mid} still in snapshot"


def test_retired_together_ids_pruned_from_catalog_and_snapshot():
    from effgen.models.together_models import TOGETHER_MODELS

    snap_ids = {m["id"] for m in (C.load_snapshot("together") or {})["models"]}
    for mid in _TOGETHER_PRUNED:
        assert mid not in TOGETHER_MODELS, f"{mid} still in catalog"
        assert mid not in snap_ids, f"{mid} still in snapshot"


def test_bundled_snapshots_have_no_finetune_or_pollution():
    # No shipped snapshot may carry a private fine-tune id; and the chat-focused
    # providers must not have leaked non-chat ids into their bundled snapshot.
    for prov in C.known_providers():
        data = C.load_snapshot(prov) or {"models": []}
        for m in data["models"]:
            assert not C.is_finetune(m["id"]), f"{prov}: {m['id']} is a fine-tune"


def test_pruned_snapshots_round_trip_against_catalog():
    # The pruned snapshots must still mirror the in-package catalog exactly.
    for prov in ("openai", "together"):
        diff = C.check_drift_against_snapshot(prov)
        assert not diff["added"], f"{prov}: snapshot missing {diff['added']}"
        assert not diff["removed"], f"{prov}: snapshot has stale {diff['removed']}"
        assert not diff["changed"], f"{prov}: snapshot changed {list(diff['changed'])}"
