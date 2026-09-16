"""
Gemini + Gemma model registry for effGen.

Rate limits and context/max-output values fetched from:
  https://ai.google.dev/gemini-api/docs/rate-limits
  https://ai.google.dev/gemini-api/docs/models
Fetch date: 2026-04-25

Per-token pricing from https://ai.google.dev/gemini-api/docs/pricing (verified 2026-05-11).
Free tier note: Gemini Flash/Flash-Lite models have free quotas (e.g. 500 RPD for
gemini-2.5-flash + gemini-2.5-flash-lite combined). Free tier data may be used by
Google for model improvement. Paid tier removes data use and raises limits.

Free-tier limits apply per Google Cloud project. Whichever metric
(RPM / TPM / RPD) is hit first triggers a 429 ``ResourceExhausted``
with an SDK-supplied ``retry_delay`` that effGen's adapter honors
(see :class:`effgen.models.gemini_adapter.GeminiAdapter`).

Only text-out and Gemma text-only models are listed here. Multi-modal
generative endpoints (Imagen, Veo, Lyria, Nano Banana, native-audio
TTS / Live API) and embedding models are out of scope for this
registry.

For models that require the ``-preview`` suffix to call (e.g.
``gemini-3.1-flash-lite``), both keys are provided so callers can use
either form.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry has:
#   family             — model family label
#   context            — context window in tokens
#   max_output         — max completion tokens
#   rpm/tpm/rpd        — free-tier requests-per-minute / tokens-per-minute /
#                        requests-per-day (None when not exposed by Google)
#   free_tier          — True if reliably callable on the free tier today
#   supports_thinking  — supports `thinking_config` (Gemini 2.5+ Pro/Flash
#                        and Gemini 3.x)
#   supports_grounding — supports the Google Search grounding tool
#   supports_native_tools — supports function-calling via FunctionDeclaration
#   supports_vision    — accepts image input
#   tier               — "free" (no payment needed) | "cheap" | "premium"
#   notes              — short human-friendly note (rendered by CLI)
# ---------------------------------------------------------------------------
GEMINI_MODELS: dict[str, dict] = {
    # ---- Gemini 3.x family ----
    "gemini-3.1-flash-lite": {
        "family": "flash-lite",
        "context": 1_000_000,
        "max_output": 32_768,
        "rpm": 15,
        "tpm": 250_000,
        "rpd": 500,
        "free_tier": True,
        "supports_thinking": True,
        "supports_grounding": False,  # Google Search grounding hits 429 on free tier for this model
        "supports_native_tools": True,
        "supports_vision": True,
        "supports_audio": True,   # native audio understanding
        "supports_video": True,   # native video via Files API
        "tier": "free",
        "notes": "Cheapest Gemini 3.x text model. Best free-tier choice. Grounding not available on free tier.",
        "pricing_per_1m_input": 0.25,
        "pricing_per_1m_output": 1.50,
    },
    "gemini-3-flash-preview": {
        "family": "flash",
        "context": 1_000_000,
        "max_output": 32_768,
        "rpm": 5,
        "tpm": 250_000,
        "rpd": 20,
        "free_tier": True,
        "supports_thinking": True,
        "supports_grounding": True,
        "supports_native_tools": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_video": True,
        "tier": "free",
        "notes": "Mid-tier Gemini 3. Free-tier RPD is tight (20/day).",
        "pricing_per_1m_input": 0.50,
        "pricing_per_1m_output": 3.00,
    },
    "gemini-3.1-pro-preview": {
        "family": "pro",
        "context": 2_000_000,
        "max_output": 65_536,
        "rpm": None,
        "tpm": None,
        "rpd": None,
        "free_tier": False,
        "supports_thinking": True,
        "supports_grounding": True,
        "supports_native_tools": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_video": True,
        "tier": "premium",
        "notes": "Gemini 3.1 Pro preview. Paid tier only.",
        "pricing_per_1m_input": 2.00,
        "pricing_per_1m_output": 12.00,
    },

    # ---- Gemini 2.5 family ----
    "gemini-2.5-flash-lite": {
        "family": "flash-lite",
        "context": 1_000_000,
        "max_output": 16_384,
        "rpm": 10,
        "tpm": 250_000,
        "rpd": 20,
        "free_tier": True,
        "supports_thinking": False,
        "supports_grounding": True,
        "supports_native_tools": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_video": True,
        "tier": "free",
        "notes": "Stable cheap Gemini 2.5. Free-tier RPD is 20/day.",
        "pricing_per_1m_input": 0.10,
        "pricing_per_1m_output": 0.40,
    },
    "gemini-2.5-flash": {
        "family": "flash",
        "context": 1_000_000,
        "max_output": 16_384,
        "rpm": 5,
        "tpm": 250_000,
        "rpd": 20,
        "free_tier": True,
        "supports_thinking": False,
        "supports_grounding": True,
        "supports_native_tools": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_video": True,
        "tier": "free",
        "notes": "Stable mid-tier Gemini 2.5. Good for tool use.",
        "pricing_per_1m_input": 0.30,
        "pricing_per_1m_output": 2.50,
    },
    "gemini-2.5-pro": {
        "family": "pro",
        "context": 2_000_000,
        "max_output": 32_768,
        "rpm": None,
        "tpm": None,
        "rpd": None,
        "free_tier": False,
        "supports_thinking": True,
        "supports_grounding": True,
        "supports_native_tools": True,
        "supports_vision": True,
        "supports_audio": True,
        "supports_video": True,
        "tier": "premium",
        "notes": "Gemini 2.5 Pro. Paid tier only.",
        "pricing_per_1m_input": 1.25,
        "pricing_per_1m_output": 10.00,
    },

    # ---- Gemma family (open weights, free-tier hosted) ----
    # Only the ids the live Gemini API serves today are listed (verified against
    # client.models.list() + a live generate on 2026-06-08).  Earlier hand-listed
    # Gemma 3 ids (gemma-3-1b/4b/12b/27b, and the never-real gemma-3-2b) and the
    # un-suffixed gemma-4-26b / gemma-4-31b now return 404 model_not_found and
    # have been dropped; the un-suffixed names resolve via GEMINI_MODEL_ALIASES.
    "gemma-4-26b-a4b-it": {
        "family": "gemma",
        "context": 65_536,
        "max_output": 8_192,
        "rpm": 15,
        "tpm": None,  # unlimited per docs
        "rpd": 1_500,
        "free_tier": True,
        "supports_thinking": False,
        "supports_grounding": False,
        "supports_native_tools": False,
        "supports_vision": False,
        "tier": "free",
        "notes": "Gemma 4 26B (A4B) instruction-tuned. TPM unlimited; lower RPD.",
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
    },
    "gemma-4-31b-it": {
        "family": "gemma",
        "context": 65_536,
        "max_output": 8_192,
        "rpm": 15,
        "tpm": None,
        "rpd": 1_500,
        "free_tier": True,
        "supports_thinking": False,
        "supports_grounding": False,
        "supports_native_tools": False,
        "supports_vision": False,
        "tier": "free",
        "notes": "Gemma 4 31B instruction-tuned. TPM unlimited; lower RPD.",
        "pricing_per_1m_input": 0.0,
        "pricing_per_1m_output": 0.0,
    },
}

# Aliases that resolve to a registered key when callers use the short form.
# Google's API also accepts these names directly.
GEMINI_MODEL_ALIASES: dict[str, str] = {
    # gemini-3.1-flash-lite is the GA id; the older -preview id was retired by
    # Google (returns 404). Short aliases resolve to the GA canonical id.
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite",
    "gemini-3-flash-lite": "gemini-3.1-flash-lite",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3.1-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-flash-lite-latest": "gemini-3.1-flash-lite",
    # The live Gemma-4 ids carry an instruction-tuned suffix; accept the short
    # forms users naturally type and resolve them to the canonical live ids.
    "gemma-4-26b": "gemma-4-26b-a4b-it",
    "gemma-4-31b": "gemma-4-31b-it",
}

# Default model for free-tier live testing (cheap + tool-calling).
GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-lite"

#: The shortest prompt prefix each model will serve from its implicit context
#: cache. A prompt shorter than this is simply not cached; the number is
#: reported so a run can say why it saw no hit, and is never used to pad a
#: prompt up to it. Models published before implicit caching existed are absent
#: and take the conservative default below.
CACHE_MIN_PREFIX_TOKENS: dict[str, int] = {
    "gemini-2.5-flash":        2048,
    "gemini-2.5-flash-lite":   2048,
    "gemini-2.5-pro":          2048,
    "gemini-3-flash-preview":  4096,
    "gemini-3.1-flash-lite":   4096,
    "gemini-3.1-pro-preview":  4096,
}

_DEFAULT_CACHE_MIN_PREFIX_TOKENS = 4096


def get_cache_min_prefix_tokens(model_id: str) -> int:
    """Return the shortest cacheable prompt prefix for *model_id*.

    Args:
        model_id: The Gemini model id, alias or canonical.

    Returns:
        The published minimum in tokens, or the larger default for an id this
        table does not carry, so the number is never optimistic.
    """
    return CACHE_MIN_PREFIX_TOKENS.get(
        _resolve(model_id), _DEFAULT_CACHE_MIN_PREFIX_TOKENS
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve(model_id: str) -> str:
    return GEMINI_MODEL_ALIASES.get(model_id, model_id)


def available_models() -> list[str]:
    """Return all registered Gemini / Gemma text-out model IDs."""
    return list(GEMINI_MODELS.keys())


def free_tier_models() -> list[str]:
    """Return models reliably callable on the free tier today."""
    return [m for m, info in GEMINI_MODELS.items() if info.get("free_tier")]


def model_info(model_id: str) -> dict:
    """Return capability + rate-limit metadata for *model_id*.

    Accepts short aliases (e.g. ``gemini-3.1-flash-lite``).

    Raises:
        KeyError: If *model_id* is not registered.
    """
    canonical = _resolve(model_id)
    if canonical not in GEMINI_MODELS:
        raise KeyError(
            f"Unknown Gemini model '{model_id}'. Available: {available_models()}"
        )
    info = dict(GEMINI_MODELS[canonical])
    info["canonical_id"] = canonical
    return info


def recommended_models(tier: str = "free") -> list[dict]:
    """Return registered models filtered by *tier*.

    Args:
        tier: One of ``"free"``, ``"cheap"``, ``"premium"``, or ``"all"``.

    Returns:
        List of ``{"id": ..., **model_info}`` dicts ordered by family + size.
    """
    if tier == "all":
        return [{"id": m, **info} for m, info in GEMINI_MODELS.items()]
    return [
        {"id": m, **info}
        for m, info in GEMINI_MODELS.items()
        if info.get("tier") == tier
    ]
