"""
Anthropic model registry for effGen.

Verified against Anthropic docs on 2026-04-26 (https://docs.anthropic.com/en/docs/about-claude/models/overview).

Notes:
- Claude Opus 4.7 supports adaptive thinking but NOT explicit extended thinking
  (the `thinking={"type": "enabled", ...}` API parameter). Use Sonnet 4.6 or
  Haiku 4.5 when you need a verifiable thinking trace.
- Claude 4.x model IDs use the short alias form (e.g., `claude-opus-4-7`); the
  date-stamped form (e.g., `claude-haiku-4-5-20251001`) is also accepted.
"""

from __future__ import annotations

#: What a prompt token served from the cache costs, as a multiple of the input
#: rate. Anthropic publishes its cache pricing this way rather than as a second
#: price list, so it is carried once instead of per model.
CACHE_READ_PRICE_MULTIPLIER = 0.1

#: What a prompt token written into the cache costs, as a multiple of the input
#: rate, for the five-minute lifetime effGen's markers ask for. The one-hour
#: lifetime is billed at 2x instead.
CACHE_WRITE_PRICE_MULTIPLIER = 1.25

# Cost per million tokens: (input_$/1M, output_$/1M)
ANTHROPIC_MODEL_COSTS: dict[str, tuple[float, float]] = {
    # ── Current Claude 4.x ──────────────────────────────────────────────────
    "claude-opus-4-7":              (5.0, 25.0),
    "claude-sonnet-4-6":            (3.0, 15.0),
    "claude-haiku-4-5-20251001":    (1.0, 5.0),
    "claude-haiku-4-5":             (1.0, 5.0),
    # ── Legacy Claude 4.x (still available) ─────────────────────────────────
    "claude-opus-4-6":              (5.0, 25.0),
    "claude-sonnet-4-5-20250929":   (3.0, 15.0),
    "claude-sonnet-4-5":            (3.0, 15.0),
    "claude-opus-4-5-20251101":     (5.0, 25.0),
    "claude-opus-4-5":              (5.0, 25.0),
    "claude-opus-4-1-20250805":     (15.0, 75.0),
    "claude-opus-4-1":              (15.0, 75.0),
    # ── Legacy Claude 3.x ───────────────────────────────────────────────────
    "claude-3-7-sonnet-20250219":   (3.0, 15.0),
    "claude-3-opus-20240229":       (15.0, 75.0),
    "claude-3-5-sonnet-20241022":   (3.0, 15.0),
    "claude-3-5-sonnet-20240620":   (3.0, 15.0),
    "claude-3-haiku-20240307":      (0.25, 1.25),
    "claude-3-sonnet-20240229":     (3.0, 15.0),
}

ANTHROPIC_MODELS: dict[str, dict] = {
    # ── Current Claude 4.x ──────────────────────────────────────────────────
    "claude-opus-4-7": {
        "context": 1_000_000,
        "max_output": 128_000,
        "family": "opus",
        # Opus 4.7 uses adaptive thinking; explicit extended thinking is not exposed.
        "supports_thinking": False,
        "supports_adaptive_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-sonnet-4-6": {
        "context": 1_000_000,
        "max_output": 64_000,
        "family": "sonnet",
        "supports_thinking": True,
        "supports_adaptive_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-haiku-4-5-20251001": {
        "context": 200_000,
        "max_output": 64_000,
        "family": "haiku",
        "supports_thinking": True,
        "supports_adaptive_thinking": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-haiku-4-5": {
        "context": 200_000,
        "max_output": 64_000,
        "family": "haiku",
        "supports_thinking": True,
        "supports_adaptive_thinking": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    # ── Legacy Claude 4.x (still generally available) ───────────────────────
    "claude-opus-4-6": {
        "context": 1_000_000,
        "max_output": 128_000,
        "family": "opus",
        "supports_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-sonnet-4-5-20250929": {
        "context": 200_000,
        "max_output": 64_000,
        "family": "sonnet",
        "supports_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-sonnet-4-5": {
        "context": 200_000,
        "max_output": 64_000,
        "family": "sonnet",
        "supports_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-opus-4-5-20251101": {
        "context": 200_000,
        "max_output": 64_000,
        "family": "opus",
        "supports_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-opus-4-5": {
        "context": 200_000,
        "max_output": 64_000,
        "family": "opus",
        "supports_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-opus-4-1-20250805": {
        "context": 200_000,
        "max_output": 32_000,
        "family": "opus",
        "supports_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-opus-4-1": {
        "context": 200_000,
        "max_output": 32_000,
        "family": "opus",
        "supports_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    # ── Claude 3.7 ──────────────────────────────────────────────────────────
    "claude-3-7-sonnet-20250219": {
        "context": 200_000,
        "max_output": 64_000,
        "family": "sonnet",
        "supports_thinking": True,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    # ── Claude 3.5 ──────────────────────────────────────────────────────────
    "claude-3-5-sonnet-20241022": {
        "context": 200_000,
        "max_output": 8_096,
        "family": "sonnet",
        "supports_thinking": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-3-5-sonnet-20240620": {
        "context": 200_000,
        "max_output": 8_096,
        "family": "sonnet",
        "supports_thinking": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    # ── Claude 3 ────────────────────────────────────────────────────────────
    "claude-3-opus-20240229": {
        "context": 200_000,
        "max_output": 4_096,
        "family": "opus",
        "supports_thinking": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-3-haiku-20240307": {
        "context": 200_000,
        "max_output": 4_096,
        "family": "haiku",
        "supports_thinking": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
    "claude-3-sonnet-20240229": {
        "context": 200_000,
        "max_output": 4_096,
        "family": "sonnet",
        "supports_thinking": False,
        "supports_native_tools": True,
        "supports_prompt_caching": True,
        "supports_vision": True,
    },
}


def get_model_info(model_name: str) -> dict:
    """Return registry entry for model_name, or sensible defaults if unknown."""
    return ANTHROPIC_MODELS.get(model_name, {
        "context": 200_000,
        "max_output": 64_000,
        "family": "unknown",
        "supports_thinking": False,
        "supports_native_tools": True,
        "supports_prompt_caching": False,
        "supports_vision": False,
    })


def supports_thinking(model_name: str) -> bool:
    """True when the catalog marks *model_name* as supporting extended thinking."""
    return get_model_info(model_name).get("supports_thinking", False)


def get_context_length(model_name: str) -> int:
    """Return the context window size (tokens) for *model_name*."""
    return get_model_info(model_name)["context"]


def get_cost_per_million(model_name: str) -> tuple[float, float]:
    """Return (input_cost, output_cost) per million tokens."""
    return ANTHROPIC_MODEL_COSTS.get(model_name, (3.0, 15.0))
