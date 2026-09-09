"""
Groq model registry for effGen.

Rate limits from Groq dashboard (Developer plan free tier):
  https://console.groq.com/settings/limits
Fetch date: 2026-04-28

Context windows, modalities and per-token pricing from the Groq Models API
(/openai/v1/models); per-model request and token allowances are also returned as
``x-ratelimit-*`` response headers.  A model whose ``supported_features`` include
``reasoning`` is flagged ``reasoning: True``: it spends output budget on hidden
reasoning, so it needs the larger default budget.  Catalog reconciled against the
live listing on 2026-09-08.
"""

from __future__ import annotations

GROQ_MODELS: dict[str, dict] = {
    # -----------------------------------------------------------------------
    # Chat Completions — text models
    # -----------------------------------------------------------------------
    "qwen/qwen3.6-27b": {
        "family": "qwen",
        "context": 131_072,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "supports_vision": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 8_000,
        "tpd": None,
        "active": True,
        "modality": "chat",
        "reasoning": True,
        "notes": "Qwen3.6 27B — accepts text and image input",
        "pricing_per_1m_input": 0.60,
        "pricing_per_1m_output": 3.00,
    },
    "qwen/qwen3.8-27b": {
        "family": "qwen",
        "context": 131_042,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 8_000,
        "tpd": None,
        "active": True,
        "modality": "chat",
        "notes": "Qwen3.8 27B — answers directly, without a reasoning chain",
    },
    "openai/gpt-oss-120b": {
        "family": "gpt-oss",
        "context": 131_072,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 8_000,
        "tpd": 200_000,
        "active": True,
        "modality": "chat",
        "reasoning": True,
        "notes": "OpenAI GPT-OSS 120B open weights",
        "pricing_per_1m_input": 0.15,
        "pricing_per_1m_output": 0.60,
    },
    "openai/gpt-oss-20b": {
        "family": "gpt-oss",
        "context": 131_072,
        "max_output": 16_384,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 8_000,
        "tpd": 200_000,
        "active": True,
        "modality": "chat",
        "reasoning": True,
        "notes": "OpenAI GPT-OSS 20B open weights",
        "pricing_per_1m_input": 0.075,
        "pricing_per_1m_output": 0.30,
    },
    "openai/gpt-oss-safeguard-20b": {
        "family": "gpt-oss",
        "context": 131_072,
        "max_output": 4_096,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 1_000,
        "tpm": 8_000,
        "tpd": 200_000,
        "active": True,
        "modality": "chat",
        "reasoning": True,
        "notes": "GPT-OSS 20B safety/guardrail variant",
        "pricing_per_1m_input": 0.075,
        "pricing_per_1m_output": 0.30,
    },
    "groq/compound": {
        "family": "compound",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 250,
        "tpm": 70_000,
        "tpd": None,
        "active": True,
        "modality": "chat",
        "notes": "Groq Compound (agentic system) — no daily token limit",
    },
    "groq/compound-mini": {
        "family": "compound",
        "context": 131_072,
        "max_output": 8_192,
        "supports_native_tools": True,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 250,
        "tpm": 70_000,
        "tpd": None,
        "active": True,
        "modality": "chat",
        "notes": "Groq Compound Mini (smaller agentic system)",
    },
    "allam-2-7b": {
        "family": "allam",
        "context": 4_096,
        "max_output": 2_048,
        "supports_native_tools": False,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 7_000,
        "tpm": 6_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "ALLaM 2 7B — Arabic/English bilingual",
    },
    "meta-llama/llama-prompt-guard-2-86m": {
        "family": "llama-guard",
        "context": 512,
        "max_output": 256,
        "supports_native_tools": False,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 14_400,
        "tpm": 15_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "Llama Prompt Guard 2 86M — safety classifier",
        "pricing_per_1m_input": 0.04,
        "pricing_per_1m_output": 0.04,
    },
    "meta-llama/llama-prompt-guard-2-22m": {
        "family": "llama-guard",
        "context": 512,
        "max_output": 256,
        "supports_native_tools": False,
        "supports_streaming": True,
        "rpm": 30,
        "rpd": 14_400,
        "tpm": 15_000,
        "tpd": 500_000,
        "active": True,
        "modality": "chat",
        "notes": "Llama Prompt Guard 2 22M — lightweight safety classifier",
        "pricing_per_1m_input": 0.03,
        "pricing_per_1m_output": 0.03,
    },
    # -----------------------------------------------------------------------
    # Speech-to-Text models (not usable via chat completions)
    # -----------------------------------------------------------------------
    "whisper-large-v3": {
        "family": "whisper",
        "context": 448,
        "max_output": 448,
        "supports_native_tools": False,
        "supports_streaming": False,
        "rpm": 20,
        "rpd": 2_000,
        "tpm": None,
        "tpd": None,
        "active": True,
        "modality": "stt",
        "notes": "Whisper Large v3 — speech-to-text only, not a chat model",
    },
    "whisper-large-v3-turbo": {
        "family": "whisper",
        "context": 448,
        "max_output": 448,
        "supports_native_tools": False,
        "supports_streaming": False,
        "rpm": 20,
        "rpd": 2_000,
        "tpm": None,
        "tpd": None,
        "active": True,
        "modality": "stt",
        "notes": "Whisper Large v3 Turbo — faster speech-to-text",
    },
    # -----------------------------------------------------------------------
    # Text-to-Speech models
    # -----------------------------------------------------------------------
    "canopylabs/orpheus-v1-english": {
        "family": "orpheus",
        "context": 4_000,
        "max_output": 4_000,
        "supports_native_tools": False,
        "supports_streaming": False,
        "rpm": 10,
        "rpd": 100,
        "tpm": 1_200,
        "tpd": 3_600,
        "active": True,
        "modality": "tts",
        "notes": "Orpheus v1 English — text-to-speech",
    },
    "canopylabs/orpheus-arabic-saudi": {
        "family": "orpheus",
        "context": 4_000,
        "max_output": 4_000,
        "supports_native_tools": False,
        "supports_streaming": False,
        "rpm": 10,
        "rpd": 100,
        "tpm": 1_200,
        "tpd": 3_600,
        "active": True,
        "modality": "tts",
        "notes": "Orpheus Arabic (Saudi) — text-to-speech",
    },
}

# Default model for new GroqAdapter instances
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"

# Chat-capable model IDs (usable via chat.completions.create)
GROQ_CHAT_MODELS = {
    k: v for k, v in GROQ_MODELS.items()
    if v.get("modality") == "chat"
}


def available_models() -> list[str]:
    """Return all registered Groq model IDs."""
    return list(GROQ_MODELS.keys())


def chat_models() -> list[str]:
    """Return model IDs usable via chat completions."""
    return list(GROQ_CHAT_MODELS.keys())


def tool_capable_models() -> list[str]:
    """Return chat models that support native function calling."""
    return [k for k, v in GROQ_CHAT_MODELS.items() if v.get("supports_native_tools")]


def model_info(model_name: str) -> dict:
    """Return registry entry for *model_name*, or raise ``KeyError``."""
    if model_name not in GROQ_MODELS:
        raise KeyError(
            f"Unknown Groq model '{model_name}'. "
            f"Available: {available_models()}"
        )
    return GROQ_MODELS[model_name]
