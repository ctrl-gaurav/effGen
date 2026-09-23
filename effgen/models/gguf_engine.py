"""GGUF model engine via llama-cpp-python (optional).

Loads GGUF-quantized models (Q2_K, Q4_K_M, Q5_K_M, Q8_0, ...) on CPU or GPU.
``llama-cpp-python`` is an optional dependency; importing this module without
it raises a clear ImportError on engine instantiation only.

Install:
    pip install llama-cpp-python                # CPU
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python  # CUDA
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from typing import Any

from ._adapter_utils import provider_runtime_error
from .base import BaseModel, GenerationConfig, GenerationResult, ModelType, TokenCount

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "GGUF support requires the optional 'llama-cpp-python' package.\n"
    "Install with:\n"
    "    pip install llama-cpp-python\n"
    "  or with CUDA:\n"
    "    CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python"
)


def is_gguf_path(path: str) -> bool:
    """Return True if *path* looks like a GGUF model file."""
    return isinstance(path, str) and path.lower().endswith(".gguf")


class GGUFEngine(BaseModel):
    """llama.cpp-backed engine for GGUF-quantized models."""

    def __init__(
        self,
        model_name: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=ModelType.TRANSFORMERS,  # closest existing enum value
            context_length=n_ctx,
        )
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.verbose = verbose
        self._extra = kwargs
        self._llm: Any = None
        # One llama.cpp context holds one sequence's KV cache and is not safe to
        # drive from two threads at once, so concurrent agents sharing this
        # engine take turns. Counting tokens reads only the vocabulary and does
        # not wait.
        self._context_lock = threading.RLock()
        #: Prompt tokens the last call took from the context's KV cache instead
        #: of evaluating again.
        self.last_reused_prompt_tokens = 0

    # ------------------------------------------------------------------ load/unload
    def load(self) -> None:
        """Load the GGUF file with llama-cpp-python (raises when absent)."""
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        if not os.path.exists(self.model_name):
            raise FileNotFoundError(f"GGUF model file not found: {self.model_name}")

        logger.info("Loading GGUF model: %s", self.model_name)
        try:
            self._llm = Llama(
                model_path=self.model_name,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=self.verbose,
                **self._extra,
            )
        except Exception as exc:
            raise provider_runtime_error(
                "gguf", self.model_name, "load", exc,
                message="GGUF model loading failed",
            ) from exc
        self._is_loaded = True
        self._metadata = {
            "engine": "llama-cpp-python",
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
        }

    def unload(self) -> None:
        """Release the llama.cpp model handle."""
        self._llm = None
        self._is_loaded = False

    # ------------------------------------------------------------------ inference
    def _to_kwargs(self, config: GenerationConfig | None) -> dict[str, Any]:
        cfg = config or GenerationConfig()
        # Normalize deterministic generation: temperature<=0 means greedy. llama.cpp
        # treats temperature=0 as greedy; clamp negatives to 0 so the effGen config
        # behaves consistently with the other backends.
        greedy = cfg.temperature is None or cfg.temperature <= 0
        return {
            "temperature": 0.0 if greedy else cfg.temperature,
            "top_p": 1.0 if greedy else cfg.top_p,
            "top_k": -1 if greedy else cfg.top_k,
            "max_tokens": cfg.max_tokens or 256,
            "stop": cfg.stop_sequences or None,
            "repeat_penalty": cfg.repetition_penalty,
            "seed": cfg.seed if cfg.seed is not None else -1,
        }

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a completion for *prompt* and stamp usage metadata.

        Args:
            prompt: The prompt to complete.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the llama.cpp backend.

        Returns:
            The completion with its usage metadata.
        """
        if not self._is_loaded:
            self.load()
        params = self._to_kwargs(config)
        params.update(kwargs)
        try:
            with self._context_lock:
                self._prepare_context(prompt, params)
                out = self._llm(prompt, **params)
        except Exception as exc:
            raise provider_runtime_error(
                "gguf", self.model_name, "generate", exc,
                message="GGUF generation failed",
            ) from exc
        choice = out["choices"][0]
        text = choice.get("text", "")
        usage = out.get("usage", {}) or {}
        return GenerationResult(
            text=text,
            tokens_used=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason", "stop") or "stop",
            model_name=self.model_name,
            metadata={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
                "cached_prompt_tokens": self.last_reused_prompt_tokens,
            },
        )

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield completion text chunks for *prompt* as they are produced.

        Args:
            prompt: The prompt to complete.
            config: Sampling and budget settings for the call.
            **kwargs: Extra parameters forwarded to the llama.cpp backend.
        """
        if not self._is_loaded:
            self.load()
        params = self._to_kwargs(config)
        params.update(kwargs)
        params["stream"] = True
        try:
            with self._context_lock:
                self._prepare_context(prompt, params)
                for chunk in self._llm(prompt, **params):
                    yield chunk["choices"][0].get("text", "")
        except Exception as exc:
            raise provider_runtime_error(
                "gguf", self.model_name, "generate_stream", exc,
                message="GGUF streaming generation failed",
            ) from exc

    def _prepare_context(self, prompt: str, params: dict[str, Any]) -> None:
        """Keep or clear the KV cache before a call, and log what is kept.

        llama.cpp keeps the previous call's tokens and evaluates only what
        follows the longest prefix a new prompt shares with them, so the turns
        of one run — each the previous prompt plus what happened since — skip
        recomputing the part they have in common. A prompt position computed
        from the cache is not bit-for-bit the one a fresh evaluation computes,
        so a call that fixes a seed starts from a clear context instead: the
        same seed on the same prompt then draws the same text every time.
        Called with the context lock held.
        """
        self.last_reused_prompt_tokens = 0
        seed = params.get("seed")
        if seed is not None and seed != -1:
            self._llm.reset()
            return
        held = int(getattr(self._llm, "n_tokens", 0) or 0)
        if held <= 0:
            return
        try:
            tokens = self._llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
            cached = list(self._llm._input_ids[:held])
        except Exception:  # noqa: BLE001 - a count for the log, never a failure
            return
        shared = 0
        for a, b in zip(cached, tokens[:-1]):
            if a != b:
                break
            shared += 1
        self.last_reused_prompt_tokens = shared
        if shared:
            logger.debug(
                "[gguf] the prompt reuses %d of %d tokens already in the KV cache",
                shared, len(tokens),
            )

    def count_tokens(self, text: str) -> TokenCount:
        """Tokenize *text* with the model's own tokenizer and return the count."""
        if not self._is_loaded:
            self.load()
        try:
            tokens = self._llm.tokenize(text.encode("utf-8"))
        except Exception as exc:
            raise provider_runtime_error(
                "gguf", self.model_name, "count_tokens", exc,
                message="GGUF token counting failed",
            ) from exc
        return TokenCount(count=len(tokens), model_name=self.model_name)

    def get_context_length(self) -> int:
        """Return the configured context window size (``n_ctx``)."""
        return self.n_ctx
