"""Model-id routing for the model loader.

Resolves a model id and the optional ``provider:``/``engine:`` prefix to the
adapter or engine that serves it, and reports an actionable error when it
cannot. Mixed into :class:`~effgen.models.model_loader.ModelLoader`; not usable
on its own.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from effgen.models.base import BaseModel, ModelType
from effgen.models.model_loader_cloud import (
    _get_cerebras_adapter,
    _get_fireworks_adapter,
    _get_groq_adapter,
    _get_hf_inference_adapter,
    _get_replicate_adapter,
    _get_together_adapter,
)

logger = logging.getLogger("effgen.models.model_loader")


def _wants_compatible_endpoint(
    engine_config: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> bool:
    """Whether this call names an endpoint other than the provider's own.

    An explicit ``base_url`` in either place counts. The environment fallback
    does not: ``OPENAI_BASE_URL`` is often set machine-wide for an unrelated
    proxy, and silently rerouting a plain ``provider="openai"`` call because of
    it would be a surprise. Ask for ``provider="openai_compatible"`` to pick up
    the environment.
    """
    for source in (engine_config or {}, kwargs):
        value = source.get("base_url")
        if value and str(value).strip():
            return True
    return False


class ModelLoaderRoutingMixin:
    """Model-id and prefix routing to a provider adapter or local engine."""

    def load_model(
        self,
        model_name: str,
        engine_config: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> BaseModel:
        """
        Load a model with automatic detection and configuration.

        Args:
            model_name: Model identifier (HuggingFace ID, local path, or API model name)
            engine_config: Optional engine-specific configuration
            **kwargs: Additional model parameters

        Returns:
            Loaded model instance ready for inference

        Raises:
            ValueError: If model_name is invalid or unsupported
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading model: {model_name}")

        # Check if already loaded
        if model_name in self.loaded_models:
            logger.info(f"Model '{model_name}' already loaded, returning cached instance")
            return self.loaded_models[model_name]

        # Explicit provider routing (e.g. provider="cerebras", provider="openai")
        provider = kwargs.pop("provider", None)

        # An explicit but unknown provider (e.g. a typo'd "grok") must fail
        # fast and loud rather than silently falling through to a local
        # HuggingFace download.  Normalize common aliases (google -> gemini).
        if provider is not None:
            _known = {
                "openai", "openai_compatible", "anthropic", "gemini", "cerebras",
                "groq", "together", "fireworks", "replicate", "hf", "hf_inference",
            }
            _aliases = {
                "google": "gemini", "huggingface": "hf", "claude": "anthropic",
                # The protocol, not the company: every spelling people reach for
                # when they mean "an OpenAI-protocol server at my own URL".
                "openai-compatible": "openai_compatible",
                "openai_compat": "openai_compatible",
                "oai_compatible": "openai_compatible",
                "compatible": "openai_compatible",
                "server": "openai_compatible",
                "vllm_server": "openai_compatible",
                "local_server": "openai_compatible",
            }
            _p = str(provider).strip().lower()
            _p = _aliases.get(_p, _p)
            if _p not in _known:
                import difflib
                _close = difflib.get_close_matches(_p, sorted(_known), n=1, cutoff=0.5)
                _hint = f" Did you mean '{_close[0]}'?" if _close else ""
                raise ValueError(
                    f"Unknown provider '{provider}'.{_hint} "
                    f"Known providers: {', '.join(sorted(_known - {'hf_inference'}))}."
                )
            provider = _p

            # An id may already carry the provider the caller passed
            # separately: a suggestion is reported as a qualified id *and* as
            # the provider it came from, and a caller that forwards both means
            # one thing, not two. Drop the redundant prefix so the adapter is
            # asked for the id its own catalog publishes. A prefix naming a
            # different provider is a real disagreement and is left in place,
            # so the error names what was actually asked for.
            if isinstance(model_name, str) and ":" in model_name:
                _dup_prefix, _dup_rest = model_name.split(":", 1)
                _dup = _dup_prefix.strip().lower()
                if _dup_rest and _aliases.get(_dup, _dup) == provider:
                    model_name = _dup_rest

        # Support "provider:model_id" prefix syntax via ProviderRegistry
        if provider is None and isinstance(model_name, str) and ":" in model_name:
            _prefix, _rest = model_name.split(":", 1)
            try:
                from effgen.models.registry import ProviderRegistry
                if _prefix in ProviderRegistry.list_providers():
                    provider = _prefix
                    model_name = _rest
            except Exception:
                logger.debug("Provider-prefix registry lookup failed", exc_info=True)

        # Support "engine:model_id" prefix syntax for local engines, mirroring
        # the cloud "provider:model_id" syntax above (e.g.
        # "transformers:Qwen/Qwen2.5-7B-Instruct"). Without this, the whole
        # string is passed to the HuggingFace repo-id validator, which rejects
        # the colon with a cryptic HFValidationError instead of running the
        # model locally with the requested engine.
        if (
            provider is None
            and isinstance(model_name, str)
            and ":" in model_name
        ):
            _eng_prefix, _eng_rest = model_name.split(":", 1)
            if _eng_prefix in self._LOCAL_ENGINE_PREFIXES and _eng_rest:
                if self.force_engine is None:
                    self.force_engine = _eng_prefix
                model_name = _eng_rest

        # A colon-prefixed id that matched neither a provider nor a local-engine
        # prefix above is a typo'd prefix (e.g. "nonexistant:foo"), not a local
        # HuggingFace repo id — HF repo ids cannot contain a colon. Fail fast
        # with the valid prefixes instead of falling through to the local
        # loader, which would reject the whole string with an unrelated
        # "invalid repo id" error that names internal loader machinery.
        if (
            provider is None
            and isinstance(model_name, str)
            and ":" in model_name
            and not os.path.exists(model_name)
        ):
            _bad_prefix, _ = model_name.split(":", 1)
            try:
                from effgen.models.registry import ProviderRegistry
                known_providers = sorted(ProviderRegistry.list_providers())
            except Exception:
                known_providers = []
            _known_engines = ", ".join(sorted(self._LOCAL_ENGINE_PREFIXES))
            # No providers at all means the registry was emptied at runtime
            # (ProviderRegistry.clear()); the prefix may well be valid, so name
            # the call that brings the built-in providers back instead of
            # blaming the model id.
            if not known_providers:
                raise ValueError(
                    f"Cannot resolve provider prefix {_bad_prefix!r} in model id "
                    f"{model_name!r}: the provider registry is empty — call "
                    "ProviderRegistry.reset() to restore the built-in providers. "
                    f"Known local engines: {_known_engines}."
                )
            raise ValueError(
                f"Unknown provider or engine prefix {_bad_prefix!r} in model id "
                f"{model_name!r}. Known providers: {', '.join(known_providers)}. "
                f"Known local engines: {_known_engines}."
            )

        # Route / disambiguate bare cloud model ids by consulting the model
        # catalog directly.  Without this, a documented
        # provider id such as ``gpt-oss-120b`` or ``allam-2-7b``
        # falls through to the local HuggingFace path and fails with a confusing
        # download error instead of calling the provider.  Bare ids that no cloud
        # catalog knows (the normal case for local HF repos / paths) are left
        # untouched so the existing local detection still runs.
        # A "/" in a bare id means an org/model HuggingFace-style repo (also how
        # Together/Fireworks/Replicate/HF list their models), so it stays on the
        # local/HF path; only slash-free cloud slugs (gpt-oss-120b,
        # allam-2-7b, zai-glm-4.7, …) are candidates for routing.
        if (
            provider is None
            and isinstance(model_name, str)
            and "/" not in model_name
            and not os.path.exists(model_name)
            and not model_name.lower().endswith(".gguf")
        ):
            try:
                from effgen.models import _catalog, _refresh
                from effgen.models.errors import AmbiguousModelError

                candidates = _catalog.providers_for(model_name)
                if len(candidates) > 1:
                    raise AmbiguousModelError(model_name, candidates)
                if len(candidates) == 1:
                    only = candidates[0]
                    if _refresh.has_credentials(only):
                        provider = only
                        logger.info(
                            "Routing bare model id %r to provider %r (catalog match)",
                            model_name, only,
                        )
                    else:
                        envs = _refresh._KEY_ENVS.get(only, ("<API_KEY>",))[0]
                        logger.warning(
                            "Model %r is a known %s model but %s is not set — "
                            "treating it as a local model. To call the provider, "
                            "pass provider=%r (or use '%s:%s') and set %s.",
                            model_name, only, envs, only, only, model_name, envs,
                        )
            except AmbiguousModelError:
                raise
            except Exception:
                logger.debug("Bare-id provider routing lookup failed", exc_info=True)

        # A base_url means the OpenAI protocol against someone else's server,
        # so provider="openai" with one routes to the compatible adapter: its
        # ids, context window and pricing are the server's, not OpenAI's.
        if provider == "openai" and _wants_compatible_endpoint(engine_config, kwargs):
            provider = "openai_compatible"

        if provider == "openai_compatible":
            model = self._load_openai_compatible_model(model_name, engine_config, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "openai":
            model = self._load_openai_model(model_name, engine_config, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "anthropic":
            model = self._load_anthropic_model(model_name, engine_config, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "gemini":
            model = self._load_gemini_model(model_name, engine_config, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "cerebras":
            CerebrasAdapter = _get_cerebras_adapter()
            api_key = kwargs.pop("api_key", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = CerebrasAdapter(model_name=model_name, api_key=api_key, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "groq":
            GroqAdapter = _get_groq_adapter()
            api_key = kwargs.pop("api_key", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = GroqAdapter(model_name=model_name, api_key=api_key, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "together":
            TogetherAdapter = _get_together_adapter()
            api_key = kwargs.pop("api_key", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = TogetherAdapter(model_name=model_name, api_key=api_key, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "fireworks":
            FireworksAdapter = _get_fireworks_adapter()
            api_key = kwargs.pop("api_key", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = FireworksAdapter(model_name=model_name, api_key=api_key, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider == "replicate":
            ReplicateAdapter = _get_replicate_adapter()
            api_token = kwargs.pop("api_token", kwargs.pop("api_key", None))
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = ReplicateAdapter(model_name=model_name, api_token=api_token, **kwargs)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model
        if provider in ("hf_inference", "hf"):
            # `hf:` is the remote HuggingFace Inference API, not local execution.
            # A privacy-conscious user expecting their data to stay on-box should
            # know they need a local engine instead — nudge once, don't spam.
            logger.info(
                "Loading '%s' via the HuggingFace Inference API (remote). For "
                "local GPU execution of this model, load it with "
                "engine='transformers' (or 'vllm') instead of the 'hf:' prefix.",
                model_name,
            )
            HFInferenceAdapter = _get_hf_inference_adapter()
            api_token = kwargs.pop("api_token", kwargs.pop("api_key", None))
            endpoint_url = kwargs.pop("endpoint_url", None)
            for k in ("apply_chat_template", "tensor_parallel_size", "gpu_memory_utilization", "trust_remote_code", "quantization", "device", "torch_dtype"):
                kwargs.pop(k, None)
            model = HFInferenceAdapter(
                model_name=model_name,
                api_token=api_token,
                endpoint_url=endpoint_url,
                **kwargs,
            )
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            return model

        # GGUF files take a dedicated path (llama-cpp-python).
        if isinstance(model_name, str) and model_name.lower().endswith(".gguf"):
            from .gguf_engine import GGUFEngine

            gguf_params = dict(kwargs)
            gguf_params.pop("apply_chat_template", None)
            model = GGUFEngine(model_name=model_name, **gguf_params)
            model.load()
            self._validate_model(model)
            self.loaded_models[model_name] = model
            logger.info(f"GGUF model '{model_name}' loaded successfully")
            return model

        # Detect model type
        model_type = self._detect_model_type(model_name)
        logger.info(f"Detected model type: {model_type.value}")

        # Load based on type
        if model_type == ModelType.OPENAI:
            model = self._load_openai_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.ANTHROPIC:
            model = self._load_anthropic_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.GEMINI:
            model = self._load_gemini_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.MLX:
            # MLX model detected (e.g., mlx-community/ prefix) — use MLX engine
            if self.force_engine is None:
                self.force_engine = "mlx"
            model = self._load_huggingface_model(model_name, engine_config, **kwargs)
        elif model_type == ModelType.MLX_VLM:
            if self.force_engine is None:
                self.force_engine = "mlx_vlm"
            model = self._load_huggingface_model(model_name, engine_config, **kwargs)
        else:
            # HuggingFace model - use Transformers by default, vLLM/MLX optional
            model = self._load_huggingface_model(model_name, engine_config, **kwargs)

        # Load the model
        model.load()

        # Validate
        self._validate_model(model)

        # Cache the loaded model
        self.loaded_models[model_name] = model

        logger.info(f"Model '{model_name}' loaded successfully")
        return model

    def _detect_model_type(self, model_name: str) -> ModelType:
        """
        Detect the type of model based on its name.

        Args:
            model_name: Model identifier

        Returns:
            ModelType enum value
        """
        model_lower = model_name.lower()

        # Check API models
        for prefix in self.OPENAI_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.OPENAI

        for prefix in self.ANTHROPIC_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.ANTHROPIC

        for prefix in self.GEMINI_MODELS:
            if model_lower.startswith(prefix):
                return ModelType.GEMINI

        # Check for MLX-community models (pre-converted for Apple Silicon)
        if "mlx-community/" in model_lower:
            logger.info(f"Detected MLX-community model: {model_name}")
            return ModelType.MLX

        # GGUF files are handled by a separate engine.
        if model_lower.endswith(".gguf"):
            logger.info(f"Detected GGUF model file: {model_name}")
            return ModelType.TRANSFORMERS  # routed to GGUFEngine in load path

        # Check if it's a local path
        if os.path.exists(model_name):
            logger.info(f"Detected local model path: {model_name}")
            return ModelType.TRANSFORMERS  # Default to Transformers for local models

        # Assume HuggingFace model ID
        return ModelType.TRANSFORMERS  # Default to Transformers for HuggingFace models
