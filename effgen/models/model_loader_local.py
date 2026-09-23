"""Local-engine construction for the model loader.

Holds the torch-availability helpers and the loader methods that build the
Transformers, vLLM, GGUF and MLX engines. Mixed into
:class:`~effgen.models.model_loader.ModelLoader`; not usable on its own.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from effgen.models.base import BaseModel

if TYPE_CHECKING:
    # Cloud adapters and the heavy local-inference deps (torch, the
    # Transformers/vLLM engines) are imported lazily inside the ``_load_*``
    # methods that actually construct them. This matches the six other adapters
    # (HF/Replicate/Cerebras/Groq/Together/Fireworks) and — importantly — breaks
    # the import cycle that arises when ``effgen.models`` is imported before
    # ``effgen.core.agent`` (anthropic_adapter -> core.messages -> core.agent ->
    # model_loader). It also keeps `from effgen import Agent` / the CLI from
    # pulling torch or transformers for a pure cloud-API workflow.
    from effgen.models.transformers_engine import TransformersEngine
    from effgen.models.vllm_engine import VLLMEngine

logger = logging.getLogger("effgen.models.model_loader")


def _require_torch(engine: str):
    """Return the ``torch`` module, or raise naming the engine and the install.

    The local engines are the only part of effGen that needs PyTorch — the cloud
    providers do not — so an install without it must say what is missing and how
    to get it, rather than surfacing the import system's ``No module named
    'torch'`` as the whole explanation.
    """
    try:
        import torch
    except ImportError as exc:
        from effgen.models._adapter_utils import missing_torch_error

        raise missing_torch_error(engine) from exc
    return torch


def _missing_module(exc: BaseException) -> str | None:
    """Name the package whose absence caused *exc*, following the cause chain.

    A lazy import layer (the one in ``transformers``) re-raises a missing
    dependency as "Could not import module 'X'", which names the symbol it was
    building rather than the package that is actually absent. The innermost
    ``ModuleNotFoundError`` still carries it.
    """
    seen: set[int] = set()
    found: str | None = None
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = getattr(current, "name", None)
        if isinstance(current, ModuleNotFoundError) and name:
            found = name
        current = current.__cause__ or current.__context__
    return found


class ModelLoaderLocalMixin:
    """Construction of the local inference engines."""

    def _load_huggingface_model(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        **kwargs
    ) -> "VLLMEngine | TransformersEngine | BaseModel":
        """
        Load HuggingFace model with intelligent engine selection.

        Engine selection priority:
        1. Explicitly requested engine (force_engine parameter)
        2. Auto-detect: MLX on Apple Silicon (when no CUDA), else Transformers

        Args:
            model_name: HuggingFace model ID or local path
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            Model engine instance (VLLMEngine, TransformersEngine, MLXEngine, or MLXVLMEngine)
        """
        params = config or {}
        params.update(kwargs)

        # Check if MLX engine is explicitly requested
        if self.force_engine == "mlx":
            logger.info("Using MLX engine (explicitly requested)")
            try:
                return self._load_with_mlx(model_name, params)
            except Exception as e:
                logger.warning(f"MLX loading failed: {e}")
                logger.info("Falling back to Transformers...")
                return self._load_with_transformers(model_name, params)

        if self.force_engine == "mlx_vlm":
            logger.info("Using MLX-VLM engine (explicitly requested)")
            try:
                return self._load_with_mlx_vlm(model_name, params)
            except Exception as e:
                logger.warning(f"MLX-VLM loading failed: {e}")
                logger.info("Falling back to Transformers...")
                return self._load_with_transformers(model_name, params)

        # Check if vLLM engine is explicitly requested
        if self.force_engine == "vllm":
            logger.info("Using vLLM engine (explicitly requested)")
            try:
                return self._load_with_vllm(model_name, params)
            except Exception as e:
                logger.warning(f"vLLM loading failed: {e}")
                logger.info("Falling back to Transformers...")
                return self._load_with_transformers(model_name, params)

        # Opt-in "auto-fast": prefer vLLM when it is importable and the GPU is
        # usable, otherwise fall back to Transformers. The default (force_engine
        # None) stays on Transformers — auto-fast must be requested explicitly.
        if self.force_engine == "auto-fast":
            if self._vllm_usable():
                logger.info("Using vLLM engine (auto-fast: vLLM available and GPU usable)")
                try:
                    return self._load_with_vllm(model_name, params)
                except Exception as e:
                    logger.warning(f"vLLM loading failed: {e}")
                    logger.info("Falling back to Transformers...")
                    return self._load_with_transformers(model_name, params)
            logger.info(
                "Using Transformers engine (auto-fast: vLLM unavailable or no usable GPU)"
            )
            return self._load_with_transformers(model_name, params)

        # Auto-detection: prefer MLX on Apple Silicon when no CUDA available
        if self.force_engine is None:
            try:
                import torch

                from effgen.hardware.platform import is_apple_silicon, is_mlx_available
                if is_apple_silicon() and is_mlx_available() and not torch.cuda.is_available():
                    logger.info("Apple Silicon detected with MLX available, using MLX engine")
                    try:
                        return self._load_with_mlx(model_name, params)
                    except Exception as e:
                        logger.warning(f"MLX auto-detection loading failed: {e}")
                        logger.info("Falling back to Transformers...")
            except ImportError:
                pass

        # Default to Transformers (more compatible, easier setup)
        logger.info("Using Transformers engine (default)")
        return self._load_with_transformers(model_name, params)

    @staticmethod
    def _vllm_usable() -> bool:
        """Return True only if vLLM can actually run here: the package imports
        cleanly (no CUDA/torch ABI mismatch) and a CUDA GPU is available. Used by
        the opt-in ``engine="auto-fast"`` path to decide vLLM vs Transformers
        without ever raising.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            return False

        # Asked without starting the CUDA driver, which would leave the engine
        # core vLLM forks unable to reach the GPU (see cuda_device_visible).
        from effgen.models._vram import cuda_device_visible

        if not cuda_device_visible():
            return False
        try:
            import importlib.util
            if importlib.util.find_spec("vllm") is None:
                return False
            # Probe the EXACT import VLLMEngine.load() performs. `import vllm`
            # alone succeeds even when the compiled extension is ABI-incompatible
            # (e.g. a missing libcudart.so) — only importing LLM/SamplingParams
            # surfaces that. Probing here keeps auto-fast on Transformers instead
            # of constructing an engine whose later load() would hard-fail.
            from vllm import LLM, SamplingParams  # noqa: F401
            return True
        except Exception:
            logger.debug("vLLM present but not usable; auto-fast will use Transformers",
                         exc_info=True)
            return False

    def _load_with_vllm(
        self,
        model_name: str,
        params: dict[str, Any]
    ) -> VLLMEngine:
        """
        Load model with vLLM.

        Args:
            model_name: Model identifier
            params: Configuration parameters

        Returns:
            VLLMEngine instance

        Raises:
            RuntimeError: If vLLM is unavailable or loading fails
        """
        _require_torch("vllm")

        from effgen.models.vllm_engine import VLLMEngine

        logger.info(f"Attempting to load with vLLM: {model_name}")

        # Check CUDA availability without starting the CUDA driver in this
        # process: vLLM forks its engine core, and a parent that started the
        # driver leaves that child unable to reach the GPU.
        from effgen.models._vram import cuda_device_visible

        if not cuda_device_visible():
            from effgen.gpu.cuda_compat import get_cuda_status
            status = get_cuda_status()
            if status.mismatch and status.message:
                raise RuntimeError(f"vLLM requires a usable GPU. {status.message}")
            raise RuntimeError("CUDA not available, vLLM requires GPU")

        # Determine quantization if not specified
        if "quantization" not in params:
            params["quantization"] = self._auto_select_quantization(model_name)

        # Determine tensor parallel size if not specified
        if "tensor_parallel_size" not in params:
            params["tensor_parallel_size"] = self._auto_select_tensor_parallel(model_name)

        # Only set download directory if explicitly specified (let vLLM use its default otherwise)
        # This avoids potential issues with path handling
        if "download_dir" not in params and self.cache_dir != os.path.expanduser("~/.cache/huggingface"):
            params["download_dir"] = self.cache_dir

        return VLLMEngine(model_name=model_name, **params)

    def _load_with_mlx(
        self,
        model_name: str,
        params: dict[str, Any]
    ) -> "BaseModel":
        """
        Load model with MLX (Apple Silicon).

        Args:
            model_name: Model identifier (mlx-community/ or HuggingFace ID)
            params: Configuration parameters

        Returns:
            MLXEngine instance

        Raises:
            RuntimeError: If MLX is unavailable or loading fails
        """
        from effgen.models.mlx_engine import MLXEngine

        logger.info(f"Attempting to load with MLX: {model_name}")

        # Filter out CUDA-specific params that don't apply to MLX
        mlx_params = {
            k: v for k, v in params.items()
            if k not in (
                "tensor_parallel_size", "gpu_memory_utilization", "quantization",
                "max_num_seqs", "max_num_batched_tokens", "download_dir",
                "device_map", "quantization_bits",
            )
        }

        return MLXEngine(model_name=model_name, **mlx_params)

    def _load_with_mlx_vlm(
        self,
        model_name: str,
        params: dict[str, Any]
    ) -> "BaseModel":
        """
        Load vision-language model with MLX-VLM (Apple Silicon).

        Args:
            model_name: Model identifier
            params: Configuration parameters

        Returns:
            MLXVLMEngine instance

        Raises:
            RuntimeError: If MLX-VLM is unavailable or loading fails
        """
        from effgen.models.mlx_vlm_engine import MLXVLMEngine

        logger.info(f"Attempting to load VLM with MLX-VLM: {model_name}")

        # Filter out CUDA-specific params
        mlx_params = {
            k: v for k, v in params.items()
            if k not in (
                "tensor_parallel_size", "gpu_memory_utilization", "quantization",
                "max_num_seqs", "max_num_batched_tokens", "download_dir",
                "device_map", "quantization_bits",
            )
        }

        return MLXVLMEngine(model_name=model_name, **mlx_params)

    def _load_with_transformers(
        self,
        model_name: str,
        params: dict[str, Any]
    ) -> TransformersEngine:
        """
        Load model with Transformers.

        Args:
            model_name: Model identifier
            params: Configuration parameters

        Returns:
            TransformersEngine instance
        """
        torch = _require_torch("transformers")

        try:
            from effgen.models.transformers_engine import TransformersEngine
        except ImportError as exc:
            missing = _missing_module(exc)
            if missing:
                raise ImportError(
                    f"The local 'transformers' engine could not be imported: the package "
                    f"'{missing}' is not installed. Install it with: pip install {missing}"
                ) from exc
            raise

        logger.info(f"Loading with Transformers: {model_name}")

        # Convert shorthand quantization="4bit"/"8bit"/"awq"/"gptq" to engine params.
        if "quantization" in params and "quantization_bits" not in params:
            q = params.pop("quantization")
            if q in ("4bit", "4"):
                params["quantization_bits"] = 4
            elif q in ("8bit", "8"):
                params["quantization_bits"] = 8
            elif q == "awq":
                # AWQ models carry their own quantization config; just verify
                # autoawq is importable so we fail with a friendly message.
                try:
                    import awq  # type: ignore  # noqa: F401
                except ImportError:
                    logger.warning(
                        "quantization='awq' requested but 'autoawq' is not installed. "
                        "Install with: pip install autoawq"
                    )
                params["quantization_method"] = "awq"
            elif q == "gptq":
                try:
                    import auto_gptq  # type: ignore  # noqa: F401
                except ImportError:
                    logger.warning(
                        "quantization='gptq' requested but 'auto-gptq' is not installed. "
                        "Install with: pip install auto-gptq"
                    )
                params["quantization_method"] = "gptq"
            elif q is not None:
                logger.warning(
                    "Unknown quantization value '%s' for Transformers engine, ignoring.", q
                )

        # Determine quantization if not specified
        if "quantization_bits" not in params:
            params["quantization_bits"] = self._auto_select_quantization_bits()

        # Set device map
        if "device_map" not in params:
            params["device_map"] = "auto" if torch.cuda.is_available() else None

        # AWQ/GPTQ quantization is encoded in the model checkpoint config; the
        # transformers engine just needs to load it as-is. Drop our internal
        # marker so it isn't forwarded to from_pretrained.
        params.pop("quantization_method", None)

        return TransformersEngine(model_name=model_name, **params)
