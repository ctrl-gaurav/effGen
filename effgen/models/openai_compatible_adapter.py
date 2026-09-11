"""Adapter for any server that speaks the OpenAI protocol.

The OpenAI chat-completions API is how most self-hosted and gateway serving
works: vLLM, SGLang, TGI, llama.cpp's server, Ollama, LM Studio, LiteLLM and
similar proxies all expose it, as do several hosted providers. This adapter
talks that protocol against a URL you supply, so effGen can drive a model you
serve yourself instead of loading its own copy of the weights.

    from effgen.models import load_model

    model = load_model(
        "Qwen/Qwen2.5-7B-Instruct",
        provider="openai_compatible",
        base_url="http://127.0.0.1:8100/v1",
    )

``base_url`` may also come from ``EFFGEN_BASE_URL``, ``OPENAI_BASE_URL`` or
``OPENAI_API_BASE``. ``api_key`` defaults to a placeholder, which is what a
local server that authenticates nothing expects; pass a real one for a gateway
that checks it.

Serving the weights once and pointing several agents at that one server means
one copy in GPU memory, continuous batching across every caller, and a GPU
whose lifetime is not tied to any agent process.

What this adapter changes relative to :class:`~effgen.models.openai_adapter.OpenAIAdapter`:

- ``base_url`` is required rather than optional.
- Model ids are the server's own, so the bundled OpenAI catalog is not
  consulted for context length, reasoning support or sampling support. Pass
  ``context_length=`` when the default does not match what you serve.
- Calls report no price. What a server you run costs is not something effGen
  can derive from a token count, so it states nothing rather than ``$0``.
"""

from __future__ import annotations

import logging
from typing import Any

from effgen.models._base_url import PLACEHOLDER_API_KEY, resolve_base_url
from effgen.models.openai_adapter import OpenAIAdapter

logger = logging.getLogger(__name__)

#: Context window assumed when the caller names none. Most instruction-tuned
#: models served this way are at least this large, and a server that is asked
#: for more than it can hold reports the real limit itself.
DEFAULT_CONTEXT_LENGTH = 32768


class OpenAICompatibleAdapter(OpenAIAdapter):
    """Call a model served over the OpenAI protocol at a URL you supply.

    Attributes:
        model_name: The id the server serves the model under. For vLLM this is
            whatever ``--served-model-name`` was given, defaulting to the
            weights' path or repo id.
        base_url: The endpoint, including the API version segment most servers
            use (``http://127.0.0.1:8000/v1``).
        api_key: Sent as the bearer credential. Defaults to a placeholder for
            servers that check nothing.
        context_length: The window to plan against, since the server does not
            publish one over this protocol.
        supports_vision: Whether this server's model takes image input. ``None``
            — the default — sends the request and lets the server answer, since
            no catalog here describes a model somebody else is serving.
    """

    #: Provider label used for metrics/error reporting (see Agent._model_provider).
    _provider = "openai_compatible"

    #: The server serves its own ids; the bundled OpenAI catalog describes none
    #: of them, so context length, pricing and capability lookups skip it.
    _catalog_backed = False

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        context_length: int | None = None,
        max_retries: int = 3,
        timeout: int = 60,
        supports_reasoning: bool = False,
        supports_vision: bool | None = None,
        **kwargs: Any,
    ) -> None:
        resolved = resolve_base_url(base_url)
        if not resolved:
            raise ValueError(
                "An OpenAI-compatible endpoint needs a base_url. Pass "
                "base_url='http://host:port/v1', or set EFFGEN_BASE_URL "
                "(or OPENAI_BASE_URL) in the environment. To call OpenAI "
                "itself, use provider='openai' instead."
            )

        # A window nobody named is a guess, and a wrong one is expensive: effGen
        # plans the history against it, so a server started with a smaller
        # `--max-model-len` truncates or refuses far from where the number was
        # chosen. Say so once, naming the value and where to set it.
        if context_length is None:
            logger.warning(
                "No context_length given for '%s' at %s; assuming %d tokens. "
                "If the server was started with a smaller window (vLLM's "
                "--max-model-len, TGI's --max-total-tokens), pass "
                "context_length=<the real number> — planning against a window "
                "the server does not have fails later, at the call.",
                model_name, resolved, DEFAULT_CONTEXT_LENGTH,
            )

        super().__init__(
            model_name=model_name,
            api_key=api_key or PLACEHOLDER_API_KEY,
            max_retries=max_retries,
            timeout=timeout,
            base_url=resolved,
            context_length=(
                context_length if context_length is not None else DEFAULT_CONTEXT_LENGTH
            ),
            **kwargs,
        )

        # The catalog's answers describe OpenAI's models, not this server's.
        # Take the caller's word for reasoning support, and allow the full
        # sampling surface, which every server implementing this protocol
        # accepts and OpenAI's own reasoning models do not.
        self._is_reasoning_model = supports_reasoning
        self._supports_sampling_params = True
        # Whether this server's model takes images is the server's to answer.
        # ``None`` sends the request and reports whatever it says; a caller who
        # knows can say so and get the refusal before the call is billed.
        self._declared_vision = supports_vision

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_served_models(self) -> list[str]:
        """Return the model ids this server reports, newest protocol field first.

        Asks the endpoint's ``/models`` route. Returns an empty list when the
        server does not implement it, which some minimal servers do not — that
        is not an error, just an endpoint with nothing to say about itself.
        """
        if self.client is None:
            self.load()
        client = self.client
        if client is None:
            # load() reports why it could not build a client; an id listing is
            # not worth raising a second error over.
            return []
        try:
            listing = client.models.list()
        except Exception as e:
            logger.debug(f"Endpoint {self.base_url} does not list its models: {e}")
            return []
        return [entry.id for entry in getattr(listing, "data", []) if getattr(entry, "id", None)]

    @classmethod
    def list_models(cls) -> list[str]:
        """Return an empty list: the ids live on the server, not in a catalog.

        Use :meth:`list_served_models` on an instance to ask the endpoint what
        it serves.
        """
        return []

    @classmethod
    def get_model_info(cls, model_id: str) -> dict:
        """Return what is known without a catalog: the id and the defaults used."""
        return {
            "model_name": model_id,
            "context_length": DEFAULT_CONTEXT_LENGTH,
            "provider": cls._provider,
            "catalog_backed": False,
        }


def _register() -> None:
    try:
        from effgen.models.capabilities import Capability
        from effgen.models.registry import ProviderRegistry
        ProviderRegistry.register(
            "openai_compatible",
            OpenAICompatibleAdapter,
            # The ids belong to whichever server the caller points at, so the
            # registry carries none.
            {},
            env_keys=["EFFGEN_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"],
            capabilities={
                Capability.chat, Capability.streaming, Capability.tools,
                Capability.json_schema,
            },
            # No pricing entry: a server the caller runs publishes no
            # per-token rate, so every surface reports these calls as unpriced
            # rather than as a free tier or a fabricated $0.
            pricing=None,
        )
    except Exception:
        logger.debug("Failed to build detailed provider info; using fallback", exc_info=True)


_register()
