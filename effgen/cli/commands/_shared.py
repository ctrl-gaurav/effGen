"""Helpers shared by more than one ``effgen`` CLI command.

A leaf module: it imports nothing from :mod:`effgen.cli`, so any command module
may import it at module scope. Every name here is re-exported from
:mod:`effgen.cli._main`, which remains the import path callers and tests use.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface


def filter_incompatible_tools(
    tools: list,
    model_id: str,
    *,
    warn: Any = None,
) -> tuple[list, list[tuple[str, str]]]:
    """Drop provider-native tools the chosen model cannot execute.

    Provider-*native* tools (Anthropic computer-use ``bash``/``text_editor``/
    ``computer``, OpenAI built-ins like ``web_search_preview``) are run
    server-side by one specific provider and raise "incompatible with model" on
    any other model. ``effgen run`` and ``effgen chat`` both attach a default
    tool set, so both must filter these out for the common case (a non-Claude,
    non-OpenAI model) instead of crashing at agent construction.

    Returns ``(kept_tools, skipped)`` where *skipped* is a list of
    ``(tool_name, reason)``. If *warn* is callable it is invoked once per
    skipped tool with a friendly one-line note.

    Args:
        tools: The tools the command wants to attach.
        model_id: The model the tools would run against.
        warn: Called once per dropped tool with a one-line note.
    """
    model_id = model_id or ""
    is_anthropic_model = model_id.startswith("claude") or "anthropic" in model_id.lower()
    is_openai_model = (
        model_id.startswith("gpt-")
        or model_id.startswith("o1")
        or model_id.startswith("o3")
        or model_id.startswith("o4")
        or "openai" in model_id.lower()
    )
    kept: list = []
    skipped: list[tuple[str, str]] = []
    for tool in tools:
        tname = str(getattr(tool, "name", "") or "")
        cls_name = type(tool).__name__
        is_anthropic_native = "AnthropicNative" in cls_name or "anthropic" in tname.lower()
        is_openai_native = "OpenAINative" in cls_name
        if is_anthropic_native and not is_anthropic_model:
            skipped.append((tname, "requires a Claude model"))
            continue
        if is_openai_native and not is_openai_model:
            skipped.append((tname, "requires a gpt/o-series model"))
            continue
        kept.append(tool)
    if warn is not None:
        for name, why in skipped:
            warn(f"Skipping native tool '{name}' ({why})")
    return kept, skipped


# Providers effGen can route a bare model id to. Keep in sync with the model
# loader / ProviderRegistry; aliases map common spellings to the canonical name.
KNOWN_PROVIDERS = (
    "openai", "anthropic", "gemini", "cerebras", "groq",
    "together", "fireworks", "replicate", "hf",
)
PROVIDER_ALIASES = {
    "google": "gemini",
    "googleai": "gemini",
    "huggingface": "hf",
    "hf_inference": "hf",
    "claude": "anthropic",
    "gpt": "openai",
    "openai-compat": "openai",
}


def resolve_provider_name(provider: str | None) -> tuple[str | None, str | None]:
    """Validate/normalize a user-supplied provider name.

    Returns ``(canonical_provider, error_message)``. On success the error is
    ``None``; on a typo (e.g. ``grok``) the canonical name is ``None`` and the
    error carries a fuzzy "did you mean" suggestion so the CLI never silently
    falls through to a local model download.
    """
    if provider is None:
        return None, None
    raw = provider.strip()
    lower = raw.lower()
    if lower in KNOWN_PROVIDERS:
        return lower, None
    if lower in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[lower], None
    import difflib
    pool = list(KNOWN_PROVIDERS) + list(PROVIDER_ALIASES)
    close = difflib.get_close_matches(lower, pool, n=1, cutoff=0.5)
    hint = ""
    if close:
        suggestion = PROVIDER_ALIASES.get(close[0], close[0])
        hint = f" Did you mean '{suggestion}'?"
    return None, (
        f"Unknown provider '{raw}'.{hint} "
        f"Known providers: {', '.join(KNOWN_PROVIDERS)}."
    )


# Config-file keys that `run` reads and applies to the AgentConfig it builds
# (directly or via the CLI arg of the same name winning first). Keep this in
# sync with every `config.get(...)` call in `run_agent`.
_RUN_CONFIG_APPLIED_KEYS = frozenset({
    "model", "provider",
    "system_prompt", "temperature", "max_iterations", "max_tokens", "guardrails",
})


def _warn_unapplied_config_keys(config: dict, cli: "CLIInterface") -> None:
    """Warn about a config-file key that names a real AgentConfig field but
    isn't one `run` currently applies from a config file.

    A `-c/--config` value the loader doesn't wire through should never be a
    silent no-op — that's a fail-open surprise for a security-relevant field
    (such as ``guardrails``) and a source of confusion for everything else.
    Anything not in :data:`_RUN_CONFIG_APPLIED_KEYS` gets a one-line heads-up
    naming the field and pointing at the matching CLI flag.
    """
    from dataclasses import fields as _dataclass_fields

    from effgen.core.agent import AgentConfig

    valid_fields = {f.name for f in _dataclass_fields(AgentConfig)}
    unapplied = sorted(
        k for k in config if k in valid_fields and k not in _RUN_CONFIG_APPLIED_KEYS
    )
    if not unapplied:
        return
    cli.print_warning(
        f"Configuration file sets {', '.join(unapplied)}, which `effgen run` "
        "does not read from a config file — pass the matching CLI flag "
        "instead, or build the agent through the Python API."
    )


def _print_group_help(args) -> int:
    """Print a command group's help when it is invoked with no subcommand.

    A bare group command (``effgen tools``, ``effgen models``, ...) has nothing
    to do on its own, so it shows the group's usage and subcommand list instead
    of an error, matching what ``--help`` prints.
    """
    parser = getattr(args, "_group_parser", None)
    if parser is not None:
        parser.print_help()
    return 0


def _invoked_command() -> str:
    """Return the command line that produced a result, for a report header.

    The header is stamped into documents that are saved and shared, and an
    invocation carries whatever the user typed — a task, a persona, a header
    argument. Anything key-shaped in it is replaced by a labelled placeholder
    on the way out, the same way the document beside it is.
    """
    from effgen.observability.redact import get_redactor

    return get_redactor().scrub(" ".join(["effgen", *sys.argv[1:]]).strip())


def _checkpoint_run_kwargs(args) -> dict:
    """Extract checkpoint run() kwargs from CLI args."""
    out: dict = {}
    if getattr(args, 'checkpoint_dir', None):
        out['checkpoint_dir'] = args.checkpoint_dir
    if getattr(args, 'checkpoint_interval', 0):
        out['checkpoint_interval'] = args.checkpoint_interval
    return out


# Cheapest well-known model per cloud provider, used to suggest a first model in
# the quickstart. Order = auto-pick preference (fast/free first).
_QUICKSTART_CLOUD_MODELS: tuple[tuple[str, str], ...] = (
    ("groq", "openai/gpt-oss-20b"),
    ("openai", "gpt-5-nano"),
    ("gemini", "gemini-3.1-flash-lite"),
    ("cerebras", "gpt-oss-120b"),
)
# The engine prefix is part of the id: the bare repo id also appears in a cloud
# provider's catalog, and a caller that auto-routes bare ids by catalog would
# send this suggestion to that provider — whose key is, by definition, absent.
_QUICKSTART_LOCAL_MODEL = "transformers:Qwen/Qwen2.5-1.5B-Instruct"


def _quickstart_suggest_model() -> tuple[str, str | None, str]:
    """Pick a sensible first model: a keyed cloud model if any, else local.

    Returns ``(model_id, provider, reason)``.
    """
    try:
        from effgen.models.auth import check_keys
        keys = check_keys()
    except Exception:  # noqa: BLE001
        keys = {}
    for provider, model_id in _QUICKSTART_CLOUD_MODELS:
        info = keys.get(provider)
        if info and info.get("available"):
            # Qualified with the provider that suggested it, for the same reason
            # the local fallback carries its engine prefix: several providers
            # serve the same id, so a caller that uses the id on its own would
            # otherwise have to guess which one was meant.
            return f"{provider}:{model_id}", provider, f"{provider} key detected"
    return _QUICKSTART_LOCAL_MODEL, None, "no cloud key found — using a small local model"


def _preflight_model_hint(cli: "CLIInterface", model_id: str, provider: str | None) -> None:
    """Surface a clean "did you mean" for an unknown model id, once, up front.

    When the user passes an explicit ``-m`` id that isn't in the local catalog,
    a high-confidence typo (e.g. ``gpt-5-nanoo``) otherwise only reveals itself
    mid-run as a provider 404 wall. This checks the local catalog first and, if
    the id is unknown but has near matches, prints a single tidy suggestion line.
    It never blocks — the catalog can be stale, so an unknown-but-real new id is
    allowed through; we only inform.
    """
    try:
        from effgen.models import _catalog

        _bare = model_id.split(":", 1)[-1]
        # Local / HF-hub ids ("org/model") are resolved by download, not via the
        # cloud catalog, so a catalog miss there is meaningless — never warn on a
        # legitimate local model (e.g. meta-llama/Llama-3.2-3B-Instruct). The hint
        # targets slash-free cloud chat ids, where a typo is the likely cause.
        if "/" in _bare:
            return
        if _catalog.lookup(model_id, provider) is not None:
            return
        alts = _catalog.nearest_alternatives(model_id, provider, n=3)
        if not alts:
            return
        names = ", ".join(r.id for r in alts)
        cli.print(
            f"Note: '{_bare}' isn't in the local catalog. Did you mean: {names}? "
            "Proceeding anyway — run 'effgen models refresh' if it's a new id."
        )
    except Exception:  # noqa: BLE001 - a hint must never break the run
        pass
