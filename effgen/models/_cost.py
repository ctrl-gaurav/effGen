"""
Cost tracker for effGen model adapters.

Accumulates prompt and completion token counts per (provider, model) pair
and converts them to USD using per-provider rate tables.  Cerebras free-tier
cost is $0; a model the catalog publishes no rate for reports ``None`` instead,
so "this was free" and "we do not know what this cost" stay distinguishable.
The process-global tracker persists events to SQLite so the ``effgen cost`` CLI
can summarize spend across restarts.

Usage::

    from effgen.models._cost import CostTracker

    tracker = CostTracker.get()
    tracker.record("cerebras", "llama3.1-8b", prompt_tokens=50, completion_tokens=20)
    print(tracker.total_cost("cerebras", "llama3.1-8b"))
    print(tracker.summary())
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

_BUDGET_CONFIG_PATH = Path.home() / ".effgen" / "budget.json"


def _budget_config_path() -> Path:
    """Resolve the budget-config file, honoring an ``EFFGEN_BUDGET_CONFIG``
    override (mirrors ``EFFGEN_COST_DB``) so a sandbox or test run can point the
    budget away from the user's real ``~/.effgen/budget.json``."""
    env_path = os.environ.get("EFFGEN_BUDGET_CONFIG")
    return Path(env_path) if env_path else _BUDGET_CONFIG_PATH


def _load_budget() -> dict:
    """Load budget config (returns empty dict if absent)."""
    try:
        path = _budget_config_path()
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        logger.debug("Failed to load budget config; treating as empty", exc_info=True)
    return {}


def _configured_budgets(budget_cfg: dict) -> list[tuple[str, float]]:
    """Return the ``(period, budget_usd)`` pairs with a valid positive cap.

    Skips a period that is absent, non-numeric, or <= 0 (logging a warning for
    the non-numeric case) so callers can iterate only the budgets that apply.
    """
    result = []
    for period in ("daily", "monthly"):
        raw_budget = budget_cfg.get(period)
        if raw_budget is None:
            continue
        try:
            budget_usd = float(raw_budget)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid %s budget value: %r", period, raw_budget)
            continue
        if budget_usd <= 0:
            continue
        result.append((period, budget_usd))
    return result


def format_usd(amount: float) -> str:
    """Format a USD amount with enough significant digits for sub-cent values.

    Amounts of ``$0.0001`` or more use the familiar 4-decimal form
    (``$1.2300``); smaller amounts switch to enough decimal places to keep at
    least two significant digits, then drop trailing zeros, so a cap like
    ``$0.00005`` prints as ``$0.00005`` rather than rounding to ``$0.0001``
    (or vanishing to ``$0.0000``) under a fixed ``:.4f``.
    """
    if amount == 0:
        return "$0.0000"
    magnitude = abs(amount)
    if magnitude >= 0.0001:
        return f"${amount:.4f}"
    exponent = math.floor(math.log10(magnitude))
    decimals = -exponent + 1
    text = f"{amount:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"${text}"


# Internal alias kept short for call sites in this module.
_format_usd = format_usd

# ---------------------------------------------------------------------------
# Pricing resolution
#
# The normalized model catalog (``effgen/models/_catalog.py``) is the single
# source of truth for per-token prices: it reads the same in-package ``*_MODELS``
# dicts the adapters call, so the cost a user is charged can never disagree with
# the catalog the CLI shows.  ``_rate()`` consults the catalog first and only
# falls back to the small legacy table below for ids no catalog knows (e.g. the
# deprecated Cerebras test ids or a bare local model name).
#
# A model is treated as:
#   * "priced"   - the catalog publishes a per-token price (use it),
#   * "free"     - the catalog flags a genuine free tier ($0, labeled "free"),
#   * "metered"  - non-token billing (e.g. Replicate per-second; cost arrives
#                  pre-computed via ``record(cost_usd=...)``),
#   * "unpriced" - no published price (we never invent one; shown as "unpriced"
#                  rather than a misleading "$0.000000").
#
# The process-global tracker persists events to SQLite. Constructing
# ``CostTracker(storage=None)`` keeps the old in-memory behavior for callers
# that need isolated accounting.
# ---------------------------------------------------------------------------

# Providers whose legacy fallback rate of $0 means "genuinely free tier" rather
# than "price unknown" (used only for ids absent from the catalog).
_FREE_TIER_PROVIDERS = {"cerebras"}

# Legacy fallback rates (USD per 1M tokens) for ids the catalog does not carry.
# Kept intentionally small — the catalog is authoritative for everything else.
#
# This table is deliberately **not** pruned to the current roster. A provider
# drops an id from its listing long before the run records naming it stop being
# read, and a stored record whose id has left the catalog would otherwise be
# reported as unpriced. So a row here outliving its catalog entry is the point,
# not drift: it is what still prices last quarter's runs. Nothing new can be
# *called* through such a row — every adapter refuses an id its catalog does not
# carry — so an extra row costs a dictionary entry and nothing else.
#
# The invariant that does matter is agreement: where this table and a catalog
# both price the same id, the numbers must match, or the price a run reports
# depends on which lookup answered first. It is gated by
# ``tests/unit/test_cost_rate_tables.py``.
_RATES: dict[str, dict[str, tuple[float, float]]] = {
    "cerebras": {
        # OFFICIAL: Cerebras free tier pricing = $0 for all models.
        "*": (0.0, 0.0),
    },
    "openai": {
        # PLACEHOLDER (verify against OpenAI pricing page before billing use)
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "*": (1.00, 3.00),
    },
    "anthropic": {
        # PLACEHOLDER
        "claude-3-opus": (15.00, 75.00),
        "claude-3-sonnet": (3.00, 15.00),
        "claude-3-haiku": (0.25, 1.25),
        "*": (3.00, 15.00),
    },
    "gemini": {
        # PLACEHOLDER
        "gemini-pro": (0.125, 0.375),
        "gemini-1.5-pro": (3.50, 10.50),
        "*": (1.00, 3.00),
    },
    "groq": {
        # OFFICIAL: Groq free tier = $0 for all models (2026-04-28)
        "*": (0.0, 0.0),
    },
    "together": {
        # OFFICIAL rates from the Together AI pricing page, reconciled against
        # the live /v1/models listing on 2026-08-07.
        # Per million tokens: (input, output)
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": (1.04, 1.04),
        "meta-llama/Meta-Llama-3-8B-Instruct": (0.20, 0.20),
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": (0.18, 0.59),
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": (0.27, 0.85),
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": (0.18, 0.18),
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": (0.88, 0.88),
        "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": (0.88, 0.88),
        "meta-llama/Meta-Llama-3.1-405B-Instruct": (3.50, 3.50),
        "meta-llama/Llama-3.2-1B-Instruct": (0.06, 0.06),
        "meta-llama/Llama-3-8b-chat-hf": (0.20, 0.20),
        "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.30),
        "Qwen/Qwen2.5-72B-Instruct-Turbo": (1.20, 1.20),
        "Qwen/Qwen2.5-72B-Instruct": (1.20, 1.20),
        "Qwen/Qwen2.5-14B-Instruct": (0.80, 0.80),
        "Qwen/Qwen2.5-Coder-32B-Instruct": (0.80, 0.80),
        "Qwen/QwQ-32B": (1.20, 1.20),
        "Qwen/Qwen3.5-9B": (0.17, 0.25),
        "Qwen/Qwen3.5-397B-A17B": (0.60, 3.60),
        "Qwen/Qwen3-Coder-Next-FP8": (0.50, 1.20),
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8": (2.00, 2.00),
        "Qwen/Qwen3-235B-A22B-Thinking-2507": (0.65, 3.00),
        "Qwen/Qwen3-Next-80B-A3B-Instruct": (0.15, 1.50),
        "Qwen/Qwen3-Next-80B-A3B-Thinking": (0.15, 1.50),
        "Qwen/Qwen3-VL-8B-Instruct": (0.18, 0.68),
        "Qwen/Qwen3-VL-32B-Instruct": (0.50, 1.50),
        "Qwen/Qwen2-VL-72B-Instruct": (1.20, 1.20),
        "Qwen/Qwen2.5-VL-72B-Instruct": (1.95, 8.00),
        "deepseek-ai/DeepSeek-V3.1": (0.60, 1.70),
        "deepseek-ai/DeepSeek-V3-0324": (1.25, 1.25),
        "deepseek-ai/DeepSeek-V4-Pro": (1.74, 3.48),
        "deepseek-ai/DeepSeek-R1": (3.00, 7.00),
        "deepseek-ai/DeepSeek-R1-0528": (3.00, 7.00),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (0.18, 0.18),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": (0.00, 0.00),
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": (1.60, 1.60),
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": (2.00, 2.00),
        "mistralai/Mixtral-8x7B-Instruct-v0.1": (0.60, 0.60),
        "mistralai/Mistral-7B-Instruct-v0.3": (0.20, 0.20),
        "mistralai/Mistral-7B-Instruct-v0.1": (0.20, 0.20),
        "mistralai/Mistral-Small-24B-Instruct-2501": (0.10, 0.30),
        "mistralai/Ministral-3-14B-Instruct-2512": (0.20, 0.20),
        "openai/gpt-oss-20b": (0.05, 0.20),
        "openai/gpt-oss-120b": (0.15, 0.60),
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": (0.88, 0.88),
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2": (0.06, 0.25),
        "moonshotai/Kimi-K2.5": (0.50, 2.80),
        "moonshotai/Kimi-K2.6": (1.20, 4.50),
        "moonshotai/Kimi-K2-Thinking": (1.20, 4.00),
        "MiniMaxAI/MiniMax-M2.5": (0.30, 1.20),
        "MiniMaxAI/MiniMax-M2.7": (0.30, 1.20),
        "MiniMaxAI/MiniMax-M2": (0.00, 0.00),
        "zai-org/GLM-4.5-Air-FP8": (0.20, 1.10),
        "zai-org/GLM-4.6": (0.60, 2.20),
        "zai-org/GLM-4.7": (0.45, 2.00),
        "zai-org/GLM-5": (1.00, 3.20),
        "zai-org/GLM-5.1": (1.40, 4.40),
        "google/gemma-4-31B-it": (0.39, 0.97),
        "google/gemma-3n-E4B-it": (0.06, 0.12),
        "arize-ai/qwen-2-1.5b-instruct": (0.10, 0.10),
        "deepcogito/cogito-v2-1-671b": (1.25, 1.25),
        "Qwen/Qwen2-1.5B-Instruct": (0.02, 0.02),
        # Free / dedicated-endpoint models → $0 in tracker
        "*": (0.0, 0.0),
    },
}


def _catalog_pricing(provider: str, model: str) -> tuple[float, float, str, str] | None:
    """Resolve pricing for *provider*/*model* from the normalized catalog.

    Returns ``(price_in_per_1m, price_out_per_1m, status, note)`` where *status*
    is one of ``priced``/``free``/``metered``/``unpriced``, or ``None`` when the
    catalog has no record for this id (so the caller can fall back).
    """
    try:
        from effgen.models import _catalog as _cat

        rec = _cat.lookup(model, provider.lower())
    except Exception:  # pragma: no cover - catalog import/lookup is best-effort
        logger.debug("Catalog pricing lookup failed for %s/%s", provider, model, exc_info=True)
        return None
    if rec is None:
        return None

    pin = rec.price_in_per_1m
    pout = rec.price_out_per_1m
    # A genuinely nonzero published price wins over free_tier: some providers
    # (e.g. Gemini) flag a model free_tier=True to mean "usable within a free
    # quota" while still publishing the real per-token rate that applies
    # beyond that quota, so a nonzero price is the more accurate label there.
    # A catalog entry with an explicit 0/0 price is a different situation —
    # some Together entries carry this without knowing the real rate — so it
    # is never treated as "priced" on its own; it falls through to the
    # free_tier / price_note / unpriced checks below instead.
    if (pin is not None and pin > 0) or (pout is not None and pout > 0):
        return (float(pin or 0.0), float(pout or 0.0), "priced", rec.price_note or "")
    if rec.free_tier:
        return (0.0, 0.0, "free", rec.price_note or "")
    if rec.price_note:
        # Non-token billing (e.g. Replicate per-second); cost comes in via cost_usd.
        return (0.0, 0.0, "metered", rec.price_note)
    return (0.0, 0.0, "unpriced", "")


def _legacy_rate(
    provider: str, model: str, *, allow_wildcard: bool = True
) -> tuple[float, float] | None:
    """Per-1M rate from the small fallback table, or None if the provider/id is unknown.

    Args:
        provider: The provider that served the call.
        model: The model id the call used.
        allow_wildcard: Whether a provider's ``"*"`` placeholder rate may
            answer. It is a rough figure for an id nobody has priced, so it is
            useful as an estimate and wrong as a published price.

    Returns:
        The (input, output) per-million rate, or None when nothing matches.
    """
    provider_rates = _RATES.get(provider.lower())
    if not provider_rates:
        return None
    if model in provider_rates:
        return provider_rates[model]
    for key in provider_rates:
        if key != "*" and model.startswith(key):
            return provider_rates[key]
    if allow_wildcard and "*" in provider_rates:
        return provider_rates["*"]
    return None


def _rate(provider: str, model: str) -> tuple[float, float]:
    """Lookup (input_per_M, output_per_M) rate for provider/model.

    Catalog-first (the single source of truth shared with the adapters); falls
    back to the small legacy table for ids the catalog does not carry.  Returns
    ``(0.0, 0.0)`` for genuinely free/unpriced/unknown ids — pricing is never
    invented.  A caller that has to tell a free tier apart from a model with no
    published price must use :func:`call_cost`, which returns ``None`` for the
    latter instead of a rate of zero.
    """
    catalog = _catalog_pricing(provider, model)
    if catalog is not None:
        # priced -> real numbers; free/metered/unpriced -> 0 token rate.
        return (catalog[0], catalog[1])
    legacy = _legacy_rate(provider, model)
    return legacy if legacy is not None else (0.0, 0.0)


def pricing_status(provider: str, model: str) -> str:
    """Return how *provider*/*model* is priced: ``priced``/``free``/``metered``/``unpriced``.

    Used by ``effgen cost`` to label a $0 row accurately — a genuine free tier
    reads "free" while a model with no published price reads "unpriced" instead
    of a misleading "$0.000000".
    """
    catalog = _catalog_pricing(provider, model)
    if catalog is not None:
        return catalog[2]
    # A provider's "*" placeholder is not a published price. Counting it as one
    # billed every id the bundled catalog had not seen — a fine-tuned `ft:`
    # model, anything released since the last refresh — at a made-up rate, and
    # reported the invented number as real.
    legacy = _legacy_rate(provider, model, allow_wildcard=False)
    if legacy is not None and (legacy[0] > 0 or legacy[1] > 0):
        return "priced"
    if provider.lower() in _FREE_TIER_PROVIDERS:
        return "free"
    return "unpriced"


def cache_rates(provider: str, model: str) -> tuple[float | None, float | None]:
    """The cache read and write rates for *provider*/*model*, per 1M tokens.

    ``None`` in either slot means the provider publishes no separate rate for
    that half, in which case those tokens are billed at the ordinary input rate:
    a discount is reported only where a provider states one, never assumed.

    Args:
        provider: The provider that served the call.
        model: The model id the call used.

    Returns:
        ``(cached_read_per_1m, cache_write_per_1m)``.
    """
    try:
        from effgen.models import _catalog as _cat

        rec = _cat.lookup(model, provider.lower())
    except Exception:  # pragma: no cover - catalog lookup is best-effort
        logger.debug("Cache-rate lookup failed for %s/%s", provider, model, exc_info=True)
        return (None, None)
    if rec is None:
        return (None, None)
    return (rec.price_cached_in_per_1m, rec.price_cache_write_in_per_1m)


def call_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Price one call, or return ``None`` when *model* has no published rate.

    This is the one place a per-call ``cost_usd`` is derived, so every adapter
    reports the same number for the same usage.  ``0.0`` means the call really
    was free — a free tier, or a model billed outside the token price
    (``metered``), whose real cost arrives through
    :meth:`CostTracker.record`'s ``cost_usd`` argument.  ``None`` means the
    catalog publishes no price for this id, so no cost can be stated; callers
    render that as "unpriced" rather than a fabricated ``$0.000000``.

    Args:
        provider: The provider that served the call.
        model: The model id the call used.
        prompt_tokens: Input tokens the provider reported, cache reads and
            writes included — they are prompt tokens, priced differently.
        completion_tokens: Output tokens the provider reported.
        cached_tokens: How many of *prompt_tokens* came from the cache.
        cache_write_tokens: How many of *prompt_tokens* were written to it.
    """
    if pricing_status(provider, model) == "unpriced":
        return None
    input_rate, output_rate = _rate(provider, model)
    cost = completion_tokens * output_rate
    cached_rate, write_rate = cache_rates(provider, model)
    read = max(0, min(int(cached_tokens or 0), prompt_tokens))
    written = max(0, min(int(cache_write_tokens or 0), prompt_tokens - read))
    plain = prompt_tokens - read - written
    cost += plain * input_rate
    cost += read * (input_rate if cached_rate is None else cached_rate)
    cost += written * (input_rate if write_rate is None else write_rate)
    return cost / 1_000_000


# Guard so the "no published price, so this spend is not counted" heads-up
# fires at most once per provider/model per process.
_UNPRICED_BUDGET_WARNED: set[str] = set()


def reset_unpriced_budget_warnings() -> None:
    """Clear the once-per-process unpriced-spend warning guard (test helper)."""
    _UNPRICED_BUDGET_WARNED.clear()


#: Ledger provider keys that are not the catalog's name for the same provider.
#: ``effgen models refresh`` only accepts the catalog name, so a heads-up that
#: names the ledger key would hand the user a command that fails.
_REFRESH_PROVIDER_ALIASES = {"hf_inference": "hf"}


def _refresh_hint(provider: str) -> str:
    """The ``effgen models refresh`` sentence for *provider*, when it has one.

    Empty for a provider the command does not accept (a local engine, a name
    with no published catalog), so the heads-up never suggests a command that
    would come back with "Unknown provider".
    """
    name = _REFRESH_PROVIDER_ALIASES.get(provider, provider)
    try:
        from effgen.models._catalog import known_providers

        refreshable = name in known_providers()
    except Exception:  # pragma: no cover - catalog import is best-effort
        logger.debug("Could not list refreshable providers", exc_info=True)
        return ""
    if not refreshable:
        return ""
    return f" Run `effgen models refresh --provider {name}` to pick up a published rate."


def _warn_unpriced_spend(provider: str, model: str) -> None:
    """Say once that *provider*/*model* spend cannot be counted toward a budget.

    A model the catalog has no rate for is still billed by the provider, so a
    configured budget silently undercounts it.  The budget is not enforced on a
    price we do not know — the call is allowed — but the user is told which
    model is missing from the total, once per process, so the heads-up does not
    repeat on every call of a loop.
    """
    key = f"{provider.lower()}:{model}"
    if key in _UNPRICED_BUDGET_WARNED:
        return
    _UNPRICED_BUDGET_WARNED.add(key)
    warnings.warn(
        f"effGen budget: no published price for '{key}', so this call's spend is "
        f"not counted toward the configured budget."
        f"{_refresh_hint(provider.lower())}",
        UserWarning,
        stacklevel=5,
    )


@dataclass
class _ModelStats:
    """Per-(provider, model) accumulated stats."""
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    total_cost_usd: float = 0.0
    #: Calls whose price the catalog does not publish. They contribute tokens
    #: but no money, so a caller can tell "spent nothing" from "cost unknown".
    unpriced_requests: int = 0


class CostTracker:
    """Thread-safe cost tracker with optional SQLite persistence and budget alerts.

    Use :meth:`get` for the process-global singleton (default: SQLite-backed).
    Pass ``storage=None`` to get a pure in-memory instance (back-compat).

    Budget alerts
    -------------
    Set budgets via ``effgen config set budget.daily 1.0`` (writes
    ``~/.effgen/budget.json``).  On each :meth:`record` call:

    - At 80% of daily or monthly budget → :class:`UserWarning` is emitted.
    - At 100% → :class:`~effgen.models.errors.BudgetExceededError` is raised
      for paid calls. Zero-cost calls are still allowed so router failover can
      land on free-tier providers.
    - A call on a model with no published price is allowed at any budget level
      and emits a :class:`UserWarning` once per model saying its spend is not
      counted, rather than being silently added as $0.

    The router catches ``BudgetExceededError`` and fails over to a free-tier
    provider when available.

    Example::

        tracker = CostTracker.get()
        cost = tracker.record("cerebras", "llama3.1-8b", 50, 20)
        # 0.000000 for the Cerebras free tier; None for a model with no rate
        print(f"Cost: ${cost:.6f}" if cost is not None else "Cost: unpriced")
    """

    _instance: "CostTracker | None" = None
    _lock: threading.Lock = threading.Lock()

    #: How long a period-spend reading stays usable, in seconds.
    _PERIOD_SPEND_TTL_S = 1.0

    def __init__(
        self,
        storage: "SQLiteCostStore | None" = None,
    ) -> None:
        self._data: dict[tuple[str, str], _ModelStats] = {}
        self._lock = threading.Lock()
        self._storage = storage
        #: period -> (monotonic time of the reading, spend). See
        #: :meth:`_period_spend` for why a reading may be reused.
        self._period_spend_cache: dict[str, tuple[float, float]] = {}
        self._period_spend_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "CostTracker":
        """Return the process-global CostTracker instance (SQLite-backed)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    from effgen.models._cost_store import SQLiteCostStore
                    cls._instance = cls(storage=SQLiteCostStore())
        return cls._instance

    @classmethod
    def instance(cls) -> "CostTracker":
        """Backward-compatible alias for :meth:`get`."""
        return cls.get()

    @classmethod
    def get_instance(cls) -> "CostTracker":
        """Backward-compatible alias for :meth:`get`."""
        return cls.get()

    @classmethod
    def reset(cls) -> None:
        """Reset the global tracker (useful in tests)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float | None:
        """Record a completed API call and return the USD cost.

        Args:
            provider: Provider name, e.g. ``"cerebras"``.
            model: Model ID, e.g. ``"llama3.1-8b"``.
            prompt_tokens: Tokens in the prompt (from API usage).
            completion_tokens: Tokens in the completion (from API usage).
            input_tokens: Alias for ``prompt_tokens`` used by some adapters.
            output_tokens: Alias for ``completion_tokens`` used by some adapters.
            cost_usd: Optional precomputed USD cost override for providers that
                do not bill by prompt/completion token price.
            cached_tokens: How many of ``prompt_tokens`` the provider served
                from its cache. Priced at the published cached rate where there
                is one, and at the input rate where there is not.
            cache_write_tokens: How many of ``prompt_tokens`` were written into
                the cache, which the providers that report writes bill above
                the input rate.

        Returns:
            USD cost for this call — ``0.0`` when the call really was free (a
            free tier), or ``None`` when the catalog publishes no price for
            this model, so no cost can be stated.  The token counts are still
            recorded in that case; only the money is unknown.

        Raises:
            BudgetExceededError: If daily budget is configured and exceeded.
        """
        import time

        if input_tokens is not None:
            prompt_tokens = input_tokens
        if output_tokens is not None:
            completion_tokens = output_tokens

        cost: float | None
        if cost_usd is not None:
            cost = float(cost_usd)
        else:
            cost = call_cost(
                provider,
                model,
                prompt_tokens,
                completion_tokens,
                cached_tokens=cached_tokens,
                cache_write_tokens=cache_write_tokens,
            )

        key = (provider.lower(), model)
        with self._lock:
            if key not in self._data:
                self._data[key] = _ModelStats(provider=provider, model=model)
            stats = self._data[key]
            stats.prompt_tokens += prompt_tokens
            stats.completion_tokens += completion_tokens
            stats.requests += 1
            if cost is None:
                stats.unpriced_requests += 1
            else:
                stats.total_cost_usd += cost

        if self._storage is not None:
            try:
                self._storage.insert(
                    provider=provider,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost or 0.0,
                    timestamp=time.time(),
                )
            except Exception as exc:
                logger.warning("CostStore insert failed: %s", exc)

        # Spend just landed. Folding it into every cached period reading —
        # before :meth:`_check_budget` reads it below — is what keeps the cache
        # in :meth:`_period_spend` a latency optimization rather than a hole in
        # the budget: the post-spend check sees this call's cost at once, and
        # no ledger read is paid to learn a number this process already knows.
        if cost:
            self._add_period_spend(cost)

        logger.debug(
            "CostTracker.record %s/%s: prompt=%d completion=%d cost=%s",
            provider, model, prompt_tokens, completion_tokens,
            "unpriced" if cost is None else f"${cost:.6f}",
        )

        self._check_budget(provider=provider, model=model, cost=cost)
        return cost

    def check_preflight(self, provider: str, model: str) -> None:
        """Refuse to start a call when a configured budget is already at/over its cap.

        Called before the provider call is made (see
        :func:`effgen.models.base._preflight_budget_check`), so a call refused
        here is never billed. This is a pre-existing-spend check only: it
        cannot foresee the cost of the call about to be made, so a call that
        pushes spend past the cap for the first time is still allowed through
        here and is caught after the fact by :meth:`_check_budget` inside
        :meth:`record`.
        """
        budget_cfg = _load_budget()
        for period, budget_usd in _configured_budgets(budget_cfg):
            spend = self._period_spend(period)
            if spend >= budget_usd:
                from effgen.models.errors import BudgetExceededError
                raise BudgetExceededError(
                    budget_usd=budget_usd,
                    actual_usd=spend,
                    period=period,
                    provider=provider,
                    model=model,
                )

    def _check_budget(self, provider: str, model: str, cost: float | None) -> None:
        """Emit a warning or raise BudgetExceededError for configured budgets.

        A call whose price is unknown (``cost is None`` — the catalog publishes
        no rate for the model) is allowed through: the gate refuses spend it can
        measure, and refusing on a price nobody published would block a model
        that may well be free.  Instead the user is told once, per model, that
        this spend is missing from the budget total.
        """
        budget_cfg = _load_budget()
        budgets = _configured_budgets(budget_cfg)
        if cost is None:
            if budgets:
                _warn_unpriced_spend(provider, model)
            return
        for period, budget_usd in budgets:
            spend = self._period_spend(period)
            previous_spend = max(0.0, spend - cost)
            ratio = spend / budget_usd

            if ratio >= 1.0:
                if cost <= 0.0:
                    continue
                from effgen.models.errors import BudgetExceededError
                raise BudgetExceededError(
                    budget_usd=budget_usd,
                    actual_usd=spend,
                    period=period,
                    provider=provider,
                    model=model,
                )

            if previous_spend < budget_usd * 0.8 <= spend:
                warnings.warn(
                    f"effGen {period} budget warning: {_format_usd(spend)} / "
                    f"{_format_usd(budget_usd)} ({ratio * 100:.0f}%) spent.",
                    UserWarning,
                    stacklevel=4,
                )

    def _period_spend(self, period: str) -> float:
        """Return spend for *period*, from a reading at most a second old.

        The caller is the per-call budget check, and a run with several agents
        in flight asks this same question many times a second. The ledger is
        read at most once a second per process; between readings, spend this
        process records is added to the cached number (see :meth:`record`), so
        the total the post-spend check sees is exact for everything this
        process has spent and at most a second behind what other processes
        writing the same ledger have added.

        Reusing a reading is safe because of what the preflight is. It guards
        spend that has **already** happened — it cannot know what the call it
        is about to allow will cost — and it is backed by the check inside
        :meth:`record`, which runs after the real cost is known against a total
        that already includes it. So the cap is enforced on the number that
        matters either way; the cache only decides whether a call that was
        going to be allowed pays a database round trip to find that out. What
        a reading can miss is spend another process landed within the last
        second, which is the same window the preflight has anyway, and
        :meth:`_check_budget` still refuses the next call.

        The ledger read runs under the lock, so a burst of callers arriving as
        a reading expires pays for one read, not one each.
        """
        import time

        with self._period_spend_lock:
            cached = self._period_spend_cache.get(period)
            if cached is not None and time.monotonic() - cached[0] < self._PERIOD_SPEND_TTL_S:
                logger.debug("budget preflight: period spend served from cache (%s)",
                             period)
                return cached[1]
            spend = self._period_spend_uncached(period)
            self._period_spend_cache[period] = (time.monotonic(), spend)
            return spend

    def _period_spend_uncached(self, period: str) -> float:
        """Read spend for *period* from storage, falling back to memory.

        Prefers the store's aggregate, which returns the one number the budget
        check needs. A store that does not implement it — a third-party
        implementation of the same duck type — keeps working through the
        row-returning queries, at the cost of building an object per row.
        """
        if self._storage is not None:
            try:
                if period == "daily":
                    spend_today = getattr(self._storage, "spend_today", None)
                    if spend_today is not None:
                        logger.debug("budget preflight: period spend summed in store "
                                     "(%s)", period)
                        return float(spend_today())
                    logger.debug("budget preflight: period spend summed from rows (%s)",
                                 period)
                    return sum(e.cost_usd for e in self._storage.query_today())
                if period == "monthly":
                    spend_month = getattr(self._storage, "spend_month", None)
                    if spend_month is not None:
                        logger.debug("budget preflight: period spend summed in store "
                                     "(%s)", period)
                        return float(spend_month())
                    logger.debug("budget preflight: period spend summed from rows (%s)",
                                 period)
                    query_month = getattr(self._storage, "query_month", None)
                    if query_month is not None:
                        return sum(e.cost_usd for e in query_month())
                    import time

                    return sum(e.cost_usd for e in self._storage.query_since(
                        time.time() - 30 * 86400.0
                    ))
            except Exception:
                logger.warning("CostStore budget query failed; falling back to memory")
        return self.total_cost()

    def _add_period_spend(self, cost: float) -> None:
        """Fold spend this process just recorded into every cached reading.

        A recorded call is stamped with the current time, so it falls inside
        every period window; adding its cost keeps each cached total exact for
        this process without a ledger read. If the insert behind it failed,
        the reading overstates by one call's cost until it next expires, which
        is the safe direction for a cap.
        """
        with self._period_spend_lock:
            for period, (taken, spend) in self._period_spend_cache.items():
                self._period_spend_cache[period] = (taken, spend + cost)

    def _invalidate_period_spend(self) -> None:
        """Drop every cached period reading, forcing the next one to the ledger."""
        with self._period_spend_lock:
            self._period_spend_cache.clear()

    def total_cost(self, provider: str | None = None, model: str | None = None) -> float:
        """Return total USD cost accumulated in memory, optionally filtered.

        Args:
            provider: Filter to this provider (None = all providers).
            model: Filter to this model (None = all models).

        Returns:
            Cumulative USD cost.
        """
        with self._lock:
            total = 0.0
            for (prov, mod), stats in self._data.items():
                if provider and prov != provider.lower():
                    continue
                if model and mod != model:
                    continue
                total += stats.total_cost_usd
        return total

    def total_tokens(self, provider: str | None = None, model: str | None = None) -> dict[str, int]:
        """Return total token counts, optionally filtered.

        Returns:
            Dict with keys ``prompt``, ``completion``, ``total``.
        """
        prompt = completion = 0
        with self._lock:
            for (prov, mod), stats in self._data.items():
                if provider and prov != provider.lower():
                    continue
                if model and mod != model:
                    continue
                prompt += stats.prompt_tokens
                completion += stats.completion_tokens
        return {"prompt": prompt, "completion": completion, "total": prompt + completion}

    def summary(self) -> list[dict]:
        """Return a list of per-(provider, model) usage summaries (in-memory).

        Returns:
            List of dicts with keys: ``provider``, ``model``, ``requests``,
            ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
            ``cost_usd``, ``unpriced_requests`` and ``pricing``.  ``cost_usd``
            is ``None`` when every recorded call was on a model with no
            published price — the tokens are known, the money is not.
        """
        with self._lock:
            rows = []
            for stats in self._data.values():
                all_unpriced = (
                    stats.requests > 0 and stats.unpriced_requests == stats.requests
                )
                rows.append({
                    "provider": stats.provider,
                    "model": stats.model,
                    "requests": stats.requests,
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "total_tokens": stats.prompt_tokens + stats.completion_tokens,
                    "cost_usd": None if all_unpriced else round(stats.total_cost_usd, 8),
                    "unpriced_requests": stats.unpriced_requests,
                    "pricing": pricing_status(stats.provider, stats.model),
                })
        return rows

    def reset_stats(self) -> None:
        """Clear all accumulated in-memory stats (does not reset singleton or DB)."""
        with self._lock:
            self._data.clear()


if TYPE_CHECKING:
    from effgen.models._cost_store import SQLiteCostStore
