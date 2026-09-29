"""What a model actually does when it is handed a tool, measured once and kept.

A model's declared tool-calling support says how tool definitions reach it —
as a request parameter, or rendered into a chat template — not whether it uses
them. Two behaviours hide behind the same declaration:

- a model that answers from memory while holding a tool that has the answer;
- a model that calls the tool, but in a form the serving stack does not carry
  back as a structured call, so the call repeats until a guard ends the run.

:func:`probe_tool_calling` measures both on a fixed set of questions about
fictional things, answerable only through a stub search tool. Each question is
one ordinary :class:`~effgen.core.agent.Agent` run, first in the native frame
(tool definitions on the request) and — only when native calls go unresolved —
again in the ReAct text frame. The outcome of each run is ``resolved`` (a call,
and the answer carries the tool's value), ``unresolved`` (a call, and the answer
does not, or a guard stopped the run) or ``skipped`` (no call).

The result is a :class:`ToolCallingProbe`, stored in one JSON file together
with facts the framework learns from real requests (a provider that rejects
stop sequences beside tools, or a ``reasoning_effort`` field). The file is
``$EFFGEN_CAPABILITY_CACHE`` when set (``off`` keeps everything in memory), else
``$EFFGEN_HOME/capabilities.json``, else ``~/.effgen/capabilities.json``.
``effgen doctor`` shows it; ``effgen doctor --probe MODEL --refresh`` re-runs a
probe.

Only an adapter that answers :meth:`~effgen.models.base.BaseModel.capability_key`
is probed — a model served behind a URL, or a local engine. The first-party
cloud adapters answer ``None`` and are never probed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "PROBE_VERSION",
    "ToolCallingProbe",
    "capability_store_path",
    "derive_defaults",
    "learned_fact",
    "probe_for_agent",
    "probe_tool_calling",
    "read_store",
    "remember_fact",
]

#: Version of the item set and the rules below. Part of every probe key, so a
#: change to either re-probes instead of reading an entry measured differently.
PROBE_VERSION = 1

#: Native runs that called a tool and did not resolve, at or above which the
#: text frame is measured too.
UNRESOLVED_THRESHOLD = 3
#: How many more items the text frame must resolve than the native frame before
#: ``auto`` resolves to the text frame.
TEXT_FRAME_MARGIN = 3
#: Native runs that answered with no call, at or above which information tools
#: become ones a run may not answer without calling.
SKIP_THRESHOLD = 1

#: Upper bounds on one probe. Over either, the probe reports nothing.
MAX_PROBE_REQUESTS = 48
MAX_PROBE_SECONDS = 120.0

#: Age after which a stored probe result is measured again.
PROBE_MAX_AGE_S = 30 * 24 * 3600
#: Age after which a learned rejection is forgotten and tried again.
LEARNED_MAX_AGE_S = 7 * 24 * 3600
#: How long a probe that could not run is not retried in this process.
FAILED_RETRY_S = 600.0

_SEED = 7
_MAX_TOKENS = 256
_MAX_ITERATIONS = 3

#: ``(shape, question, what the search tool returns, tokens the answer must carry)``.
#: Every entity is fictional, so a right answer can only come from the tool.
_ITEMS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("year", "In what year was the town of Veldmark-on-Oster founded?",
     "Veldmark-on-Oster is a small market town. It was founded in 1712 by Hanseatic traders.",
     ("1712",)),
    ("person", "Who designed the Harrowgate Lantern Bridge in Coldwell?",
     "The Harrowgate Lantern Bridge in Coldwell was designed by the engineer Mirela Oskarsdottir.",
     ("oskarsdottir",)),
    ("count", "What was the population of the island of Tessavar at its 2020 census?",
     "Tessavar census 2020: population 4,381.",
     ("4381",)),
    ("place", "In which city does the Brannock Ceramics Company have its headquarters?",
     "The Brannock Ceramics Company is headquartered in the city of Quillhaven.",
     ("quillhaven",)),
    ("code", "What is the station code of Pellbury Junction railway station?",
     "Pellbury Junction railway station (station code PJX) opened on the Varrow line.",
     ("pjx",)),
    ("date", "On what date did the Selvane Canal open to traffic?",
     "The Selvane Canal opened to traffic on 14 March 1897.",
     ("14", "march", "1897")),
    ("quantity", "How long is the Mount Pellan railway tunnel, in metres?",
     "The Mount Pellan railway tunnel is 2,764 metres long.",
     ("2764",)),
    ("name", "What is the name of the flagship ferry of the Kestrel Isles Line?",
     "The flagship ferry of the Kestrel Isles Line is the MV Aurora Venn.",
     ("auroravenn",)),
)

#: Stop reasons that mean a guard ended the run rather than the model answering.
_GUARD_STOPS = frozenset({
    "max_iterations_partial", "max_iterations_exhausted", "loop_detected",
    "repeated_tool_result", "null_final_from_model", "written_tool_call",
    "tool_failed",
})

#: Stop reasons that mean the run could not be carried out at all.
_FAILED_STOPS = frozenset({"generation_failed", "run_failed", "empty_task", "guardrail_blocked"})

RESOLVED = "resolved"
UNRESOLVED = "unresolved"
SKIPPED = "skipped"

#: ``strategy`` value for "resolve ``auto`` as the model's declaration says".
STRATEGY_DECLARED = "declared"
#: ``strategy`` value for "resolve ``auto`` to the ReAct text frame".
STRATEGY_REACT = "react"


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallingProbe:
    """What one model did with a tool, and the defaults derived from it.

    Attributes:
        key: The cache key the adapter answered (weights, template, endpoint).
        model: The model's name.
        endpoint: The URL it is served at, or ``None`` for a local engine.
        backend: The adapter or engine class that answered.
        native: Outcome counts in the native frame: ``resolved``,
            ``unresolved`` and ``skipped``.
        text: The same counts in the ReAct text frame, or ``None`` when it was
            not measured (it is only measured when native calls go unresolved).
        strategy: ``"react"`` when ``auto`` resolves to the text frame for this
            model, ``"declared"`` when it resolves as the model declares.
        required_categories: Tool categories a run may not answer without
            calling, for this model, when the caller stated no ``tool_use``.
        items: One ``(frame, shape, outcome)`` per run, in order.
        requests: Model requests the probe made.
        prompt_tokens: Prompt tokens those requests sent.
        completion_tokens: Completion tokens they returned.
        wall_s: Seconds the probe took.
        probed_at: When it ran (seconds since the epoch).
        probe_version: :data:`PROBE_VERSION` at the time.
        effgen_version: The effGen version that ran it.
        source: ``"ran"`` when this call measured it, ``"cached"`` when it was
            read from the store. Not stored.
    """

    key: str
    model: str
    endpoint: str | None
    backend: str
    native: dict[str, int]
    text: dict[str, int] | None
    strategy: str
    required_categories: tuple[str, ...]
    items: tuple[tuple[str, str, str], ...] = ()
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    probed_at: float = 0.0
    probe_version: int = PROBE_VERSION
    effgen_version: str = ""
    source: str = field(default="ran", compare=False)

    def to_dict(self) -> dict[str, Any]:
        """The stored form (everything but :attr:`source`)."""
        return {
            "key": self.key, "model": self.model, "endpoint": self.endpoint,
            "backend": self.backend, "native": dict(self.native),
            "text": dict(self.text) if self.text is not None else None,
            "strategy": self.strategy,
            "required_categories": list(self.required_categories),
            "items": [list(i) for i in self.items],
            "requests": self.requests, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "wall_s": self.wall_s,
            "probed_at": self.probed_at, "probe_version": self.probe_version,
            "effgen_version": self.effgen_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "cached") -> ToolCallingProbe:
        """Rebuild a stored probe.

        Raises:
            KeyError, TypeError, ValueError: When *data* is not a stored probe.
        """
        return cls(
            key=str(data["key"]), model=str(data["model"]),
            endpoint=data.get("endpoint"), backend=str(data.get("backend", "")),
            native={k: int(v) for k, v in dict(data["native"]).items()},
            text=({k: int(v) for k, v in dict(data["text"]).items()}
                  if data.get("text") is not None else None),
            strategy=str(data["strategy"]),
            required_categories=tuple(str(c) for c in data.get("required_categories") or ()),
            items=tuple(tuple(str(x) for x in i) for i in data.get("items") or ()),  # type: ignore[misc]
            requests=int(data.get("requests", 0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            wall_s=float(data.get("wall_s", 0.0)),
            probed_at=float(data.get("probed_at", 0.0)),
            probe_version=int(data.get("probe_version", 0)),
            effgen_version=str(data.get("effgen_version", "")),
            source=source,
        )

    def summary(self) -> str:
        """One line: the counts per frame."""
        n = self.native
        line = (f"native resolved={n.get(RESOLVED, 0)} unresolved={n.get(UNRESOLVED, 0)} "
                f"skipped={n.get(SKIPPED, 0)}")
        if self.text is not None:
            line += f" text resolved={self.text.get(RESOLVED, 0)}"
        return line


def derive_defaults(
    native: dict[str, int], text: dict[str, int] | None,
) -> tuple[str, tuple[str, ...]]:
    """The strategy and must-call categories a probe's counts imply.

    - Native calls that do not resolve, where the text frame resolves clearly
      more of the same items: ``auto`` resolves to the text frame.
    - Any native run that answered without calling the search tool: tools of
      the information-retrieval category become ones a run may not answer
      without calling. Computation and code tools are never moved.

    Args:
        native: Native-frame outcome counts.
        text: Text-frame outcome counts, or ``None`` when not measured.

    Returns:
        ``(strategy, required_categories)``.
    """
    strategy = STRATEGY_DECLARED
    if (
        text is not None
        and native.get(UNRESOLVED, 0) >= UNRESOLVED_THRESHOLD
        and text.get(RESOLVED, 0) >= native.get(RESOLVED, 0) + TEXT_FRAME_MARGIN
    ):
        strategy = STRATEGY_REACT
    categories: tuple[str, ...] = ()
    if native.get(SKIPPED, 0) >= SKIP_THRESHOLD:
        categories = ("information_retrieval",)
    return strategy, categories


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

_STORE_LOCK = threading.Lock()
#: In-process memo of the store: ``path -> (mtime_ns, size, data)``.
_MEMO: dict[str, tuple[int, int, dict[str, Any]]] = {}
#: The store used when persistence is off (``EFFGEN_CAPABILITY_CACHE=off``).
_MEMORY_STORE: dict[str, Any] = {"probes": {}, "learned": {}}
_CORRUPT_WARNED: set[str] = set()


def capability_store_path() -> Path | None:
    """Where the store lives, or ``None`` when persistence is off."""
    explicit = os.environ.get("EFFGEN_CAPABILITY_CACHE", "").strip()
    if explicit:
        if explicit.lower() in ("off", "0", "none", "false"):
            return None
        return Path(explicit).expanduser()
    home = os.environ.get("EFFGEN_HOME", "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".effgen"
    return base / "capabilities.json"


def _empty() -> dict[str, Any]:
    return {"version": 1, "probes": {}, "learned": {}}


def read_store() -> dict[str, Any]:
    """The whole store: ``{"probes": {key: entry}, "learned": {identity: facts}}``.

    A missing file is an empty store. A file that cannot be read or parsed is
    logged once and treated as empty; the next write replaces it.
    """
    path = capability_store_path()
    if path is None:
        return _MEMORY_STORE
    try:
        st = path.stat()
    except OSError:
        return _empty()
    memo = _MEMO.get(str(path))
    if memo is not None and memo[0] == st.st_mtime_ns and memo[1] == st.st_size:
        return memo[2]
    data: dict[str, Any]
    problem: str | None = None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        problem, data = str(exc), _empty()
    if problem is None and not isinstance(data, dict):
        problem, data = "not a JSON object", _empty()
    if problem is None:
        data.setdefault("probes", {})
        data.setdefault("learned", {})
        if not isinstance(data["probes"], dict) or not isinstance(data["learned"], dict):
            problem, data = "no 'probes' and 'learned' objects", _empty()
    if problem is not None:
        if str(path) not in _CORRUPT_WARNED:
            _CORRUPT_WARNED.add(str(path))
            logger.warning(
                "capability store: %s could not be read (%s); ignoring it — it is "
                "rewritten on the next probe", path, problem,
            )
        data = _empty()
    _MEMO[str(path)] = (st.st_mtime_ns, st.st_size, data)
    return data


@contextlib.contextmanager
def _file_lock(lock_path: Path, timeout: float) -> Iterator[bool]:
    """An advisory lock on *lock_path*; yields whether it was acquired."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - platforms without fcntl
        yield True
        return
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+")  # noqa: SIM115 - held for the block
    except OSError:
        yield True
        return
    acquired = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
        yield acquired
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _update_store(mutate: Any) -> None:
    """Apply *mutate(data)* to the store and write it back atomically."""
    path = capability_store_path()
    with _STORE_LOCK:
        if path is None:
            mutate(_MEMORY_STORE)
            return
        with _file_lock(path.with_name(path.name + ".lock"), timeout=10.0):
            data = read_store()
            data = json.loads(json.dumps(data))  # a private copy to mutate
            mutate(data)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=".capabilities-", dir=str(path.parent))
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=1, sort_keys=True)
                os.replace(tmp, path)
            except OSError as exc:
                logger.warning("capability store: could not write %s (%s)", path, exc)
                with contextlib.suppress(Exception):
                    os.unlink(tmp)
                return
            _MEMO.pop(str(path), None)


# ---------------------------------------------------------------------------
# Learned facts (from real requests, never from a request made on purpose)
# ---------------------------------------------------------------------------


def _unwrap(model: Any) -> Any:
    inner = getattr(model, "_inner", None)
    return inner if inner is not None else model


def model_identity(model: Any) -> str:
    """``"<endpoint or backend>|<model name>"`` — what a learned fact is kept under."""
    model = _unwrap(model)
    where = (getattr(model, "base_url", None) or getattr(model, "_provider", None)
             or type(model).__name__)
    return f"{where}|{getattr(model, 'model_name', '') or ''}"


def learned_fact(model: Any, name: str) -> bool | None:
    """What was learned about *model* for capability *name*, or ``None``.

    Read on every turn that could use it, so it is a dictionary lookup against
    an in-process copy of the store that is re-read only when the file changes.
    """
    if model is None:
        return None
    try:
        facts = read_store().get("learned", {}).get(model_identity(model)) or {}
        fact = facts.get(name)
        if not isinstance(fact, dict):
            return None
        if time.time() - float(fact.get("at", 0)) > LEARNED_MAX_AGE_S:
            return None
        value = fact.get("value")
        return value if isinstance(value, bool) else None
    except Exception:  # noqa: BLE001 - an unreadable fact is no fact
        logger.debug("learned capability lookup failed", exc_info=True)
        return None


def remember_fact(model: Any, name: str, value: bool, *, detail: str = "") -> None:
    """Record that *model* answered capability *name* with *value*.

    Args:
        model: The model the fact is about.
        name: The capability, e.g. ``"stop_with_tools"``.
        value: What the provider was seen to do.
        detail: A short note on what was observed, shown by ``effgen doctor``.
    """
    identity = model_identity(model)

    def mutate(data: dict[str, Any]) -> None:
        learned = data.setdefault("learned", {})
        facts = learned.setdefault(identity, {})
        facts[name] = {"value": bool(value), "at": time.time(), "detail": detail}

    try:
        _update_store(mutate)
    except Exception:  # noqa: BLE001 - failing to remember costs one retry later
        logger.debug("could not record a learned capability", exc_info=True)


# ---------------------------------------------------------------------------
# Running the probe
# ---------------------------------------------------------------------------

_KEY_LOCKS: dict[str, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()
_FAILED: dict[str, float] = {}


def _key_lock(key: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = _KEY_LOCKS[key] = threading.Lock()
        return lock


def _norm(text: str) -> str:
    return re.sub(r"[\s,]+", "", (text or "").lower())


def _full_key(model: Any) -> str | None:
    try:
        key = model.capability_key()
    except Exception:  # noqa: BLE001 - an adapter that cannot say is not probed
        logger.debug("capability_key failed", exc_info=True)
        return None
    if not key:
        return None
    return hashlib.sha256(f"{key}|probe-v{PROBE_VERSION}".encode()).hexdigest()[:32]


class _ProbeAborted(Exception):
    pass


class _Run:
    """One probe's counters, shared with the worker thread."""

    def __init__(self) -> None:
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.items: list[tuple[str, str, str]] = []
        self.error: str | None = None
        self.abort = False


def _search_stub(fact: str, calls: list[int]) -> Any:
    from ..tools.base_tool import ToolCategory
    from ..tools.function_tool import tool

    @tool(name="web_search", category=ToolCategory.INFORMATION_RETRIEVAL)
    def web_search(query: str) -> str:
        """Search the web and return the most relevant result."""
        calls.append(1)
        return f"[Result 1] {fact}"

    return web_search


def _run_frame(model: Any, mode: str, frame: str, run: _Run) -> dict[str, int]:
    from ..core.agent import Agent
    from ..core.agent_config import AgentConfig

    counts = {RESOLVED: 0, UNRESOLVED: 0, SKIPPED: 0}
    for shape, question, fact, needles in _ITEMS:
        if run.abort:
            raise _ProbeAborted("stopped")
        if run.requests >= MAX_PROBE_REQUESTS:
            raise _ProbeAborted(f"over the budget of {MAX_PROBE_REQUESTS} requests")
        calls: list[int] = []
        agent = Agent(config=AgentConfig(
            name="capability_probe", model=model, tools=[_search_stub(fact, calls)],
            tool_calling_mode=mode, temperature=0.0, seed=_SEED,
            max_tokens=_MAX_TOKENS, max_iterations=_MAX_ITERATIONS,
            capability_probe=False, raise_on_error=False, enable_memory=False,
            enable_sub_agents=False,
            # The thresholds were calibrated against the loop without the
            # answer request or the closing request; a probe that measured
            # with them would read a different model.
            max_turns_without_progress=None,
            # Likewise calibrated against the strict reader.
            recover_lost_tool_calls=False,
        ))
        try:
            response = agent.run(question)
        finally:
            agent.close()
        stop = getattr(response, "stop_reason", None)
        if stop in _FAILED_STOPS:
            # A refused or failing request is not something the model did.
            detail = (getattr(response, "metadata", None) or {}).get("error") or stop
            raise _ProbeAborted(f"a request failed: {detail}")
        ledger = getattr(response, "ledger", None)
        run.requests += int(getattr(ledger, "llm_calls", 0) or 0)
        run.prompt_tokens += int(getattr(ledger, "prompt_tokens", 0) or 0)
        run.completion_tokens += int(getattr(ledger, "completion_tokens", 0) or 0)
        answer = _norm(str(getattr(response, "output", "") or ""))
        if not calls:
            outcome = SKIPPED
        elif all(n in answer for n in needles) and stop not in _GUARD_STOPS:
            outcome = RESOLVED
        else:
            outcome = UNRESOLVED
        counts[outcome] += 1
        run.items.append((frame, shape, outcome))
    return counts


def _measure(model: Any, key: str) -> ToolCallingProbe | None:
    """Run the probe under its budget; ``None`` when it could not finish."""
    from .. import __version__

    run = _Run()
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            native = _run_frame(model, "native", "native", run)
            text = None
            if native[UNRESOLVED] >= UNRESOLVED_THRESHOLD:
                text = _run_frame(model, "react", "text", run)
            result["native"], result["text"] = native, text
        except Exception as exc:  # noqa: BLE001 - any failure means "no probe"
            run.error = f"{type(exc).__name__}: {exc}"[:300]

    started = time.monotonic()
    worker = threading.Thread(target=work, name="effgen-capability-probe", daemon=True)
    worker.start()
    worker.join(MAX_PROBE_SECONDS)
    name = getattr(_unwrap(model), "model_name", "?")
    if worker.is_alive():
        run.abort = True
        logger.warning(
            "capability probe: could not run for %s (over %.0f s); using the declared "
            "capability", name, MAX_PROBE_SECONDS,
        )
        return None
    if run.error is not None or "native" not in result:
        logger.warning(
            "capability probe: could not run for %s (%s); using the declared capability",
            name, run.error or "no result",
        )
        return None
    strategy, categories = derive_defaults(result["native"], result["text"])
    inner = _unwrap(model)
    return ToolCallingProbe(
        key=key, model=str(name), endpoint=getattr(inner, "base_url", None),
        backend=type(inner).__name__, native=result["native"], text=result["text"],
        strategy=strategy, required_categories=categories, items=tuple(run.items),
        requests=run.requests, prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
        wall_s=round(time.monotonic() - started, 2), probed_at=time.time(),
        effgen_version=__version__,
    )


def _cached(key: str) -> ToolCallingProbe | None:
    entry = read_store().get("probes", {}).get(key)
    if not isinstance(entry, dict):
        return None
    try:
        probe = ToolCallingProbe.from_dict(entry)
    except (KeyError, TypeError, ValueError):
        return None
    if probe.probe_version != PROBE_VERSION:
        return None
    if time.time() - probe.probed_at > PROBE_MAX_AGE_S:
        return None
    return probe


def probe_tool_calling(model: Any, *, refresh: bool = False) -> ToolCallingProbe | None:
    """Measure how *model* uses a tool, or return what was measured before.

    At most one probe runs per key at a time, across threads and processes:
    the others wait for it and read its entry.

    Args:
        model: A loaded model. Only one whose adapter answers
            :meth:`~effgen.models.base.BaseModel.capability_key` is probed.
        refresh: Measure again even when a stored result exists.

    Returns:
        The probe, with ``source`` ``"cached"`` or ``"ran"``; ``None`` when the
        model is not one that is probed, or when the probe could not run (a
        refused or failing request, or over its budget), in which case nothing
        is stored and one WARNING says why.
    """
    key = _full_key(model)
    if key is None:
        return None
    if not refresh:
        hit = _cached(key)
        if hit is not None:
            return hit
        failed_at = _FAILED.get(key)
        if failed_at is not None and time.monotonic() - failed_at < FAILED_RETRY_S:
            return None
    with _key_lock(key):
        if not refresh:
            hit = _cached(key)
            if hit is not None:
                return hit
        path = capability_store_path()
        lock_path = (path.with_name(f"{path.name}.{key[:16]}.probe.lock")
                     if path is not None else None)
        with (_file_lock(lock_path, MAX_PROBE_SECONDS + 30.0) if lock_path is not None
              else contextlib.nullcontext(True)):
            if not refresh:
                hit = _cached(key)
                if hit is not None:
                    return hit
            probe = _measure(model, key)
            if probe is None:
                _FAILED[key] = time.monotonic()
                return None
            _FAILED.pop(key, None)
            entry = probe.to_dict()

            def mutate(data: dict[str, Any]) -> None:
                data.setdefault("probes", {})[key] = entry

            _update_store(mutate)
            return probe


def probe_for_agent(model: Any) -> ToolCallingProbe | None:
    """The probe an agent at ``tool_calling_mode="auto"`` resolves against.

    ``None`` whenever the model is not probed; any unexpected failure is
    logged and also answers ``None``, so building an agent never fails here.
    """
    try:
        return probe_tool_calling(model)
    except Exception as exc:  # noqa: BLE001 - the declared capability still works
        logger.warning(
            "capability probe: could not run for %s (%s); using the declared capability",
            getattr(_unwrap(model), "model_name", "?"), exc,
        )
        return None


def local_capability_key(engine: Any) -> str | None:
    """The key of a local engine: weights, their revision and the chat template.

    ``None`` until the engine has loaded its tokenizer — a model that is not
    loaded yet is not probed.
    """
    tokenizer = getattr(engine, "_hf_tokenizer", None) or getattr(engine, "tokenizer", None)
    if tokenizer is None or not getattr(engine, "_is_loaded", False):
        return None
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        template = json.dumps(template, sort_keys=True, default=str)
    config = getattr(getattr(engine, "model", None), "config", None)
    revision = getattr(config, "_commit_hash", None) or ""
    digest = hashlib.sha256(template.encode()).hexdigest()
    return f"{type(engine).__name__}|{getattr(engine, 'model_name', '')}|{revision}|{digest}"

