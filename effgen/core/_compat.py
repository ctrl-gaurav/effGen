"""Backward-compatibility helpers for loading serialized data across versions.

Saved artifacts (sessions, agent state, memory snapshots, configs) may have been
written by an older or newer effGen build whose field set differs slightly from the
running one. Naively splatting such a dict into a constructor raises a cryptic
``TypeError: __init__() got an unexpected keyword argument 'x'`` and makes the file
unloadable. The helpers here load forgivingly instead:

* unknown fields (renamed/removed since the file was written) are dropped, with a
  single, quiet :class:`DeprecationWarning` that points at the migration path;
* missing optional fields fall back to their defaults;
* a field whose value is not the type the class declares is replaced by that
  field's empty value, so the object handed back is usable by every caller;
* a genuinely missing *required* field raises a clear, named error.

This keeps v0.2.x files loadable on newer builds (and vice-versa) without silently
losing data. :func:`thread_from_saved` is the same idea one level up: it returns the
run's steps from a document that carries them, and reconstructs them from the flat
transcript of a document written before they were kept.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import warnings
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Remember which (target, unknown-keys) combinations we have already warned about so a
# loop loading many records emits at most one warning per distinct drift, not one per row.
_warned_drift: set[tuple[str, tuple[str, ...]]] = set()


def _accepted_param_names(target: Callable[..., Any]) -> tuple[set[str], bool]:
    """Return (accepted keyword names, accepts **kwargs) for ``target``'s signature."""
    sig = inspect.signature(target)
    accepts_var_kw = False
    names: set[str] = set()
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_var_kw = True
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        names.add(name)
    return names, accepts_var_kw


def coerce_kwargs(
    target: Callable[..., Any],
    data: dict[str, Any],
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Filter ``data`` to the keyword arguments ``target`` actually accepts.

    Unrecognized keys are dropped and reported once via :class:`DeprecationWarning`.
    If ``target`` accepts ``**kwargs`` the dict is returned unchanged.

    Args:
        target: The callable whose signature decides what is kept.
        data: The keyword arguments to filter.
        label: Name used in the warning about dropped keys.

    Returns:
        The subset of *data* that *target* accepts.
    """
    label = label or getattr(target, "__name__", str(target))
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{label}: cannot load saved data — expected a JSON object of fields, got "
            f"{type(data).__name__}. The file is corrupt or is not a {label} file."
        )
    accepted, accepts_var_kw = _accepted_param_names(target)
    if accepts_var_kw:
        return dict(data)

    known = {k: v for k, v in data.items() if k in accepted}
    unknown = sorted(k for k in data if k not in accepted)
    if unknown:
        drift_key = (label, tuple(unknown))
        if drift_key not in _warned_drift:
            _warned_drift.add(drift_key)
            warnings.warn(
                f"{label}: ignoring unrecognized field(s) {unknown} while loading saved "
                f"data written by a different effGen version. They were dropped; re-save "
                f"the file to migrate it to the current format.",
                DeprecationWarning,
                stacklevel=3,
            )
    return known


def _annotation_text(annotation: Any) -> str:
    """Return a declared field annotation as text, however it was written.

    A module using ``from __future__ import annotations`` gives a string; one
    that does not gives the type object.
    """
    if isinstance(annotation, str):
        return annotation.strip()
    text = str(annotation)
    return text if "[" in text else str(getattr(annotation, "__name__", text))


def _declared_kind(annotation: Any) -> str | None:
    """Classify a field annotation as one of the shapes a loaded value must take.

    Only the shapes whose mismatch breaks a caller are classified: a list, a
    list of mappings, a mapping, text, and a whole number. A union, a float, a
    timestamp or anything else returns ``None`` and is left as written.
    """
    text = _annotation_text(annotation)
    if "|" in text or text.startswith(("Optional", "Union")):
        return None
    if text in ("list", "List") or text.startswith(("list[", "List[")):
        inner = text[text.index("[") + 1 : -1].strip() if "[" in text else ""
        return "mappings" if inner.startswith(("dict", "Dict", "Mapping")) else "list"
    if text in ("dict", "Dict") or text.startswith(("dict[", "Dict[", "Mapping[")):
        return "mapping"
    if text == "str":
        return "text"
    if text == "int":
        return "whole"
    return None


def _repair(value: Any, kind: str) -> tuple[Any, bool]:
    """Return ``(value-of-the-declared-shape, was-repaired)``."""
    if kind in ("list", "mappings"):
        if not isinstance(value, list):
            return [], True
        if kind == "mappings":
            usable = [item for item in value if isinstance(item, Mapping)]
            return usable, len(usable) != len(value)
        return value, False
    if kind == "mapping":
        return (value, False) if isinstance(value, Mapping) else ({}, True)
    if kind == "text":
        return (value, False) if isinstance(value, str) else (
            "" if value is None else str(value), True
        )
    if kind == "whole":
        if isinstance(value, bool) or not isinstance(value, int):
            try:
                return int(value), True  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return 0, True
        return value, False
    return value, False


def repair_field_types(
    target: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Replace values whose type does not match what ``target`` declares.

    A saved document is JSON another build, another process or a hand edit wrote,
    so a field can hold any type. Every reader of the resulting object — a CLI
    listing, a resumed agent, an export — otherwise gets an ``AttributeError``
    from deep inside itself instead of the conversation it asked for. A value of
    the wrong shape is replaced by that field's empty value and named in one log
    line; ``target`` is left alone when it is not a dataclass.

    Args:
        target: The dataclass whose declared field types are the reference.
        kwargs: The values to check and repair.
        label: Name used in the log line about a replaced value.

    Returns:
        The same mapping with mistyped values replaced by their empty value.
    """
    if not dataclasses.is_dataclass(target):
        return kwargs
    repaired: list[str] = []
    for field in dataclasses.fields(target):
        if field.name not in kwargs:
            continue
        kind = _declared_kind(field.type)
        if kind is None:
            continue
        value, changed = _repair(kwargs[field.name], kind)
        if changed:
            kwargs[field.name] = value
            repaired.append(field.name)
    if repaired:
        logger.warning(
            "%s: saved data held the wrong type for %s; those field(s) were "
            "replaced with a value of the declared type so the rest of the "
            "record still loads.",
            label,
            ", ".join(repaired),
        )
    return kwargs


def load_from_dict(
    target: Callable[..., T],
    data: dict[str, Any],
    *,
    label: str | None = None,
) -> T:
    """Construct ``target`` from a possibly-drifted dict, forgiving extra keys.

    Drops unknown keys (one warning), lets optional fields default, replaces a
    value whose type does not match the declared field, and turns a missing
    *required* field into a clear ``ValueError`` naming the field rather than a
    bare ``TypeError``.

    Args:
        target: The callable to construct.
        data: The stored document to build from.
        label: Name used in the warning about dropped keys.

    Returns:
        The constructed object.

    Raises:
        ValueError: A required field is missing from *data*.
    """
    label = label or getattr(target, "__name__", str(target))
    kwargs = repair_field_types(
        target, coerce_kwargs(target, data, label=label), label=label
    )
    try:
        return target(**kwargs)
    except TypeError as exc:  # missing required positional/keyword argument
        msg = str(exc)
        if "required" in msg or "missing" in msg or "argument" in msg:
            raise ValueError(
                f"{label}: cannot load saved data — it is missing a required field "
                f"({msg}). The file may be from an incompatible version or corrupt."
            ) from exc
        raise


#: Logged once per reconstruction, so a run that resumed a pre-thread checkpoint
#: is greppable in a log file.
_REBUILT_FROM_TRANSCRIPT = (
    "[compat] rebuilt a thread from a flat transcript written before a run's "
    "steps were kept"
)


def thread_from_saved(data: Mapping[str, Any], *, label: str = "checkpoint") -> Any:
    """Return the :class:`~effgen.core.thread.AgentThread` a saved document carries.

    A document written by a build that kept a run's steps carries them under
    ``thread``; one written before that carries only the flat transcript under
    ``scratchpad``. Both load, so a saved run resumes on either build.

    **What the reconstruction from a transcript loses.** The transcript is the
    text the run's steps rendered to, and four things never reached it:

    * a tool call's id — the reconstructed call is renamed from its position, so
      it answers its own observation but no longer names what the provider named;
    * a tool's argument names, when the call's input was not written as JSON —
      the text carries what was sent, so an input the model wrote as prose comes
      back whole under one key rather than as the arguments it stood for. An
      input that rendered as JSON is read back as JSON, types and all;
    * a line the framework injected after an observation — it comes back as part
      of that observation rather than as the step it was, so nothing afterwards
      can tell which half the run wrote;
    * the step the run ended on: the answer it reached and the reason it stopped
      are not in the transcript and do not return;
    * every step field the text does not render: the run's frame (the persona,
      the tool contract, the session's earlier turns and the task), a turn's
      reasoning alongside a provider-native call, whether a call was declined,
      whether an observation was an error, and the thread's own metadata.

    Rendering the reconstruction gives back the transcript it was read from, so a
    resumed run sees the prompt it would have seen. What it does not get back is
    the structure underneath that text.

    Args:
        data: A saved document, as a checkpoint or a session turn stores one.
        label: Name used in the log line about a reconstruction.

    Returns:
        The thread the document carries, reconstructed when it carries only text.
    """
    from .thread import AgentThread

    if not isinstance(data, Mapping):
        return AgentThread()
    saved = data.get("thread")
    if isinstance(saved, Mapping) and saved.get("steps") is not None:
        return AgentThread.from_dict(dict(saved))
    transcript = data.get("scratchpad")
    if not isinstance(transcript, str) or not transcript:
        return AgentThread()
    logger.info("%s (%s)", _REBUILT_FROM_TRANSCRIPT, label)
    return AgentThread.from_scratchpad(transcript)
