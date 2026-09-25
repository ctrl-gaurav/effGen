"""Scorers: how a bench task's answer is compared with what was expected.

A suite names one scorer. Built in:

``exact``
    The answer equals ``expected`` after normalisation (case, surrounding
    whitespace and a trailing full stop are ignored; when the answer has a line
    starting ``Answer:`` or ``Final answer:``, the text after the last such
    line is what is compared).
``contains``
    ``expected`` appears in the answer, ignoring case.
``number``
    The last number in the answer equals ``expected`` within ``tolerance``
    (relative, default ``1e-6``). Thousands separators are ignored.
``regex``
    ``expected`` is a regular expression searched for in the answer, ignoring
    case.
``choice``
    The answer names the option letter ``expected`` (``A`` to ``Z``): the letter
    after the last ``Answer:``, else a letter standing on its own.

``expected`` may be a list for ``exact``, ``contains`` and ``regex``: any match
scores 1.

A suite can also name a function of its own::

    scorer:
      callable: my_scorers.py:score      # or a module path, pkg.mod:score

called as ``score(answer, expected, task)`` and returning a bool or a number in
``[0, 1]``. A relative file path is resolved against the suite file.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

__all__ = ["BUILTIN_SCORERS", "make_scorer", "validate_scorer"]

_ANSWER_LINE = re.compile(r"^\s*(?:final\s+answer|answer)\s*[:：]\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+")
_LETTER_AFTER_ANSWER = re.compile(r"answer\s*(?:is)?\s*[:：]?\s*\(?([A-Za-z])\)?(?![A-Za-z])", re.IGNORECASE)
_LONE_LETTER = re.compile(r"(?<![A-Za-z])\(?([A-Z])\)?(?![A-Za-z])")

Scorer = Callable[[str, Any, dict[str, Any]], float]


def _final_segment(text: str) -> str:
    matches = _ANSWER_LINE.findall(text or "")
    return matches[-1] if matches else (text or "")


def _normalise(value: Any) -> str:
    text = " ".join(str(value).split()).casefold()
    return text[:-1].rstrip() if text.endswith(".") else text


def _options(expected: Any) -> list[Any]:
    return list(expected) if isinstance(expected, (list, tuple)) else [expected]


def _exact(answer: str, expected: Any, task: dict[str, Any]) -> float:
    got = _normalise(_final_segment(answer))
    return float(any(got == _normalise(e) for e in _options(expected)))


def _contains(answer: str, expected: Any, task: dict[str, Any]) -> float:
    got = _normalise(answer)
    return float(any(_normalise(e) in got for e in _options(expected)))


def _regex(answer: str, expected: Any, task: dict[str, Any]) -> float:
    return float(any(re.search(str(e), answer or "", re.IGNORECASE) for e in _options(expected)))


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _number_scorer(tolerance: float) -> Scorer:
    def _number(answer: str, expected: Any, task: dict[str, Any]) -> float:
        want = expected if isinstance(expected, (int, float)) else _to_float(str(expected))
        if want is None:
            return 0.0
        found = _NUMBER.findall(_final_segment(answer))
        if not found:
            return 0.0
        got = _to_float(found[-1])
        if got is None:
            return 0.0
        return float(abs(got - float(want)) <= tolerance * max(1.0, abs(float(want))))
    return _number


def _choice(answer: str, expected: Any, task: dict[str, Any]) -> float:
    text = answer or ""
    after = _LETTER_AFTER_ANSWER.findall(text)
    if after:
        got = after[-1].upper()
    else:
        lone = _LONE_LETTER.findall(_final_segment(text))
        if not lone:
            return 0.0
        got = lone[-1].upper()
    return float(got == str(expected).strip().upper())


#: The scorers a suite can name by string.
BUILTIN_SCORERS: dict[str, Scorer] = {
    "exact": _exact,
    "contains": _contains,
    "number": _number_scorer(1e-6),
    "regex": _regex,
    "choice": _choice,
}

_OPTIONS = {"name", "tolerance", "callable"}


def _load_callable(reference: str, base_dir: Path | None) -> Callable[..., Any]:
    if ":" not in reference:
        raise ValueError(f"Callable scorer {reference!r} names no function. Use 'file.py:function' or 'module:function'.")
    target, _, attr = reference.rpartition(":")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        if not path.is_file():
            raise ValueError(f"The scorer file {path} was not found. Check the path; it is relative to the suite file.")
        spec = importlib.util.spec_from_file_location(f"effgen_bench_scorer_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"The scorer file {path} could not be imported. Check that it is a Python file.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    func = getattr(module, attr, None)
    if not callable(func):
        raise ValueError(f"{reference}: {attr!r} is not a function. Name a function defined in that module.")
    return cast("Callable[..., Any]", func)


def validate_scorer(spec: Any, base_dir: Path | None = None) -> None:
    """Raise ``SuiteError`` unless *spec* names a scorer that can be built."""
    from .suite import SuiteError

    try:
        make_scorer(spec, base_dir=base_dir)
    except (ValueError, TypeError, ImportError) as exc:
        raise SuiteError(f"The scorer entry cannot be used ({exc}). Fix the 'scorer' key in the suite file.") from exc


def make_scorer(spec: Any, base_dir: Path | None = None) -> Scorer:
    """Build the scoring function a suite's ``scorer`` entry names.

    Args:
        spec: A built-in name, or a mapping ``{"name": ..., "tolerance": ...}``
            or ``{"callable": "file.py:function"}``.
        base_dir: Where a relative scorer file is looked for.

    Returns:
        ``score(answer, expected, task) -> float`` in ``[0, 1]``.
    """
    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict):
        raise TypeError(f"The scorer is a {type(spec).__name__}. Use a scorer name or a mapping.")
    unknown = sorted(set(spec) - _OPTIONS)
    if unknown:
        raise ValueError(f"The scorer has unknown option(s) {', '.join(unknown)}. Use only name, tolerance or callable.")
    if "callable" in spec:
        func = _load_callable(str(spec["callable"]), base_dir)

        def _custom(answer: str, expected: Any, task: dict[str, Any]) -> float:
            value = func(answer, expected, task)
            score = float(value)
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"The scorer returned {value!r}, outside [0, 1]. Make it return a bool or a number from 0 to 1.")
            return score
        return _custom
    name = str(spec.get("name", ""))
    if name == "number" and "tolerance" in spec:
        return _number_scorer(float(spec["tolerance"]))
    if "tolerance" in spec:
        raise ValueError("'tolerance' was given to a scorer other than number. Remove it, or use the number scorer.")
    if name not in BUILTIN_SCORERS:
        raise ValueError(
            f"unknown scorer {name!r}. Use one of the built-in scorers ({', '.join(BUILTIN_SCORERS)}) or {{callable: ...}}."
        )
    return BUILTIN_SCORERS[name]
