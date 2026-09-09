#!/usr/bin/env python3
"""Real-model smoke for one hosted provider, run by the CI workflows.

Runs the checks the push-time and nightly workflows need against a real
endpoint: plain generation, streaming, a calculator call through an Agent, and
a yes/no reasoning question. Exit status:

* ``0`` — every check passed, **or the provider refused to serve**. A refusal
  (no credit, HTTP 402; a bad or absent key, 401/403; a model id the account
  cannot see, 404) says nothing about effGen, so it is printed as
  ``NOT MEASURED`` with a workflow warning and opens no issue.
* ``1`` — the provider accepted a call and effGen produced the wrong result.

Transient pressure — 429, 503, timeouts — is retried with exponential backoff
and jitter before it counts as anything.

Usage::

    python scripts/ci_provider_smoke.py --provider groq --model openai/gpt-oss-20b
    python scripts/ci_provider_smoke.py --provider cerebras --model gpt-oss-120b --checks stream
"""

from __future__ import annotations

import argparse
import functools
import os
import random
import sys
import time

PROVIDERS = {
    "cerebras": ("CEREBRAS_API_KEY", "effgen.models.cerebras_adapter", "CerebrasAdapter"),
    "groq": ("GROQ_API_KEY", "effgen.models.groq_adapter", "GroqAdapter"),
}

RETRIABLE_MARKS = (
    "429", "rate limit", "rate_limit", "queue_exceeded", "too_many_requests",
    "high traffic", "timeout", "timed out", "503", "service unavailable",
    "maximum iterations", "empty response",
)
REFUSAL_MARKS = (
    "402", "payment required", "payment_required", "billing", "insufficient",
    "no credit", "quota", "401", "403", "unauthorized", "forbidden",
    "invalid api key", "invalid_api_key", "authentication", "404",
    "model_not_found", "does not exist",
)

#: Words that show the model agreed roses need water.
AFFIRM = ("yes", "they do", "they need", "do need", "need water")


class Refused(RuntimeError):
    """The provider would not serve the call; nothing about effGen was measured."""


def _text(exc: BaseException) -> str:
    return str(exc).lower()


def is_retriable(exc: BaseException) -> bool:
    return any(mark in _text(exc) for mark in RETRIABLE_MARKS)


def _refusal_types() -> tuple[type[BaseException], ...]:
    try:
        from effgen.models.errors import ModelAuthError, ModelNotFoundError
    except ImportError:  # pragma: no cover - the package is installed in CI
        return ()
    return (ModelAuthError, ModelNotFoundError)


def is_refusal(exc: BaseException) -> bool:
    if isinstance(exc, _refusal_types()):
        return True
    return not is_retriable(exc) and any(mark in _text(exc) for mark in REFUSAL_MARKS)


def with_retry(label: str, fn, attempts: int, base_delay: float = 3.0, max_delay: float = 60.0):
    """Call *fn*; retry transient pressure, refuse on a refusal, raise otherwise."""
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - every kind is classified below
            last = exc
            if is_refusal(exc):
                raise Refused(f"{label}: {exc}") from exc
            retriable = is_retriable(exc)
            print(f"WARN[{label}] attempt {attempt + 1}/{attempts} "
                  f"{'RETRIABLE' if retriable else 'FATAL'}: {exc}")
            if not retriable or attempt == attempts - 1:
                break
            delay = random.uniform(0.5, min(max_delay, base_delay * (2 ** attempt)))
            print(f"     sleeping {delay:.1f}s before retry")
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")


def check_generate(adapter, model_id: str) -> str:
    from effgen.models.base import GenerationConfig

    # A reasoning model spends tokens before any visible content, so the cap
    # leaves room for that pass as well as the answer.
    out = adapter.generate("Reply with a single word: ALIVE",
                           config=GenerationConfig(temperature=0.0, max_tokens=512))
    text = (out.text or "").strip()
    if not text:
        raise RuntimeError("empty response: generate returned no visible text")
    return f"generate -> {text!r}"


def check_stream(adapter, model_id: str) -> str:
    chunks = list(adapter.generate_stream("Count: 1, 2, 3."))
    text = "".join(chunks)
    if not text:
        raise RuntimeError("empty response: stream returned no text")
    return f"stream -> {len(chunks)} chunks, {len(text)} chars"


def check_tools(adapter, model_id: str) -> str:
    from effgen.core.agent import Agent, AgentConfig
    from effgen.tools.builtin.calculator import Calculator

    config = AgentConfig(
        name=f"smoke-{model_id}",
        model=adapter,
        tools=[Calculator()],
        system_prompt="You are a math assistant. Use the calculator tool. Answer concisely.",
        max_iterations=5,
        temperature=0.1,
    )
    response = Agent(config).run("What is 12 * 12?")
    text = str(response.output or "")
    if not text.strip():
        raise RuntimeError("empty response: the agent returned no text")
    if "144" not in text:
        raise RuntimeError(f"'144' missing from output: {text!r}")
    return f"tool-calling -> '144' present ({getattr(response, 'tool_calls', 0) or 0} tool calls)"


def check_reasoning(adapter, model_id: str) -> str:
    from effgen.models.base import GenerationConfig

    result = adapter.generate(
        "If all roses are flowers and all flowers need water, do roses need water? "
        "Answer with just 'yes' or 'no' first, then a one-sentence explanation.",
        config=GenerationConfig(temperature=0.0, max_tokens=2048),
    )
    text = (result.text or "").strip()
    if not text:
        raise RuntimeError("empty response: no visible content "
                           f"(finish_reason={getattr(result, 'finish_reason', '?')})")
    if not any(word in text.lower() for word in AFFIRM):
        raise RuntimeError(f"unexpected text: {text!r}")
    return f"reasoning -> {text[:80]!r}"


CHECKS = {
    "generate": check_generate,
    "stream": check_stream,
    "tools": check_tools,
    "reasoning": check_reasoning,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model", required=True, help="model id on that provider")
    parser.add_argument("--checks", default=",".join(CHECKS),
                        help="comma-separated subset of: " + ", ".join(CHECKS))
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args(argv)

    key_var, module_name, class_name = PROVIDERS[args.provider]
    if not os.getenv(key_var, ""):
        print(f"SKIPPED: {key_var} not set for {args.provider}/{args.model}")
        return 0
    names = [name.strip() for name in args.checks.split(",") if name.strip()]
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        parser.error(f"unknown check(s): {', '.join(unknown)}. Choose from: {', '.join(CHECKS)}")

    import importlib

    adapter_cls = getattr(importlib.import_module(module_name), class_name)
    label = f"{args.provider}/{args.model}"
    try:
        # A model id the account cannot see is refused at construction, before
        # any call is made; that is a refusal like any other.
        adapter = adapter_cls(args.model, enable_rate_limiting=False)
        adapter.load()
    except Exception as exc:  # noqa: BLE001 - classified below
        if is_refusal(exc):
            print(f"NOT MEASURED[{label}]: the provider refused to serve, so nothing "
                  f"about effGen was checked. {exc}")
            print(f"::warning title=Provider smoke not measured::{label} refused: {str(exc)[:300]}")
            return 0
        print(f"FAIL[{label}]: could not load the adapter: {exc}")
        return 1
    try:
        for name in names:
            check = CHECKS[name]
            try:
                print(f"OK[{label}]: "
                      f"{with_retry(name, functools.partial(check, adapter, args.model), args.max_attempts)}")
            except Refused as refused:
                print(f"NOT MEASURED[{label}]: the provider refused to serve, so nothing "
                      f"about effGen was checked. {refused}")
                print(f"::warning title=Provider smoke not measured::{label} refused: "
                      f"{str(refused)[:300]}")
                return 0
            except Exception as exc:  # noqa: BLE001 - the verdict is the exit code
                print(f"FAIL[{label}]: {exc}")
                return 1
    finally:
        try:
            adapter.unload()
        except Exception:  # noqa: BLE001 - nothing left to report
            pass
    print(f"All checks passed for {label}: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
