"""Building the agent under measurement, and reading its answer back.

Copied in behaviour from the adapter that produced the recorded runs, so a fresh
run and a recorded one are the same measurement. Each choice below is one that
would otherwise move the numbers:

* the tools are the same implementations with the same names, descriptions and
  schemas, so the only thing that varies between two runs is the agent;
* a fresh agent per sample, because an agent kept across a set carries its own
  short-term memory into the next prompt — which makes samples stop being
  independent and makes the prompt grow with the sample index, in a measurement
  that is partly about prompt size;
* ``raise_on_error`` off, because turning "the loop ran out of iterations" into
  an exception scores an unfinished run differently from a wrong answer, and the
  recorded runs it is compared against returned their partial text;
* the answer read is the model's own text. When a run does not finish, the
  library reports that in ``output`` and keeps what the model produced in the
  metadata. That is the right thing for a caller; it is the wrong thing to score,
  because it measures the reporting style rather than the agent.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .harness.tools.spec import ToolSpec

#: Which tool belongs to which category. An agent is allowed to choose its
#: behaviour from a tool's declared category, so a measurement that got these
#: wrong would be measuring a different decision from the one it named.
TOOL_CATEGORIES = {
    "calculator": "COMPUTATION",
    "python_exec": "CODE_EXECUTION",
    "web_search": "INFORMATION_RETRIEVAL",
    "knowledge_search": "INFORMATION_RETRIEVAL",
}


def build_tools(specs: list[ToolSpec]) -> list:
    """Wrap the harness tool specs as tools the agent can be handed."""
    from effgen.tools.base_tool import (
        BaseTool,
        ParameterSpec,
        ParameterType,
        ToolCategory,
        ToolMetadata,
    )

    tools = []
    for spec in specs:
        properties = spec.parameters.get("properties", {})
        required = set(spec.parameters.get("required", []))
        parameters = [
            ParameterSpec(
                name=name,
                type=ParameterType.STRING,
                description=schema.get("description", ""),
                required=name in required,
            )
            for name, schema in properties.items()
        ]
        category = getattr(
            ToolCategory,
            TOOL_CATEGORIES.get(spec.name, "COMPUTATION"),
            ToolCategory.COMPUTATION,
        )

        class _Wrapped(BaseTool):
            def __init__(self, spec=spec, parameters=parameters, category=category):
                self._spec = spec
                super().__init__(
                    metadata=ToolMetadata(
                        name=spec.name,
                        description=spec.description,
                        category=category,
                        parameters=parameters,
                    )
                )

            async def _execute(self, **kwargs: Any) -> str:
                return self._spec.call(**kwargs)

        tools.append(_Wrapped())
    return tools


def agent_config_kwargs(
    config_cls: type,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Generation settings, filtered to the fields this release actually has."""
    wanted = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "raise_on_error": raise_on_error,
        "enable_memory": False,
    }
    known = {f.name for f in dataclasses.fields(config_cls)}
    return {k: v for k, v in wanted.items() if k in known and v is not None}


#: Text a run puts in ``output`` when it did not finish. It is a report about
#: the run, not an answer, and scoring it as one turns "this stopped early" into
#: "this got it wrong" without saying so.
_RUN_DIAGNOSTICS = (
    "maximum iterations reached without final answer",
    "max iterations reached",
)


def _is_run_diagnostic(text: str) -> bool:
    return text.strip().lower().rstrip(".") in _RUN_DIAGNOSTICS


def answer_text(response) -> str:
    """The model's own text, not the report about the run."""
    if getattr(response, "success", True) is False:
        metadata = getattr(response, "metadata", None) or {}
        partial = metadata.get("partial_output")
        if isinstance(partial, str) and partial.strip():
            return partial.strip()
        detail = metadata.get("error")
        if isinstance(detail, dict):
            preview = detail.get("answer_preview")
            if isinstance(preview, str) and preview.strip():
                return preview.strip()

    text = getattr(response, "output", None)
    if text is None:
        text = getattr(response, "content", None)
    if text is None:
        text = str(response)
    text = str(text)
    if _is_run_diagnostic(text):
        return ""
    return text


def read_response(response) -> tuple[str, list[dict[str, Any]], int | None]:
    """The answer, the tool calls, and a call count when the calls are not kept."""
    text = answer_text(response)
    raw_calls = getattr(response, "tool_calls", None)

    count = getattr(response, "tool_call_count", None)
    if count is None and isinstance(raw_calls, int):
        count = raw_calls
    if count is None:
        count = getattr(raw_calls, "total", None)

    calls: list[dict[str, Any]] = []
    if isinstance(raw_calls, list | tuple):
        for call in raw_calls:
            if isinstance(call, dict):
                calls.append(
                    {
                        "name": call.get("tool") or call.get("name") or "?",
                        "arguments": call.get("input") or call.get("arguments"),
                        "result": str(call.get("output") or call.get("result") or "")[:2000],
                    }
                )
            else:
                calls.append(
                    {
                        "name": getattr(call, "name", "?"),
                        "arguments": getattr(call, "arguments", None),
                        "result": str(
                            getattr(call, "error", None)
                            or getattr(call, "result", "")
                            or ""
                        )[:2000],
                    }
                )

    if not calls:
        recovered = _calls_from_trace(getattr(response, "execution_trace", None) or [])
        if recovered:
            calls, count = recovered, None

    if calls and count is not None and int(count) == len(calls):
        count = None

    return str(text).strip(), calls, (int(count) if count is not None else None)


def read_stop_reason(response) -> str | None:
    """Why the loop ended, as the run reports it."""
    metadata = getattr(response, "metadata", None) or {}
    reason = metadata.get("reason") or metadata.get("answer_source")
    if reason:
        return str(reason)
    if getattr(response, "success", True) is False:
        return "failed"
    return None


def read_outcome(response) -> str | None:
    """What became of the run: ``answered``, ``stopped`` or ``failed``.

    Read from the response when it reports one. A response from a build that
    does not carry the field is classified from what it does carry, so a cell
    recorded before and a cell recorded after can be compared like for like.
    """
    outcome = getattr(response, "outcome", None)
    if isinstance(outcome, str) and outcome:
        return outcome
    success = getattr(response, "success", None)
    if success is None:
        return None
    if success:
        return "answered"
    reason = (getattr(response, "metadata", None) or {}).get("reason")
    stopped = {
        "max_iterations_partial", "max_iterations_exhausted", "loop_detected",
        "repeated_tool_result", "null_final_from_model",
    }
    return "stopped" if reason in stopped else "failed"


def read_iterations(response) -> int | None:
    """How many loop turns the run spent, when the response reports them.

    Accuracy is bought with iterations, so the count travels with every record
    and the trade stays visible. A response from a build that does not carry
    the field records ``None`` rather than a guess.
    """
    value = getattr(response, "iterations", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _calls_from_trace(trace) -> list[dict[str, Any]]:
    calls = []
    for event in trace:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("type", ""))
        if "tool_call" not in kind or kind.endswith("start"):
            continue
        data = event.get("data") or {}
        calls.append(
            {
                "name": data.get("tool_name") or data.get("tool") or "?",
                "arguments": data.get("arguments") or data.get("input"),
                "result": str(data.get("result") or data.get("output") or "")[:2000],
            }
        )
    return calls


__all__ = [
    "TOOL_CATEGORIES",
    "agent_config_kwargs",
    "answer_text",
    "build_tools",
    "read_iterations",
    "read_outcome",
    "read_response",
    "read_stop_reason",
]
