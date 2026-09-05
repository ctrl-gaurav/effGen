"""A run that produced no answer says so, and never returns its notes instead.

Three loop exits used to hand the caller internal state as the answer with
``success=True``: a model repeating the same compute call, a computing tool
reproducing its own result, and a model answering "N/A" after its tools ran.
What arrived in ``output`` was a pipe-joined list of tool observations, or one
raw tool result, and nothing on the response distinguished it from an answer the
model wrote.

All three now report the same shape as the iteration cap already did: an
``outcome`` of ``"stopped"``, the statement of what happened in ``output``, and
the tool results under ``partial``. Under the default ``raise_on_error=True``
they raise :class:`~effgen.errors.RunStoppedError`, which carries the response so
the progress survives the raise.

The model here is scripted in-process, so each terminal path is reached
deterministically without a network call.

Two of the assertions below are guards rather than proofs of this change:
``test_the_statement_names_the_model_and_what_to_do[max_iterations_partial]``
(the iteration cap already stated its outcome) and
``test_a_failed_run_does_not_raise_the_stopped_error`` (a failure has never
raised the stopped error). Both hold before and after; they are here so a later
change cannot quietly take them away.
"""

from __future__ import annotations

import json
import re

import pytest

from effgen.core.agent import Agent, AgentConfig, AgentResponse, PartialResult
from effgen.core.agent_response import STOP_REASONS, STOPPED_REASONS
from effgen.errors import RunStoppedError
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from effgen.tools.builtin.calculator import Calculator

# --------------------------------------------------------------------------- #
# The mechanical shape classifier
# --------------------------------------------------------------------------- #
#: A tool echo the scratchpad writes in front of a batched result.
_ECHO = re.compile(r"^\[[^\]]*\]\s*(?:→|->)\s*")


def _norm(text: str) -> str:
    return " ".join(_ECHO.sub("", (text or "")).split()).strip().lower()


def classify_output(text: str, tool_results: list[str]) -> str:
    """Name the shape of *text* against what the run's tools returned.

    ``pipe_dump`` — two or more ``" | "``-separated segments, at least two of
    which are a tool result the run recorded. ``raw_tool_result`` — the whole
    text is one recorded result. Both are the shapes a caller must never receive
    as an answer, so this is what the assertions below count.
    """
    out = (text or "").strip()
    results = [_norm(r) for r in tool_results if r]
    if not out:
        return "empty"
    if " | " in out:
        segments = [_norm(s) for s in out.split(" | ") if s.strip()]
        if len(segments) >= 2:
            hits = sum(
                1
                for s in segments
                if any(s and (s == r or s in r or r in s) for r in results)
            )
            return "pipe_dump" if hits >= 2 else "pipe_other"
    normalized = _norm(out)
    if results and any(normalized == r for r in results):
        return "raw_tool_result"
    return "other"


# --------------------------------------------------------------------------- #
# The scripted model and the two tool categories
# --------------------------------------------------------------------------- #
class _Scripted(BaseModel):
    """Replays one ReAct turn per ``generate()``; the last turn repeats."""

    def __init__(self, turns: list[str], *, native: bool = False) -> None:
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)
        self._turns = turns
        self._native = native
        self.calls = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def generate(self, prompt, config=None, **kwargs) -> GenerationResult:
        text = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs):
        yield self.generate(prompt).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 4096

    def supports_function_calling(self) -> bool:
        return self._native

    def supports_tool_calling(self) -> bool:
        return self._native


class _Passages(BaseTool):
    """A retrieval-category tool that returns the same passage every call."""

    BODY = "Refunds are accepted within 30 days of the charge."

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="knowledge_base",
            description="Search the handbook.",
            category=ToolCategory.INFORMATION_RETRIEVAL,
            parameters=[ParameterSpec(
                name="query", type=ParameterType.STRING,
                description="what to look up", required=True,
            )],
        ))

    async def _execute(self, query: str = "", **kwargs) -> str:
        return self.BODY


def _calc(expression: str) -> str:
    return (
        "Thought: compute it.\nAction: calculator\n"
        f"Action Input: {json.dumps({'expression': expression})}"
    )


def _search(query: str) -> str:
    return (
        "Thought: look it up.\nAction: knowledge_base\n"
        f"Action Input: {json.dumps({'query': query})}"
    )


def _agent(turns, *, tools=None, raise_on_error=False, **cfg) -> Agent:
    return Agent(config=AgentConfig(
        name="stopped-outcome-test",
        model=_Scripted(turns),
        tools=[Calculator()] if tools is None else tools,
        tool_calling_mode="react",
        max_iterations=cfg.pop("max_iterations", 6),
        raise_on_error=raise_on_error,
        enable_memory=False,
        **cfg,
    ))


#: A question that names two computations, so the loop reaches its stop with
#: more than one observation in hand. "Explain" keeps the direct-calculator
#: shortcut out of the way.
TASK = (
    "Explain step by step: a shop sells 12 pens at 3 dollars and 7 pens at "
    "5 dollars; what is the total revenue?"
)

#: Each stopped exit, as (turns, kwargs, stop_reason).
STOPPED_PATHS = [
    pytest.param(
        [_calc("12*3"), _calc("7*5"), _calc("7*5")], {}, "loop_detected",
        id="loop_detected",
    ),
    pytest.param(
        # 36*1 and 18*2 both return 36, so the tool reproduces its own result
        # with a fresh input each time. The first repeat buys a turn to state
        # the answer; the second finds that turn spent and stops the run.
        [_calc("12*3"), _calc("36*1"), _calc("18*2")], {}, "repeated_tool_result",
        id="repeated_tool_result",
    ),
    pytest.param(
        [_calc("12*3"), _calc("7*5"), "Thought: done.\nFinal Answer: N/A"], {},
        "null_final_from_model", id="null_final_from_model",
    ),
    pytest.param(
        [_calc("12*3"), _calc("7*5"), _calc("36+35"), _calc("71*2")],
        {"max_iterations": 3}, "max_iterations_partial", id="max_iterations_partial",
    ),
]


# --------------------------------------------------------------------------- #
# The contract at every stopped exit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("turns", "kwargs", "reason"), STOPPED_PATHS)
def test_a_stopped_run_reports_the_stop_and_keeps_its_progress(turns, kwargs, reason):
    response = _agent(turns, **kwargs).run(TASK)

    assert response.success is False
    assert response.outcome == "stopped"
    assert response.stop_reason == reason
    assert response.metadata["reason"] == reason
    assert response.partial is not None
    assert response.partial.text
    assert response.output != response.partial.text
    # The statement is a report about the run, not one of the tool results.
    assert response.metadata["partial_output"] == response.partial.text
    assert response.metadata["partial"] is True


@pytest.mark.parametrize(("turns", "kwargs", "reason"), STOPPED_PATHS)
def test_no_stopped_run_returns_a_dump_as_its_answer(turns, kwargs, reason):
    """The shape the caller used to receive is gone from ``output``."""
    response = _agent(turns, **kwargs).run(TASK)
    results = [c.result or "" for c in response.tool_calls]

    assert classify_output(response.output, results) == "other"
    # And the progress it kept is exactly the shape that used to be the answer.
    assert classify_output(response.partial.text, results) in (
        "pipe_dump", "raw_tool_result",
    )


@pytest.mark.parametrize(("turns", "kwargs", "reason"), STOPPED_PATHS)
def test_the_statement_names_the_model_and_what_to_do(turns, kwargs, reason):
    response = _agent(turns, **kwargs).run(TASK)

    assert "scripted-model" in response.output
    assert "max_tokens" in response.output or "max_iterations" in response.output
    detail = response.metadata["error"]
    assert detail["category"] == reason or detail["category"] == "max_iterations"
    assert detail["retryable"] is False
    assert detail["message"] == response.output


@pytest.mark.parametrize(("turns", "kwargs", "reason"), STOPPED_PATHS)
def test_the_partial_agrees_with_the_tool_calls_one_for_one(turns, kwargs, reason):
    response = _agent(turns, **kwargs).run(TASK)

    assert len(response.partial.observations) == len(list(response.tool_calls))
    assert response.partial.observations == tuple(
        c.result or "" for c in response.tool_calls
    )
    assert response.partial.last_observation == response.partial.observations[-1]
    assert response.partial.tool_calls == response.tool_call_count


def test_a_run_stopped_with_nothing_to_show_carries_no_partial():
    response = _agent(
        ["Thought: hmm.\nAction: (continue reasoning)"], max_iterations=2
    ).run(TASK)

    assert response.outcome == "stopped"
    assert response.stop_reason == "max_iterations_exhausted"
    assert response.partial is None
    assert response.metadata.get("partial_output") is None


def test_a_retrieval_loop_reports_the_same_shape_as_a_compute_one():
    """Both categories stop the same way; only the wording differs."""
    response = _agent(
        [_search("refunds")], tools=[_Passages()], max_iterations=8,
    ).run("Explain the refund window.")

    assert response.outcome == "stopped"
    assert response.stop_reason in ("loop_detected", "repeated_tool_result")
    assert _Passages.BODY in response.partial.text
    assert _Passages.BODY not in response.output
    assert "knowledge_base" in response.output


# --------------------------------------------------------------------------- #
# What still answers
# --------------------------------------------------------------------------- #
def test_a_model_that_writes_an_answer_still_answers():
    response = _agent([
        _calc("12*3"), "Thought: I know it.\nFinal Answer: 71 dollars.",
    ]).run(TASK)

    assert response.outcome == "answered"
    assert response.success is True
    assert response.stop_reason == "final_answer"
    assert response.partial is None
    assert response.output == "71 dollars."


def test_a_run_with_no_tools_answers_and_reports_final_answer():
    agent = Agent(config=AgentConfig(
        name="direct", model=_Scripted(["The total is 71 dollars."]),
        tools=[], raise_on_error=False, enable_memory=False,
    ))
    response = agent.run("What is the total revenue?")

    assert response.outcome == "answered"
    assert response.stop_reason == "final_answer"
    assert response.partial is None


def test_the_direct_calculator_shortcut_is_still_an_answer():
    response = _agent([_calc("17*23"), _calc("17*23")]).run("What is 17 * 23?")

    assert response.outcome == "answered"
    assert response.stop_reason == "final_answer"
    assert response.metadata["answer_source"] == "direct_calculator_result"


# --------------------------------------------------------------------------- #
# raise_on_error
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("turns", "kwargs", "reason"), STOPPED_PATHS)
def test_the_default_raises_and_the_exception_carries_the_run(turns, kwargs, reason):
    agent = _agent(turns, raise_on_error=True, **kwargs)

    with pytest.raises(RunStoppedError) as caught:
        agent.run(TASK)

    exc = caught.value
    assert isinstance(exc, RuntimeError), "except RuntimeError must keep working"
    assert exc.stop_reason == reason
    assert exc.response.outcome == "stopped"
    assert exc.partial is not None
    assert exc.partial.text == exc.response.metadata["partial_output"]
    assert str(exc) == exc.response.output


def test_a_failed_run_does_not_raise_the_stopped_error():
    """A written-out tool call is a failure, not a stop, and keeps its error."""
    agent = _agent(
        ['<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'],
        raise_on_error=True, max_iterations=4,
    )
    with pytest.raises(RuntimeError) as caught:
        agent.run(TASK)
    assert not isinstance(caught.value, RunStoppedError)


# --------------------------------------------------------------------------- #
# written_tool_call keeps what its tools returned
# --------------------------------------------------------------------------- #
def test_a_written_call_after_a_tool_ran_keeps_the_observations():
    """The run did no work the answer describes, but it did have tool results.

    Driven at the seam because the loop only reaches this exit when a model
    writes a call out *after* its tools ran, which no scripted ReAct turn can
    produce reliably — the same reason no recorded sample reaches it either.
    """
    from effgen.core.tool_call_record import ToolCall

    agent = _agent([_calc("12*3")])
    calls = [ToolCall(name="calculator", arguments="12*3", result="36")]
    response = agent._written_tool_call_response(
        "calculator", "<tool_call>…</tool_call>",
        iterations=2, tool_calls=1, tokens_used=10, tool_ran=True,
        calls=calls, scratchpad="Thought: compute.\nObservation: 36",
    )

    assert response.outcome == "failed"
    assert response.stop_reason == "written_tool_call"
    assert response.partial is not None
    assert response.partial.observations == ("36",)
    assert response.metadata["partial_output"] == response.partial.text
    assert list(response.tool_calls) == calls


def test_a_written_call_with_no_tool_run_has_nothing_to_keep():
    written = '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'
    response = _agent([written, written], max_iterations=4).run(TASK)

    assert response.outcome == "failed"
    assert response.stop_reason == "written_tool_call"
    assert response.partial is None


# --------------------------------------------------------------------------- #
# Every response carries a stop reason
# --------------------------------------------------------------------------- #
def test_a_bare_response_derives_its_stop_reason():
    assert AgentResponse(output="x").stop_reason == "final_answer"
    assert AgentResponse(output="", success=False).stop_reason == "run_failed"
    assert AgentResponse(
        output="", success=False, metadata={"guardrail_blocked": True}
    ).stop_reason == "guardrail_blocked"
    assert AgentResponse(
        output="", success=False, metadata={"reason": "loop_detected"}
    ).stop_reason == "loop_detected"


def test_the_stop_reason_and_the_metadata_reason_agree():
    response = AgentResponse(output="x", stop_reason="final_answer")
    assert response.metadata["reason"] == "final_answer"
    assert response.outcome == "answered"


def test_marking_a_response_failed_moves_both_fields_together():
    response = AgentResponse(output="x")
    response.mark_failed("structured_output_failed", {"message": "no"})

    assert response.success is False
    assert response.stop_reason == "structured_output_failed"
    assert response.metadata["reason"] == "structured_output_failed"
    assert response.metadata["error"] == {"message": "no"}
    assert response.outcome == "failed"


@pytest.mark.parametrize(("turns", "kwargs", "reason"), STOPPED_PATHS)
def test_every_run_reports_a_stop_reason_from_the_published_vocabulary(
    turns, kwargs, reason,
):
    response = _agent(turns, **kwargs).run(TASK)
    assert response.stop_reason in STOP_REASONS
    assert (response.outcome == "stopped") is (response.stop_reason in STOPPED_REASONS)


# --------------------------------------------------------------------------- #
# The document a caller saves
# --------------------------------------------------------------------------- #
def test_to_dict_carries_the_three_new_keys():
    response = _agent(
        [_calc("12*3"), _calc("7*5"), _calc("7*5")]
    ).run(TASK)
    document = response.to_dict()

    assert document["stop_reason"] == "loop_detected"
    assert document["outcome"] == "stopped"
    assert document["partial"]["text"] == response.partial.text
    assert document["partial"]["observations"] == list(response.partial.observations)
    assert PartialResult.from_dict(document["partial"]) == response.partial


def test_an_answered_run_carries_no_partial_in_its_document():
    response = _agent([
        _calc("12*3"), "Thought: I know it.\nFinal Answer: 71 dollars.",
    ]).run(TASK)

    assert response.to_dict()["partial"] is None
    assert response.to_dict()["outcome"] == "answered"


# --------------------------------------------------------------------------- #
# The three outcomes, told apart without reading ``output``
# --------------------------------------------------------------------------- #
def test_a_caller_tells_the_three_outcomes_apart_from_the_response_alone():
    answered = _agent(
        [_calc("12*3"), "Thought: I know it.\nFinal Answer: 71 dollars."]
    ).run(TASK)
    stopped = _agent([_calc("12*3"), _calc("7*5"), _calc("7*5")]).run(TASK)
    failed = Agent(config=AgentConfig(
        name="empty", model=_Scripted(["x"]), tools=[], raise_on_error=False,
        enable_memory=False,
    )).run("   ")

    assert [r.outcome for r in (answered, stopped, failed)] == [
        "answered", "stopped", "failed",
    ]
    assert [r.partial is not None for r in (answered, stopped, failed)] == [
        False, True, False,
    ]


# --------------------------------------------------------------------------- #
# The surfaces a user reads
# --------------------------------------------------------------------------- #
@pytest.fixture
def stopped():
    return _agent([_calc("12*3"), _calc("7*5"), _calc("7*5")]).run(TASK)


def test_the_cli_frames_the_stop_and_the_progress_separately(stopped):
    import io

    pytest.importorskip("rich")
    from effgen.cli.commands.run import PARTIAL_PROGRESS_TITLE, present_response
    from effgen.ui.theme import get_console

    class _Cli:
        def __init__(self, console):
            self.console = console
            self.panels: list[str] = []

        def print_error_panel(self, text, title=""):
            self.panels.append(f"ERROR:{title}")

    buffer = io.StringIO()
    cli = _Cli(get_console(file=buffer, force_terminal=False, width=100))
    present_response(cli, stopped)
    printed = buffer.getvalue()

    assert not cli.panels, "a stopped run is not an error panel"
    assert "Stopped" in printed
    assert PARTIAL_PROGRESS_TITLE.split(" (")[0] in printed


def test_the_footer_names_what_stopped_the_run(stopped):
    from effgen.ui.render import summary_line

    plain, _markup = summary_line(stopped)
    assert "loop_detected" in plain


def test_the_run_card_badge_names_the_stop(stopped):
    from effgen.ui.report_html import build_html_report

    document = {
        k: v for k, v in stopped.to_dict().items() if k != "execution_trace"
    }
    document["metadata"] = {
        k: v for k, v in stopped.metadata.items() if k != "debug_trace"
    }
    html = build_html_report(document, kind="run")

    assert "stopped (loop_detected)" in html
    assert ">failed<" not in html


def test_the_run_history_records_the_stop_not_an_error(tmp_path, monkeypatch):
    import json as _json

    from effgen.observability import run_log

    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(run_log, "_pruned", True)
    response = _agent([_calc("12*3"), _calc("7*5"), _calc("7*5")]).run(TASK)

    files = sorted((tmp_path / "runs").glob("*.jsonl"))
    records = [_json.loads(line) for f in files for line in f.read_text().splitlines()]
    record = records[-1]

    assert record["status"] == "stopped"
    assert record["stop_reason"] == "loop_detected"
    assert record["output"] is None
    assert record["error"] == response.output


def test_the_coding_result_reports_the_stop(tmp_path, stopped):
    from effgen.cli.code.engine import CodeEngine
    from effgen.cli.code.permissions import PermissionMode

    engine = CodeEngine(model="scripted", workspace=tmp_path, mode=PermissionMode.PLAN)
    document = engine.result_from_response("do it", stopped).to_dict()

    assert document["stop_reason"] == "loop_detected"
    assert document["outcome"] == "stopped"
    assert document["success"] is False
    assert document["partial_output"] == stopped.partial.text


def test_a_batch_row_names_the_outcome(stopped):
    from effgen.core.batch import BatchRunner

    row = BatchRunner._result_row(0, stopped, query="q")

    assert row["outcome"] == "stopped"
    assert row["stop_reason"] == "loop_detected"
    assert row["success"] is False
    assert row["error"] == stopped.output


# --------------------------------------------------------------------------- #
# The server envelope
# --------------------------------------------------------------------------- #
def _stopped_response() -> AgentResponse:
    return AgentResponse(
        output="'m' did not write an answer: the run stopped (loop_detected).",
        success=False,
        metadata={
            "reason": "loop_detected",
            "partial": True,
            "partial_output": "36 | 35",
            "error": {"type": "UnsynthesizedToolResult", "category": "loop_detected",
                      "message": "stopped", "retryable": False},
        },
        partial=PartialResult(
            observations=("36", "35"), last_observation="35", text="36 | 35",
            iterations=3, tool_calls=2,
        ),
    )


def _client(monkeypatch, *, streaming: bool):
    """A test client whose agent always ends its run stopped."""
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    import effgen.core.agent as agent_mod
    import effgen.server.app as app_mod

    response = _stopped_response()

    class _Model(_Scripted):
        def __init__(self, *a, **k):
            super().__init__(["x"])

    monkeypatch.setattr(app_mod, "_get_pooled_model", lambda resolved: _Model())
    if streaming:
        monkeypatch.setattr(
            agent_mod.Agent, "stream", lambda self, *a, **k: iter(("36 | 35",)),
            raising=False,
        )
        monkeypatch.setattr(
            agent_mod.Agent, "last_stream_response", property(lambda self: response),
            raising=False,
        )
    else:
        def _raise(self, *a, **k):
            raise RunStoppedError(response)

        monkeypatch.setattr(agent_mod.Agent, "run", _raise, raising=False)
    return TestClient(app_mod.create_app(api_key="k")), response


def test_the_server_serves_a_stopped_run_as_200_cut_short(monkeypatch):
    """The request was valid and the server did its job; the model did not finish."""
    client, response = _client(monkeypatch, streaming=False)

    reply = client.post(
        "/v1/chat/completions", headers={"X-API-Key": "k"},
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["message"]["content"] == response.output
    extension = body["effgen"]
    assert extension["stop_reason"] == "loop_detected"
    assert extension["outcome"] == "stopped"
    assert extension["partial"]["text"] == "36 | 35"


def test_the_streamed_server_reply_ends_cut_short(monkeypatch):
    client, _response = _client(monkeypatch, streaming=True)

    reply = client.post(
        "/v1/chat/completions", headers={"X-API-Key": "k"},
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}],
              "stream": True},
    )

    assert reply.status_code == 200, reply.text
    finishes = [
        json.loads(line[6:])["choices"][0]["finish_reason"]
        for line in reply.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
        and json.loads(line[6:]).get("choices")
    ]
    assert finishes[-1] == "length"
