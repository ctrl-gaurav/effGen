"""``effgen run`` hands back the conversation, as steps and as a document.

Three things a user needs from a finished run and could not get before: the
steps the run took printed without turning on debug logging, a ``--json``
document that parses for a run that called a tool, and a guarantee that
whatever a tool or an instruction happened to contain, a provider key is not in
what gets piped, saved or shared.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

#: A key-shaped string that is not a key: the scrubber matches it on shape.
FAKE_KEY = "sk-ZZZZfakefakefakefakefakefake00"


class Calc(BaseTool):
    """A deterministic tool, so a turn's call has one right answer."""

    def __init__(self, result: str = "36") -> None:
        super().__init__(metadata=ToolMetadata(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="expression", type=ParameterType.STRING,
                description="the expression", required=True,
            )],
        ))
        self._result = result

    async def _execute(self, **kwargs: Any) -> str:
        return self._result


class Scripted(BaseModel):
    """Replays a fixed script, so a command-line run is deterministic."""

    def __init__(self, script: list[dict]) -> None:
        super().__init__(model_name="scripted", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.script = script
        self.index = 0

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return True

    def tool_call_support(self) -> str:
        return "api"

    def streams_tool_calls(self) -> bool:
        return False

    def supports_conversation(self) -> bool:
        return True

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        turn = self.script[min(self.index, len(self.script) - 1)]
        self.index += 1
        calls = turn.get("calls")
        return GenerationResult(
            text=turn.get("text", ""),
            tokens_used=5,
            finish_reason="tool_calls" if calls else "stop",
            model_name=self.model_name,
            metadata={"tool_calls": calls} if calls else {},
        )

    def generate_stream(self, prompt: Any, config: Any = None, **kwargs: Any):
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


def _script(call_id: str) -> list[dict]:
    return [
        {"text": "Thought: multiply.",
         "calls": [{"id": call_id, "type": "function", "function": {
             "name": "calculator", "arguments": '{"expression": "6*6"}'}}]},
        {"text": "Thought: done.\nFinal Answer: 36"},
    ]


@pytest.fixture
def cli_agent(monkeypatch, tmp_path):
    """Give the command a scripted agent and a history directory of its own."""
    from effgen.cli import _main

    monkeypatch.setenv("EFFGEN_RUN_HISTORY", "0")
    monkeypatch.setenv("EFFGEN_WORKSPACE", str(tmp_path))

    def build(call_id: str = "call-1", tool_result: str = "36") -> None:
        def factory(config: Any = None, **kwargs: Any) -> Agent:
            settings: dict[str, Any] = {
                "name": "cli-agent",
                "model": Scripted(_script(call_id)),
                "tools": [Calc(tool_result)],
                "max_iterations": 4,
                "tool_calling_mode": "hybrid",
                "raise_on_error": False,
            }
            # Keep the persona the command line asked for; the model and the
            # tools are this fixture's, so the run is deterministic.
            persona = getattr(config, "system_prompt", None)
            if persona:
                settings["system_prompt"] = persona
            return Agent(AgentConfig(**settings))

        monkeypatch.setattr(_main, "Agent", factory)

    build()
    return build


def _run(argv: list[str]) -> tuple[int, str, str]:
    """Run one ``effgen`` command in this process, capturing both streams."""
    out, err = io.StringIO(), io.StringIO()
    rc = 0
    argv_before = sys.argv
    sys.argv = ["effgen", *argv]
    from effgen.cli._main import main as cli_main

    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli_main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        sys.argv = argv_before
    return rc, out.getvalue(), err.getvalue()


TASK = "what is 6*6?"
BASE = ["run", TASK, "-m", "scripted", "-t", "calculator", "-q"]


# --- --show-thread -----------------------------------------------------------


def test_show_thread_prints_every_step_the_run_took(cli_agent):
    """The conversation is printed after the answer, with no debug logging on."""
    rc, out, err = _run([*BASE, "--show-thread"])
    printed = out + err

    assert rc == 0
    assert "Conversation" in printed
    assert "action (calculator)" in printed
    assert "observation" in printed
    assert "6*6" in printed


def test_without_the_flag_the_conversation_is_not_printed(cli_agent):
    """The flag is what asks for it; a plain run reads as it did before."""
    _, out, err = _run(BASE)
    assert "action (calculator)" not in out + err


def test_two_runs_of_the_same_path_print_the_same_conversation(cli_agent):
    """Two renderings diff cleanly: no ids, no timings, no run id."""
    cli_agent(call_id="call-aaa")
    _, first_out, first_err = _run([*BASE, "--show-thread"])
    cli_agent(call_id="call-zzz")
    _, second_out, second_err = _run([*BASE, "--show-thread"])

    def conversation(text: str) -> list[str]:
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if "Conversation" in line)
        return lines[start + 1:]

    assert conversation(first_out + first_err) == conversation(second_out + second_err)


# --- --json on a run that used a tool ----------------------------------------


def test_json_parses_for_a_run_that_called_a_tool(cli_agent):
    """The defect this closes: the document used to be unserialisable."""
    rc, out, _ = _run([*BASE, "--json"])
    document = json.loads(out)

    assert rc == 0
    assert document["tool_calls"] == 1
    assert document["execution_tree"]
    assert document["tool_call_details"][0]["name"] == "calculator"


def test_json_carries_the_conversation(cli_agent):
    """Everything ``--show-thread`` prints is in the document as data."""
    _, out, _ = _run([*BASE, "--json"])
    steps = json.loads(out)["metadata"]["thread"]["steps"]

    kinds = [s["kind"] for s in steps]
    assert kinds[:2] == ["system", "task"]
    assert "action" in kinds and "observation" in kinds


def test_the_output_file_and_the_card_are_written_from_the_same_document(
    cli_agent, tmp_path,
):
    """``-o`` and ``--card`` compose, and neither raises on a tool call."""
    saved = tmp_path / "run.json"
    card = tmp_path / "run.html"
    rc, _, _ = _run([*BASE, "-o", str(saved), "--card", str(card)])

    assert rc == 0
    document = json.loads(saved.read_text(encoding="utf-8"))
    assert document["tool_call_details"][0]["name"] == "calculator"
    html = card.read_text(encoding="utf-8")
    assert "Conversation" in html
    assert "action (calculator)" in html


# --- secrets ------------------------------------------------------------------


def test_a_key_in_a_tool_result_reaches_neither_the_print_nor_the_document(
    cli_agent, tmp_path,
):
    """The scrubber runs on every path a run's text leaves the process by."""
    cli_agent(tool_result=f"36 (billed to {FAKE_KEY})")
    saved = tmp_path / "run.json"
    card = tmp_path / "run.html"
    rc, out, err = _run([*BASE, "--show-thread", "-o", str(saved), "--card", str(card)])

    printed = out + err
    conversation = printed[printed.index("Conversation"):]

    assert rc == 0
    assert FAKE_KEY not in conversation
    assert "<REDACTED:openai_key>" in conversation
    assert FAKE_KEY not in saved.read_text(encoding="utf-8")
    assert FAKE_KEY not in card.read_text(encoding="utf-8")
    # The answer panel prints what the run answered, unredacted — the tool put
    # the key in the answer itself. Every document the command writes is
    # scrubbed; the terminal answer is the run's own words.
    assert FAKE_KEY in printed[:printed.index("Conversation")]


def test_a_key_in_the_system_prompt_is_redacted_too(cli_agent):
    """An instruction is as likely a place for a key as a tool's output."""
    rc, out, _ = _run([
        *BASE, "--json", "--system-prompt", f"Authenticate with {FAKE_KEY}.",
    ])

    assert rc == 0
    assert FAKE_KEY not in out
    assert "<REDACTED:openai_key>" in out


def test_a_key_typed_on_the_command_line_is_not_stamped_into_the_card(
    cli_agent, tmp_path,
):
    """The card's header records the invocation, and that is a document too.

    A task or a persona typed on the command line is as likely a place for a
    key as a tool's output, and the header is stamped into a page meant to be
    shared.
    """
    card = tmp_path / "run.html"
    rc, _, _ = _run([
        "run", f"what is 6*6? my key is {FAKE_KEY}",
        "-m", "scripted", "-t", "calculator", "-q",
        "--system-prompt", f"Authenticate with {FAKE_KEY}.",
        "--card", str(card),
    ])

    html = card.read_text(encoding="utf-8")

    assert rc == 0
    assert FAKE_KEY not in html
    # The header is HTML-escaped, so the placeholder appears escaped with it.
    assert "&lt;REDACTED:openai_key&gt;" in html
