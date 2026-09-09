"""A tool call the model writes out as text is reported as a failed turn.

A model that is offered tools but answers with the call as literal prose —
``<file_operations> {"operation": "write", …}`` — runs no tool, yet the answer
reads like the work happened. These pin the detector
(:func:`effgen.core.agent_runtime.find_written_tool_call`), the failure the tool
loop reports when it fires, and the tool-calling path every tool-loop result
names.

The fixtures are the strings small models actually produced (groq
``openai/gpt-oss-20b`` driving the coding agent). The loop is driven by an
in-process scripted model so the checks are deterministic and offline — live
behavior is proven separately against real providers.
"""

from __future__ import annotations

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.agent_runtime import find_written_tool_call, written_call_only
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools import get_registry
from effgen.tools.builtin.calculator import Calculator

# The answer groq's llama-3.1-8b returned instead of calling the tools.
WRITTEN_CALL_ANSWER = (
    '<file_operations> {"operation": "write", "path": "greet.py", "content": '
    '"def greet(name): return \'Hello, \' + name + \'!\'"} </file_operations>\n\n'
    '<code_executor> {"code": "greet.py", "language": "python", "timeout": "30"} '
    "</code_executor>"
)

CODE_TOOL_NAMES = {"file_operations", "code_executor", "python_repl", "bash"}


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        (WRITTEN_CALL_ANSWER, "file_operations"),
        # The tag with no space before the arguments.
        ('<code_executor>{"code": "print(1)"}</code_executor>', "code_executor"),
        # A ``function=`` wrapper whose JSON the parser could not read.
        ('<function=file_operations>{"operation": "read", "path": }', "file_operations"),
        # No tag at all — the bare name in front of its arguments.
        ('file_operations {"operation": "list"}\nI listed the directory.', "file_operations"),
        # A tagged call naming its tool inside the object.
        ('<tool_call>{"name": "bash", "arguments": {"command": "ls"', "bash"),
        # Mid-answer, after prose.
        ('I will write the file now.\n<file_operations> {"operation": "write"}', "file_operations"),
    ],
)
def test_written_call_is_detected(text, expected):
    assert find_written_tool_call(text, CODE_TOOL_NAMES) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The 10th Fibonacci number is 55.",
        # A tool the agent does not have is not this failure.
        'web_search {"query": "python"}',
        # Documentation of a call, fenced or inline, is an answer about a tool.
        'Call it like `file_operations {"operation": "read"}` from your script.',
        '```\nfile_operations {"operation": "read"}\n```\nThat is the shape.',
        # Ordinary JSON in an answer.
        'Here is the config you asked for: {"retries": 3}',
        # A markdown table row that happens to name a tool.
        "| file_operations | reads and writes files |",
        # Code the agent was asked to write.
        'def main():\n    return {"ok": True}',
    ],
)
def test_ordinary_answer_is_not_flagged(text):
    assert find_written_tool_call(text, CODE_TOOL_NAMES) is None


def test_no_tools_never_flags():
    assert find_written_tool_call(WRITTEN_CALL_ANSWER, set()) is None
    assert find_written_tool_call(None, CODE_TOOL_NAMES) is None


# --- the loop reports it ---------------------------------------------------

class _ScriptedModel(BaseModel):
    """Returns one fixed response per ``generate()`` call, repeating the last."""

    def __init__(self, responses: list[str], *, native: bool = False):
        super().__init__(model_name="scripted-model", model_type=ModelType.OPENAI)
        self._responses = responses
        self._native = native
        self.calls = 0

    def load(self) -> None:  # pragma: no cover - trivial
        pass

    def generate(self, prompt, config=None, **kwargs):
        idx = min(self.calls, len(self._responses) - 1)
        text = self._responses[idx]
        self.calls += 1
        return GenerationResult(
            text=text, tokens_used=7, finish_reason="stop",
            model_name=self.model_name, metadata={},
        )

    def generate_stream(self, prompt, config=None, **kwargs):  # pragma: no cover
        yield self.generate(prompt).text

    def count_tokens(self, text: str) -> TokenCount:  # pragma: no cover
        return TokenCount(count=len(text.split()), model_name=self.model_name)

    def get_context_length(self) -> int:  # pragma: no cover
        return 4096

    def unload(self) -> None:  # pragma: no cover
        pass

    def generate_batch(self, prompts, config=None, **kwargs):  # pragma: no cover
        return [self.generate(p, config=config) for p in prompts]

    def generate_with_tools(self, prompt, tools, config=None, **kwargs):  # pragma: no cover
        return self.generate(prompt, config=config)

    def supports_function_calling(self) -> bool:
        return self._native

    def supports_tool_calling(self) -> bool:
        return self._native


def _agent(responses, *, mode="react", native=False, max_iterations=3) -> Agent:
    return Agent(config=AgentConfig(
        raise_on_error=False,  # these assert the failure response, not the raise
        name="written-call-test",
        model=_ScriptedModel(responses, native=native),
        tools=[Calculator()],
        max_iterations=max_iterations,
        tool_calling_mode=mode,
    ))


CALC_CALL_ANSWER = 'calculator {"operation": "calculate", "expression": "379*68"}'


def test_written_call_turn_fails_instead_of_answering():
    """The answer that only describes the call is a failure, not a result."""
    agent = _agent([f"Final Answer: {CALC_CALL_ANSWER}"])
    resp = agent.run("What is 379 * 68?")
    assert resp.success is False
    assert resp.tool_calls == 0
    assert resp.metadata["reason"] == "written_tool_call"
    detail = resp.metadata["error"]
    assert detail["tool"] == "calculator"
    assert detail["category"] == "written_tool_call"
    assert detail["retryable"] is False
    assert "calculator" in resp.output
    assert "instead of calling the tool" in resp.output


def test_tagged_call_block_fails_on_the_first_turn():
    """The shape a small model emits under native tool calling costs one call.

    ``<calculator> {…} </calculator>`` survives answer sanitization, so without
    the guard it is returned as a successful answer. It is deterministic for the
    model that produces it, so the turn is reported at once rather than retried.
    """
    agent = _agent(
        ['<calculator> {"operation": "calculate", "expression": "379*68"} </calculator>'],
        mode="hybrid", native=True,
    )
    resp = agent.run("What is 379 * 68?")
    assert resp.success is False
    assert resp.metadata["reason"] == "written_tool_call"
    assert resp.metadata["error"]["tool"] == "calculator"
    assert agent.model.calls == 1, "a deterministic outcome must not be re-billed"


def test_message_names_a_model_that_calls_tools():
    agent = _agent([f"Final Answer: {CALC_CALL_ANSWER}"], native=False)
    message = agent.run("What is 379 * 68?").output
    assert "does not advertise native tool calling" in message
    assert "gpt-5-nano" in message


def test_react_path_on_a_native_capable_model_points_at_native_mode():
    """A model that advertises tool calling is asked for the native path."""
    agent = _agent([f"Final Answer: {CALC_CALL_ANSWER}"], mode="react", native=True)
    resp = agent.run("What is 379 * 68?")
    assert resp.metadata["error"]["tool_calling_strategy"] == "react"
    assert "tool_calling_mode='native'" in resp.output


def test_native_path_recommends_a_different_model():
    """Tools already sent to the provider's API: the model is the problem."""
    agent = _agent([CALC_CALL_ANSWER], mode="hybrid", native=True)
    resp = agent.run("What is 379 * 68?")
    assert resp.success is False
    assert resp.metadata["error"]["tool_calling_strategy"] == "hybrid"
    assert "provider's tool-calling API" in resp.output


def test_real_answer_after_a_tool_call_still_succeeds():
    """The guard never turns a genuine tool-backed answer into a failure."""
    agent = _agent([
        (
            'Thought: compute it.\nAction: calculator\n'
            'Action Input: {"operation": "calculate", "expression": "379*68"}'
        ),
        "Final Answer: 379 * 68 is 25772.",
    ])
    resp = agent.run("Explain and compute 379 times 68")
    assert resp.success is True
    assert resp.tool_calls == 1
    assert "25772" in resp.output


def test_answer_documenting_a_call_still_succeeds():
    """An answer *about* a tool call is an answer, not a written-out call."""
    agent = _agent([
        "Final Answer: Call it as `calculator {\"expression\": \"2+2\"}` from your code.",
    ])
    resp = agent.run("How do I call the calculator tool?")
    assert resp.success is True
    assert "calculator" in resp.output


# --- a call the run really made -------------------------------------------

def _fs_agent(responses, tmp_path, *, mode="react", native=False) -> Agent:
    """An agent over the real file tools, whose model is scripted turn by turn."""
    registry = get_registry()
    return Agent(config=AgentConfig(
        raise_on_error=False,  # these assert the failure response, not the raise
        name="written-call-fs",
        model=_ScriptedModel(responses, native=native),
        tools=[
            registry.get_tool_sync("file_operations"),
            registry.get_tool_sync("code_executor"),
        ],
        max_iterations=4,
        tool_calling_mode=mode,
    ))


def _listing_call(directory) -> str:
    return (
        "Thought: list the directory.\nAction: file_operations\n"
        'Action Input: {"operation": "list", "path": "' + str(directory) + '"}'
    )


def test_recap_of_a_call_that_ran_is_an_answer(tmp_path):
    """An answer that recaps a call the run made keeps its result.

    Asking for the arguments back is a real request, and a model that answers it
    after doing the work has answered: reporting that nothing was carried out
    would contradict the files the run wrote.
    """
    (tmp_path / "notes.txt").write_text("hello\n")
    agent = _fs_agent([
        _listing_call(tmp_path),
        (
            'Final Answer: I called file_operations {"operation": "list", "path": "."} '
            "and the directory holds notes.txt."
        ),
    ], tmp_path)
    resp = agent.run(f"List {tmp_path} and show the arguments you used.")
    assert resp.success is True
    assert resp.tool_calls == 1
    assert "notes.txt" in resp.output


def test_answer_that_is_only_the_call_fails_even_after_the_tool_ran(tmp_path):
    """A call block returned *as* the answer is still a failed turn.

    The tool ran, so the message says the answer carried no result rather than
    claiming the tool never ran.
    """
    (tmp_path / "notes.txt").write_text("hello\n")
    agent = _fs_agent([
        _listing_call(tmp_path),
        'Final Answer: file_operations {"operation": "list", "path": "."}',
    ], tmp_path)
    resp = agent.run(f"List the files in {tmp_path}.")
    assert resp.success is False
    assert resp.metadata["reason"] == "written_tool_call"
    assert "as its answer" in resp.output


def test_written_call_for_a_tool_that_never_ran_fails(tmp_path):
    """A second tool's call, written out beside real content, is still the failure."""
    (tmp_path / "notes.txt").write_text("hello\n")
    agent = _fs_agent([
        _listing_call(tmp_path),
        (
            "Final Answer: The directory holds notes.txt. Next I will run "
            'code_executor {"code": "print(1)", "language": "python"}'
        ),
    ], tmp_path)
    resp = agent.run(f"List {tmp_path} and run something.")
    assert resp.success is False
    assert resp.metadata["error"]["tool"] == "code_executor"
    assert "never ran" in resp.output


def test_written_call_only_reads_a_bare_block():
    """``written_call_only`` separates a bare call block from a recap."""
    block = (
        '<file_operations> {"operation": "write", "path": "a.py", "content": "x=1"} '
        "</file_operations>"
    )
    assert written_call_only(block, CODE_TOOL_NAMES) is True
    assert written_call_only(f"I ran {block} and wrote a.py.", CODE_TOOL_NAMES) is False
    assert written_call_only("Nothing to report.", CODE_TOOL_NAMES) is False


# --- shapes sanitization mangles first ------------------------------------

@pytest.mark.parametrize(
    "answer",
    [
        # A ``function=`` call cut off by the token cap: sanitizing leaves the
        # arguments behind as a bare JSON fragment.
        '<function=calculator>{"operation": "calculate", "expression": "379*',
        # A tagged call whose arguments survive as an object.
        '<|python_tag|>{"name": "calculator", "parameters": {"expression": "379*68"}}',
    ],
)
def test_a_call_sanitization_mangles_is_still_reported(answer):
    """The text as the model wrote it is scanned, not only the cleaned answer."""
    agent = _agent([answer], mode="react")
    resp = agent.run("What is 379 * 68?")
    assert resp.success is False
    assert resp.metadata["reason"] == "written_tool_call"
    assert resp.metadata["error"]["tool"] == "calculator"


def test_react_text_path_reports_the_cause_not_the_cap():
    """A turn that writes the call out names the cause instead of the cap.

    Under the ReAct text protocol the block parses as neither an action nor an
    answer, so without this the loop spent its whole budget and reported only
    that the iteration cap was reached.
    """
    agent = _agent(
        ['<calculator> {"operation": "calculate", "expression": "379*68"} </calculator>'],
        mode="react", max_iterations=6,
    )
    resp = agent.run("What is 379 * 68?")
    assert resp.success is False
    assert resp.metadata["reason"] == "written_tool_call"
    assert agent.model.calls == 2, "one nudge, then the failure is reported"


# --- the run names the path it took ---------------------------------------

def test_successful_run_names_the_tool_calling_path():
    agent = _agent(["Final Answer: 25772."])
    resp = agent.run("What is 379 * 68?")
    assert resp.metadata["tool_calling_strategy"] == "react"


def test_iteration_cap_names_the_tool_calling_path():
    agent = _agent(["Thought: still thinking."], max_iterations=2)
    resp = agent.run("What is 379 * 68?")
    assert resp.success is False
    assert resp.metadata["reason"].startswith("max_iterations")
    assert resp.metadata["tool_calling_strategy"] == "react"


# ---------------------------------------------------------------------------
# A tagged call whose body is a query string, not JSON
# ---------------------------------------------------------------------------
#
# Measured live on groq:openai/gpt-oss-20b driving `effgen code`: one run in
# two produced
#
#   <file_operations>operation=write&path=greet.py&content=greet(name)=print(&quot;…
#
# The tag is there but the body is a query string, sometimes HTML-escaped, so
# both JSON readers found nothing and the turn was reported as a successful
# answer with `files_written=[]` — the exact failure this detector exists to
# stop, in a shape it did not cover.


class TestQueryStringTaggedCall:
    TOOLS = {"file_operations", "code_executor", "calculator"}

    def test_the_shape_measured_live_is_detected(self):
        from effgen.core.agent_runtime import find_written_tool_call

        answer = (
            "<file_operations>operation=write&path=greet.py&"
            "content=greet(name)=print(&quot;Hello, &quot;+name+&quot;!&quot;)"
        )
        assert find_written_tool_call(answer, self.TOOLS) == "file_operations"

    def test_prose_that_merely_names_the_tool_is_not_flagged(self):
        from effgen.core.agent_runtime import find_written_tool_call

        for answer in (
            "The <file_operations> tool writes files.",
            "Use file_operations to write a file; operation=write is the argument.",
            "<file_operations> is one of the tools available here.",
        ):
            assert find_written_tool_call(answer, self.TOOLS) is None, answer

    def test_a_code_span_is_still_exempt(self):
        """A coding agent is often asked to *show* a call."""
        from effgen.core.agent_runtime import find_written_tool_call

        answer = "```\n<file_operations>operation=write&path=x.py\n```"
        assert find_written_tool_call(answer, self.TOOLS) is None

    def test_a_tool_the_agent_does_not_hold_is_not_flagged(self):
        from effgen.core.agent_runtime import find_written_tool_call

        assert find_written_tool_call(
            "<unknown_thing>operation=write&x=1", self.TOOLS
        ) is None


# ---------------------------------------------------------------------------
# Python call syntax — only when the call IS the answer
# ---------------------------------------------------------------------------
#
# ``name(...)`` is ordinary prose and ordinary code, and a coding agent is
# frequently *asked* to write or explain a call, so this shape cannot be flagged
# wherever it appears — it carries no JSON object to anchor on the way the other
# shapes do. The rule is the one the report proposes: with every such call
# removed, an answer that merely mentions one still has words left; an answer
# that is nothing but calls does not.


class TestStandalonePythonCallSyntax:
    TOOLS = {"file_operations", "code_executor", "calculator"}

    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("file_operations('write', 'greet.py', 'print(1)')", "file_operations"),
            ('file_operations(operation="write", path="greet.py")', "file_operations"),
            (
                (
                    'file_operations(operation="write", path="greet.py")\n'
                    'code_executor("python greet.py")'
                ),
                "file_operations",
            ),
            ("calculator(expression='1367 * 89')", "calculator"),
        ],
    )
    def test_a_call_that_is_the_whole_answer_is_flagged(self, answer, expected):
        from effgen.core.agent_runtime import find_written_tool_call

        assert find_written_tool_call(answer, self.TOOLS) == expected

    @pytest.mark.parametrize(
        "answer",
        [
            "I used calculator(2+2) and got 4.",
            (
                'To write a file, call file_operations(operation="write", '
                'path=...) with the path you want.'
            ),
            "Here is the diff:\n- calculator(1)\n+ calculator(2)\nThat fixes it.",
            "The pytest function asserts that calculator(expression='2+2') is 4.",
            "The answer is 42.",
            "unknown_tool('x')",
        ],
    )
    def test_writing_about_a_call_is_not_flagged(self, answer):
        """The precision profile the detector was built to protect."""
        from effgen.core.agent_runtime import find_written_tool_call

        assert find_written_tool_call(answer, self.TOOLS) is None

    def test_a_call_inside_a_code_span_is_not_flagged(self):
        from effgen.core.agent_runtime import find_written_tool_call

        assert find_written_tool_call(
            "```python\ncalculator(expression='2+2')\n```", self.TOOLS
        ) is None


class TestXmlTaggedWrittenCall:
    """A call written as nested tags is a written-out call, not an answer."""

    TOOLS = ("calculator", "retrieval")

    def test_a_tagged_xml_call_is_flagged(self):
        from effgen.core.agent_runtime import find_written_tool_call

        assert find_written_tool_call(
            "<tool_call><function=calculator><parameter=expression>6*7</parameter>"
            "</function></tool_call>",
            self.TOOLS,
        ) == "calculator"

    def test_a_tool_the_agent_does_not_hold_is_not_flagged(self):
        from effgen.core.agent_runtime import find_written_tool_call

        assert find_written_tool_call(
            "<function=send_email><parameter=to>a@b.c</parameter></function>",
            self.TOOLS,
        ) is None

    def test_prose_about_the_tags_is_not_flagged(self):
        from effgen.core.agent_runtime import find_written_tool_call

        assert find_written_tool_call(
            "A template may render <function=NAME> tags around each parameter.",
            self.TOOLS,
        ) is None
