"""Layout of the ``effgen.core.agent_react`` split.

The ReAct module keeps the reasoning loop and the instructions the loop injects
into a scratchpad. Reading a model turn and running a provider's own tool loop
are separate concerns and live in their own modules, contributed to
:class:`~effgen.core.agent.Agent` as bases of ``AgentReActMixin``.

Three invariants make that split safe:

* ``AgentReActMixin`` stays the single class ``agent.py`` imports, and the new
  mixins are *bases* of it, so a subclass of ``AgentReActMixin`` alone still
  inherits every method;
* the loop and every nudge it appends stay in ``agent_react``, so the nudge
  strings and the strip-list that removes them can be compared in one place;
* each new module reads the config, response and runtime leaves, never
  ``agent.py``, so the import graph stays acyclic.
"""

from __future__ import annotations

import ast
import inspect

import pytest

import effgen.core.agent_citations as agent_citations
import effgen.core.agent_native_tools as agent_native_tools
import effgen.core.agent_react as agent_react
import effgen.core.agent_react_parsing as agent_react_parsing
import effgen.core.agent_stream_native as agent_stream_native
import effgen.core.agent_tool_execution as agent_tool_execution
import effgen.core.agent_tool_loop as agent_tool_loop
import effgen.core.result_relay as result_relay
from effgen.core.agent import Agent
from effgen.core.agent_react import AgentReActMixin

# Members that moved out of the ReAct module, and the module that owns each.
OWNERS = {
    # reading a model turn
    "_parse_react_response": "effgen.core.agent_react_parsing",
    "_parse_native_tool_calls": "effgen.core.agent_react_parsing",
    "_tool_call_result_to_dict": "effgen.core.agent_react_parsing",
    "_text_parse_strategy": "effgen.core.agent_react_parsing",
    "_model_advertises_tool_calling": "effgen.core.agent_react_parsing",
    "_model_tool_call_support": "effgen.core.agent_react_parsing",
    "_extract_partial_answer": "effgen.core.agent_react_parsing",
    "_partial_result": "effgen.core.agent_react_parsing",
    "_should_return_direct_calculator_result": "effgen.core.agent_react_parsing",
    # provider-native tool loops
    "_has_native_tools": "effgen.core.agent_native_tools",
    "_run_with_native_tools": "effgen.core.agent_native_tools",
    "_reasoning_only_native_response": "effgen.core.agent_native_tools",
    "_has_gemini_native_tools": "effgen.core.agent_native_tools",
    "_run_with_gemini_native_tools": "effgen.core.agent_native_tools",
    # earlier splits of the same module, pinned here so they cannot drift back
    "_execute_tool": "effgen.core.agent_tool_execution",
    "_execute_tool_once": "effgen.core.agent_tool_execution",
    "_map_input_to_parameters": "effgen.core.agent_tool_execution",
    "_native_tool_loop_hint": "effgen.core.agent_tool_execution",
    "_collect_citations": "effgen.core.agent_citations",
    "_attach_citations": "effgen.core.agent_citations",
    "_is_retrieval_tool": "effgen.core.agent_citations",
}

# The loop, and the helpers that build the text it injects or the failure it
# reports, stay in the ReAct module.
KEPT = (
    "_run_single_agent",
    "_run_with_sub_agents",
    "_is_context_retrieval_tool",
    "_context_answer_instruction",
    "_answer_shape_instruction",
    "_compose_closing",
    "_continuation_instruction",
    "_written_tool_call_detail",
    "_written_tool_call_response",
    "_stopped_outcome_response",
    "_repeated_tool_detail",
    "_iteration_cap_detail",
)

NEW_MIXINS = (
    agent_react_parsing.AgentReActParsingMixin,
    agent_native_tools.AgentNativeToolsMixin,
    agent_tool_execution.AgentToolExecutionMixin,
    agent_citations.AgentCitationsMixin,
)

NUDGES = (
    "NUDGE_CONTINUE",
    "NUDGE_HAVE_ANSWER",
    "NUDGE_HAVE_RESULTS",
    "NUDGE_ALREADY_COMPUTED",
    "NUDGE_NO_TOOLS",
    "NUDGE_NOT_USABLE",
)


def _defining_class(name: str) -> type:
    for cls in Agent.__mro__:
        if name in vars(cls):
            return cls
    raise AssertionError(f"{name!r} is not defined anywhere in Agent.__mro__")


@pytest.mark.parametrize(("name", "module"), sorted(OWNERS.items()))
def test_moved_members_are_owned_by_one_module(name: str, module: str) -> None:
    assert _defining_class(name).__module__ == module


@pytest.mark.parametrize("name", KEPT)
def test_the_loop_and_its_instructions_stay_in_the_react_module(name: str) -> None:
    assert _defining_class(name).__module__ == "effgen.core.agent_react"


@pytest.mark.parametrize("mixin", NEW_MIXINS, ids=lambda c: c.__name__)
def test_the_react_mixin_inherits_every_split_out_mixin(mixin: type) -> None:
    # ``class _StubAgent(AgentReActMixin)`` must keep the full surface, so the
    # mixins are bases of AgentReActMixin rather than extra bases on Agent.
    assert issubclass(AgentReActMixin, mixin)
    assert mixin in Agent.__mro__


def _imported_modules(module) -> set[str]:
    """Every module name *module* imports, however the import is written.

    Read from the syntax tree rather than the text: ``from effgen.core import
    agent`` and ``import effgen.core.agent`` name the same module as ``from
    effgen.core.agent import X`` but share no substring with it, and an import
    inside a function body is a dependency just as much as one at the top.
    """
    names: set[str] = set()
    package = module.__name__.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = f"{package}.{base}" if base else package
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize(
    "module",
    [
        agent_react_parsing,
        agent_native_tools,
        agent_tool_loop,
        agent_stream_native,
        result_relay,
    ],
    ids=lambda m: m.__name__,
)
def test_new_modules_do_not_import_the_agent_module(module) -> None:
    assert "effgen.core.agent" not in _imported_modules(module)


@pytest.mark.parametrize("nudge", NUDGES)
def test_every_nudge_is_injected_from_the_loop(nudge: str) -> None:
    """A nudge is appended by the loop or by the policy both loops share.

    ``test_answer_sanitization`` compares the injected nudges against the
    strip-list by reading those sources; a nudge injected from somewhere else
    would escape that comparison.
    """
    from effgen.core.agent_tool_loop import NativeToolLoop

    sources = (
        inspect.getsource(Agent._run_single_agent),
        inspect.getsource(NativeToolLoop),
    )
    assert any(nudge in src for src in sources)


def test_the_react_module_no_longer_defines_the_moved_members() -> None:
    src = inspect.getsource(agent_react)
    for name in OWNERS:
        assert f"    def {name}(" not in src, f"{name!r} is still defined in agent_react.py"


def test_the_native_stream_no_longer_builds_its_own_prompt() -> None:
    """The streamed native loop calls the shared builder instead of copying it.

    The two assemblies were line-for-line copies, and a line added to one of
    them reached only that path. Nothing prevents that drift while both exist.
    """
    assert _defining_class("_native_tool_prompt").__module__ == "effgen.core.agent_runtime"
    assert "self._native_tool_prompt(" in inspect.getsource(agent_stream_native)
    assert "self._native_tool_prompt(" in inspect.getsource(agent_react)


def test_a_bare_react_subclass_keeps_the_moved_methods() -> None:
    class _Stub(AgentReActMixin):
        pass

    for name in OWNERS:
        assert hasattr(_Stub, name), f"{name} unreachable from a bare AgentReActMixin subclass"
