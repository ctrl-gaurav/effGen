"""Layout of the ``effgen.core.agent`` package split.

The agent module keeps construction, per-call state and lifecycle; the run
funnel, the prompt assembly and the result assembly are contributed by mixins in
their own modules. Two invariants make that split safe to read and to extend:

* ``effgen.core.agent`` stays the import path for every public name and for the
  attributes tests and callers patch, whichever module now defines them;
* every attribute reaches :class:`~effgen.core.agent.Agent` from exactly one
  class in its MRO, so adding a mixin can never silently shadow a method.
"""

from __future__ import annotations

import inspect

import pytest

import effgen.core.agent as agent_mod
import effgen.core.agent_config as agent_config
import effgen.core.agent_orchestration as agent_orchestration
import effgen.core.agent_prompting as agent_prompting
import effgen.core.agent_response as agent_response
import effgen.core.agent_result as agent_result
import effgen.core.agent_runtime as agent_runtime
from effgen.core.agent import Agent


class _EmptyClassBody:
    """A class body that declares nothing, used to read what the interpreter adds."""


#: Attributes the interpreter itself puts on every class body, which therefore
#: appear on every class in a MRO and are not definitions anyone wrote. Read off
#: an empty class rather than listed, so an interpreter that stamps a new one —
#: 3.13 added ``__firstlineno__`` and ``__static_attributes__`` — does not read
#: as a name defined twice. The explicit names cover slots a class only carries
#: when it does not inherit them.
INTERPRETER_CLASS_ATTRIBUTES = frozenset(vars(_EmptyClassBody)) | {
    # Stamped on any class whose body contains an annotation anywhere, including
    # one inside a ``TYPE_CHECKING`` block the interpreter never runs. Two
    # mixins that each declare what a sibling contributes would otherwise read
    # as defining the same name twice.
    "__annotations__",
    "__module__",
    "__doc__",
    "__dict__",
    "__weakref__",
    "__qualname__",
}

# Members that moved out of the agent module, and the module that owns each.
OWNERS = {
    # run entry points and everything wrapped around a single run
    "run": "effgen.core.agent_orchestration",
    "run_async": "effgen.core.agent_orchestration",
    "run_batch": "effgen.core.agent_orchestration",
    "run_background": "effgen.core.agent_orchestration",
    "get_task_status": "effgen.core.agent_orchestration",
    "get_task_result": "effgen.core.agent_orchestration",
    "cancel_task": "effgen.core.agent_orchestration",
    "pause_task": "effgen.core.agent_orchestration",
    "resume_task": "effgen.core.agent_orchestration",
    "resume": "effgen.core.agent_orchestration",
    "synthesize": "effgen.core.agent_orchestration",
    # prompt and context assembly
    "REACT_PROMPT_TEMPLATE": "effgen.core.agent_prompting",
    "_build_system_prompt": "effgen.core.agent_prompting",
    "_auto_detect_verbose": "effgen.core.agent_prompting",
    "_get_tools_description": "effgen.core.agent_prompting",
    "_format_conversation_history": "effgen.core.agent_prompting",
    "_get_anthropic_system": "effgen.core.agent_prompting",
    "_get_anthropic_tools": "effgen.core.agent_prompting",
    # the native/hybrid turn prompt, shared by the blocking and streamed loops
    "_native_tool_prompt": "effgen.core.agent_runtime",
    "_tool_contract": "effgen.core.agent_runtime",
    # source mining, beside the methods that read it
    "_RETRIEVAL_TOOL_TAGS": "effgen.core.agent_citations",
    # result assembly
    "_stamp_run_identity": "effgen.core.agent_result",
    "_resolve_provider": "effgen.core.agent_result",
    "_session_turn_metadata": "effgen.core.agent_result",
    "_record_dashboard_run": "effgen.core.agent_result",
}

# Members the agent module itself must keep: construction, per-call state,
# lifecycle, and the native-tool guard that callers patch on the class.
KEPT = (
    "__init__",
    "_validate_native_tool_compatibility",
    "_agent_call_scope",
    "_get_call_state",
    "execution_tracker",
    "add_tool",
    "remove_tool",
    "reset_memory",
    "save_state",
    "load_state",
    "close",
    "aclose",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    "__del__",
    "get_execution_summary",
    "__repr__",
)

NEW_MODULES = (agent_orchestration, agent_prompting, agent_result)


def _defining_class(name: str) -> type:
    for cls in Agent.__mro__:
        if name in vars(cls):
            return cls
    raise AssertionError(f"{name!r} is not defined anywhere in Agent.__mro__")


@pytest.mark.parametrize(("name", "module"), sorted(OWNERS.items()))
def test_moved_members_are_owned_by_one_module(name: str, module: str) -> None:
    assert _defining_class(name).__module__ == module


@pytest.mark.parametrize("name", KEPT)
def test_construction_and_lifecycle_stay_in_the_agent_module(name: str) -> None:
    assert _defining_class(name).__module__ == "effgen.core.agent"


def test_agent_class_is_defined_in_the_agent_module() -> None:
    # ``patch("effgen.core.agent.Agent")`` and
    # ``patch("effgen.core.agent.Agent._validate_native_tool_compatibility")``
    # both resolve against this module.
    assert Agent.__module__ == "effgen.core.agent"
    assert agent_mod.Agent is Agent
    assert "_validate_native_tool_compatibility" in vars(Agent)


def test_public_names_still_resolve_from_the_agent_module() -> None:
    assert agent_mod.AgentConfig is agent_config.AgentConfig
    assert agent_mod.AgentMode is agent_config.AgentMode
    assert agent_mod.AgentResponse is agent_response.AgentResponse
    assert agent_mod.StreamEvent is agent_response.StreamEvent
    assert agent_mod.sanitize_final_answer is agent_runtime.sanitize_final_answer
    assert agent_mod._RUN_KWARGS is agent_config._RUN_KWARGS
    assert agent_mod._MODEL_LOAD_KWARGS is agent_config._MODEL_LOAD_KWARGS


@pytest.mark.parametrize("module", NEW_MODULES, ids=lambda m: m.__name__)
def test_new_modules_do_not_import_the_agent_module(module) -> None:
    # Keeps the dependency graph acyclic: the mixins read the config, response
    # and runtime leaves, never ``agent.py``.
    src = inspect.getsource(module)
    assert "from .agent import" not in src
    assert "from effgen.core.agent import" not in src


def test_every_agent_attribute_is_defined_once_across_the_mro() -> None:
    """No mixin may shadow a name another class in the MRO already defines.

    Two definitions of the same name would make the answer depend on the base
    order, so a split that duplicates one is rejected here rather than in a
    behaviour test that happens to notice.
    """
    seen: dict[str, list[str]] = {}
    for cls in Agent.__mro__:
        if cls is object:
            continue
        for name in vars(cls):
            if name in INTERPRETER_CLASS_ATTRIBUTES:
                continue
            seen.setdefault(name, []).append(f"{cls.__module__}.{cls.__name__}")
    duplicates = {n: where for n, where in seen.items() if len(where) > 1}
    assert not duplicates, f"defined more than once across Agent.__mro__: {duplicates}"


def test_retrieval_tags_still_reach_the_source_miner() -> None:
    """A class attribute a mixin reads must land on the agent, wherever it lives."""
    assert "wikipedia" in Agent._RETRIEVAL_TOOL_TAGS

    class _Meta:
        category = "search"
        tags = ["wikipedia"]

    class _Tool:
        metadata = _Meta()

    assert Agent._is_retrieval_tool(Agent, _Tool(), "wiki_lookup") is True
    assert Agent._is_retrieval_tool(Agent, object(), "calculator") is False


def test_agent_module_is_a_composition_layer() -> None:
    """The agent module no longer carries the members it delegates."""
    src = inspect.getsource(agent_mod)
    for name in ("def run(", "def run_async(", "def run_batch(", "def synthesize(",
                 "def _stamp_run_identity(", "def _get_tools_description("):
        assert name not in src, f"{name!r} is still defined in agent.py"
