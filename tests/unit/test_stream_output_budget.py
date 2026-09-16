"""Every path that generates honours the same output-token budget.

`Agent.run()` folded `AgentConfig(max_tokens=...)` into the call; the two older
streaming paths skipped that step and went straight from a per-call value to the
model default. The same agent then answered at two different lengths depending
on whether `run()` or `stream()` was called.
"""

from __future__ import annotations

import inspect

import pytest

from effgen.core import agent_loop, agent_stream_native, agent_streaming
from effgen.core.agent_runtime import resolve_output_budget


class _Model:
    """A model with no opinion, so the fallback is the module default."""


class _StubAgent:
    """The least an agent has to declare for its turn to be resolved."""

    model = _Model()

    class _Strategy:
        name = "react"

    _tool_calling_strategy = _Strategy()

    class config:
        temperature = 0.2
        max_tokens = 333
        top_p = 0.31
        top_k = 17
        seed = 4242
        presence_penalty = 0.5
        frequency_penalty = 0.25
        repetition_penalty = 1.15
        system_prompt_template = None
        answer_style = None

    def _interleaves_reasoning(self, model):
        return False

    def _effective_output_schema(self):
        return None


class TestPrecedence:
    def test_a_per_call_value_wins(self):
        assert resolve_output_budget(333, 777, _Model()) == 333

    def test_the_configured_default_is_used_when_no_per_call_value(self):
        assert resolve_output_budget(None, 777, _Model()) == 777

    def test_the_model_default_is_the_last_resort(self):
        from effgen.models._adapter_utils import default_max_output_tokens

        model = _Model()
        assert resolve_output_budget(None, None, model) == default_max_output_tokens(model)

    def test_a_configured_zero_is_honoured_not_treated_as_unset(self):
        """`0` is a value; only `None` means "not set"."""
        assert resolve_output_budget(None, 0, _Model()) == 0

    def test_a_declared_schema_bounds_the_budget_below_the_constant(self):
        """A run that asked for one integer stops asking for a report."""
        one_integer = {"type": "object", "properties": {"n": {"type": "integer"}}}
        assert resolve_output_budget(None, None, _Model(),
                                     output_schema=one_integer) == 256

    def test_a_bigger_declared_schema_asks_for_more(self):
        wide = {"type": "object", "properties": {
            "summary": {"type": "string"},
            "items": {"type": "array", "items": {"type": "string"}},
        }}
        assert resolve_output_budget(None, None, _Model(), output_schema=wide) > 1024

    def test_a_pinned_value_still_wins_over_a_declared_schema(self):
        one_integer = {"type": "object", "properties": {"n": {"type": "integer"}}}
        assert resolve_output_budget(9, None, _Model(),
                                     output_schema=one_integer) == 9
        assert resolve_output_budget(None, 9, _Model(),
                                     output_schema=one_integer) == 9

    def test_a_reasoning_model_is_never_budgeted_below_its_floor(self):
        """The schema bound may not starve a model that reasons before it answers."""
        from effgen.models._adapter_utils import REASONING_OUTPUT_FLOOR

        class _Reasons(_Model):
            _is_reasoning_model = True

        one_integer = {"type": "object", "properties": {"n": {"type": "integer"}}}
        assert resolve_output_budget(
            None, None, _Reasons(), output_schema=one_integer
        ) == REASONING_OUTPUT_FLOOR


class TestEveryPathUsesIt:
    """The call sites drifting apart is how this defect arose.

    There are two left, and both go through the shared helper: the loop every
    run takes, and the direct stream a tool-free agent takes.
    """

    @pytest.mark.parametrize(
        "module", [agent_loop, agent_streaming], ids=["loop", "direct-stream"]
    )
    def test_the_module_resolves_through_the_shared_helper(self, module):
        source = inspect.getsource(module)
        assert "resolve_output_budget(" in source or "resolve_turn_config(" in source

    @pytest.mark.parametrize(
        "module",
        [agent_loop, agent_streaming, agent_stream_native],
        ids=["loop", "streaming", "native"],
    )
    def test_no_path_reaches_for_the_model_default_directly(self, module):
        """Calling `default_max_output_tokens` here is what skipped the config."""
        source = inspect.getsource(module)
        assert "default_max_output_tokens(" not in source

    def test_the_budget_is_resolved_once_for_every_turn(self):
        """One resolver, called from the one place a turn's settings are built."""
        assert inspect.getsource(agent_loop).count("resolve_output_budget(") == 1

    def test_a_streamed_run_and_a_blocking_run_ask_for_the_same_budget(self):
        """The same agent, both entry points, one number."""
        from effgen.core.agent_loop import resolve_turn_config

        cfg, _ = resolve_turn_config(_StubAgent(), {})
        assert cfg.max_tokens == 333
        assert cfg.top_p == 0.31 and cfg.seed == 4242 and cfg.top_k == 17
