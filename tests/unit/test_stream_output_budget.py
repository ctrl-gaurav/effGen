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

        class _Agent:
            model = _Model()

            class config:
                temperature = 0.2
                max_tokens = 333
                top_p = 0.31
                top_k = 17
                seed = 4242
                presence_penalty = 0.5
                frequency_penalty = 0.25
                repetition_penalty = 1.15

            def _interleaves_reasoning(self, model):
                return False

        cfg, _ = resolve_turn_config(_Agent(), {})
        assert cfg.max_tokens == 333
        assert cfg.top_p == 0.31 and cfg.seed == 4242 and cfg.top_k == 17
