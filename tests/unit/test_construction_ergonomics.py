"""Construction/run input ergonomics — the obvious call works or fails clearly.

Covers three natural first-user calls that previously hit cryptic failures:

* ``run(output_schema=PydanticClass)`` returned a buried
  ``success=False, "Object of type ModelMetaclass is not JSON serializable"``
  instead of converting the class (it ships ``pydantic_model_to_schema``).
* ``Agent("groq:model")`` (a bare model string) raised a deferred
  ``AttributeError: 'str' object has no attribute 'name'`` from inside run().
* ``create_agent("minimal", model=..., engine="transformers")`` raised
  ``TypeError: AgentConfig.__init__() got an unexpected keyword argument 'engine'``.

These are offline, lightweight checks of the input validation/normalization
paths — no live provider calls. Live proof lives in
``tests/integration/test_construction_ergonomics_live.py``.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from effgen import Agent, create_agent
from effgen.core.agent import AgentConfig
from effgen.core.structured_output import normalize_output_schema


class _Capital(BaseModel):
    country: str
    capital: str


# ── output_schema accepts a JSON-Schema dict or a Pydantic class ──────────────

class TestOutputSchemaNormalization:
    def test_none_passthrough(self):
        assert normalize_output_schema(None) is None

    def test_dict_passthrough_identity(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert normalize_output_schema(schema) is schema

    def test_pydantic_class_converts_to_schema(self):
        out = normalize_output_schema(_Capital)
        assert isinstance(out, dict)
        assert out["type"] == "object"
        assert set(out["properties"]) == {"country", "capital"}
        # title fields are stripped so SLMs don't echo them back as data
        assert "title" not in out
        assert all("title" not in p for p in out["properties"].values())

    @pytest.mark.parametrize("bad", [12345, "a string", ["list"], 3.14, object()])
    def test_bad_type_raises_clear_typeerror(self, bad):
        with pytest.raises(TypeError) as exc:
            normalize_output_schema(bad)
        msg = str(exc.value)
        assert "output_schema" in msg
        assert "dict" in msg and "Pydantic" in msg

    def test_run_with_pydantic_class_does_not_crash_in_serialization(self):
        # require_model=False keeps construction cheap; the schema is normalized
        # before any model call, so a bad value errors immediately.
        agent = Agent(AgentConfig(name="t", model="x", require_model=False))
        with pytest.raises(TypeError):
            agent.run("hi", output_schema=12345)


# ── Agent(...) constructor validates its config ───────────────────────────────

class TestAgentConstructorGuard:
    @pytest.mark.parametrize("bad", ["groq:openai/gpt-oss-20b", None, 123, ["x"]])
    def test_non_agentconfig_raises_typeerror(self, bad):
        with pytest.raises(TypeError) as exc:
            Agent(bad)
        msg = str(exc.value)
        assert "AgentConfig" in msg
        assert "create_agent" in msg

    def test_valid_config_still_builds(self):
        agent = Agent(AgentConfig(name="t", model="x", require_model=False))
        assert agent.name == "t"

    def test_no_arg_constructor_teaches(self):
        # ``Agent()`` (a natural newcomer guess) must teach the fix the same way
        # ``Agent("model")`` does, not raise a bare "missing argument 'config'".
        with pytest.raises(TypeError) as exc:
            Agent()
        msg = str(exc.value)
        assert "AgentConfig" in msg
        assert "create_agent" in msg

    @pytest.mark.parametrize("bad_model", [12345, ["x"], _Capital])
    def test_unsupported_model_type_fails_fast(self, bad_model):
        # AgentConfig.model that is neither a str id nor a loaded instance must
        # fail at construction, not silently build a model-less agent.
        with pytest.raises(TypeError) as exc:
            Agent(AgentConfig(name="t", model=bad_model))
        assert "AgentConfig.model" in str(exc.value)

    def test_failed_construction_does_not_warn_on_gc(self, caplog):
        # A construction that fails (e.g. missing model / unloadable id) builds a
        # partial agent that is never returned to the caller; it must mark itself
        # closed so __del__ doesn't tail the clean error with a confusing
        # "garbage-collected without calling close()" warning. (A unique name
        # isolates this from other tests' unclosed agents during gc.)
        import gc

        probe = "gc-close-guard-probe"
        caplog.set_level("WARNING", logger="effgen.core.agent")
        with pytest.raises(ValueError):
            Agent(AgentConfig(name=probe, model=None, require_model=True))
        gc.collect()
        offenders = [r for r in caplog.records
                     if "garbage-collected" in r.message and probe in r.message]
        assert not offenders, offenders


# ── create_agent routes load_model kwargs / errors clearly ────────────────────

class TestCreateAgentKwargs:
    def test_engine_kwarg_with_loaded_instance_raises(self):
        # `engine` only applies to a model-id string; with a pre-loaded instance
        # it must raise a clear error rather than a cryptic AgentConfig TypeError.
        class _FakeModel:  # stand-in loaded model (not a str)
            model_name = "fake"

        with pytest.raises(TypeError) as exc:
            create_agent("minimal", model=_FakeModel(), engine="transformers")
        assert "engine" in str(exc.value)

    def test_unknown_kwarg_lists_accepted_options(self):
        with pytest.raises(TypeError) as exc:
            create_agent("minimal", "x", not_a_real_field=1, require_model=False)
        msg = str(exc.value)
        assert "not_a_real_field" in msg
        # the actionable message points at both load-model options and config fields
        assert "engine" in msg
        assert "temperature" in msg or "max_iterations" in msg

    def test_real_config_field_passthrough_still_works(self):
        agent = create_agent("minimal", "x", require_model=False, max_context_length=2048)
        assert agent.config.max_context_length == 2048

    def test_name_is_accepted_as_alias_for_agent_name(self):
        # `name` is the natural way to name an agent (and a real AgentConfig
        # field). It must set the agent name rather than colliding with the
        # internally-supplied name and raising "got multiple values for 'name'".
        agent = create_agent("minimal", "x", require_model=False, name="researcher")
        assert agent.name == "researcher"
        assert agent.config.name == "researcher"

    def test_agent_name_takes_precedence_over_name_alias(self):
        # When both are given, the explicit `agent_name=` wins.
        agent = create_agent(
            "minimal", "x", require_model=False, agent_name="winner", name="loser"
        )
        assert agent.name == "winner"

    def test_default_name_when_neither_given(self):
        agent = create_agent("minimal", "x", require_model=False)
        assert agent.name.endswith("-agent")


# ── create_agent accepts tool *names* and a `tools=` alias ────────────────────

class TestCreateAgentTools:
    def test_extra_tools_accepts_name_strings(self):
        # The CLI's -t/--tools accepts names; the Python extra_tools= path must
        # too — resolving "calculator" via the registry rather than crashing with
        # "'str' object has no attribute 'metadata'".
        agent = create_agent(
            "minimal", "x", require_model=False, extra_tools=["calculator"]
        )
        assert "calculator" in {t.metadata.name for t in agent.config.tools}

    def test_extra_tools_mixes_names_and_instances(self):
        from effgen.tools import get_registry

        calc = get_registry().get_tool_sync("calculator")
        agent = create_agent(
            "minimal", "x", require_model=False, extra_tools=[calc, "wikipedia"]
        )
        names = {t.metadata.name for t in agent.config.tools}
        assert {"calculator", "wikipedia"} <= names

    def test_unknown_tool_name_raises_with_suggestion(self):
        with pytest.raises(ValueError) as exc:
            create_agent(
                "minimal", "x", require_model=False, extra_tools=["calcualtor"]
            )
        msg = str(exc.value)
        assert "extra_tools" in msg
        assert "calculator" in msg  # close-match suggestion

    def test_tools_kwarg_is_alias_for_extra_tools(self):
        # The LangChain-style `tools=` guess must work as an alias for
        # extra_tools, not collide into "got multiple values for 'tools'".
        agent = create_agent(
            "minimal", "x", require_model=False, tools=["calculator"]
        )
        assert "calculator" in {t.metadata.name for t in agent.config.tools}

    def test_tools_and_extra_tools_together_raise(self):
        with pytest.raises(TypeError) as exc:
            create_agent(
                "minimal", "x", require_model=False,
                tools=["calculator"], extra_tools=["calculator"],
            )
        assert "extra_tools" in str(exc.value)
