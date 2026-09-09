"""Which provider a model call is attributed to.

A local Transformers or vLLM run used to fall through to the family-name
guesses and be labelled with whichever cloud provider serves a model of that
family: `transformers:Qwen/...` matched "qwen" and was reported as `cerebras`,
so on-device GPU work appeared as spend at a company the user never called.
The label is what the dashboard span stream shows, what the OpenTelemetry
`effgen.model.provider` attribute carries, and what the provider-labelled
Prometheus series are keyed on.
"""

from __future__ import annotations

import pytest

from effgen.core.agent_runtime import _infer_provider_from_model
from effgen.observability.tracing import _infer_provider

LOCAL_ENGINES = ["transformers", "vllm", "gguf", "mlx"]


class _Bare:
    """A model object carrying no provider signal of its own."""


class TestLocalRunsAreNotAttributedToACloud:
    @pytest.mark.parametrize("engine", LOCAL_ENGINES)
    @pytest.mark.parametrize("family", ["Qwen/Qwen2.5-1.5B-Instruct",
                                        "meta-llama/Llama-3.2-3B-Instruct",
                                        "mistralai/Mistral-7B-Instruct"])
    def test_an_engine_prefixed_id_names_the_engine(self, engine, family):
        assert _infer_provider_from_model(_Bare(), f"{engine}:{family}") == engine

    @pytest.mark.parametrize("engine", LOCAL_ENGINES)
    def test_the_tracing_shim_agrees(self, engine):
        """The two tables drifting is how the labels diverged before."""
        model_id = f"{engine}:Qwen/Qwen2.5-1.5B-Instruct"
        assert _infer_provider(model_id) == _infer_provider_from_model(_Bare(), model_id)

    @pytest.mark.parametrize(
        "class_name, expected",
        [("TransformersEngine", "transformers"), ("VLLMEngine", "vllm"),
         ("GGUFEngine", "gguf"), ("MLXEngine", "mlx")],
    )
    def test_the_engine_class_alone_is_enough(self, class_name, expected):
        """A bare repo id with no prefix still must not read as a cloud."""
        engine = type(class_name, (), {})()
        assert _infer_provider_from_model(engine, "Qwen/Qwen2.5-1.5B-Instruct") == expected

    def test_a_local_run_is_never_unknown(self):
        engine = type("TransformersEngine", (), {})()
        assert _infer_provider_from_model(engine, "Qwen/Qwen2.5-1.5B") != "unknown"


class TestCloudAttributionIsUnchanged:
    @pytest.mark.parametrize(
        "model_id, expected",
        [
            ("gpt-4o-mini", "openai"),
            ("claude-opus-4-7", "anthropic"),
            ("gemini-2.5-flash", "google"),
            ("llama3.1-8b", "cerebras"),
            ("mixtral-8x7b", "groq"),
        ],
    )
    def test_a_hosted_id_keeps_its_provider(self, model_id, expected):
        assert _infer_provider_from_model(_Bare(), model_id) == expected
        assert _infer_provider(model_id) == expected

    def test_an_explicit_attribute_still_wins_over_everything(self):
        model = _Bare()
        model.provider = "fireworks"
        assert _infer_provider_from_model(model, "transformers:Qwen/Qwen2.5") == "fireworks"


class TestTheTwoResolversAgree:
    """The agent has a second resolver, and it answered differently.

    ``AgentGenerationMixin._model_provider`` feeds the error detail, the model
    span and the provider-labelled metric series; ``_infer_provider_from_model``
    feeds the run store. For a local engine the first answered ``"unknown"`` and
    the second ``"transformers"``, so one run was recorded under two providers
    and a question asked of the metrics could not be answered from the runs.
    """

    @staticmethod
    def _resolve(model) -> str:
        from effgen.core.agent_generation import AgentGenerationMixin

        return AgentGenerationMixin._model_provider(None, model)

    @pytest.mark.parametrize(
        "class_name, expected",
        [("TransformersEngine", "transformers"), ("VLLMEngine", "vllm"),
         ("GGUFEngine", "gguf"), ("MLXEngine", "mlx")],
    )
    def test_a_local_engine_is_labelled_with_its_engine(self, class_name, expected):
        engine = type(class_name, (), {"model_name": "Qwen/Qwen2.5-1.5B-Instruct"})()
        assert self._resolve(engine) == expected

    def test_a_local_run_is_never_unknown(self):
        engine = type("TransformersEngine", (), {"model_name": "Qwen/Qwen2.5-1.5B"})()
        assert self._resolve(engine) != "unknown"

    @pytest.mark.parametrize("engine", LOCAL_ENGINES)
    def test_both_resolvers_give_one_answer(self, engine):
        model_id = f"{engine}:Qwen/Qwen2.5-1.5B-Instruct"
        model = type("Bare", (), {"model_name": model_id})()
        assert self._resolve(model) == _infer_provider_from_model(model, model_id)

    def test_an_explicit_attribute_still_wins(self):
        model = _Bare()
        model.provider = "fireworks"
        model.model_name = "transformers:Qwen/Qwen2.5"
        assert self._resolve(model) == "fireworks"

    @pytest.mark.parametrize(
        "model_id, expected",
        [("gpt-4o-mini", "openai"), ("claude-opus-4-7", "anthropic"),
         ("gemini-2.5-flash", "google")],
    )
    def test_a_hosted_id_keeps_its_provider(self, model_id, expected):
        model = type("Bare", (), {"model_name": model_id})()
        assert self._resolve(model) == expected

    def test_a_model_with_no_signal_at_all_is_still_unknown(self):
        assert self._resolve(_Bare()) == "unknown"
        assert self._resolve(None) == "unknown"
