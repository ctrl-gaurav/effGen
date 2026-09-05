"""Model-generation and structured-output internals for :class:`Agent`.

Extracted from ``agent.py`` without behaviour change: the provider call,
retry/error classification, speculative execution, structured-output coercion,
and the no-tool direct-inference path. Mixed into :class:`Agent`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from ..errors import quote_for_message
from ..models._adapter_utils import (
    FINISH_LENGTH,
    apply_stop_sequences,
    default_max_output_tokens,
    needs_reasoning_headroom,
    normalize_stop_sequences,
)
from ..models.base import BaseModel, GenerationConfig
from ..models.errors import (
    REMEDIATION_BY_CATEGORY,
    BackendUnreachableError,
    InvalidRequestError,
    ModelAuthError,
    ModelNotFoundError,
    ModelRefusalError,
    classify_provider_error,
    generation_failure_text,
    simplify_embedded_provider_error,
)
from ..observability import get_logger as _get_obs_logger
from ..observability.spans import ModelAttrs
from ..observability.tracing import (
    mark_span_error,
    start_model_call,
)
from ..observability.tracing import (
    stamp_call_cost as _stamp_call_cost,
)
from ..utils.structured_logging import (
    get_structured_logger,
)

# When a reasoning model returns an empty, "length"-truncated result with the
# default budget, grow the budget and retry once (×4, capped) before giving up
# with an actionable message — the user never pinned max_tokens, so escalating
# is safe and far better than a misleading "empty response" they were billed for.
_TRUNCATION_ESCALATION_FACTOR = 4
_TRUNCATION_MAX_TOKENS_CEILING = 8192

# A reasoning model asked for structured output needs headroom well beyond the
# reasoning-family default (it spends budget on hidden reasoning *and* the
# JSON/schema structure before any field value). Below this, warn at call time
# instead of only after a billed, truncated failure.
_REASONING_STRUCTURED_OUTPUT_MIN_TOKENS = 8192

# Models observed to return their reasoning chain and their answer through one
# token stream, keyed ``"provider:model"``. A stop sequence sent to such a model
# is matched against the chain too, so generation can end at the boundary
# between the chain and the answer — before the first visible token, with a
# ``"stop"`` finish reason and an empty, still-billed result. For a model in
# this set the agent holds its stop sequences back and applies them to the text
# the model returns instead (see ``apply_stop_sequences``), which is the same
# visible result without the collision. Populated from the catalog flag at call
# time and from an observed collision, so the wasted call happens at most once
# per model per process.
_reasoning_stream_models: set[str] = set()

#: Generation parameters the agent forwards to the adapter as keyword arguments,
#: beyond the sampling settings :class:`GenerationConfig` carries.
#:
#: ``GenerationConfig`` covers temperature, budget, penalties and stop
#: sequences. Everything else that shapes a request — how the tools are
#: presented and whether the model may decline to call one — travels beside it,
#: and the agent has to copy it across explicitly. It used to copy exactly one
#: name, so every other parameter was dropped between the loop and the provider
#: without an error or a log line: the guard that decided to require a tool call
#: recorded that it had, and the request went out without it.
#:
#: Both model-call paths build their adapter kwargs from this set through
#: :func:`model_call_kwargs`, so a parameter added here reaches the provider on
#: every path at once and none of them can drift.
MODEL_CALL_KWARGS: frozenset[str] = frozenset({"tools", "tool_choice"})


def model_call_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The subset of *kwargs* that is forwarded to the adapter's ``generate``.

    Only the names in :data:`MODEL_CALL_KWARGS` travel: the rest of the agent's
    keyword arguments are its own bookkeeping — the iteration cap, the
    checkpoint knobs, the multimodal inputs — and a provider asked to accept one
    of those rejects the whole request.

    A ``tool_choice`` is dropped when no tool definitions travel with it.
    Requiring a call from a request that offers nothing to call is rejected by
    every provider that enforces the choice, and the two come apart on their
    own: a run steered toward an answer withholds the definitions, and a caller
    who asked for a required call on an agent whose tools reach the model as
    prose has asked for something that request cannot carry. Dropping it here
    covers every adapter at once, which is where the pair is decided.

    Args:
        kwargs: The keyword arguments the agent was called with.

    Returns:
        A new dict holding the forwardable parameters that were supplied.
    """
    picked = {name: kwargs[name] for name in MODEL_CALL_KWARGS if name in kwargs}
    if not picked.get("tools"):
        picked.pop("tool_choice", None)
    return picked


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
_slog = get_structured_logger(__name__)
# (model_name, kind) pairs already warned about in this process — a call-time
# reasoning-budget heads-up fires once per model/kind, not on every call.
_reasoning_budget_warned: set[tuple[str, str]] = set()
# Canonical structured observability logger — emits redacted JSON lines with OTel context
_obs_log = _get_obs_logger(__name__)

from .agent import AgentMode, AgentResponse  # noqa: E402
from .agent_runtime import (  # noqa: E402
    _infer_provider_from_model,
    sanitize_final_answer,
)
from .tool_call_record import ToolCallList  # noqa: E402


class AgentGenerationMixin:
    """Generation / error-handling / structured-output methods for :class:`Agent`."""

    def _apply_structured_output(
        self,
        response: AgentResponse,
        schema: dict[str, Any],
        output_model: Any | None,
        task: str,
    ) -> AgentResponse:
        """Post-process response to ensure structured output matches schema.

        If the agent's free-text output already contains valid JSON matching
        the schema, it is extracted and returned. Otherwise, the model is
        re-prompted to produce conforming JSON.

        Args:
            response: The original AgentResponse.
            schema: JSON Schema dict.
            output_model: Optional Pydantic model class for parsing.
            task: Original task (used for re-prompting).

        Returns:
            AgentResponse with validated structured output.
        """
        from .structured_output import (
            StructuredOutputConfig,
            extract_json_from_text,
            structured_generate,
            validate_json_schema,
        )

        raw_output = response.output

        # Fast path (0 extra calls): the agent's own answer already validates.
        json_str = extract_json_from_text(response.output)
        if json_str:
            try:
                parsed = json.loads(json_str)
                valid, err = validate_json_schema(parsed, schema)
                if valid:
                    response.output = json_str
                    response.metadata["structured_output"] = True
                    response.metadata["structured_output_attempts"] = 0
                    response.metadata["structured_output_method"] = "agent_output"
                    if output_model is not None:
                        response.metadata["parsed"] = self._parse_with_pydantic(
                            output_model, parsed,
                        )
                    else:
                        response.metadata["parsed"] = parsed
                    if self._structured_all_empty(parsed):
                        response.metadata["structured_output_empty"] = True
                    return response
            except (json.JSONDecodeError, TypeError):
                pass

        # The free-text answer didn't validate. Constrain via a native JSON mode
        # (one call) before any text reprompt, and surface the attempt count so
        # callers can see when/how much repair happened.
        if self.model is None:
            response.metadata["structured_output"] = False
            response.metadata["structured_output_attempts"] = 0
            response.metadata["structured_output_error"] = (
                "structured output requested but no model is available to produce it"
            )
            response.metadata["raw_output"] = raw_output
            response.mark_failed("structured_output_failed")
            return response

        structured_prompt = (
            f"Based on this task and result, produce structured output.\n"
            f"Task: {task}\n"
            f"Result: {raw_output}"
        )
        outcome = structured_generate(
            self.model, structured_prompt, schema,
            StructuredOutputConfig(schema=schema),
        )
        response.metadata["structured_output_attempts"] = outcome.attempts
        response.metadata["structured_output_method"] = outcome.method
        if outcome.success and outcome.json_str is not None:
            response.output = outcome.json_str
            response.metadata["structured_output"] = True
            # True only when a text reprompt actually ran — grammar/native
            # constraint attempts also count toward `attempts` but are not a
            # reprompt, so `attempts > 0` alone would mislabel them as one.
            response.metadata["structured_output_reprompted"] = outcome.method == "reprompt"
            if output_model is not None:
                response.metadata["parsed"] = self._parse_with_pydantic(
                    output_model, outcome.parsed,
                )
            else:
                response.metadata["parsed"] = outcome.parsed
            if self._structured_all_empty(outcome.parsed):
                response.metadata["structured_output_empty"] = True
        else:
            # Keep the raw answer, mark the run unsuccessful, and explain why —
            # consistent with the framework's no-silent-failure rule.
            logger.warning(
                "Structured output constraint failed after %d attempt(s): %s",
                outcome.attempts, outcome.error,
            )
            response.output = raw_output
            response.metadata["structured_output"] = False
            response.metadata["structured_output_error"] = (
                outcome.error or "could not produce schema-valid output"
            )
            response.metadata["raw_output"] = raw_output
            response.mark_failed("structured_output_failed")
            if outcome.raw_text:
                response.metadata["structured_output_raw_attempt"] = outcome.raw_text

        return response

    @staticmethod
    def _structured_all_empty(data: Any) -> bool:
        """Return True when a schema-valid object carries no extracted values.

        A reasoning model or a nested schema under a tight ``max_tokens`` can
        return an object that validates but has every field empty
        (``{"diagnosis": "", "medications": []}``). Flagging it lets a caller
        distinguish "nothing to extract" from "budget ran out before the model
        filled the fields" — a larger ``max_tokens`` usually resolves the latter.
        """
        def _empty(v: Any) -> bool:
            if v is None or v == "" or v == [] or v == {}:
                return True
            if isinstance(v, dict):
                return all(_empty(x) for x in v.values())
            if isinstance(v, list | tuple):
                return all(_empty(x) for x in v)
            return False

        if isinstance(data, dict):
            return len(data) > 0 and all(_empty(v) for v in data.values())
        if isinstance(data, list | tuple):
            return len(data) == 0
        return False

    @staticmethod
    def _parse_with_pydantic(model_class: Any, data: Any) -> Any:
        """Parse data into a Pydantic model instance.

        Supports both Pydantic v1 and v2.
        """
        try:
            if hasattr(model_class, 'model_validate'):
                # Pydantic v2
                return model_class.model_validate(data)
            else:
                # Pydantic v1
                return model_class(**data)
        except Exception as e:
            logger.warning(f"Pydantic parsing failed: {e}")
            return None

    def _generate(self, prompt: Any, **kwargs) -> dict[str, Any]:
        """Generate from the model, with any configured middleware around it.

        A ``before_model_call`` hook may rewrite the prompt or the generation
        options, or return a result to use without calling the model at all;
        ``after_model_call`` sees whichever result was produced. With no
        middleware configured this is one boolean test and a direct call.

        Every retry and every failover hop is one call as far as the hooks are
        concerned, because each is a separate request to a provider.
        """
        chain = self._middleware_chain()
        if not chain:
            return self._generate_instrumented(prompt, **kwargs)

        from .middleware import ModelCallContext

        ctx = ModelCallContext(
            prompt=prompt,
            model_name=getattr(self, "model_name", "") or "",
            kwargs=kwargs,
            run=getattr(self, "_active_run_context", None),
        )
        short_circuit = chain.before_model_call(ctx)
        if short_circuit is not None:
            answered: dict[str, Any] = chain.after_model_call(ctx, short_circuit)
            return answered
        result = self._generate_instrumented(ctx.prompt, **ctx.kwargs)
        final: dict[str, Any] = chain.after_model_call(ctx, result)
        return final

    def _generate_instrumented(self, prompt: Any, **kwargs) -> dict[str, Any]:
        """
        Generate response from model with retry logic for empty responses.

        Retries up to 3 times on empty responses with exponential backoff
        and slightly increasing temperature.

        When a model router is configured (multi-model mode), the router
        selects the best model for the query. On failure, the agent
        automatically fails over to the next model in the pool.

        If speculative_execution is enabled, runs on two models in parallel
        and returns the first successful result.

        Args:
            prompt: Input prompt. May be a plain string or a structured
                multimodal Message/list of Messages accepted by API adapters.
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Dictionary with 'text', 'tokens_used', and other metadata

        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None and not self._all_models:
            raise RuntimeError(
                f"Agent '{self.name}' has no model loaded. "
                "Provide a model in AgentConfig or use a mock for testing."
            )

        # Speculative execution: run on 2 models, return first success
        if self._speculative_execution and len(self._all_models) >= 2:
            result = self._generate_speculative(prompt, **kwargs)
            if result is not None:
                return result
            # Fall through to normal path if speculative failed

        # Select model via router if available
        active_model = self.model
        if self._model_router is not None:
            try:
                task_hint = kwargs.pop("_task_hint", self._prompt_to_task_hint(prompt))
                tools_list = list(self.tools.values()) if self.tools else None
                decision = self._model_router.select(task_hint, tools_list)
                active_model = decision.model
                logger.info(
                    "Router selected '%s' (reason: %s)",
                    decision.model_name, decision.reason,
                )
            except Exception as e:
                logger.warning("Router selection failed, using default model: %s", e)
                active_model = self.model

        if active_model is None:
            active_model = self.model

        max_retries = 3
        backoff_delays = [0.5, 1.0, 2.0]
        base_temperature = kwargs.get('temperature', self.config.temperature)

        default_stop_sequences = [
            "\nObservation:",
            "\nQuestion:",
            "\nHuman:",
            "\nUser:",
        ]

        last_error = None
        deterministic_detail: dict[str, Any] | None = None
        # A non-retryable failure (auth/not-found/invalid) is already logged once,
        # in full, at ERROR below. Track that so the final summary doesn't re-log
        # the same failure a second time at WARNING (the "logged 3×" noise).
        nonretryable_logged = False
        # Same for a deterministic empty outcome (truncation / reasoning-only):
        # the branch that detects it logs the full message once.
        deterministic_logged = False
        total_tokens = 0
        # Whether the caller pinned max_tokens; if they didn't we may grow the
        # budget to recover a reasoning model from a "length"-truncated empty.
        user_pinned_max_tokens = kwargs.get("max_tokens") is not None
        # Build ordered list of models to try: selected first, then others
        failover_models = [active_model] + [
            m for m in self._all_models if m is not active_model
        ] if len(self._all_models) > 1 else [active_model]

        for model_idx, current_model in enumerate(failover_models):
            if current_model is None:
                continue

            # Model-aware default budget: reasoning families (gpt-5*, o-series)
            # burn output budget on hidden reasoning, so 1024 can leave zero
            # visible tokens. Give them room unless the caller pinned a value.
            current_max_tokens = (
                kwargs["max_tokens"] if user_pinned_max_tokens
                else default_max_output_tokens(current_model)
            )

            # A bare string is the shape the OpenAI API accepts, so a caller
            # writes ``run(task, stop_sequences="END")``. Normalized here as
            # well as in GenerationConfig, because the local-trim list below is
            # built from this value directly and ``list("END")`` is three
            # single-character sequences.
            requested_stop_sequences = normalize_stop_sequences(
                kwargs.get('stop_sequences', default_stop_sequences)
            )
            # A model that streams its reasoning chain and its answer through one
            # token stream matches stop sequences against the chain as well, so
            # sending them can end generation before the first visible token.
            # Cut the returned answer here instead.
            local_stop_sequences = (
                list(requested_stop_sequences or [])
                if self._interleaves_reasoning(current_model) else None
            )

            for attempt in range(max_retries):
                try:
                    # Slightly increase temperature on retries to get different output
                    retry_temperature = min(base_temperature + (attempt * 0.1), 1.0)

                    gen_config = GenerationConfig(
                        temperature=retry_temperature,
                        max_tokens=current_max_tokens,
                        top_p=kwargs.get('top_p', self.config.top_p),
                        top_k=kwargs.get('top_k', self.config.top_k),
                        seed=kwargs.get('seed', self.config.seed),
                        presence_penalty=kwargs.get('presence_penalty', self.config.presence_penalty),
                        frequency_penalty=kwargs.get('frequency_penalty', self.config.frequency_penalty),
                        repetition_penalty=kwargs.get('repetition_penalty', self.config.repetition_penalty),
                        stop_sequences=(
                            None if local_stop_sequences else requested_stop_sequences
                        ),
                    )

                    # Generation parameters that travel beside GenerationConfig
                    # (the tool definitions, and whether a call is required).
                    extra_gen_kwargs = model_call_kwargs(kwargs)

                    result = current_model.generate(prompt, config=gen_config, **extra_gen_kwargs)

                    response_text = result.text if result and result.text else ""
                    if local_stop_sequences:
                        response_text = apply_stop_sequences(
                            response_text, local_stop_sequences,
                        )
                    tokens_used = result.tokens_used if result and hasattr(result, 'tokens_used') else 0
                    finish_reason = result.finish_reason if result and hasattr(result, 'finish_reason') else "unknown"
                    total_tokens += tokens_used
                    result_metadata = result.metadata if result and hasattr(result, 'metadata') else {}
                    # Surface per-run cost/token usage on the eventual response
                    # (every call counts — including a billed empty one).
                    self._accumulate_run_cost(result_metadata)

                    # Native tool-calls can arrive with empty text (finish_reason="tool_calls").
                    # Return the call to the agent loop instead of treating it as an empty
                    # response that needs retrying.
                    native_tool_calls = (result_metadata or {}).get("tool_calls") or []

                    # If we got non-empty text OR a native tool call, return it
                    if response_text.strip() or native_tool_calls:
                        return {
                            "text": response_text,
                            "tokens_used": total_tokens,
                            "finish_reason": finish_reason,
                            "tool_calls": native_tool_calls,
                            "metadata": result_metadata or {},
                        }

                    # Empty result with finish_reason="length" is deterministic
                    # truncation, not a flaky empty: the whole budget was spent
                    # (reasoning models on internal reasoning) before any visible
                    # token. Retrying at the same budget can never recover and
                    # bills again, so grow the budget once (caller didn't pin it)
                    # or fail with an actionable hint — never the misleading
                    # "empty response after retries".
                    if finish_reason == FINISH_LENGTH:
                        grown = min(
                            current_max_tokens * _TRUNCATION_ESCALATION_FACTOR,
                            _TRUNCATION_MAX_TOKENS_CEILING,
                        )
                        if not user_pinned_max_tokens and grown > current_max_tokens:
                            logger.info(
                                "Empty 'length'-truncated response from '%s' at "
                                "max_tokens=%d; retrying with max_tokens=%d",
                                getattr(current_model, "model_name", "?"),
                                current_max_tokens, grown,
                            )
                            current_max_tokens = grown
                            continue
                        deterministic_detail = self._truncation_error_detail(
                            current_model, current_max_tokens, result_metadata,
                        )
                        deterministic_logged = True
                        logger.warning("Generation failed: %s", deterministic_detail["message"])
                        break  # deterministic for this model; outer loop may failover

                    # Empty content beside a reasoning chain: the model spent the
                    # turn reasoning and emitted no visible token. Deterministic
                    # at these settings, so retrying identical ones bills again
                    # for the same outcome. When the stop sequences went upstream
                    # they can have ended generation at the boundary between the
                    # chain and the answer — retry that once with them applied
                    # locally instead; otherwise report it.
                    if (result_metadata or {}).get("reasoning_only"):
                        if local_stop_sequences is None and requested_stop_sequences:
                            self._remember_reasoning_stream_model(current_model)
                            local_stop_sequences = list(requested_stop_sequences)
                            logger.info(
                                "'%s' returned only reasoning with stop sequences in "
                                "force; retrying once with them applied to the answer "
                                "instead of sent to the provider",
                                getattr(current_model, "model_name", "?"),
                            )
                            continue
                        deterministic_detail = self._reasoning_only_error_detail(
                            current_model, current_max_tokens, result_metadata,
                        )
                        deterministic_logged = True
                        logger.warning("Generation failed: %s", deterministic_detail["message"])
                        break  # deterministic for this model; outer loop may failover

                    # Empty response — retry
                    if attempt < max_retries - 1:
                        logger.info(
                            f"Empty response on attempt {attempt + 1}/{max_retries}, "
                            f"retrying in {backoff_delays[attempt]}s with temperature={retry_temperature:.2f}"
                        )
                        time.sleep(backoff_delays[attempt])
                    else:
                        logger.warning(f"Empty response after {max_retries} attempts")

                except Exception as e:
                    last_error = e
                    err_class = classify_provider_error(e)
                    # Only retry errors that could plausibly succeed on retry
                    # (transient/timeout/rate-limited/unknown). Auth, not-found,
                    # refusal and invalid-request errors fail fast — no retry
                    # storm and no wasted latency.
                    if not err_class.should_retry:
                        logger.error(
                            "Generation failed with non-retryable %s error on '%s': %s",
                            err_class.category,
                            getattr(current_model, "model_name", "?"),
                            e,
                        )
                        nonretryable_logged = True
                        break  # stop retrying this model; outer loop may failover
                    if err_class.rate_limited:
                        # A rate limit is the one failure where retrying harder
                        # makes it worse: the quota is already spent, and every
                        # extra request spends it further. It has also already
                        # been through the layer that knows the delay the
                        # provider stated — the adapter's own backoff, or the
                        # SDK's where the adapter has none. Retrying here as
                        # well multiplies the attempts instead of sharing the
                        # budget, and it is what turned one client request into
                        # twelve upstream requests and a twenty-second wait at a
                        # stated two-second delay. The caller gets the typed
                        # rate-limit error with the provider's own retry-after.
                        logger.warning(
                            "Rate limited on '%s'; not retrying at the agent layer "
                            "(the provider's stated delay was already honoured): %s",
                            getattr(current_model, "model_name", "?"), e,
                        )
                        break
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Generation error on attempt {attempt + 1}/{max_retries} "
                            f"({err_class.category}): {e}, retrying in {backoff_delays[attempt]}s"
                        )
                        time.sleep(backoff_delays[attempt])
                    else:
                        logger.error(f"Generation failed after {max_retries} attempts: {e}")

            # If we have more models to try, failover
            if model_idx < len(failover_models) - 1:
                next_name = getattr(failover_models[model_idx + 1], 'model_name', '?')
                logger.warning(
                    "Failing over to model '%s' after '%s' exhausted retries",
                    next_name, getattr(current_model, 'model_name', '?'),
                )

        # All models and retries exhausted — return a structured, redacted error
        # so callers (both the tool loop and the direct path) can fail explicitly.
        if last_error is not None:
            detail = self._build_error_detail(last_error, current_model)
        elif deterministic_detail is not None:
            detail = deterministic_detail
        else:
            detail = {
                "type": "EmptyResponse",
                "category": "empty_response",
                "provider": self._model_provider(current_model),
                "model": getattr(current_model, "model_name", None) or self.model_name or "unknown",
                "message": "Model returned an empty response after retries.",
                "retryable": True,
            }
        # Avoid re-logging a failure already reported in full above.
        if nonretryable_logged or deterministic_logged:
            logger.debug("Generation failed: %s", detail["message"])
        else:
            logger.warning("Generation failed: %s", detail["message"])
        return {
            "text": "",
            "tokens_used": total_tokens,
            "finish_reason": "error",
            "metadata": {"error": detail["message"], "error_detail": detail},
        }

    def _model_provider(self, model: Any) -> str:
        """Best-effort provider name for a model object.

        Feeds the error detail, the model span and the provider-labelled
        metric series. A local engine carries no provider attribute — it has an
        *engine* — so this used to answer ``"unknown"`` for on-device work while
        :func:`_infer_provider_from_model`, which feeds the run store for the
        same run, answered ``transformers``. Two labels for one call is a
        question the dashboard cannot answer, so the shared resolver is the
        fallback here too.
        """
        for attr in ("_provider", "provider", "provider_name"):
            val = getattr(model, attr, None)
            if isinstance(val, str) and val:
                return val
        name = getattr(model, "model_name", None) or getattr(model, "name", None)
        return _infer_provider_from_model(model, name if isinstance(name, str) else None)

    def _record_provider_metrics(
        self,
        *,
        execution_time: float,
        outcome: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Feed the provider/model-labeled Prometheus series for one run.

        Populates ``effgen_model_call_latency_seconds{provider,model,outcome}``
        and ``effgen_tokens_total{provider,model,kind}`` (declared in
        ``effgen.observability.metrics`` but otherwise never written), so a
        server operator can graph latency/error-rate per provider and model
        and cost by model — not just the flat per-``agent_name`` aggregate.
        """
        try:
            from ..observability.metrics import record_model_call, record_tokens
        except Exception:  # pragma: no cover - metrics module always ships
            return
        provider = self._model_provider(self.model)
        model_name = getattr(self.model, "model_name", None) or self.model_name or "unknown"
        record_model_call(
            provider=provider,
            model=model_name,
            outcome=outcome,
            latency=max(0.0, execution_time or 0.0),
        )
        if prompt_tokens or completion_tokens:
            record_tokens(
                provider=provider,
                model=model_name,
                input_tokens=int(prompt_tokens or 0),
                output_tokens=int(completion_tokens or 0),
            )

    def _warn_reasoning_budget(self, max_tokens: int | None, structured_output: bool) -> None:
        """Warn once per model when a reasoning model's budget looks too tight.

        gpt-5 / o-series models spend part of ``max_tokens`` on hidden
        reasoning before any visible token, so a small pinned budget — or a
        generous one paired with structured output, which needs headroom for
        the reasoning *and* the JSON/schema structure — can produce nothing
        and still be billed. Logged before the call so the caller sees it
        without first burning a failed, billed round-trip.
        """
        model = self.model
        if model is None or not needs_reasoning_headroom(model):
            return
        name = getattr(model, "model_name", None) or self.model_name or "unknown"

        if structured_output:
            min_tokens = _REASONING_STRUCTURED_OUTPUT_MIN_TOKENS
            if max_tokens is not None and max_tokens >= min_tokens:
                return
            kind, hint = "structured", (
                f"'{name}' is a reasoning model asked for structured output with "
                f"max_tokens={max_tokens if max_tokens is not None else 'default'} — "
                f"reasoning plus the JSON structure can consume the whole budget "
                f"before any field is filled. Consider max_tokens>={min_tokens}."
            )
        else:
            min_tokens = default_max_output_tokens(model)
            if max_tokens is None or max_tokens >= min_tokens:
                return
            kind, hint = "tight_budget", (
                f"'{name}' is a reasoning model with max_tokens={max_tokens} — it can "
                f"spend the whole budget on hidden reasoning and return no visible "
                f"text. Consider max_tokens>={min_tokens}. This can happen even at "
                f"temperature=0: the reasoning-token allocation is provider-controlled "
                f"and is not made deterministic by a fixed temperature."
            )

        warn_key = (name, kind)
        if warn_key in _reasoning_budget_warned:
            return
        _reasoning_budget_warned.add(warn_key)
        logger.warning(hint)

    def _reasoning_stream_key(self, model: Any) -> str:
        """Key a model by provider and name for the reasoning-stream registry."""
        name = getattr(model, "model_name", None) or self.model_name or "unknown"
        return f"{self._model_provider(model)}:{name}"

    def _interleaves_reasoning(self, model: Any) -> bool:
        """Return True if *model* streams its reasoning chain with its answer.

        True for the reasoning families the catalog flags, and for any model
        already observed ending a turn at a stop sequence before its first
        visible token. The agent applies its stop sequences to such a model's
        returned answer rather than sending them to the provider.
        """
        if model is None:
            return False
        return (
            needs_reasoning_headroom(model)
            or self._reasoning_stream_key(model) in _reasoning_stream_models
        )

    def _remember_reasoning_stream_model(self, model: Any) -> None:
        """Record that *model* interleaves its reasoning chain with its answer."""
        if model is not None:
            _reasoning_stream_models.add(self._reasoning_stream_key(model))

    @staticmethod
    def _reasoning_budget_note(result_metadata: dict[str, Any] | None) -> str:
        """Describe the reasoning budget one call spent, or ``""`` if unknown."""
        meta = result_metadata or {}
        tokens = meta.get("reasoning_tokens") or 0
        chars = meta.get("reasoning_chars") or 0
        if not tokens and not chars:
            return ""
        if tokens and chars:
            spent = f"{tokens} reasoning tokens ({chars} characters of reasoning)"
        elif tokens:
            spent = f"{tokens} reasoning tokens"
        else:
            spent = f"{chars} characters of reasoning"
        return f" It spent the budget on {spent} and returned no answer."

    def _truncation_error_detail(
        self,
        model: Any,
        max_tokens: int,
        result_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Structured error for a deterministic ``max_tokens`` truncation.

        Distinct from the generic empty-response error: the model produced no
        visible text because it hit ``max_tokens`` (reasoning models can spend
        the whole budget on internal reasoning). The message tells the caller
        exactly how to recover, and it is *not* retryable at the same budget.
        """
        name = getattr(model, "model_name", None) or self.model_name or "unknown"
        hint = (
            f"Model '{name}' hit the max_tokens limit ({max_tokens}) and "
            "produced no output (finish_reason='length'). Increase max_tokens — "
            "e.g. agent.run(task, max_tokens=8192); reasoning models can spend "
            "the whole budget on internal reasoning before any visible token."
            + self._reasoning_budget_note(result_metadata)
        )
        return {
            "type": "TruncatedResponse",
            "category": "truncation",
            "provider": self._model_provider(model),
            "model": name,
            "message": hint,
            "retryable": False,
        }

    def _reasoning_only_error_detail(
        self,
        model: Any,
        max_tokens: int,
        result_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Structured error for a turn that produced reasoning and no answer.

        The model was billed for output the caller cannot see, and the outcome
        is deterministic at these settings — so the message names the model, the
        cap in force and the reasoning budget spent instead of the generic
        "empty response after retries", and it is not retryable unchanged. The
        reasoning chain is never returned as the answer.
        """
        name = getattr(model, "model_name", None) or self.model_name or "unknown"
        meta = result_metadata or {}
        hint = meta.get("empty_response_reason") or (
            f"Model '{name}' returned no visible text: it produced only internal "
            f"reasoning under a max_tokens cap of {max_tokens}. Raise max_tokens, "
            "drop any stop sequence the reasoning chain can match, or use a model "
            "that answers without an extended reasoning chain."
        )
        return {
            "type": "ReasoningOnlyResponse",
            "category": "reasoning_only",
            "provider": self._model_provider(model),
            "model": name,
            "message": hint,
            "max_tokens": max_tokens,
            "reasoning_tokens": meta.get("reasoning_tokens", 0),
            "retryable": False,
        }

    def _accumulate_run_cost(self, result_metadata: dict[str, Any] | None) -> None:
        """Fold one model call's cost/token usage into the per-run accumulator.

        Lets :meth:`Agent.run` surface ``cost_usd`` and prompt/completion token
        counts on the response metadata for every run (tool loops included),
        without each call site reaching into the underlying GenerationResult.

        A call on a model with no published price reports ``cost_usd`` as
        ``None``. Those calls are counted separately rather than summed as
        zero, so a run's total is never quietly understated.
        """
        if not result_metadata:
            return
        accum = getattr(self, "_run_cost_accum", None)
        if accum is None:
            return
        if "cost_usd" in result_metadata and result_metadata["cost_usd"] is None:
            accum["unpriced_calls"] = accum.get("unpriced_calls", 0) + 1
        for key in ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens"):
            val = result_metadata.get(key)
            if isinstance(val, int | float):
                accum[key] = accum.get(key, 0) + val
        # Where the model ran (local engines report 'cuda'/'cpu'/'mixed'), so a
        # headless caller can detect a CPU fallback. Not summed — the last call
        # wins, and cloud calls that don't report it leave it untouched.
        device = result_metadata.get("device")
        if device:
            accum["device"] = device
        accum["calls"] = accum.get("calls", 0) + 1

    def _finalize_cost_metadata(self, response: AgentResponse) -> None:
        """Fold per-run latency + the cost accumulator onto ``response.metadata``.

        Surfaces ``latency_ms`` / ``duration_s`` (always available) and, when the
        run made billable model calls, ``cost_usd`` plus ``prompt_tokens`` /
        ``completion_tokens`` / ``total_tokens`` (every call summed, tool loops
        included). Existing keys win, so a path that already set its own value is
        left untouched. Cost is rounded to whole microdollars. A run whose model
        has no published price carries no ``cost_usd`` key at all and reports
        ``unpriced_calls`` instead.
        """
        meta = response.metadata
        # Per-run latency, mirrored from execution_time so callers can read it
        # off metadata alongside cost/tokens — no need to wrap each call in a
        # timer. (The eval layer already measures the same number.)
        if response.execution_time:
            meta.setdefault("latency_ms", round(response.execution_time * 1000.0, 1))
            meta.setdefault("duration_s", round(response.execution_time, 4))
        accum = getattr(self, "_run_cost_accum", None)
        if not accum or not accum.get("calls"):
            return
        if "cost_usd" in accum and "cost_usd" not in meta:
            meta["cost_usd"] = round(float(accum["cost_usd"]), 8)
        # Calls the catalog publishes no price for. A run made only of those
        # reports no cost_usd at all; a mixed run reports the cost of the calls
        # that were priced, and this says how many are missing from it.
        if accum.get("unpriced_calls") and "unpriced_calls" not in meta:
            meta["unpriced_calls"] = int(accum["unpriced_calls"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in accum and key not in meta:
                meta[key] = int(accum[key])
        if accum.get("device") and "device" not in meta:
            meta["device"] = accum["device"]
        # response.tokens_used is documented as the run's total token count;
        # the value set along the generation path is completion-tokens-only
        # (each provider adapter's GenerationResult.tokens_used). Correct it
        # here, once, from the same prompt+completion sum already surfaced on
        # metadata["total_tokens"] — covers Prometheus token_usage/tokens_used
        # and the effgen.tokens_used span, which both read response.tokens_used
        # after this point.
        if "total_tokens" in accum:
            response.tokens_used = int(accum["total_tokens"])

    def _build_error_detail(self, exc: Exception, model: Any) -> dict[str, Any]:
        """Build a structured, redacted error record from an exception.

        Shape: ``{type, category, provider, model, message, remediation,
        retryable}`` — used identically by the tool-loop and direct-inference
        paths so a failure looks the same regardless of which path produced it.
        """
        ec = classify_provider_error(exc)
        # Prefer an explicit .provider, then the structured .error_context the
        # adapters attach (provider_runtime_error), then the model's own.
        _ctx = getattr(exc, "error_context", None)
        provider = (
            getattr(exc, "provider", None)
            or (_ctx.get("provider") if isinstance(_ctx, dict) else None)
            or self._model_provider(model)
        )
        model_name = (
            getattr(exc, "model_name", None)
            or getattr(model, "model_name", None)
            or self.model_name
            or "unknown"
        )
        # effGen's typed provider errors (ModelAuthError, ModelNotFoundError,
        # InvalidRequestError, ...) keep the raw cause on ``.message`` and only
        # add the "<provider> error (model=...):" prefix in their ``str()``.
        # Use the raw form here so a reconstructed exception (``raise_on_error``,
        # the server's error envelope) doesn't get that prefix stacked twice.
        raw = getattr(exc, "message", None) or getattr(exc, "refusal_message", None)
        raw_message = raw if isinstance(raw, str) and raw else str(exc)
        # An adapter's actionable message can embed a raw SDK error body
        # ("Error code: 413 - {'error': {...}}") verbatim — collapse that to
        # its inner text so the response shows prose, not a dumped structure.
        raw_message = simplify_embedded_provider_error(raw_message)
        # Redacted, then cut: a provider that echoes the rejected request back
        # sends kilobytes, and this text is read in a terminal panel and a log
        # line. ``remediation`` is kept beside the cause rather than folded
        # into it, so rebuilding a typed error from this record does not stack
        # two copies of the same guidance.
        message = quote_for_message(raw_message)
        return {
            "type": type(exc).__name__,
            "category": ec.category,
            "provider": provider or "unknown",
            "model": model_name,
            "message": message,
            "remediation": REMEDIATION_BY_CATEGORY.get(
                ec.category, REMEDIATION_BY_CATEGORY["unknown"]
            ),
            "retryable": ec.should_retry,
        }

    def _generation_failure_response(
        self,
        gen_result: dict[str, Any],
        *,
        iterations: int = 1,
        tool_calls: int = 0,
        tokens: int = 0,
        debug_trace: Any = None,
    ) -> AgentResponse:
        """Build the canonical failure AgentResponse from a ``_generate`` result.

        Both the no-tool direct path and the ReAct tool loop call this so a
        generation failure returns an identical shape: ``success=False``, a
        clear redacted message, ``metadata["reason"]="generation_failed"`` and
        a structured ``metadata["error"]={type,provider,model,message,...}``.
        Never returns ``success=True`` with empty output.
        """
        meta_src = gen_result.get("metadata") or {}
        detail = meta_src.get("error_detail")
        if not isinstance(detail, dict):
            message = quote_for_message(meta_src.get("error") or "generation_failed")
            detail = {
                "type": "GenerationError",
                "category": "unknown",
                "provider": self._model_provider(self.model),
                "model": self.model_name or "unknown",
                "message": message,
                "remediation": REMEDIATION_BY_CATEGORY["unknown"],
                "retryable": False,
            }
        message = detail.get("message") or "generation failed"
        meta: dict[str, Any] = {"reason": "generation_failed", "error": detail}
        if debug_trace is not None:
            debug_trace.total_tokens = tokens
            debug_trace.success = False
            meta["debug_trace"] = debug_trace
        return AgentResponse(
            output=generation_failure_text(detail),
            success=False,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=ToolCallList(total=tool_calls),
            tokens_used=tokens,
            metadata=meta,
        )

    @staticmethod
    def _reconstruct_error(
        metadata: dict[str, Any] | None, response: Any = None
    ) -> Exception:
        """Rebuild a typed exception from a failure response's metadata.

        Used by ``raise_on_error`` so callers get a typed error rather than a
        bare AgentResponse. Falls back to ``RuntimeError`` when the failure was
        not a classified provider error (e.g. guardrail block, max-iterations).

        When *response* is given and the loop stopped the run before the model
        wrote an answer, the exception is a
        :class:`~effgen.errors.RunStoppedError` carrying that response, so the
        progress the run had reached survives the raise.
        """
        if response is not None and getattr(response, "outcome", None) == "stopped":
            from effgen.errors import RunStoppedError

            return RunStoppedError(response)
        metadata = metadata or {}
        detail = metadata.get("error")
        if not isinstance(detail, dict):
            if metadata.get("guardrail_blocked"):
                return RuntimeError(
                    f"Blocked by guardrail: {metadata.get('guardrail_reason', 'policy')}"
                )
            reason = metadata.get("reason")
            if reason in ("max_iterations_exhausted", "max_iterations_reached"):
                return RuntimeError("Maximum iterations reached without a final answer.")
            if reason == "max_iterations_partial":
                return RuntimeError(
                    "Stopped at the iteration cap with a partial result; "
                    "raise max_iterations to let the run finish."
                )
            return RuntimeError(str(detail) if detail else "Agent run failed")
        category = detail.get("category")
        provider = detail.get("provider", "") or ""
        model = detail.get("model", "") or ""
        message = detail.get("message", "") or ""
        if category == "auth":
            return ModelAuthError(provider, model, message)
        if category == "not_found":
            return ModelNotFoundError(provider, model, message)
        if category == "refusal":
            return ModelRefusalError(message, model)
        if category == "invalid_request":
            return InvalidRequestError(provider, model, message)
        if category == "unreachable":
            return BackendUnreachableError(provider, model, message)
        return RuntimeError(f"{detail.get('type', 'Error')}: {message}")

    def _generate_speculative(self, prompt: str, **kwargs) -> dict[str, Any] | None:
        """Run generation on 2 models concurrently, return first success.

        Uses asyncio.gather with return_when=FIRST_COMPLETED semantics via
        asyncio.wait. Returns None if both fail.
        """
        if len(self._all_models) < 2:
            return None

        models_to_run = self._all_models[:2]
        base_temperature = kwargs.get('temperature', self.config.temperature)

        default_stop_sequences = [
            "\nObservation:", "\nQuestion:", "\nHuman:", "\nUser:",
        ]

        gen_config = GenerationConfig(
            temperature=base_temperature,
            max_tokens=kwargs.get('max_tokens', default_max_output_tokens(self.model)),
            top_p=kwargs.get('top_p', self.config.top_p),
            top_k=kwargs.get('top_k', self.config.top_k),
            seed=kwargs.get('seed', self.config.seed),
            presence_penalty=kwargs.get('presence_penalty', self.config.presence_penalty),
            frequency_penalty=kwargs.get('frequency_penalty', self.config.frequency_penalty),
            repetition_penalty=kwargs.get('repetition_penalty', self.config.repetition_penalty),
            stop_sequences=kwargs.get('stop_sequences', default_stop_sequences),
        )

        extra_gen_kwargs = model_call_kwargs(kwargs)

        async def _run_model(model: BaseModel) -> dict[str, Any]:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: model.generate(prompt, config=gen_config, **extra_gen_kwargs)
            )
            text = result.text if result and result.text else ""
            if not text.strip():
                raise RuntimeError("Empty response")
            return {
                "text": text,
                "tokens_used": result.tokens_used if result else 0,
                "finish_reason": result.finish_reason if result else "unknown",
                "metadata": result.metadata if result and hasattr(result, 'metadata') else {},
            }

        async def _speculate() -> dict[str, Any] | None:
            tasks = [asyncio.create_task(_run_model(m)) for m in models_to_run]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Cancel remaining
            for t in pending:
                t.cancel()

            for t in done:
                if not t.cancelled() and t.exception() is None:
                    return t.result()

            # All failed
            return None

        try:
            return self._run_coroutine_sync(_speculate())
        except Exception as e:
            logger.warning("Speculative execution failed: %s", e)
            return None

    def _run_direct_inference(self,
                               task: str,
                               context: dict[str, Any],
                               **kwargs) -> AgentResponse:
        """
        Run direct inference without ReAct loop (for when no tools are available).

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        inputs = kwargs.pop("inputs", None)
        if inputs is not None:
            # The multimodal builder already adds config.system_prompt as a
            # dedicated system message, so the persona is honored there.
            prompt = self._build_multimodal_prompt(task, inputs)
        else:
            # Include conversation history from short-term memory for multi-turn
            # context. A plain string prompt has no separate system slot, so a
            # custom persona leads the prompt (and owns the response contract)
            # instead of the framework's "answer directly" framing — without
            # this a "respond only in French" / Socratic tutor persona is
            # silently ignored on every provider. Default agents are unchanged.
            conversation_history = self._format_conversation_history()
            prompt = self._direct_prompt(task, conversation_history)

        try:
            # Time the call as a model span so a tool-free run still shows its
            # inner structure (the ReAct path already does this).
            model_name = getattr(self, "model_name", None) or "unknown"
            with start_model_call(
                provider=_infer_provider_from_model(self.model, model_name),
                model=model_name,
            ) as _mspan:
                response = self._generate(prompt, _task_hint=task, **kwargs)
                tokens_used = response.get("tokens_used", 0)
                _meta = response.get("metadata") or {}
                try:
                    _mspan.set_attribute(
                        ModelAttrs.INPUT_TOKENS, int(_meta.get("prompt_tokens", 0) or 0)
                    )
                    _mspan.set_attribute(ModelAttrs.OUTPUT_TOKENS, int(tokens_used or 0))
                    _stamp_call_cost(_mspan, _meta)
                except Exception:  # noqa: BLE001 - telemetry is best-effort
                    logger.debug("Failed to set model span token attributes", exc_info=True)

                # Mirror the tool-loop's check: a generation error must NOT be
                # reported as success=True with empty output. Both paths
                # return the identical failure shape via
                # _generation_failure_response.
                if response.get("finish_reason") == "error":
                    failure = self._generation_failure_response(
                        response, iterations=1, tool_calls=0, tokens=tokens_used,
                    )
                    # The adapter reports this failure by returning, not by
                    # raising, so record it on the span explicitly.
                    mark_span_error(str(failure.output)[:300])
                    return failure

            answer = response["text"].strip()
            answer = sanitize_final_answer(answer) or answer
            return AgentResponse(
                output=answer,
                success=True,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=ToolCallList(),
                tokens_used=tokens_used,
                metadata={"reason": "final_answer", "multimodal_inputs": inputs is not None},
            )

        except Exception as e:
            logger.error(f"Direct inference failed: {e}")
            detail = self._build_error_detail(e, self.model)
            return AgentResponse(
                output=generation_failure_text(detail),
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=ToolCallList(),
                tokens_used=0,
                metadata={"reason": "generation_failed", "error": detail},
            )
