"""
Span attribute constants for effGen OpenTelemetry instrumentation.

Every span attribute name used in effGen adapters, tools, agent loop, and
router **must** reference a constant from this module.  String literals in
instrumentation code are forbidden — the type-checker will catch typos.

Span names
----------
- ``effgen.agent.run``           — top-level agent execution
- ``effgen.agent.iteration``     — a single ReAct iteration
- ``effgen.model.call``          — one model inference call
- ``effgen.tool.call``           — one tool execution
- ``effgen.router.decision``     — policy-based routing decision
- ``effgen.retry.attempt``       — retry attempt (event on parent span)

Attribute keys
--------------
All keys are namespaced strings.  Import the relevant group class and use its
class attributes::

    from effgen.observability.spans import SpanName, AgentAttrs, ModelAttrs, ToolAttrs

    with tracer.start_as_current_span(SpanName.AGENT_RUN) as span:
        span.set_attribute(AgentAttrs.PRESET, "default")
        span.set_attribute(AgentAttrs.ITERATION, 1)
"""

from __future__ import annotations

__all__ = [
    "SpanName",
    "AgentAttrs",
    "ExecutionAttrs",
    "ModelAttrs",
    "ToolAttrs",
    "RouterAttrs",
    "RetryAttrs",
]


# ---------------------------------------------------------------------------
# Span names
# ---------------------------------------------------------------------------


class SpanName:
    """Canonical span names emitted by effGen."""

    #: Top-level agent execution (wraps the full agent.run() call)
    AGENT_RUN: str = "effgen.agent.run"

    #: A single ReAct iteration inside the agent loop
    AGENT_ITERATION: str = "effgen.agent.iteration"

    #: One call to a model provider (generate / generate_with_tools)
    MODEL_CALL: str = "effgen.model.call"

    #: One tool execution
    TOOL_CALL: str = "effgen.tool.call"

    #: A routing decision by the policy-based router
    ROUTER_DECISION: str = "effgen.router.decision"

    #: Retry attempt event name (added as a span *event*, not a child span)
    RETRY_ATTEMPT: str = "effgen.retry.attempt"


# ---------------------------------------------------------------------------
# Attribute key groups
# ---------------------------------------------------------------------------


class AgentAttrs:
    """Attributes attached to ``effgen.agent.run`` and ``effgen.agent.iteration`` spans."""

    #: Agent preset/name (string)
    PRESET: str = "effgen.agent.preset"

    #: Current iteration number inside the ReAct loop (int)
    ITERATION: str = "effgen.agent.iteration"

    #: Run identifier (UUID string, set on agent.run spans)
    RUN_ID: str = "effgen.agent.run_id"

    #: Task text (truncated to ≤500 chars)
    TASK: str = "effgen.agent.task"

    #: Failure message when the run reported a failure without raising (string)
    ERROR: str = "effgen.agent.error"


class ExecutionAttrs:
    """Attributes that tie the runs of one multi-agent execution together.

    A team run or a workflow run issues one execution id and every sub-agent
    run inside it carries that id, the agent that delegated the work, and the
    role the agent played. Spans and stored run records both carry these, so N
    sub-agent traces can be grouped back into the single execution they came
    from.
    """

    #: Identifier shared by every run in one team/workflow execution (string)
    ID: str = "effgen.execution.id"

    #: What issued the execution: ``"team"`` | ``"workflow"`` (string)
    KIND: str = "effgen.execution.kind"

    #: Name of the team or workflow (string)
    NAME: str = "effgen.execution.name"

    #: Name of the agent that delegated this work, when one did (string)
    PARENT_AGENT: str = "effgen.execution.parent_agent"

    #: The agent's role in the execution, e.g. ``"manager"`` | ``"worker"``
    #: | ``"node"`` (string)
    ROLE: str = "effgen.execution.role"


class ModelAttrs:
    """Attributes attached to ``effgen.model.call`` spans."""

    #: Provider name, e.g. ``"openai"``, ``"cerebras"`` (string)
    PROVIDER: str = "effgen.model.provider"

    #: Model identifier, e.g. ``"gpt-4o-mini"`` (string)
    NAME: str = "effgen.model.name"

    #: Number of cached input tokens read from the provider's cache (int)
    CACHED_TOKENS: str = "effgen.model.cached_tokens"

    #: Number of prompt tokens written into the provider's cache (int)
    CACHE_WRITE_TOKENS: str = "effgen.model.cache_write_tokens"

    #: Number of prompt / input tokens (int)
    INPUT_TOKENS: str = "effgen.model.input_tokens"

    #: Number of completion / output tokens (int)
    OUTPUT_TOKENS: str = "effgen.model.output_tokens"

    #: Reasoning effort level, e.g. ``"low"`` / ``"medium"`` / ``"high"`` (string)
    REASONING_EFFORT: str = "effgen.model.reasoning_effort"

    #: Thinking budget in tokens (int, reasoning models only)
    THINKING_BUDGET: str = "effgen.model.thinking_budget"

    #: Number of multimodal content parts (int, multimodal calls only)
    PARTS_COUNT: str = "effgen.model.parts_count"

    #: Total cost of this call in USD (float; 0.0 on a free tier, and the
    #: attribute is absent when the model publishes no price)
    COST_USD: str = "effgen.model.cost_usd"

    #: Generation outcome: ``"ok"`` | ``"error"`` | ``"timeout"`` (string)
    OUTCOME: str = "effgen.model.outcome"

    #: Latency in milliseconds (float)
    LATENCY_MS: str = "effgen.model.latency_ms"


class ToolAttrs:
    """Attributes attached to ``effgen.tool.call`` spans."""

    #: Tool name (string)
    NAME: str = "effgen.tool.name"

    #: Execution status: ``"ok"`` | ``"error"`` | ``"timeout"`` (string)
    STATUS: str = "effgen.tool.status"

    #: Truncated input arguments (string, ≤500 chars)
    INPUT: str = "effgen.tool.input"

    #: Latency in milliseconds (float)
    LATENCY_MS: str = "effgen.tool.latency_ms"


class RouterAttrs:
    """Attributes attached to ``effgen.router.decision`` spans."""

    #: Routing policy name, e.g. ``"cost"`` | ``"latency"`` | ``"capability"`` (string)
    POLICY: str = "effgen.router.policy"

    #: Selected provider, e.g. ``"cerebras"`` (string)
    SELECTED_PROVIDER: str = "effgen.router.selected_provider"

    #: Comma-separated list of considered (provider, model) candidates (string)
    CONSIDERED: str = "effgen.router.considered"

    #: Number of candidates eliminated by the policy (int)
    ELIMINATED_COUNT: str = "effgen.router.eliminated_count"


class RetryAttrs:
    """Attributes attached to ``effgen.retry.attempt`` span *events*."""

    #: Attempt number (1-based int)
    ATTEMPT: str = "effgen.retry.attempt"

    #: Reason for retry, e.g. ``"http_429"`` | ``"network_error"`` (string)
    REASON: str = "effgen.retry.reason"

    #: Delay before this attempt in seconds (float)
    DELAY_S: str = "effgen.retry.delay_s"
