"""
Multi-agent orchestration for effGen.

Coordinates multiple agents using various patterns:
- Sequential: Agents work one after another
- Hierarchical: Manager agent coordinates worker agents
- Collaborative: Agents discuss and reach consensus
- Competitive: Multiple agents solve same task, select best
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..observability.tracing import execution_scope, new_execution_id
from .agent import Agent, AgentMode
from .execution_tracker import EventType, ExecutionEvent, ExecutionTracker
from .lifecycle import AgentRegistry
from .message_bus import AgentMessage, MessageBus, MessageType
from .shared_state import SharedState

logger = logging.getLogger(__name__)


def _redact(text: str) -> str:
    """Scrub secrets from an error string before surfacing/logging it."""
    try:
        from ..observability.redact import get_redactor
        return get_redactor().scrub(text)
    except Exception:  # pragma: no cover - redaction is best-effort
        return text


def _error_text(err: Any) -> str:
    """Render a team/workflow ``metadata["error"]`` value as display text.

    That value is either a plain string (an already-redacted message) or the
    structured ``{type, category, provider, model, message, retryable}`` dict
    an ``AgentResponse`` attaches on failure — pull its ``message`` field
    instead of falling back to the dict's ``repr()``.
    """
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err)


def _response_cost(response: Any) -> float | None:
    """Pull the per-run cost (USD) off an ``AgentResponse``'s metadata.

    The single-agent surface records cost under ``metadata["cost_usd"]`` (older
    runs used ``"cost"``). Returns ``None`` when the run carries no usable
    number — a local engine, or a model the catalog publishes no rate for —
    so a team tab never presents an unknown cost as a free one.
    """
    meta = getattr(response, "metadata", None) or {}
    raw = meta.get("cost_usd", meta.get("cost"))
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _response_time(response: Any) -> float:
    """Wall-clock seconds an agent's run took (0.0 when not recorded)."""
    try:
        return round(float(getattr(response, "execution_time", 0.0) or 0.0), 3)
    except (TypeError, ValueError):
        return 0.0


def _agent_node(agent: Any, *, role: str) -> dict[str, Any]:
    """Describe one team member: its name, model, tools and role."""
    tools = getattr(agent, "tools", None)
    if isinstance(tools, dict):
        tool_names = sorted(str(t) for t in tools)
    elif isinstance(tools, list | tuple | set):
        tool_names = sorted(str(getattr(t, "name", t)) for t in tools)
    else:
        tool_names = []
    return {
        "name": str(getattr(agent, "name", "agent")),
        "model": str(getattr(agent, "model_name", None) or "unknown"),
        "tools": tool_names,
        "role": role,
    }


def _aggregate_usage(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum ``cost_usd``/``tokens_used`` across member-agent result dicts.

    Mirrors the per-run ``AgentResponse.metadata`` cost surface so a
    budget-conscious caller can read team/workflow spend without re-summing by
    hand. The tab sums the members that reported a cost; when none did — every
    member local or on a model with no published rate — ``cost_usd`` is
    ``None`` rather than ``$0``. Non-numeric values are skipped.
    """
    cost = 0.0
    priced = 0
    tokens = 0
    for r in responses:
        raw_cost = r.get("cost_usd")
        if raw_cost is not None:
            try:
                cost += float(raw_cost)
                priced += 1
            except (TypeError, ValueError):
                pass
        try:
            tokens += int(r.get("tokens_used") or 0)
        except (TypeError, ValueError):
            pass
    return {"cost_usd": round(cost, 6) if priced else None, "tokens_used": tokens}


class OrchestrationPattern(Enum):
    """Available orchestration patterns."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    PIPELINE = "pipeline"


@dataclass
class TeamConfig:
    """
    Configuration for a team of agents.

    Attributes:
        name: Team name
        pattern: Orchestration pattern
        agents: List of agents in team
        manager_agent: Optional manager agent (for hierarchical)
        voting_strategy: Voting strategy for collaborative/competitive
        timeout: Team execution timeout
        max_rounds: Maximum collaboration rounds
        metadata: Additional metadata
    """
    name: str
    pattern: OrchestrationPattern
    agents: list[Agent] = field(default_factory=list)
    manager_agent: Agent | None = None
    voting_strategy: str = "majority"  # majority, unanimous, weighted
    timeout: int = 600
    max_rounds: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the team's topology — members, their tools, and the edges.

        Mirrors :meth:`~effgen.core.workflow.WorkflowDAG.to_dict`: the shape is
        available before the team runs, so a team can be drawn or reviewed
        without spending anything on a run.

        Returns a dict with ``name``, ``pattern``, ``manager`` (or ``None``),
        ``agents`` — each with its ``name``, ``model`` and ``tools`` — and
        ``edges``. The edges the pattern implies are ``delegation``
        (manager → worker, hierarchical), ``peer`` (worker ↔ worker,
        collaborative), and ``handoff`` (stage → next stage, sequential and
        pipeline). Parallel and competitive teams run every member on the same
        task, so they carry no inter-agent edges.

        Example::

            team = orch.create_team("desk", [a, b], pattern=OrchestrationPattern.SEQUENTIAL)
            shape = team.to_dict()
            print(shape["edges"])   # [{'source': 'a', 'target': 'b', 'kind': 'handoff'}]
        """
        nodes = [_agent_node(a, role="worker") for a in self.agents]
        manager = _agent_node(self.manager_agent, role="manager") if self.manager_agent else None

        edges: list[dict[str, str]] = []
        names = [n["name"] for n in nodes]
        if self.pattern == OrchestrationPattern.HIERARCHICAL and manager is not None:
            edges = [
                {"source": manager["name"], "target": n, "kind": "delegation"}
                for n in names
            ]
        elif self.pattern == OrchestrationPattern.COLLABORATIVE:
            edges = [
                {"source": a, "target": b, "kind": "peer"}
                for i, a in enumerate(names)
                for b in names[i + 1:]
            ]
        elif self.pattern in (
            OrchestrationPattern.SEQUENTIAL, OrchestrationPattern.PIPELINE,
        ):
            edges = [
                {"source": a, "target": b, "kind": "handoff"}
                for a, b in zip(names, names[1:], strict=False)
            ]

        return {
            "name": self.name,
            "pattern": self.pattern.value,
            "manager": manager,
            "agents": nodes,
            "edges": edges,
        }

    def diagram(self, response: TeamResponse | None = None) -> str:
        """Return the team's shape as plain text, one line per member.

        Pass the :class:`TeamResponse` from a run to annotate each member with
        its status, duration, cost and tokens; without one the shape renders as
        pending structure. Status glyphs carry the meaning, so the output stays
        readable when it is piped or captured::

            print(team.diagram(orch.assign_task("Draft the brief.", team)))
        """
        from ..ui.team_viz import team_diagram_lines

        results = list(response.agent_responses) if response is not None else None
        if response is not None and self.manager_agent is not None and not any(
            r.get("agent_name") == self.manager_agent.name for r in (results or [])
        ):
            # A manager coordinates rather than answering a subtask, so it has no
            # entry in agent_responses; its status comes from the team's own
            # outcome, and a worker failure does not mark the manager failed.
            manager_ok = response.success or response.metadata.get("reason") == "sub_agent_failed"
            results.append({"agent_name": self.manager_agent.name, "success": manager_ok})
        return "\n".join(text for _style, text in team_diagram_lines(self.to_dict(), results))


@dataclass
class TeamResponse:
    """
    Response from team execution.

    Attributes:
        output: Final team output
        success: Whether execution succeeded
        pattern: Orchestration pattern used
        agent_responses: Individual agent responses
        execution_time: Total execution time
        rounds: Number of rounds (for collaborative)
        selected_response: Selected response (for competitive)
        consensus_score: Consensus score (for collaborative)
        metadata: Additional metadata
    """
    output: str
    success: bool = True
    pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL
    agent_responses: list[dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    rounds: int = 1
    selected_response: dict[str, Any] | None = None
    consensus_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "success": self.success,
            "pattern": self.pattern.value,
            "agent_responses": self.agent_responses,
            "execution_time": round(self.execution_time, 2),
            "rounds": self.rounds,
            "selected_response": self.selected_response,
            "consensus_score": self.consensus_score,
            "metadata": self.metadata
        }


class MultiAgentOrchestrator:
    """
    Coordinate multiple agents with various orchestration patterns.

    Features:
    - Agent registry
    - Task routing
    - Inter-agent communication
    - Result aggregation
    - Conflict resolution
    - Load balancing

    Example::

        from effgen import (
            Agent, AgentConfig, MultiAgentOrchestrator, OrchestrationPattern, load_model,
        )

        m = load_model("gpt-5-nano")
        writer = Agent(AgentConfig(name="writer", model=m))
        editor = Agent(AgentConfig(name="editor", model=m))

        orch = MultiAgentOrchestrator()
        team = orch.create_team(
            "blog", [writer, editor], pattern=OrchestrationPattern.SEQUENTIAL,
        )
        result = orch.assign_task("Write one sentence about the ocean.", team)
        print(result.success, result.output)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize orchestrator.

        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.teams: dict[str, TeamConfig] = {}
        self.execution_tracker = ExecutionTracker()
        self.agent_registry: dict[str, Agent] = {}
        self.message_bus = MessageBus(persist=True)
        self.shared_state = SharedState()
        self.lifecycle_registry = AgentRegistry()
        # Per-team cooperative-cancellation flags. cancel_workflow() sets these;
        # the execution loops check them before starting the next agent so a
        # cancel reliably stops not-yet-started in-flight work and returns the
        # partial results gathered so far.
        self._cancel_events: dict[str, threading.Event] = {}

    def register_agent(self, agent: Agent) -> None:
        """
        Register an agent.

        Args:
            agent: Agent to register
        """
        self.agent_registry[agent.name] = agent
        # Also register in lifecycle registry (ignore if already registered)
        try:
            self.lifecycle_registry.register(agent.name, agent)
        except ValueError:
            pass

    def create_team(self,
                   name: str,
                   agents: list[Agent],
                   pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL,
                   manager_agent: Agent | None = None,
                   **kwargs: Any) -> TeamConfig:
        """
        Create a team of agents.

        Args:
            name: Team name
            agents: List of agents
            pattern: Orchestration pattern
            manager_agent: Optional manager agent
            **kwargs: Additional configuration

        Returns:
            TeamConfig
        """
        team = TeamConfig(
            name=name,
            pattern=pattern,
            agents=agents,
            manager_agent=manager_agent,
            **kwargs
        )
        self.teams[name] = team

        # Register agents
        for agent in agents:
            if agent.name not in self.agent_registry:
                self.register_agent(agent)

        return team

    def assign_task(self,
                   task: str,
                   team: TeamConfig | str,
                   context: dict[str, Any] | None = None) -> TeamResponse:
        """
        Assign task to team and coordinate execution.

        Args:
            task: Task description (a plain string).
            team: Either the ``TeamConfig`` returned by :meth:`create_team`, or
                the **name** of an already-registered team (so you can pass the
                name you just registered, not only the config object).
            context: Optional context

        Returns:
            TeamResponse with results. ``success`` is ``False`` (never a silent
            success) when the team is empty or any agent fails; per-agent
            outputs and errors are preserved in ``agent_responses``.
        """
        start_time = time.time()
        context = context or {}

        # Accept a team *name* as well as a TeamConfig (additive ergonomics).
        if isinstance(team, str):
            resolved = self.teams.get(team)
            if resolved is None:
                known = ", ".join(self.teams) or "<none>"
                raise KeyError(
                    f"No team named {team!r} is registered. Known teams: {known}. "
                    f"Create one with orchestrator.create_team(name, agents, pattern=...)."
                )
            team = resolved
        elif not isinstance(team, TeamConfig):
            raise TypeError(
                "assign_task() expects a TeamConfig or a registered team name "
                f"(str); got {type(team).__name__}. Build one with "
                "orchestrator.create_team(name, agents, pattern=...)."
            )

        if not isinstance(task, str):
            raise TypeError(
                f"assign_task() task must be a string; got {type(task).__name__}."
            )

        # Empty-team guard: a team with no agents cannot succeed.
        if not team.agents:
            return TeamResponse(
                output="Error: team has no agents to run the task.",
                success=False,
                pattern=team.pattern,
                execution_time=time.time() - start_time,
                metadata={"reason": "empty_team", "error": "Team has no agents."},
            )

        execution_id = new_execution_id()

        # Arm a fresh cancellation flag for this run.
        cancel_event = threading.Event()
        self._cancel_events[team.name] = cancel_event

        # Track team task start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TASK_START,
            agent_id=f"team_{team.name}",
            message=f"Team {team.name} starting task with {team.pattern.value} pattern",
            data={
                "task": task,
                "pattern": team.pattern.value,
                "num_agents": len(team.agents)
            }
        ))

        try:
            # One id for the whole team run: every member's spans and stored run
            # records carry it, so the runs regroup into the execution they
            # belong to instead of reading as unrelated traces.
            with execution_scope(
                kind="team", name=team.name, execution_id=execution_id,
            ) as execution:
                # Execute based on pattern
                if team.pattern == OrchestrationPattern.SEQUENTIAL:
                    response = self._execute_sequential(task, team, context)
                elif team.pattern == OrchestrationPattern.PARALLEL:
                    response = self._execute_parallel(task, team, context)
                elif team.pattern == OrchestrationPattern.HIERARCHICAL:
                    response = self._execute_hierarchical(task, team, context)
                elif team.pattern == OrchestrationPattern.COLLABORATIVE:
                    response = self._execute_collaborative(task, team, context)
                elif team.pattern == OrchestrationPattern.COMPETITIVE:
                    response = self._execute_competitive(task, team, context)
                elif team.pattern == OrchestrationPattern.PIPELINE:
                    response = self._execute_pipeline(task, team, context)
                else:
                    raise ValueError(f"Unknown pattern: {team.pattern}")

            response.metadata.setdefault("execution_id", execution["execution_id"])
            response.metadata.setdefault("topology", team.to_dict())
            response.execution_time = time.time() - start_time

            # Track completion
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_COMPLETE,
                agent_id=f"team_{team.name}",
                message=f"Team completed in {response.execution_time:.2f}s",
                data={"execution_time": response.execution_time}
            ))

            return response

        except Exception as e:
            # Redact before surfacing/logging so an orchestration error never
            # leaks secrets, mirroring the single-agent failure path.
            safe = _redact(str(e))
            logger.error("Team '%s' failed: %s", team.name, safe)
            # Track failure
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_FAILED,
                agent_id=f"team_{team.name}",
                message=f"Team failed: {safe}",
                data={"error": safe}
            ))

            return TeamResponse(
                output=f"Error: {safe}",
                success=False,
                pattern=team.pattern,
                execution_time=time.time() - start_time,
                metadata={
                    "reason": "team_error",
                    "error": safe,
                    "execution_id": execution_id,
                    "topology": team.to_dict(),
                },
            )

    def _execute_sequential(self,
                           task: str,
                           team: TeamConfig,
                           context: dict[str, Any]) -> TeamResponse:
        """
        Execute agents one after another.

        Output of agent N becomes input of agent N+1.
        Messages are published on the bus and results stored in shared state.
        """
        current_task = task
        responses = []
        cancel_event = self._cancel_events.get(team.name)
        cancelled = False
        parent: str | None = None  # the stage that handed work to this one

        for i, agent in enumerate(team.agents):
            # Cooperative cancellation: stop before launching the next agent.
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break

            # Send task assignment via message bus
            self.message_bus.send(AgentMessage(
                sender=f"orchestrator_{team.name}",
                recipient=agent.name,
                type=MessageType.TASK_ASSIGNMENT,
                payload=current_task[:200],
                topic=f"team.{team.name}.task",
            ))

            # Track agent start
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_START,
                agent_id=agent.name,
                message=f"Agent {i+1}/{len(team.agents)} starting",
                data={"task": current_task[:100]}
            ))

            # Execute agent
            with execution_scope(role="stage", parent_agent=parent):
                response = agent.run(current_task, mode=AgentMode.AUTO, context=context)
            parent = agent.name

            # Track completion
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_COMPLETE,
                agent_id=agent.name,
                message=f"Agent {i+1}/{len(team.agents)} completed"
            ))

            agent_result = {
                "agent_name": agent.name,
                "output": response.output,
                "success": response.success,
                "tokens_used": response.tokens_used,
                "cost_usd": _response_cost(response),
                "execution_time": _response_time(response),
            }
            responses.append(agent_result)

            # Publish result on message bus
            self.message_bus.send(AgentMessage(
                sender=agent.name,
                recipient=f"orchestrator_{team.name}",
                type=MessageType.RESULT,
                payload=agent_result,
                topic=f"team.{team.name}.result",
            ))

            # Store result in shared state
            self.shared_state.set(
                f"team_{team.name}", f"result_{agent.name}",
                agent_result, agent_id=agent.name,
            )

            if not response.success:
                # Capture the typed, redacted error from the failing agent so the
                # team response carries a real per-agent failure, not a silent
                # success. Labeled partials (the responses so far) are preserved.
                err_detail = (getattr(response, "metadata", None) or {}).get("error")
                agent_result["error"] = err_detail or _redact(str(response.output))
                # Publish error
                self.message_bus.send(AgentMessage(
                    sender=agent.name,
                    recipient=f"orchestrator_{team.name}",
                    type=MessageType.ERROR,
                    payload=_redact(str(response.output)),
                    topic=f"team.{team.name}.error",
                ))
                break

            # Use output as input for next agent
            current_task = response.output

        # success only when at least one agent ran AND all that ran succeeded
        # AND the run was not cancelled — never a silent True on an empty/aborted run.
        success = bool(responses) and all(r["success"] for r in responses) and not cancelled
        meta: dict[str, Any] = dict(_aggregate_usage(responses))
        if cancelled:
            meta["reason"] = "cancelled"
            meta["error"] = "Workflow cancelled before completion."
        elif not success:
            meta["reason"] = "sub_agent_failed"
            failed = next((r for r in responses if not r["success"]), None)
            if failed is not None:
                meta["error"] = failed.get("error", "A sub-agent failed.")
        # On failure, never echo the customer's own input back as the answer —
        # a caller who reads .output without checking .success must not mistake
        # the input (or a stale partial) for a real result. Successful partials
        # remain discoverable in agent_responses.
        output = (
            current_task if success
            else f"Error: {_error_text(meta.get('error', 'team run did not succeed.'))}"
        )
        return TeamResponse(
            output=output,
            success=success,
            pattern=OrchestrationPattern.SEQUENTIAL,
            agent_responses=responses,
            metadata=meta,
        )

    def _execute_parallel(self,
                         task: str,
                         team: TeamConfig,
                         context: dict[str, Any]) -> TeamResponse:
        """
        Execute all agents in parallel on the same task.

        Synthesize results at the end.
        """
        # Run all agents in parallel
        cancel_event = self._cancel_events.get(team.name)
        responses = asyncio.run(
            self._parallel_execution(task, team.agents, context, cancel_event)
        )

        # Synthesize results
        synthesis = self._synthesize_parallel_results(task, responses)

        # Success only if at least one agent actually succeeded.
        success = any(r["success"] for r in responses)
        meta: dict[str, Any] = dict(_aggregate_usage(responses))
        if not success:
            meta["reason"] = "sub_agent_failed"
            failed = next((r for r in responses if r.get("error")), None)
            if failed is not None:
                meta["error"] = failed["error"]
        return TeamResponse(
            output=synthesis,
            success=success,
            pattern=OrchestrationPattern.PARALLEL,
            agent_responses=responses,
            metadata=meta,
        )

    async def _parallel_execution(self,
                                  task: str,
                                  agents: list[Agent],
                                  context: dict[str, Any],
                                  cancel_event: threading.Event | None = None,
                                  ) -> list[dict[str, Any]]:
        """Execute agents in parallel."""
        role = "member"  # parallel and competitive members are peers

        async def run_agent(agent: Agent):
            # Skip not-yet-started work if the run was cancelled.
            if cancel_event is not None and cancel_event.is_set():
                return {
                    "agent_name": agent.name,
                    "output": "Cancelled before start.",
                    "success": False,
                    "tokens_used": 0,
                    "error": "Workflow cancelled before completion.",
                }
            # Track start
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_START,
                agent_id=agent.name,
                message="Agent starting (parallel)"
            ))

            # Run agent
            with execution_scope(role=role):
                response = await agent.run_async(task, mode=AgentMode.AUTO, context=context)

            # Track completion
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_COMPLETE,
                agent_id=agent.name,
                message="Agent completed (parallel)"
            ))

            result = {
                "agent_name": agent.name,
                "output": response.output,
                "success": response.success,
                "tokens_used": response.tokens_used,
                "cost_usd": _response_cost(response),
                "execution_time": _response_time(response),
            }
            if not response.success:
                detail = (getattr(response, "metadata", None) or {}).get("error")
                result["error"] = detail or _redact(str(response.output))
            return result

        # Execute all in parallel
        results = await asyncio.gather(*[run_agent(agent) for agent in agents])
        return list(results)

    def _execute_hierarchical(self,
                             task: str,
                             team: TeamConfig,
                             context: dict[str, Any]) -> TeamResponse:
        """
        Manager agent coordinates worker agents.

        Manager delegates subtasks and synthesizes results.
        """
        if not team.manager_agent:
            raise ValueError("Hierarchical pattern requires manager_agent")

        worker_names = [agent.name for agent in team.agents]
        manager_name = team.manager_agent.name

        # Defensive language note, mirroring the fix applied to
        # DecompositionEngine's templates (PR #151): team.manager_agent.run()
        # already applies the manager's own system_prompt as a normal Agent
        # call (unlike DecompositionEngine._llm_decompose, which bypasses the
        # Agent wrapper via a raw llm_client.generate()), so this path is
        # less exposed to begin with. Still, the instruction text below is
        # fixed English and asks for a specific delegation format, which can
        # pull a small model back toward English for the subtask lines
        # themselves — this note removes that ambiguity explicitly rather
        # than relying on the system_prompt alone to override it.
        manager_system_prompt = getattr(getattr(team.manager_agent, "config", None), "system_prompt", None)
        language_note = ""
        if manager_system_prompt:
            language_note = (
                "\nWrite the subtask lines in the same language as this "
                f"instruction: \"{manager_system_prompt}\"\n"
            )

        # Manager decomposes task. Ask it to *name the worker* on each subtask
        # line ("<worker>: <what to do>") so subtasks are routed to the specialist
        # the manager intended, not by list position.
        decomposition_prompt = f"""You are a manager coordinating a team. Break down this task into subtasks for your team.
{language_note}
Task: {task}

Available workers: {', '.join(worker_names)}

Provide subtasks as a numbered list. Start EACH line with the name of the worker \
who should handle it, followed by a colon — for example "{worker_names[0]}: <subtask>". \
Use only the worker names listed above."""

        with execution_scope(role="manager"):
            manager_response = team.manager_agent.run(
                decomposition_prompt,
                mode=AgentMode.SINGLE,
                context=context
            )

        # Parse subtasks (simple heuristic)
        subtasks = self._parse_subtasks(manager_response.output)

        # Route each subtask to the worker(s) the manager named (by label),
        # falling back to round-robin only when a line carries no recognizable
        # worker name. A subtask addressed to several workers runs on each of
        # them rather than being narrowed to one. Every subtask runs — none are
        # dropped because there are more subtasks than agents.
        responses = []
        rr = 0  # round-robin cursor for unlabeled subtasks
        for subtask in subtasks:
            targets = self._route_subtask_all(subtask, team.agents)
            if not targets:
                targets = [team.agents[rr % len(team.agents)]]
                rr += 1
            co_assigned = [a.name for a in targets] if len(targets) > 1 else []
            for agent in targets:
                with execution_scope(role="worker", parent_agent=manager_name):
                    response = agent.run(subtask, mode=AgentMode.AUTO, context=context)
                entry: dict[str, Any] = {
                    "agent_name": agent.name,
                    "subtask": subtask,
                    "output": response.output,
                    "success": response.success,
                    "tokens_used": response.tokens_used,
                    "cost_usd": _response_cost(response),
                    "execution_time": _response_time(response),
                }
                if co_assigned:
                    # The subtask named more than one worker; record who else
                    # received it so the split is visible in the result.
                    entry["co_assigned"] = co_assigned
                responses.append(entry)
                if not response.success:
                    detail = (getattr(response, "metadata", None) or {}).get("error")
                    entry["error"] = detail or _redact(str(response.output))

        # Manager synthesizes
        synthesis_prompt = f"""Synthesize the results from your team into a final answer for: {task}
{language_note}
Team results:
{self._format_team_results(responses)}

Provide a comprehensive final answer."""

        with execution_scope(role="manager"):
            final_response = team.manager_agent.run(
                synthesis_prompt,
                mode=AgentMode.SINGLE,
                context=context
            )

        # Success requires the manager's synthesis to succeed AND no worker to
        # have failed (a dropped/failed specialist must not pass silently).
        worker_ok = all(r["success"] for r in responses) if responses else True
        success = final_response.success and worker_ok
        meta: dict[str, Any] = dict(_aggregate_usage(
            [{"tokens_used": manager_response.tokens_used,
              "cost_usd": _response_cost(manager_response)},
             *responses,
             {"tokens_used": final_response.tokens_used,
              "cost_usd": _response_cost(final_response)}]
        ))
        meta["manager_decomposition"] = manager_response.output
        if not success:
            meta["reason"] = "synthesis_failed" if not final_response.success else "sub_agent_failed"
            failed = next((r for r in responses if not r["success"]), None)
            if not final_response.success:
                detail = (getattr(final_response, "metadata", None) or {}).get("error")
                meta["error"] = detail or _redact(str(final_response.output))
            elif failed is not None:
                meta["error"] = failed.get("error", "A worker agent failed.")
        output = (
            final_response.output if success
            else f"Error: {_error_text(meta.get('error', 'team run did not succeed.'))}"
        )
        return TeamResponse(
            output=output,
            success=success,
            pattern=OrchestrationPattern.HIERARCHICAL,
            agent_responses=responses,
            metadata=meta,
        )

    def _execute_collaborative(self,
                              task: str,
                              team: TeamConfig,
                              context: dict[str, Any]) -> TeamResponse:
        """
        Agents discuss and reach consensus.

        Multiple rounds of discussion until consensus.
        """
        max_rounds = team.max_rounds
        current_responses = []
        any_failure = False
        first_error: str | None = None
        discussion_rounds: list[dict[str, Any]] = []
        previous_speaker: str | None = None

        for round_num in range(1, max_rounds + 1):
            round_responses = []

            for agent in team.agents:
                # Build prompt with previous responses
                if current_responses:
                    discussion = "\n\n".join([
                        f"{r['agent_name']}: {r['output']}"
                        for r in current_responses
                    ])
                    prompt = f"""Task: {task}

Previous discussion:
{discussion}

Consider the above viewpoints and provide your perspective or refined answer."""
                else:
                    prompt = task

                # The previous round's last speaker is the response this one was
                # shown most recently, so the discussion links up in telemetry
                # instead of reading as unconnected runs.
                with execution_scope(role="collaborator", parent_agent=previous_speaker):
                    response = agent.run(prompt, mode=AgentMode.AUTO, context=context)
                # Capture per-agent success/error like the other patterns — a
                # failed agent must be visible, never an invisible silent pass.
                entry = {
                    "agent_name": agent.name,
                    "output": response.output,
                    "round": round_num,
                    # The responses this one was shown, so the discussion can be
                    # read (and drawn) as who answered whom.
                    "responds_to": [r["agent_name"] for r in current_responses],
                    "success": response.success,
                    "tokens_used": response.tokens_used,
                    "cost_usd": _response_cost(response),
                    "execution_time": _response_time(response),
                }
                if not response.success:
                    any_failure = True
                    detail = (getattr(response, "metadata", None) or {}).get("error")
                    entry["error"] = detail or _redact(str(response.output))
                    if first_error is None:
                        first_error = entry["error"]
                round_responses.append(entry)

            discussion_rounds.append({
                "round": round_num,
                "agents": [r["agent_name"] for r in round_responses],
            })
            current_responses = round_responses
            previous_speaker = round_responses[-1]["agent_name"] if round_responses else None

            # Check for consensus
            consensus_score = self._calculate_consensus(round_responses)
            if consensus_score > 0.8:
                break

        # Success requires at least one agent to have run with no agent failing at any point
        # (fail-closed — a failing collaborator must not be hidden by a True).
        success = bool(current_responses) and not any_failure
        meta: dict[str, Any] = dict(_aggregate_usage(current_responses))
        meta["discussion_rounds"] = discussion_rounds
        if not success:
            meta["reason"] = "sub_agent_failed" if any_failure else "empty_team"
            meta["error"] = first_error or "A collaborating agent failed."

        # Synthesize final output (on failure, surface the error not a half answer).
        if success:
            final_output = self._synthesize_collaborative_results(task, current_responses)
        else:
            final_output = f"Error: {_error_text(meta['error'])}"

        return TeamResponse(
            output=final_output,
            success=success,
            pattern=OrchestrationPattern.COLLABORATIVE,
            agent_responses=current_responses,
            rounds=round_num,
            consensus_score=consensus_score,
            metadata=meta,
        )

    def _execute_competitive(self,
                            task: str,
                            team: TeamConfig,
                            context: dict[str, Any]) -> TeamResponse:
        """
        Multiple agents solve same task, select best solution.

        Can use voting or scoring.
        """
        # All agents work on same task
        cancel_event = self._cancel_events.get(team.name)
        responses = asyncio.run(
            self._parallel_execution(task, team.agents, context, cancel_event)
        )

        # Select best response
        best_response = self._select_best_response(task, responses, team.voting_strategy)

        meta: dict[str, Any] = dict(_aggregate_usage(responses))
        meta["voting_strategy"] = team.voting_strategy
        return TeamResponse(
            output=best_response["output"],
            success=best_response["success"],
            pattern=OrchestrationPattern.COMPETITIVE,
            agent_responses=responses,
            selected_response=best_response,
            metadata=meta,
        )

    def _execute_pipeline(self,
                         task: str,
                         team: TeamConfig,
                         context: dict[str, Any]) -> TeamResponse:
        """
        Pipeline processing: each agent is a stage; the output of one stage is
        the input to the next.

        ``PIPELINE`` is currently an alias for ``SEQUENTIAL`` — the two run
        identically (agents execute in order, threading each output forward).
        Give each stage agent a role-specific ``system_prompt`` to specialize
        the stages. To route a ticket to a *single* chosen specialist instead of
        running every stage, use ``HIERARCHICAL`` (a manager names the worker per
        subtask) — see ``docs/tutorials/multi-agent-workflows.md``.
        """
        result = self._execute_sequential(task, team, context)
        result.pattern = OrchestrationPattern.PIPELINE
        return result

    def _synthesize_parallel_results(self,
                                    task: str,
                                    responses: list[dict[str, Any]]) -> str:
        """Synthesize results from parallel execution."""
        synthesis_parts = [f"Results from {len(responses)} agents:\n"]

        for i, response in enumerate(responses, 1):
            synthesis_parts.append(f"\n{i}. {response['agent_name']}:")
            synthesis_parts.append(f"   {response['output'][:200]}...")

        # Simple concatenation
        return "\n".join(synthesis_parts)

    def _synthesize_collaborative_results(self,
                                         task: str,
                                         responses: list[dict[str, Any]]) -> str:
        """Synthesize results from collaborative discussion."""
        # Use last round responses
        latest_round = max(r["round"] for r in responses)
        latest_responses = [r for r in responses if r["round"] == latest_round]

        synthesis = f"Collaborative consensus after {latest_round} rounds:\n\n"

        for response in latest_responses:
            synthesis += f"{response['agent_name']}: {response['output']}\n\n"

        return synthesis

    def _calculate_consensus(self, responses: list[dict[str, Any]]) -> float:
        """
        Calculate a consensus score in ``[0, 1]`` across agent responses.

        Uses the mean pairwise Jaccard overlap of the responses' word sets — a
        cheap, dependency-free lexical-agreement signal (1.0 = identical wording,
        0.0 = no shared vocabulary). One or zero responses trivially agree.
        """
        import re

        def _tokens(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9']+", str(text).lower()))

        token_sets = [_tokens(r.get("output", "")) for r in responses]
        token_sets = [t for t in token_sets if t]
        if len(token_sets) < 2:
            return 1.0

        scores: list[float] = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                a, b = token_sets[i], token_sets[j]
                union = a | b
                scores.append(len(a & b) / len(union) if union else 0.0)

        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def _select_best_response(self,
                             task: str,
                             responses: list[dict[str, Any]],
                             strategy: str) -> dict[str, Any]:
        """
        Select best response using voting strategy.

        Args:
            task: Original task
            responses: Agent responses
            strategy: Voting strategy (majority, weighted, etc.)

        Returns:
            Best response
        """
        if strategy == "majority":
            # Simple: first successful response
            for response in responses:
                if response["success"]:
                    return response
            return responses[0] if responses else {}

        elif strategy == "weighted":
            # Could weight by agent performance, tokens used, etc.
            # For now, same as majority
            return self._select_best_response(task, responses, "majority")

        else:
            # Default to first response
            return responses[0] if responses else {}

    def _route_subtask(self, subtask: str, agents: list[Agent]) -> Agent | None:
        """Pick the worker a subtask is addressed to, by name.

        The manager is asked to prefix each subtask with the responsible
        worker's name (``"billing: issue the refund"``). We match that leading
        label — or any worker name mentioned in the line — against the team
        (case-insensitive) so the *named* specialist gets the work, not whoever
        happens to sit at the same list index. Returns ``None`` when no worker is
        recognizable so the caller can fall back to round-robin.
        """
        named = self._route_subtask_all(subtask, agents)
        return named[0] if named else None

    def _route_subtask_all(self, subtask: str, agents: list[Agent]) -> list[Agent]:
        """Return every worker a subtask names, in the order they are named.

        A manager sometimes addresses one line to several specialists
        ("research & writing: draft the brief"). Each named worker is returned
        so the caller can give all of them the work instead of narrowing to one
        and leaving the others out. A name that is contained in another matched
        name (``analyst`` inside ``market-analyst``) is dropped, so the longer
        specific name wins. Returns an empty list when no worker is recognizable.
        """
        text = subtask.strip()
        lowered = text.lower()
        by_name = {a.name.lower(): a for a in agents if a.name}

        # Prefer the explicit "<worker>:" label at the start of the line, and
        # only fall back to the whole line when the label names nobody.
        scope = text.split(":", 1)[0].strip().lower() if ":" in text else lowered
        matches = [(scope.find(n), n, a) for n, a in by_name.items() if n in scope]
        if not matches:
            matches = [(lowered.find(n), n, a) for n, a in by_name.items() if n in lowered]

        matched_names = {n for _pos, n, _a in matches}
        keep = [
            (pos, agent)
            for pos, name, agent in matches
            if not any(name != other and name in other for other in matched_names)
        ]
        return [agent for _pos, agent in sorted(keep, key=lambda pair: pair[0])]

    def _parse_subtasks(self, text: str) -> list[str]:
        """Parse subtasks from numbered list."""
        import re
        # Find numbered items
        pattern = r'\d+[\.)]\s+(.+?)(?=\n\d+[\.)]|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches] if matches else [text]

    def _format_team_results(self, responses: list[dict[str, Any]]) -> str:
        """Format team results for display."""
        formatted = []
        for i, response in enumerate(responses, 1):
            formatted.append(f"{i}. {response['agent_name']}:")
            formatted.append(f"   Subtask: {response.get('subtask', 'N/A')}")
            formatted.append(f"   Result: {response['output']}")
        return "\n".join(formatted)

    async def assign_task_async(self,
                                task: str,
                                team: TeamConfig | str,
                                context: dict[str, Any] | None = None) -> TeamResponse:
        """Await :meth:`assign_task` from async code.

        Takes the same arguments and returns the same :class:`TeamResponse`.
        The team runs in a worker thread, so an event loop already running in
        the caller keeps serving while the team works::

            response = await orch.assign_task_async("Draft the brief.", team)

        Args:
            task: Task description (a plain string).
            team: A ``TeamConfig`` or the name of a registered team.
            context: Optional context.

        Returns:
            TeamResponse with results.
        """
        return await asyncio.to_thread(self.assign_task, task, team, context)

    def get_team(self, name: str) -> TeamConfig | None:
        """Get team by name."""
        return self.teams.get(name)

    def list_teams(self) -> list[str]:
        """List all team names."""
        return list(self.teams.keys())

    def remove_team(self, name: str) -> None:
        """Remove a team."""
        if name in self.teams:
            del self.teams[name]

    def cancel_workflow(self, team_name: str | None = None) -> int:
        """
        Cancel running/queued work, optionally scoped to a team.

        Sets the cooperative cancellation flag so an in-progress
        :meth:`assign_task` stops before launching its next agent and returns
        the partial results gathered so far, and signals the lifecycle registry
        so any tracked agents are marked terminated.

        Args:
            team_name: If given, only cancel this team's run; otherwise cancel
                every armed team run.

        Returns:
            Number of agents signalled to cancel.
        """
        if team_name:
            event = self._cancel_events.get(team_name)
            if event is not None:
                event.set()
            team = self.teams.get(team_name)
            if not team:
                return 0
            count = 0
            for agent in team.agents:
                if self.lifecycle_registry.cancel(agent.name):
                    count += 1
            return count
        else:
            for event in self._cancel_events.values():
                event.set()
            return self.lifecycle_registry.cancel_all()

    def __repr__(self) -> str:
        """String representation."""
        return f"MultiAgentOrchestrator(teams={len(self.teams)}, agents={len(self.agent_registry)})"
