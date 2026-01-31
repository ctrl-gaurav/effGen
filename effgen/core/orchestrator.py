"""
Multi-agent orchestration for effGen.

Coordinates multiple agents using various patterns:
- Sequential: Agents work one after another
- Hierarchical: Manager agent coordinates worker agents
- Collaborative: Agents discuss and reach consensus
- Competitive: Multiple agents solve same task, select best
"""

import time
import asyncio
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .agent import Agent, AgentResponse, AgentMode
from .task import Task, TaskStatus
from .execution_tracker import ExecutionTracker, ExecutionEvent, EventType


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
    agents: List[Agent] = field(default_factory=list)
    manager_agent: Optional[Agent] = None
    voting_strategy: str = "majority"  # majority, unanimous, weighted
    timeout: int = 600
    max_rounds: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    agent_responses: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    rounds: int = 1
    selected_response: Optional[Dict[str, Any]] = None
    consensus_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize orchestrator.

        Args:
            config: Optional configuration
        """
        self.config = config or {}
        self.teams: Dict[str, TeamConfig] = {}
        self.execution_tracker = ExecutionTracker()
        self.agent_registry: Dict[str, Agent] = {}

    def register_agent(self, agent: Agent):
        """
        Register an agent.

        Args:
            agent: Agent to register
        """
        self.agent_registry[agent.name] = agent

    def create_team(self,
                   name: str,
                   agents: List[Agent],
                   pattern: OrchestrationPattern = OrchestrationPattern.SEQUENTIAL,
                   manager_agent: Optional[Agent] = None,
                   **kwargs) -> TeamConfig:
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
                   team: TeamConfig,
                   context: Optional[Dict[str, Any]] = None) -> TeamResponse:
        """
        Assign task to team and coordinate execution.

        Args:
            task: Task description
            team: Team configuration
            context: Optional context

        Returns:
            TeamResponse with results
        """
        start_time = time.time()
        context = context or {}

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
            # Track failure
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_FAILED,
                agent_id=f"team_{team.name}",
                message=f"Team failed: {str(e)}",
                data={"error": str(e)}
            ))

            return TeamResponse(
                output=f"Error: {str(e)}",
                success=False,
                pattern=team.pattern,
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def _execute_sequential(self,
                           task: str,
                           team: TeamConfig,
                           context: Dict[str, Any]) -> TeamResponse:
        """
        Execute agents one after another.

        Output of agent N becomes input of agent N+1.
        """
        current_task = task
        responses = []

        for i, agent in enumerate(team.agents):
            # Track agent start
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_START,
                agent_id=agent.name,
                message=f"Agent {i+1}/{len(team.agents)} starting",
                data={"task": current_task[:100]}
            ))

            # Execute agent
            response = agent.run(current_task, mode=AgentMode.AUTO, context=context)

            # Track completion
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_COMPLETE,
                agent_id=agent.name,
                message=f"Agent {i+1}/{len(team.agents)} completed"
            ))

            responses.append({
                "agent_name": agent.name,
                "output": response.output,
                "success": response.success,
                "tokens_used": response.tokens_used
            })

            if not response.success:
                # Stop on failure
                break

            # Use output as input for next agent
            current_task = response.output

        return TeamResponse(
            output=current_task,
            success=all(r["success"] for r in responses),
            pattern=OrchestrationPattern.SEQUENTIAL,
            agent_responses=responses
        )

    def _execute_parallel(self,
                         task: str,
                         team: TeamConfig,
                         context: Dict[str, Any]) -> TeamResponse:
        """
        Execute all agents in parallel on the same task.

        Synthesize results at the end.
        """
        # Run all agents in parallel
        responses = asyncio.run(self._parallel_execution(task, team.agents, context))

        # Synthesize results
        synthesis = self._synthesize_parallel_results(task, responses)

        return TeamResponse(
            output=synthesis,
            success=any(r["success"] for r in responses),
            pattern=OrchestrationPattern.PARALLEL,
            agent_responses=responses
        )

    async def _parallel_execution(self,
                                  task: str,
                                  agents: List[Agent],
                                  context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute agents in parallel."""
        async def run_agent(agent: Agent):
            # Track start
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_START,
                agent_id=agent.name,
                message=f"Agent starting (parallel)"
            ))

            # Run agent
            response = await agent.run_async(task, mode=AgentMode.AUTO, context=context)

            # Track completion
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_COMPLETE,
                agent_id=agent.name,
                message="Agent completed (parallel)"
            ))

            return {
                "agent_name": agent.name,
                "output": response.output,
                "success": response.success,
                "tokens_used": response.tokens_used
            }

        # Execute all in parallel
        results = await asyncio.gather(*[run_agent(agent) for agent in agents])
        return list(results)

    def _execute_hierarchical(self,
                             task: str,
                             team: TeamConfig,
                             context: Dict[str, Any]) -> TeamResponse:
        """
        Manager agent coordinates worker agents.

        Manager delegates subtasks and synthesizes results.
        """
        if not team.manager_agent:
            raise ValueError("Hierarchical pattern requires manager_agent")

        # Manager decomposes task
        decomposition_prompt = f"""You are a manager coordinating a team. Break down this task into subtasks for your team:

Task: {task}

Available workers: {', '.join(agent.name for agent in team.agents)}

Provide subtasks as a numbered list."""

        manager_response = team.manager_agent.run(
            decomposition_prompt,
            mode=AgentMode.SINGLE,
            context=context
        )

        # Parse subtasks (simple heuristic)
        subtasks = self._parse_subtasks(manager_response.output)

        # Assign to workers
        responses = []
        for i, (subtask, agent) in enumerate(zip(subtasks, team.agents)):
            response = agent.run(subtask, mode=AgentMode.AUTO, context=context)
            responses.append({
                "agent_name": agent.name,
                "subtask": subtask,
                "output": response.output,
                "success": response.success
            })

        # Manager synthesizes
        synthesis_prompt = f"""Synthesize the results from your team into a final answer for: {task}

Team results:
{self._format_team_results(responses)}

Provide a comprehensive final answer."""

        final_response = team.manager_agent.run(
            synthesis_prompt,
            mode=AgentMode.SINGLE,
            context=context
        )

        return TeamResponse(
            output=final_response.output,
            success=final_response.success,
            pattern=OrchestrationPattern.HIERARCHICAL,
            agent_responses=responses,
            metadata={"manager_decomposition": manager_response.output}
        )

    def _execute_collaborative(self,
                              task: str,
                              team: TeamConfig,
                              context: Dict[str, Any]) -> TeamResponse:
        """
        Agents discuss and reach consensus.

        Multiple rounds of discussion until consensus.
        """
        max_rounds = team.max_rounds
        current_responses = []
        consensus_reached = False

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

                response = agent.run(prompt, mode=AgentMode.AUTO, context=context)
                round_responses.append({
                    "agent_name": agent.name,
                    "output": response.output,
                    "round": round_num
                })

            current_responses = round_responses

            # Check for consensus
            consensus_score = self._calculate_consensus(round_responses)
            if consensus_score > 0.8:
                consensus_reached = True
                break

        # Synthesize final output
        final_output = self._synthesize_collaborative_results(task, current_responses)

        return TeamResponse(
            output=final_output,
            success=True,
            pattern=OrchestrationPattern.COLLABORATIVE,
            agent_responses=current_responses,
            rounds=round_num,
            consensus_score=consensus_score
        )

    def _execute_competitive(self,
                            task: str,
                            team: TeamConfig,
                            context: Dict[str, Any]) -> TeamResponse:
        """
        Multiple agents solve same task, select best solution.

        Can use voting or scoring.
        """
        # All agents work on same task
        responses = asyncio.run(self._parallel_execution(task, team.agents, context))

        # Select best response
        best_response = self._select_best_response(task, responses, team.voting_strategy)

        return TeamResponse(
            output=best_response["output"],
            success=best_response["success"],
            pattern=OrchestrationPattern.COMPETITIVE,
            agent_responses=responses,
            selected_response=best_response,
            metadata={"voting_strategy": team.voting_strategy}
        )

    def _execute_pipeline(self,
                         task: str,
                         team: TeamConfig,
                         context: Dict[str, Any]) -> TeamResponse:
        """
        Pipeline processing with specialized stages.

        Similar to sequential but each agent has specific role.
        """
        # Similar to sequential but with role awareness
        return self._execute_sequential(task, team, context)

    def _synthesize_parallel_results(self,
                                    task: str,
                                    responses: List[Dict[str, Any]]) -> str:
        """Synthesize results from parallel execution."""
        synthesis_parts = [f"Results from {len(responses)} agents:\n"]

        for i, response in enumerate(responses, 1):
            synthesis_parts.append(f"\n{i}. {response['agent_name']}:")
            synthesis_parts.append(f"   {response['output'][:200]}...")

        # Simple concatenation
        return "\n".join(synthesis_parts)

    def _synthesize_collaborative_results(self,
                                         task: str,
                                         responses: List[Dict[str, Any]]) -> str:
        """Synthesize results from collaborative discussion."""
        # Use last round responses
        latest_round = max(r["round"] for r in responses)
        latest_responses = [r for r in responses if r["round"] == latest_round]

        synthesis = f"Collaborative consensus after {latest_round} rounds:\n\n"

        for response in latest_responses:
            synthesis += f"{response['agent_name']}: {response['output']}\n\n"

        return synthesis

    def _calculate_consensus(self, responses: List[Dict[str, Any]]) -> float:
        """
        Calculate consensus score.

        Simple heuristic: similarity of responses.
        """
        # Placeholder - would use semantic similarity
        # For now, return fixed score
        return 0.7

    def _select_best_response(self,
                             task: str,
                             responses: List[Dict[str, Any]],
                             strategy: str) -> Dict[str, Any]:
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

    def _parse_subtasks(self, text: str) -> List[str]:
        """Parse subtasks from numbered list."""
        import re
        # Find numbered items
        pattern = r'\d+[\.)]\s+(.+?)(?=\n\d+[\.)]|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches] if matches else [text]

    def _format_team_results(self, responses: List[Dict[str, Any]]) -> str:
        """Format team results for display."""
        formatted = []
        for i, response in enumerate(responses, 1):
            formatted.append(f"{i}. {response['agent_name']}:")
            formatted.append(f"   Subtask: {response.get('subtask', 'N/A')}")
            formatted.append(f"   Result: {response['output']}")
        return "\n".join(formatted)

    def get_team(self, name: str) -> Optional[TeamConfig]:
        """Get team by name."""
        return self.teams.get(name)

    def list_teams(self) -> List[str]:
        """List all team names."""
        return list(self.teams.keys())

    def remove_team(self, name: str):
        """Remove a team."""
        if name in self.teams:
            del self.teams[name]

    def __repr__(self) -> str:
        """String representation."""
        return f"MultiAgentOrchestrator(teams={len(self.teams)}, agents={len(self.agent_registry)})"
