"""
Sub-agent management system for effGen.

Manages the lifecycle of specialized sub-agents including:
- Spawning specialized sub-agents
- Parallel and sequential execution
- Result synthesis
- Error handling and recovery
- Resource management
"""

import asyncio
import time
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback

from .task import SubTask, TaskStatus
from .execution_tracker import ExecutionTracker, ExecutionEvent, EventType
from .router import RoutingStrategy


class SubAgentSpecialization(Enum):
    """Available sub-agent specializations."""
    GENERAL = "general"
    RESEARCH = "research"
    CODING = "coding"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    DATA = "data"
    CREATIVE = "creative"


@dataclass
class SubAgentConfig:
    """
    Configuration for a specialized sub-agent.

    Attributes:
        specialization: Type of specialization
        tools: Available tools for this specialization
        system_prompt: Specialized system prompt
        model: Model to use (or "inherit" from parent)
        max_iterations: Maximum reasoning iterations
        temperature: Generation temperature
        timeout: Execution timeout in seconds
    """
    specialization: SubAgentSpecialization
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""
    model: str = "inherit"
    max_iterations: int = 5
    temperature: float = 0.7
    timeout: int = 300

    @classmethod
    def get_default_config(cls, specialization: str) -> "SubAgentConfig":
        """Get default configuration for a specialization."""
        specialization_enum = SubAgentSpecialization(specialization)

        configs = {
            SubAgentSpecialization.RESEARCH: SubAgentConfig(
                specialization=SubAgentSpecialization.RESEARCH,
                tools=["web_search", "web_fetch", "file_operations"],
                system_prompt=(
                    "You are a research specialist. Your role is to gather comprehensive "
                    "information from various sources. Be thorough, cite sources, and "
                    "extract key findings. Focus on accuracy and completeness."
                ),
                max_iterations=5,
                temperature=0.5
            ),
            SubAgentSpecialization.CODING: SubAgentConfig(
                specialization=SubAgentSpecialization.CODING,
                tools=["code_executor", "python_repl", "file_operations"],
                system_prompt=(
                    "You are a coding specialist. Write clean, tested, documented code. "
                    "Always test your code before returning results. Follow best practices "
                    "and handle edge cases appropriately."
                ),
                max_iterations=8,
                temperature=0.3
            ),
            SubAgentSpecialization.ANALYSIS: SubAgentConfig(
                specialization=SubAgentSpecialization.ANALYSIS,
                tools=["calculator", "python_repl", "data_tools"],
                system_prompt=(
                    "You are an analysis specialist. Perform thorough data analysis, "
                    "calculate metrics, and identify patterns and insights. Be precise "
                    "with numbers and provide clear interpretations."
                ),
                max_iterations=5,
                temperature=0.4
            ),
            SubAgentSpecialization.SYNTHESIS: SubAgentConfig(
                specialization=SubAgentSpecialization.SYNTHESIS,
                tools=[],
                system_prompt=(
                    "You are a synthesis specialist. Combine information from multiple "
                    "sources into coherent, well-structured outputs. Resolve conflicts, "
                    "highlight key insights, and provide comprehensive summaries."
                ),
                max_iterations=3,
                temperature=0.6
            ),
            SubAgentSpecialization.GENERAL: SubAgentConfig(
                specialization=SubAgentSpecialization.GENERAL,
                tools=["web_search", "calculator", "python_repl"],
                system_prompt=(
                    "You are a general-purpose assistant. Complete the assigned task "
                    "efficiently and accurately. Use available tools as needed."
                ),
                max_iterations=5,
                temperature=0.7
            )
        }

        return configs.get(specialization_enum, configs[SubAgentSpecialization.GENERAL])


@dataclass
class SubAgentResult:
    """
    Result from sub-agent execution.

    Attributes:
        subtask_id: ID of completed subtask
        agent_id: ID of sub-agent
        success: Whether execution succeeded
        result: Result data
        error: Error message if failed
        execution_time: Time taken in seconds
        tokens_used: Tokens consumed
        tool_calls: Number of tool calls made
        metadata: Additional metadata
    """
    subtask_id: str
    agent_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tokens_used: int = 0
    tool_calls: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subtask_id": self.subtask_id,
            "agent_id": self.agent_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": round(self.execution_time, 2),
            "tokens_used": self.tokens_used,
            "tool_calls": self.tool_calls,
            "metadata": self.metadata
        }


class SubAgentManager:
    """
    Manage sub-agent lifecycle and coordination.

    Responsibilities:
    - Spawn specialized sub-agents
    - Execute tasks in parallel or sequential
    - Track progress and handle failures
    - Synthesize results
    - Manage resources
    """

    def __init__(self,
                 parent_agent: Any = None,
                 config: Optional[Dict[str, Any]] = None,
                 execution_tracker: Optional[ExecutionTracker] = None):
        """
        Initialize sub-agent manager.

        Args:
            parent_agent: Parent agent instance
            config: Optional configuration
            execution_tracker: Optional execution tracker
        """
        self.parent_agent = parent_agent
        self.config = config or {}
        self.execution_tracker = execution_tracker or ExecutionTracker()
        self.active_sub_agents: Dict[str, Any] = {}
        self.sub_agent_results: Dict[str, SubAgentResult] = {}
        self.max_parallel = self.config.get("max_parallel_agents", 5)

    def spawn_sub_agent(self,
                       subtask: SubTask,
                       specialization: Optional[str] = None) -> Any:
        """
        Create a specialized sub-agent.

        Args:
            subtask: Subtask to execute
            specialization: Required specialization (or infer from subtask)

        Returns:
            Sub-agent instance
        """
        # Determine specialization
        if specialization is None:
            specialization = subtask.required_specialization or "general"

        # Get configuration for specialization
        sub_agent_config = SubAgentConfig.get_default_config(specialization)

        # Create sub-agent (will be implemented by importing Agent class)
        # For now, we create a placeholder that will be replaced
        agent_id = f"sub_agent_{subtask.id}"

        # Track spawning event
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.SUB_AGENT_SPAWN,
            agent_id=agent_id,
            message=f"Spawning {specialization} sub-agent for: {subtask.description[:50]}...",
            data={
                "subtask_id": subtask.id,
                "specialization": specialization,
                "tools": sub_agent_config.tools,
                "agent_name": f"{specialization.capitalize()} Specialist"
            }
        ))

        # Store reference
        sub_agent_info = {
            "id": agent_id,
            "subtask": subtask,
            "config": sub_agent_config,
            "status": "spawned"
        }
        self.active_sub_agents[agent_id] = sub_agent_info

        return sub_agent_info

    async def execute_parallel(self,
                               subtasks: List[SubTask],
                               progress_callback: Optional[Callable] = None) -> List[SubAgentResult]:
        """
        Execute subtasks in parallel using sub-agents.

        Args:
            subtasks: List of subtasks to execute
            progress_callback: Optional callback for progress updates

        Returns:
            List of SubAgentResult
        """
        # Spawn sub-agents for each subtask
        sub_agents = []
        for subtask in subtasks:
            agent = self.spawn_sub_agent(subtask)
            sub_agents.append(agent)

        # Execute in parallel with concurrency limit
        results = []
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_with_semaphore(agent_info, subtask):
            async with semaphore:
                return await self._execute_sub_agent_async(agent_info, subtask, progress_callback)

        # Create tasks
        tasks = [
            execute_with_semaphore(agent, subtask)
            for agent, subtask in zip(sub_agents, subtasks)
        ]

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed result
                final_results.append(SubAgentResult(
                    subtask_id=subtasks[i].id,
                    agent_id=sub_agents[i]["id"],
                    success=False,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        return final_results

    def execute_sequential(self,
                          subtasks: List[SubTask],
                          progress_callback: Optional[Callable] = None) -> List[SubAgentResult]:
        """
        Execute subtasks sequentially.

        Args:
            subtasks: List of subtasks to execute in order
            progress_callback: Optional callback for progress updates

        Returns:
            List of SubAgentResult
        """
        results = []

        for subtask in subtasks:
            # Spawn sub-agent
            agent_info = self.spawn_sub_agent(subtask)

            # Execute synchronously
            result = self._execute_sub_agent(agent_info, subtask, progress_callback)
            results.append(result)

            # Check if failed and should stop
            if not result.success and self.config.get("stop_on_failure", False):
                # Mark remaining as cancelled
                for remaining in subtasks[len(results):]:
                    results.append(SubAgentResult(
                        subtask_id=remaining.id,
                        agent_id="cancelled",
                        success=False,
                        error="Cancelled due to previous failure"
                    ))
                break

        return results

    def execute_hybrid(self,
                      subtasks: List[SubTask],
                      progress_callback: Optional[Callable] = None) -> List[SubAgentResult]:
        """
        Execute with hybrid strategy (parallel groups + sequential stages).

        Args:
            subtasks: List of subtasks
            progress_callback: Optional callback

        Returns:
            List of SubAgentResult
        """
        # Group by dependencies
        stages = self._group_by_dependencies(subtasks)

        all_results = []

        # Execute each stage
        for stage_subtasks in stages:
            if len(stage_subtasks) == 1:
                # Single task - execute sequentially
                result = self.execute_sequential(stage_subtasks, progress_callback)
                all_results.extend(result)
            else:
                # Multiple tasks - execute in parallel
                result = asyncio.run(self.execute_parallel(stage_subtasks, progress_callback))
                all_results.extend(result)

        return all_results

    def _group_by_dependencies(self, subtasks: List[SubTask]) -> List[List[SubTask]]:
        """
        Group subtasks into stages based on dependencies.

        Returns list of stages where each stage can execute in parallel.
        """
        # Simple implementation: group by dependency depth
        stages = []
        remaining = subtasks.copy()
        completed_ids = set()

        while remaining:
            # Find tasks with all dependencies met
            ready = [
                st for st in remaining
                if all(dep in completed_ids for dep in st.depends_on)
            ]

            if not ready:
                # Circular dependency or error - put all remaining in one stage
                stages.append(remaining)
                break

            stages.append(ready)
            completed_ids.update(st.id for st in ready)
            remaining = [st for st in remaining if st not in ready]

        return stages

    async def _execute_sub_agent_async(self,
                                      agent_info: Dict,
                                      subtask: SubTask,
                                      progress_callback: Optional[Callable] = None) -> SubAgentResult:
        """Execute sub-agent asynchronously."""
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._execute_sub_agent,
            agent_info,
            subtask,
            progress_callback
        )

    def _execute_sub_agent(self,
                          agent_info: Dict,
                          subtask: SubTask,
                          progress_callback: Optional[Callable] = None) -> SubAgentResult:
        """
        Execute a sub-agent on a subtask.

        Args:
            agent_info: Sub-agent information
            subtask: Subtask to execute
            progress_callback: Optional progress callback

        Returns:
            SubAgentResult
        """
        agent_id = agent_info["id"]
        config = agent_info["config"]
        start_time = time.time()

        # Track start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.SUB_AGENT_START,
            agent_id=agent_id,
            message=f"Starting execution: {subtask.description[:50]}...",
            data={"subtask_id": subtask.id}
        ))

        try:
            # Update status
            subtask.status = TaskStatus.RUNNING

            # Simulate execution (in real implementation, would call agent.run())
            # This is a placeholder that will be replaced when Agent class is available
            result_data = self._simulate_execution(subtask, config)

            execution_time = time.time() - start_time

            # Track completion
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_COMPLETE,
                agent_id=agent_id,
                message=f"Completed in {execution_time:.2f}s",
                data={
                    "subtask_id": subtask.id,
                    "execution_time": execution_time
                }
            ))

            # Update subtask
            subtask.status = TaskStatus.COMPLETED
            subtask.result = result_data

            # Create result
            result = SubAgentResult(
                subtask_id=subtask.id,
                agent_id=agent_id,
                success=True,
                result=result_data,
                execution_time=execution_time,
                tokens_used=result_data.get("tokens_used", 0) if isinstance(result_data, dict) else 0,
                tool_calls=result_data.get("tool_calls", 0) if isinstance(result_data, dict) else 0
            )

            # Store result
            self.sub_agent_results[subtask.id] = result

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"

            # Track failure
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.SUB_AGENT_FAILED,
                agent_id=agent_id,
                message=f"Failed: {error_msg}",
                data={
                    "subtask_id": subtask.id,
                    "error": error_msg,
                    "traceback": traceback.format_exc()
                }
            ))

            # Update subtask
            subtask.status = TaskStatus.FAILED
            subtask.error = error_msg

            # Create failed result
            result = SubAgentResult(
                subtask_id=subtask.id,
                agent_id=agent_id,
                success=False,
                error=error_msg,
                execution_time=execution_time
            )

            self.sub_agent_results[subtask.id] = result

            return result

    def _simulate_execution(self, subtask: SubTask, config: SubAgentConfig) -> Dict[str, Any]:
        """
        Simulate sub-agent execution (placeholder).

        This will be replaced with actual agent.run() call when Agent class is available.
        """
        # Placeholder simulation
        time.sleep(0.1)  # Simulate some work

        return {
            "output": f"Completed: {subtask.description}",
            "summary": f"Successfully executed {config.specialization.value} task",
            "tokens_used": 500,
            "tool_calls": 2
        }

    def synthesize_results(self,
                          results: List[SubAgentResult],
                          original_task: str,
                          strategy: RoutingStrategy) -> Dict[str, Any]:
        """
        Combine sub-agent results into final answer.

        Args:
            results: List of sub-agent results
            original_task: Original task description
            strategy: Routing strategy used

        Returns:
            Synthesized final result
        """
        # Track synthesis start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.RESULT_SYNTHESIS,
            message="Synthesizing results from sub-agents",
            data={
                "num_results": len(results),
                "strategy": strategy.value
            }
        ))

        # Separate successful and failed results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Build synthesis
        synthesis = {
            "original_task": original_task,
            "strategy": strategy.value,
            "total_subtasks": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "results": []
        }

        # Add successful results
        for result in successful:
            synthesis["results"].append({
                "subtask_id": result.subtask_id,
                "output": result.result
            })

        # Add failure information
        if failed:
            synthesis["failures"] = [
                {
                    "subtask_id": r.subtask_id,
                    "error": r.error
                } for r in failed
            ]

        # Aggregate metrics
        synthesis["metrics"] = {
            "total_execution_time": sum(r.execution_time for r in results),
            "total_tokens_used": sum(r.tokens_used for r in results),
            "total_tool_calls": sum(r.tool_calls for r in results),
            "avg_execution_time": sum(r.execution_time for r in results) / len(results) if results else 0
        }

        # Use parent agent to synthesize if available
        if self.parent_agent and hasattr(self.parent_agent, "synthesize"):
            synthesis["final_output"] = self.parent_agent.synthesize(synthesis)
        else:
            # Simple concatenation fallback
            synthesis["final_output"] = self._simple_synthesis(synthesis)

        return synthesis

    def _simple_synthesis(self, synthesis_data: Dict[str, Any]) -> str:
        """Simple synthesis by concatenating results."""
        parts = []
        parts.append(f"Task: {synthesis_data['original_task']}\n")
        parts.append(f"Strategy: {synthesis_data['strategy']}")
        parts.append(f"Completed {synthesis_data['successful']}/{synthesis_data['total_subtasks']} subtasks\n")

        for result in synthesis_data["results"]:
            parts.append(f"\nSubtask {result['subtask_id']}:")
            if isinstance(result["output"], dict):
                parts.append(result["output"].get("output", str(result["output"])))
            else:
                parts.append(str(result["output"]))

        if synthesis_data.get("failures"):
            parts.append("\n\nFailures:")
            for failure in synthesis_data["failures"]:
                parts.append(f"- {failure['subtask_id']}: {failure['error']}")

        return "\n".join(parts)

    def get_active_count(self) -> int:
        """Get number of active sub-agents."""
        return len([a for a in self.active_sub_agents.values() if a["status"] == "running"])

    def cleanup(self):
        """Cleanup resources and terminate sub-agents."""
        # In real implementation, would terminate running agents
        self.active_sub_agents.clear()
        self.sub_agent_results.clear()

    def __repr__(self) -> str:
        """String representation."""
        return (f"SubAgentManager(active={self.get_active_count()}, "
                f"max_parallel={self.max_parallel})")
