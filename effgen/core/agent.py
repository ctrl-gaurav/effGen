"""
Core Agent implementation for effGen.

The main Agent class with:
- ReAct loop (Reason + Act)
- Tool selection and execution
- Sub-agent integration via router
- Memory management
- Streaming support
- State persistence
"""

import time
import json
import re
import asyncio
import logging
from typing import List, Dict, Optional, Any, Iterator, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from ..models.base import BaseModel, GenerationConfig, GenerationResult
from ..models.model_loader import ModelLoader
from ..tools.base_tool import BaseTool
from .state import AgentState
from .task import Task, TaskStatus, SubTask
from .router import SubAgentRouter, RoutingDecision, RoutingStrategy
from .sub_agent_manager import SubAgentManager
from .execution_tracker import ExecutionTracker, ExecutionEvent, EventType

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Agent execution modes."""
    SINGLE = "single"  # Single agent execution
    SUB_AGENTS = "sub_agents"  # Use sub-agents for complex tasks
    AUTO = "auto"  # Automatically decide based on router


@dataclass
class AgentConfig:
    """
    Agent configuration.

    Attributes:
        name: Agent name/identifier
        model: Model instance or name
        tools: List of available tools
        system_prompt: System-level instructions
        max_iterations: Maximum tool-use loop iterations
        temperature: Generation temperature
        enable_sub_agents: Enable sub-agent spawning
        enable_memory: Enable memory systems
        enable_streaming: Enable response streaming
        max_context_length: Maximum context window
        router_config: Configuration for sub-agent router
        sub_agent_config: Configuration for sub-agent manager
        model_config: Optional model engine configuration
        require_model: Whether model loading is required (raise error on failure)
    """
    name: str
    model: Union[BaseModel, str]
    tools: List[BaseTool] = field(default_factory=list)
    system_prompt: str = "You are a helpful AI assistant."
    max_iterations: int = 10
    temperature: float = 0.7
    enable_sub_agents: bool = True
    enable_memory: bool = True
    enable_streaming: bool = False
    max_context_length: Optional[int] = None
    router_config: Dict[str, Any] = field(default_factory=dict)
    sub_agent_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Optional[Dict[str, Any]] = None
    require_model: bool = False


@dataclass
class AgentResponse:
    """
    Response from agent execution.

    Attributes:
        output: Final output text
        success: Whether execution succeeded
        mode: Execution mode used
        iterations: Number of iterations performed
        tool_calls: Number of tool calls made
        tokens_used: Total tokens consumed
        execution_time: Time taken in seconds
        execution_trace: Full execution trace
        execution_tree: Hierarchical execution tree
        routing_decision: Routing decision (if sub-agents used)
        metadata: Additional metadata
    """
    output: str
    success: bool = True
    mode: AgentMode = AgentMode.SINGLE
    iterations: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    execution_time: float = 0.0
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    execution_tree: Dict[str, Any] = field(default_factory=dict)
    routing_decision: Optional[RoutingDecision] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output": self.output,
            "success": self.success,
            "mode": self.mode.value,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "execution_time": round(self.execution_time, 2),
            "execution_trace": self.execution_trace,
            "execution_tree": self.execution_tree,
            "routing_decision": self.routing_decision.to_dict() if self.routing_decision else None,
            "metadata": self.metadata
        }


class Agent:
    """
    Main Agent implementation with ReAct loop and sub-agent support.

    The agent can:
    - Execute tasks using ReAct (Reason + Act) pattern
    - Intelligently spawn sub-agents for complex tasks
    - Use tools to interact with external systems
    - Manage conversation memory
    - Stream responses
    - Save/load state
    """

    # Default ReAct prompt template
    REACT_PROMPT_TEMPLATE = """You are a helpful AI assistant that can reason step-by-step and use tools.
{conversation_history}
Available tools:
{tools_description}

IMPORTANT: If there is previous conversation context above, use that information to answer questions about past interactions.

Use the following format:

Question: the input question or task
Thought: think step-by-step about what to do next
Action: the tool to use (or "Final Answer" when ready to respond)
Action Input: the input for the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the complete response to the original question

Begin!

Question: {task}
{scratchpad}"""

    def __init__(self, config: AgentConfig):
        """
        Initialize agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.name = config.name

        # Model initialization
        self.model_loader = ModelLoader()
        if isinstance(config.model, BaseModel):
            # Model instance provided directly
            self.model = config.model
            self.model_name = getattr(config.model, 'model_name', 'custom')
        elif isinstance(config.model, str):
            # Model name provided - load it
            self.model_name = config.model
            try:
                logger.info(f"Loading model: {self.model_name}")
                self.model = self.model_loader.load_model(
                    self.model_name,
                    engine_config=config.model_config
                )
                logger.info(f"Model loaded successfully: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model '{self.model_name}': {e}")
                self.model = None
                if config.require_model:
                    raise RuntimeError(f"Failed to load required model: {e}")
        else:
            # No model provided
            self.model_name = None
            self.model = None

        # Tools
        self.tools = {tool.name: tool for tool in config.tools}

        # State management
        self.state = AgentState(agent_id=self.name)

        # Sub-agent components
        self.router = None
        self.sub_agent_manager = None
        if config.enable_sub_agents:
            self.router = SubAgentRouter(
                config=config.router_config,
                llm_client=self  # Pass self as LLM client
            )
            self.sub_agent_manager = SubAgentManager(
                parent_agent=self,
                config=config.sub_agent_config
            )

        # Execution tracker
        self.execution_tracker = ExecutionTracker()

        # Memory (placeholder for now)
        self.short_term_memory = []
        self.long_term_memory = None  # Would integrate vector store

    def run(self,
            task: str,
            mode: AgentMode = AgentMode.AUTO,
            context: Optional[Dict[str, Any]] = None,
            **kwargs) -> AgentResponse:
        """
        Execute a task.

        Args:
            task: Task description
            mode: Execution mode (single, sub_agents, auto)
            context: Optional context
            **kwargs: Additional arguments

        Returns:
            AgentResponse with results
        """
        start_time = time.time()
        context = context or {}

        # Track task start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TASK_START,
            agent_id=self.name,
            message=f"Starting task: {task[:100]}...",
            data={"task": task, "mode": mode.value}
        ))

        try:
            # Determine execution mode
            if mode == AgentMode.AUTO and self.config.enable_sub_agents:
                # Use router to decide
                routing_decision = self.router.route(task, context)

                if routing_decision.use_sub_agents:
                    response = self._run_with_sub_agents(task, routing_decision, context, **kwargs)
                else:
                    response = self._run_single_agent(task, context, **kwargs)
            elif mode == AgentMode.SUB_AGENTS and self.config.enable_sub_agents:
                # Force sub-agent mode
                routing_decision = self.router.route(task, context)
                response = self._run_with_sub_agents(task, routing_decision, context, **kwargs)
            else:
                # Single agent mode
                response = self._run_single_agent(task, context, **kwargs)

            # Add execution metadata
            response.execution_time = time.time() - start_time
            response.execution_trace = self.execution_tracker.get_trace()
            response.execution_tree = self.execution_tracker.generate_execution_tree()

            # Track completion
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_COMPLETE,
                agent_id=self.name,
                message=f"Task completed in {response.execution_time:.2f}s",
                data={
                    "execution_time": response.execution_time,
                    "tokens_used": response.tokens_used,
                    "tool_calls": response.tool_calls
                }
            ))

            # Store conversation in short-term memory for context retention
            if response.success and response.output:
                self.short_term_memory.append({
                    'user': task,
                    'assistant': response.output,
                    'timestamp': time.time()
                })
                logger.debug(f"Stored conversation turn in memory (total turns: {len(self.short_term_memory)})")

            return response

        except Exception as e:
            # Track failure
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TASK_FAILED,
                agent_id=self.name,
                message=f"Task failed: {str(e)}",
                data={"error": str(e)}
            ))

            return AgentResponse(
                output=f"Error: {str(e)}",
                success=False,
                execution_time=time.time() - start_time,
                execution_trace=self.execution_tracker.get_trace(),
                metadata={"error": str(e)}
            )

    def _run_single_agent(self,
                         task: str,
                         context: Dict[str, Any],
                         **kwargs) -> AgentResponse:
        """
        Execute task using single agent with ReAct loop or direct inference.

        Args:
            task: Task description
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # If no tools available, use direct inference instead of ReAct
        if not self.tools:
            return self._run_direct_inference(task, context, **kwargs)

        iterations = 0
        tool_calls = 0
        tokens_used = 0
        scratchpad = ""
        max_iterations = kwargs.get("max_iterations", self.config.max_iterations)

        # Build tools description
        tools_description = self._get_tools_description()

        # Format conversation history
        conversation_history = self._format_conversation_history()

        # ReAct loop
        while iterations < max_iterations:
            iterations += 1

            # Build prompt
            prompt = self.REACT_PROMPT_TEMPLATE.format(
                tools_description=tools_description,
                conversation_history=conversation_history,
                task=task,
                scratchpad=scratchpad
            )

            # Debug: log first iteration prompt to see if history is included
            if iterations == 1 and conversation_history:
                logger.info(f"[Memory] Including conversation history ({len(self.short_term_memory)} turns)")

            # Track reasoning step
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.REASONING_STEP,
                agent_id=self.name,
                message=f"Iteration {iterations}: Reasoning...",
                data={"iteration": iterations}
            ))

            # Generate response
            response = self._generate(prompt, **kwargs)
            tokens_used += response.get("tokens_used", 0)

            # Debug: Log the raw response
            logger.info(f"[Iteration {iterations}] Raw model output: {response['text'][:300]}...")
            logger.debug(f"[Iteration {iterations}] Full model output: {response['text']}")

            # Parse response
            parsed = self._parse_react_response(response["text"])

            # Debug: Log what was parsed
            logger.info(f"[Iteration {iterations}] Parsed - Action: {parsed.get('action')}, Input: {parsed.get('action_input')}, Final: {parsed.get('final_answer')}")

            # Add to scratchpad
            scratchpad += f"\nThought: {parsed.get('thought', '')}"

            # Check for final answer
            if parsed.get("final_answer"):
                return AgentResponse(
                    output=parsed["final_answer"],
                    success=True,
                    mode=AgentMode.SINGLE,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    tokens_used=tokens_used
                )

            # Check if model is stating an answer without "Final Answer:" keyword
            # This happens when model provides result after tool execution
            if tool_calls > 0 and not parsed.get("action"):
                # No action and we've used tools - model might be stating the answer
                response_text = response["text"].strip()
                # Check for answer-like patterns
                if any(phrase in response_text.lower() for phrase in ["the answer is", "the result is", "the sum is", "equals", "="]):
                    logger.info("Detected answer statement without 'Final Answer:' keyword")
                    return AgentResponse(
                        output=response_text,
                        success=True,
                        mode=AgentMode.SINGLE,
                        iterations=iterations,
                        tool_calls=tool_calls,
                        tokens_used=tokens_used
                    )

            # Execute action if present
            if parsed.get("action") and parsed.get("action_input"):
                action = parsed["action"]
                action_input = parsed["action_input"]

                # Check if tool is available (handle no-tool mode gracefully)
                if not self.tools or action not in self.tools:
                    # No tools available - model is hallucinating tools
                    # Guide it to provide direct answer
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: No tools available. Please provide your answer directly using 'Final Answer:'."
                else:
                    # Execute tool
                    tool_result = self._execute_tool(action, action_input)
                    tool_calls += 1

                    # Add observation to scratchpad
                    scratchpad += f"\nAction: {action}"
                    scratchpad += f"\nAction Input: {action_input}"
                    scratchpad += f"\nObservation: {tool_result}"

                    # Log the observation for debugging
                    logger.info(f"Tool result added to scratchpad: {tool_result[:100]}...")

            else:
                # No action specified, prompt to continue
                scratchpad += "\nAction: (continue reasoning)"

        # Max iterations reached
        return AgentResponse(
            output="Maximum iterations reached without final answer.",
            success=False,
            mode=AgentMode.SINGLE,
            iterations=iterations,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            metadata={"reason": "max_iterations_reached"}
        )

    def _run_with_sub_agents(self,
                            task: str,
                            routing_decision: RoutingDecision,
                            context: Dict[str, Any],
                            **kwargs) -> AgentResponse:
        """
        Execute task using sub-agents based on routing decision.

        Args:
            task: Task description
            routing_decision: Router's decision
            context: Context dictionary
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # Track decomposition
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TASK_DECOMPOSITION,
            agent_id=self.name,
            message=f"Decomposed into {routing_decision.num_sub_agents} subtasks using {routing_decision.strategy.value}",
            data={
                "strategy": routing_decision.strategy.value,
                "num_subtasks": routing_decision.num_sub_agents,
                "specializations": routing_decision.specializations
            }
        ))

        # Execute based on strategy
        strategy = routing_decision.strategy
        subtasks = routing_decision.decomposition

        if strategy == RoutingStrategy.PARALLEL_SUB_AGENTS:
            # Execute in parallel
            results = asyncio.run(
                self.sub_agent_manager.execute_parallel(subtasks)
            )
        elif strategy == RoutingStrategy.SEQUENTIAL_SUB_AGENTS:
            # Execute sequentially
            results = self.sub_agent_manager.execute_sequential(subtasks)
        elif strategy == RoutingStrategy.HYBRID:
            # Execute with hybrid approach
            results = self.sub_agent_manager.execute_hybrid(subtasks)
        else:
            # Default to sequential
            results = self.sub_agent_manager.execute_sequential(subtasks)

        # Synthesize results
        synthesis = self.sub_agent_manager.synthesize_results(
            results,
            task,
            strategy
        )

        # Calculate totals
        total_tokens = synthesis["metrics"]["total_tokens_used"]
        total_tool_calls = synthesis["metrics"]["total_tool_calls"]

        return AgentResponse(
            output=synthesis["final_output"],
            success=synthesis["successful"] > 0,
            mode=AgentMode.SUB_AGENTS,
            iterations=len(subtasks),
            tool_calls=total_tool_calls,
            tokens_used=total_tokens,
            routing_decision=routing_decision,
            metadata={
                "synthesis": synthesis,
                "failed_subtasks": synthesis["failed"]
            }
        )

    def _generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate response from model.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Dictionary with 'text', 'tokens_used', and other metadata

        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError(
                f"Agent '{self.name}' has no model loaded. "
                "Provide a model in AgentConfig or use a mock for testing."
            )

        try:
            # Create generation config from kwargs
            # Use smart stop sequences to prevent the model from hallucinating observations
            # IMPORTANT: Stop AFTER the model generates Action Input, so we can execute the tool
            # and provide the real observation
            default_stop_sequences = [
                "\nObservation:",  # Stop before model hallucinates observation
                "\nQuestion:",     # Model starting a new question (runaway)
                "\nHuman:",        # Model simulating conversation
                "\nUser:",         # Model simulating conversation
                "\n\n\n"           # Multiple blank lines (hallucination signal)
            ]

            gen_config = GenerationConfig(
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', 512),  # Enough for thought + action + input + some buffer
                top_p=kwargs.get('top_p', 0.9),
                stop_sequences=kwargs.get('stop_sequences', default_stop_sequences)
            )

            # Generate response
            result = self.model.generate(prompt, config=gen_config)

            # Robust response validation
            response_text = result.text if result and result.text else ""
            tokens_used = result.tokens_used if result and hasattr(result, 'tokens_used') else 0
            finish_reason = result.finish_reason if result and hasattr(result, 'finish_reason') else "unknown"

            return {
                "text": response_text,
                "tokens_used": tokens_used,
                "finish_reason": finish_reason,
                "metadata": result.metadata if result and hasattr(result, 'metadata') else {}
            }

        except Exception as e:
            logger.error(f"Generation failed in agent '{self.name}': {e}")
            # Return empty response instead of crashing - more resilient
            logger.warning(f"Returning empty response due to generation failure")
            return {
                "text": "",
                "tokens_used": 0,
                "finish_reason": "error",
                "metadata": {"error": str(e)}
            }

    def _parse_react_response(self, text: str) -> Dict[str, Any]:
        """
        Parse ReAct formatted response with robust error handling.

        Args:
            text: Response text

        Returns:
            Dictionary with parsed components

        Notes:
            This parser handles various formats and edge cases:
            - Case-insensitive matching
            - Multiple thought/action patterns
            - Malformed responses
            - Missing fields
        """
        parsed = {
            "thought": None,
            "action": None,
            "action_input": None,
            "final_answer": None
        }

        if not text or not isinstance(text, str):
            logger.warning(f"Invalid response text for parsing: {type(text)}")
            return parsed

        try:
            # Check for final answer first (highest priority)
            final_patterns = [
                r"Final Answer:\s*(.+)",
                r"Answer:\s*(.+)",
                r"The answer is:\s*(.+)"
            ]

            for pattern in final_patterns:
                try:
                    final_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if final_match:
                        answer = final_match.group(1).strip()
                        # Stop at next section marker or observation
                        answer = re.split(r'\n(?:Question|Thought|Action):', answer, maxsplit=1)[0].strip()
                        parsed["final_answer"] = answer
                        logger.debug(f"Extracted final answer: {answer[:100]}...")
                        return parsed
                except Exception as e:
                    logger.warning(f"Error matching final answer pattern '{pattern}': {e}")
                    continue

            # Extract thought
            thought_patterns = [
                r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer|Question):|$)",
                r"Thought:\s*(.+?)(?:\n\n|\n[A-Z]|$)"
            ]

            for pattern in thought_patterns:
                try:
                    thought_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                    if thought_match:
                        thought = thought_match.group(1).strip()
                        parsed["thought"] = thought
                        logger.debug(f"Extracted thought: {thought[:100]}...")
                        break
                except Exception as e:
                    logger.warning(f"Error matching thought pattern '{pattern}': {e}")
                    continue

            # Extract action
            action_patterns = [
                r"Action:\s*([^\n]+)",
                r"Tool:\s*([^\n]+)",
                r"Use tool:\s*([^\n]+)"
            ]

            for pattern in action_patterns:
                try:
                    action_match = re.search(pattern, text, re.IGNORECASE)
                    if action_match:
                        action = action_match.group(1).strip()
                        # Clean up common artifacts
                        action = action.replace('"', '').replace("'", "")

                        # Check if action is actually "Final Answer" - treat it as final answer, not tool
                        if action.lower() in ["final answer", "finalanswer", "answer"]:
                            logger.debug(f"Action '{action}' detected as Final Answer indicator")
                            # Extract what comes after "Action: Final Answer"
                            final_match = re.search(r"Action:\s*Final\s*Answer[:\s]+(.+)", text, re.IGNORECASE | re.DOTALL)
                            if final_match:
                                parsed["final_answer"] = final_match.group(1).strip()
                                logger.debug(f"Extracted final answer from Action line")
                                return parsed  # Return early since we have final answer

                        # Handle function-call format: tool_name(args) or tool_name("args")
                        # Extract just the tool name and put args into action_input
                        func_call_match = re.match(r'^(\w+)\s*\((.+)\)$', action, re.DOTALL)
                        if func_call_match:
                            tool_name = func_call_match.group(1).strip()
                            embedded_args = func_call_match.group(2).strip()
                            # Remove surrounding quotes if present
                            embedded_args = embedded_args.strip('"\'')
                            parsed["action"] = tool_name
                            # Only set action_input if not already set
                            if "action_input" not in parsed or not parsed["action_input"]:
                                parsed["action_input"] = embedded_args
                            logger.debug(f"Extracted function-call style: action={tool_name}, input={embedded_args[:100]}...")
                        else:
                            parsed["action"] = action
                            logger.debug(f"Extracted action: {action}")
                        break
                except Exception as e:
                    logger.warning(f"Error matching action pattern '{pattern}': {e}")
                    continue

            # Extract action input (only if not already set from function-call style)
            # Skip if we already have embedded args from tool_name(args) format
            if "action_input" not in parsed or not parsed.get("action_input"):
                input_patterns = [
                    r"Action Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question|Final Answer):|$)",
                    r"Input:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                    r"Parameters?:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)"
                ]

                for pattern in input_patterns:
                    try:
                        input_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                        if input_match:
                            action_input = input_match.group(1).strip()
                            # Remove trailing observation text if present
                            action_input = re.split(r'\nObservation:', action_input, maxsplit=1)[0].strip()
                            parsed["action_input"] = action_input
                            logger.debug(f"Extracted action input: {action_input[:100]}...")
                            break
                    except Exception as e:
                        logger.warning(f"Error matching action input pattern '{pattern}': {e}")
                        continue

        except Exception as e:
            logger.error(f"Critical error in parse_react_response: {e}", exc_info=True)
            # Return partial parse results even if there was an error

        return parsed

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool with robust error handling.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool (JSON string or plain text)

        Returns:
            Tool output as string

        Notes:
            - Handles both sync and async tool execution
            - Parses JSON or plain text input
            - Provides detailed error messages
            - Tracks execution for debugging
        """
        # Track tool call start
        self.execution_tracker.track_event(ExecutionEvent(
            type=EventType.TOOL_CALL_START,
            agent_id=self.name,
            message=f"Calling tool: {tool_name}",
            data={"tool_name": tool_name, "tool_input": tool_input}
        ))

        try:
            # Validate tool exists
            if not tool_name:
                raise ValueError("Tool name cannot be empty")

            if tool_name not in self.tools:
                available_tools = ", ".join(self.tools.keys())
                raise ValueError(
                    f"Tool '{tool_name}' not available. "
                    f"Available tools: {available_tools}"
                )

            tool = self.tools[tool_name]

            # Parse input intelligently
            input_dict = {}
            if tool_input:
                try:
                    # Try parsing as JSON first
                    input_dict = json.loads(tool_input)
                    if not isinstance(input_dict, dict):
                        # JSON parsed but not a dict - need to intelligently map to tool parameters
                        input_dict = self._map_input_to_parameters(tool, input_dict)
                except json.JSONDecodeError:
                    # Not valid JSON - use as plain text and intelligently map to parameters
                    input_dict = self._map_input_to_parameters(tool, tool_input)
                except Exception as e:
                    logger.warning(f"Error parsing tool input, using as plain text: {e}")
                    input_dict = self._map_input_to_parameters(tool, tool_input)

            logger.debug(f"Executing tool '{tool_name}' with input: {input_dict}")

            # Execute tool (handle both sync and async)
            try:
                result = tool.execute(**input_dict)

                # Handle async results if necessary
                if asyncio.iscoroutine(result):
                    logger.debug(f"Tool '{tool_name}' returned coroutine, handling async execution")
                    try:
                        # Try to get the current event loop
                        try:
                            loop = asyncio.get_running_loop()
                            # Loop is running - we're in async context
                            # We need to run the coroutine synchronously
                            # Use asyncio.run_coroutine_threadsafe or nest_asyncio
                            logger.debug("Running loop detected, using synchronous execution pattern")

                            # Import nest_asyncio for nested async support
                            try:
                                import nest_asyncio
                                nest_asyncio.apply()
                                result = asyncio.run(result)
                            except ImportError:
                                # Fallback: run in new thread
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(asyncio.run, result)
                                    result = future.result(timeout=30)
                        except RuntimeError:
                            # No running loop - we can use run_until_complete
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                result = loop.run_until_complete(result)
                            finally:
                                loop.close()
                    except Exception as e:
                        logger.error(f"Error executing async tool: {e}", exc_info=True)
                        result = f"Error executing async tool: {str(e)}"

            except TypeError as e:
                # Handle parameter mismatch
                logger.error(f"Tool parameter error: {e}")
                raise ValueError(
                    f"Tool '{tool_name}' parameter error: {str(e)}. "
                    f"Input provided: {input_dict}"
                )

            # Convert result to string safely
            if result is None:
                result_str = "No result returned"
            elif hasattr(result, 'output'):
                # ToolResult object - extract output
                if hasattr(result, 'success') and not result.success:
                    # Tool failed - include error message
                    error_msg = getattr(result, 'error', 'Unknown error')
                    result_str = f"Tool execution failed: {error_msg}"
                else:
                    # Tool succeeded - extract output
                    output = result.output
                    if isinstance(output, dict):
                        # For dict outputs, try to get the main result
                        result_str = str(output.get('result', output.get('output', output)))
                    else:
                        result_str = str(output)
            elif hasattr(result, 'result'):
                # Legacy format
                result_str = str(result.result)
            else:
                result_str = str(result)

            # Track success
            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_COMPLETE,
                agent_id=self.name,
                message=f"Tool {tool_name} completed",
                data={"tool_name": tool_name, "result": result_str[:200]}
            ))

            logger.info(f"Tool '{tool_name}' executed successfully")
            return result_str

        except ValueError as e:
            # ValueError is expected for hallucinated/invalid tool names - log cleanly
            error_msg = str(e)
            logger.debug(f"Tool '{tool_name}' execution failed: {error_msg}")

            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_FAILED,
                agent_id=self.name,
                message=f"Tool {tool_name} failed: {error_msg}",
                data={"tool_name": tool_name, "error": error_msg, "input": tool_input}
            ))

            # Return informative error message
            return f"Error executing tool '{tool_name}': {error_msg}"

        except Exception as e:
            # Unexpected errors - log with traceback for debugging
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Tool '{tool_name}' execution failed: {error_msg}", exc_info=True)

            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_FAILED,
                agent_id=self.name,
                message=f"Tool {tool_name} failed: {error_msg}",
                data={"tool_name": tool_name, "error": error_msg, "input": tool_input}
            ))

            # Return informative error message
            return f"Error executing tool '{tool_name}': {error_msg}"

    def _map_input_to_parameters(self, tool, input_value):
        """
        Intelligently map input value to tool parameters.

        This handles cases where the model provides plain text or non-dict JSON
        and we need to map it to the tool's expected parameter names.

        Args:
            tool: The tool object with metadata
            input_value: The input value (string or other type)

        Returns:
            Dict mapping parameter names to values
        """
        # Get tool parameters
        if not hasattr(tool, 'metadata') or not hasattr(tool.metadata, 'parameters'):
            # No metadata - use generic "input"
            return {"input": input_value}

        params = tool.metadata.parameters
        required_params = [p for p in params if p.required]

        # Case 1: Single parameter tool
        if len(params) == 1:
            param_name = params[0].name
            return {param_name: input_value}

        # Case 2: Multiple parameters - need to be smarter
        # Common patterns for specific tools

        # Code executor pattern: expects "code" and "language"
        if any(p.name == "code" for p in params):
            # Clean markdown code fences from code input
            code_str = str(input_value)
            # Remove markdown code fences (```python, ```py, ```javascript, etc.)
            import re
            code_str = re.sub(r'^```(?:python|py|javascript|js|bash|sh)?\n?', '', code_str, flags=re.MULTILINE)
            code_str = re.sub(r'\n?```$', '', code_str, flags=re.MULTILINE)
            code_str = code_str.strip()

            result = {"code": code_str}
            # Add default language if required
            if any(p.name == "language" and p.required for p in params):
                result["language"] = "python"  # Default to Python
            return result

        # Calculator pattern: expects "expression"
        if any(p.name == "expression" for p in params):
            return {"expression": str(input_value)}

        # Python REPL pattern: expects "code"
        if any(p.name == "code" for p in required_params):
            return {"code": str(input_value)}

        # File ops pattern: expects "operation" and "path"
        if any(p.name == "operation" for p in required_params):
            import re
            input_str = str(input_value).lower()

            # Try to extract operation and path from the input
            result = {}

            # Detect operation from keywords
            operation = None
            if any(word in input_str for word in ["read", "reading", "show", "display", "cat", "get content"]):
                operation = "read"
            elif any(word in input_str for word in ["write", "writing", "create", "save"]):
                operation = "write"
            elif any(word in input_str for word in ["list", "listing", "ls", "dir", "show files"]):
                operation = "list"
            elif any(word in input_str for word in ["search", "find", "grep"]):
                operation = "search"
            elif any(word in input_str for word in ["metadata", "info", "stat"]):
                operation = "metadata"
            elif any(word in input_str for word in ["convert", "transform"]):
                operation = "convert"

            if operation:
                result["operation"] = operation
            else:
                # Default to read if unclear
                result["operation"] = "read"

            # Extract path - look for file paths or filenames
            path_str = str(input_value)

            # Remove file:/// prefix if present
            path_str = re.sub(r'file://+', '', path_str)

            # Try to find filenames with extensions (prefer last occurrence to avoid "path/to/file")
            # Look for patterns like: file.txt, /path/file.txt, ./file.txt
            path_matches = re.findall(r'[\w\-\.\/]+\.\w+', path_str)
            if path_matches:
                # Use the last match (most likely the actual filename)
                path_candidate = path_matches[-1]
                # If it contains slashes, prefer just the filename part unless it starts with / or ./
                if '/' in path_candidate and not path_candidate.startswith(('/', './')):
                    # Extract just the filename
                    result["path"] = path_candidate.split('/')[-1]
                else:
                    result["path"] = path_candidate
            else:
                # Look for any path-like string
                path_match = re.search(r'(?:file|path)[\s:=]+([^\s]+)', path_str, re.IGNORECASE)
                if path_match:
                    result["path"] = path_match.group(1)
                else:
                    # Just use the input as path (remove operation keywords)
                    cleaned_path = path_str
                    for op_word in ["read", "write", "list", "search", "metadata", "convert", "operation="]:
                        cleaned_path = cleaned_path.replace(op_word, "").replace(op_word.upper(), "")
                    result["path"] = cleaned_path.strip()

            # Clean up path - remove leading/trailing slashes that might be artifacts
            if result.get("path"):
                result["path"] = result["path"].strip().lstrip('/')
                # If path doesn't start with / or ./ or ../, it's relative to current dir
                if not result["path"].startswith(('//', './', '../')):
                    # It's just a filename, keep as is
                    pass

            return result

        # Search pattern: expects "query"
        if any(p.name == "query" for p in params):
            return {"query": str(input_value)}

        # Default: Use first required parameter name, or first parameter name
        if required_params:
            return {required_params[0].name: str(input_value)}
        elif params:
            return {params[0].name: str(input_value)}
        else:
            # Fallback
            return {"input": str(input_value)}

    def _run_direct_inference(self,
                               task: str,
                               context: Dict[str, Any],
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
        # Simple direct prompt
        prompt = f"Answer this question directly and concisely:\n\n{task}\n\nAnswer:"

        try:
            response = self._generate(prompt, **kwargs)
            answer = response["text"].strip()
            tokens_used = response.get("tokens_used", 0)

            return AgentResponse(
                output=answer,
                success=True,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=tokens_used
            )

        except Exception as e:
            logger.error(f"Direct inference failed: {e}")
            return AgentResponse(
                output=f"Failed to answer: {str(e)}",
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=0,
                tokens_used=0,
                metadata={"error": str(e)}
            )

    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        if not self.tools:
            return "No tools available."

        descriptions = []
        for tool_name, tool in self.tools.items():
            desc = f"- {tool_name}: {tool.description}"
            descriptions.append(desc)

        return "\n".join(descriptions)

    def _format_conversation_history(self, max_turns: int = 5) -> str:
        """
        Format conversation history for inclusion in prompt.

        Args:
            max_turns: Maximum number of previous turns to include

        Returns:
            Formatted conversation history string
        """
        if not self.short_term_memory:
            return ""

        # Get last N turns
        recent_turns = self.short_term_memory[-max_turns:]

        if not recent_turns:
            return ""

        history = "\n\n=== Previous Conversation Context ===\n"
        for i, turn in enumerate(recent_turns, 1):
            history += f"[Turn {i}]\n"
            history += f"User: {turn['user']}\n"
            history += f"Assistant: {turn['assistant']}\n\n"
        history += "=== End of Previous Context ===\n"

        return history

    def add_tool(self, tool: BaseTool):
        """
        Add a tool to the agent.

        Args:
            tool: Tool instance to add
        """
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        """
        Remove a tool from the agent.

        Args:
            tool_name: Name of tool to remove
        """
        if tool_name in self.tools:
            del self.tools[tool_name]

    def reset_memory(self):
        """Clear conversation and tool history."""
        self.state.clear_history()
        self.short_term_memory = []

    def save_state(self, filepath: str, format: str = "json"):
        """
        Save agent state.

        Args:
            filepath: Path to save to
            format: Format (json or pickle)
        """
        self.state.save(filepath, format)

    def load_state(self, filepath: str, format: str = "json"):
        """
        Load agent state.

        Args:
            filepath: Path to load from
            format: Format (json or pickle)
        """
        self.state = AgentState.load(filepath, format)

    def synthesize(self, synthesis_data: Dict[str, Any]) -> str:
        """
        Synthesize results from sub-agents.

        Args:
            synthesis_data: Data to synthesize

        Returns:
            Synthesized output
        """
        # Build synthesis prompt
        results_text = []
        for result in synthesis_data.get("results", []):
            output = result.get("output", {})
            if isinstance(output, dict):
                results_text.append(output.get("output", str(output)))
            else:
                results_text.append(str(output))

        prompt = f"""Synthesize the following results into a comprehensive answer for: {synthesis_data['original_task']}

Results from sub-agents:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(results_text))}

Provide a well-structured, comprehensive response that integrates all findings."""

        # Generate synthesis
        response = self._generate(prompt, temperature=0.6)
        return response.get("text", "").strip()

    async def run_async(self,
                       task: str,
                       mode: AgentMode = AgentMode.AUTO,
                       context: Optional[Dict[str, Any]] = None,
                       **kwargs) -> AgentResponse:
        """
        Asynchronous version of run().

        Args:
            task: Task description
            mode: Execution mode
            context: Optional context
            **kwargs: Additional arguments

        Returns:
            AgentResponse
        """
        # Run in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.run,
            task,
            mode,
            context,
            **kwargs
        )

    def stream(self,
               task: str,
               mode: AgentMode = AgentMode.AUTO,
               context: Optional[Dict[str, Any]] = None,
               **kwargs) -> Iterator[str]:
        """
        Stream response token by token.

        Args:
            task: Task description
            mode: Execution mode
            context: Optional context
            **kwargs: Additional arguments

        Yields:
            Response tokens
        """
        # Placeholder for streaming
        # Would integrate with model streaming
        response = self.run(task, mode, context, **kwargs)
        for char in response.output:
            yield char
            time.sleep(0.01)  # Simulate streaming delay

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of execution.

        Returns:
            Summary dictionary
        """
        return self.execution_tracker.get_summary()

    def __repr__(self) -> str:
        """String representation."""
        return f"Agent(name={self.name}, tools={len(self.tools)}, sub_agents={self.config.enable_sub_agents})"
