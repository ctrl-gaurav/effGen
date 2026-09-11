"""Prompt and context assembly for :class:`effgen.core.agent.Agent`.

Builds the pieces of text a request is assembled from: the system prompt, the
tool descriptions whose verbosity follows the model's size, the recent
conversation history, and the Anthropic prompt-caching wrappers for the system
prompt and the tool specs. Mixed into :class:`Agent`; this module imports
nothing from ``agent.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..memory.short_term import MessageRole

if TYPE_CHECKING:
    from .thread import TurnStep


class AgentPromptingMixin:
    """System prompt, tool descriptions, and conversation-history formatting."""

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

IMPORTANT: You do NOT have to use a tool for every question.
If you can answer directly from your knowledge or from the conversation history above, skip the Action step entirely:
Thought: I can answer this directly without any tools.
Final Answer: [your answer here]
Only use tools when you NEED external computation, data, or system access.

Example (no tool needed):
Question: Tell me a joke about programming.
Thought: This is a creative request. I can answer directly without tools.
Final Answer: Why do programmers prefer dark mode? Because light attracts bugs!

Begin!

Question: {task}
{scratchpad}"""

    def _get_anthropic_system(self) -> str | list | None:
        """
        Return the system prompt for Anthropic requests.

        When ``AgentConfig.cache_system_prompt=True`` and the model is an
        ``AnthropicAdapter``, the system prompt is returned as a list of
        content blocks with ``cache_control`` on the last block so that it is
        cached across sequential requests.
        """
        system = self.config.system_prompt if self.config.stable_system_prompt else None
        if system is None:
            return None
        try:
            from ..models.anthropic_adapter import AnthropicAdapter
            from ..models.anthropic_cache import apply_cache_to_system
        except ImportError:
            return system
        if isinstance(self.model, AnthropicAdapter) and self.config.cache_system_prompt:
            return apply_cache_to_system(system)
        return system

    def _get_anthropic_tools(self, tools: list[dict]) -> list[dict]:
        """
        Apply ``cache_control`` to the last tool spec when appropriate.

        Only active when ``AgentConfig.cache_tools=True`` and the model is an
        ``AnthropicAdapter``.
        """
        if not tools:
            return tools
        try:
            from ..models.anthropic_adapter import AnthropicAdapter
            from ..models.anthropic_cache import apply_cache_to_last_tool
        except ImportError:
            return tools
        if isinstance(self.model, AnthropicAdapter) and self.config.cache_tools:
            return apply_cache_to_last_tool(tools)
        return tools

    def _build_system_prompt(self) -> str:
        """Build a dynamic system prompt based on agent configuration and tools."""
        return self._system_prompt_builder.build(
            tools=self.config.tools,
            agent_name=self.name,
            base_system_prompt=None,  # Will generate default role
            enable_fallback=self._enable_fallback,
            verbose=self._verbose_tools,
        )

    def _auto_detect_verbose(self) -> bool:
        """Auto-detect whether to use verbose tool descriptions based on model size."""
        name = (self.model_name or "").lower()
        # Check for known small models (< 3B) -> full verbose with examples
        # Check for medium models (3B-7B) -> verbose without examples
        # Check for large models (> 7B) or API models -> compact
        for indicator in ["0.5b", "1b", "1.5b", "2b"]:
            if indicator in name:
                return True
        for indicator in ["3b", "4b", "5b", "7b"]:
            if indicator in name:
                return True
        # API models
        for indicator in ["gpt", "claude", "gemini"]:
            if indicator in name:
                return False
        # Default: verbose (safe for SLMs)
        return True

    def _get_tools_description(self, verbose: bool | None = None) -> str:
        """
        Get formatted description of available tools.

        Args:
            verbose: Override verbosity. If None, uses self._verbose_tools.

        Returns:
            Formatted tools description string.
        """
        if not self.tools:
            return "No tools available."

        use_verbose = verbose if verbose is not None else self._verbose_tools
        return self._tool_prompt_generator.generate_tools_section(verbose=use_verbose)

    def _prior_turn_steps(self, max_turns: int = 25) -> "list[TurnStep]":
        """The session's earlier messages, as steps this run's thread carries.

        Read from :class:`~effgen.memory.short_term.ShortTermMemory`, newest
        ``max_turns`` exchanges, with any summary of the turns that were
        dropped stated first. Each message keeps the role it was spoken in, so
        a model that takes a conversation is sent one instead of a block of
        text pasted into the current question.

        Args:
            max_turns: How many user/assistant exchanges to carry.

        Returns:
            The steps, oldest first, or an empty list for a first turn.
        """
        from .thread import TurnStep

        summaries = self.short_term_memory.summaries
        messages = self.short_term_memory.get_recent_messages(n=max_turns * 2)
        steps: list[TurnStep] = []
        for summary in summaries:
            steps.append(
                TurnStep(text=f"Earlier context summary: {summary.summary}", role="user")
            )
        for message in messages:
            if message.role not in (MessageRole.USER, MessageRole.ASSISTANT):
                continue
            role: Literal["user", "assistant"] = (
                "assistant" if message.role == MessageRole.ASSISTANT else "user"
            )
            steps.append(TurnStep(text=str(message.content), role=role))
        return steps

    def _format_conversation_history(self, max_turns: int = 25) -> str:
        """The session's earlier turns rendered for a frame that takes a string.

        The turns themselves are :class:`~effgen.core.thread.TurnStep` steps
        (:meth:`_prior_turn_steps`); this is
        :meth:`~effgen.core.thread.AgentThread.history_text` over them, which
        is what a prompt template's conversation-history field receives when
        the model is reached with one string rather than a conversation.

        Args:
            max_turns: How many user/assistant exchanges to carry.

        Returns:
            The rendering, or ``""`` when the session has no earlier turns.
        """
        from .thread import AgentThread

        steps = self._prior_turn_steps(max_turns)
        if not steps:
            return ""
        return AgentThread(steps=list(steps)).history_text()
