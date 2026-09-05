"""
Dynamic system prompt generation for tool-aware agents.

Generates optimized system prompts based on the agent's configuration,
including tool-specific instructions, tips, and ReAct format guidance.
Designed for Small Language Models (1B-7B parameters).
"""

from __future__ import annotations

import logging

from ..tools.base_tool import BaseTool, ToolCategory

logger = logging.getLogger(__name__)


# Operational hints per tool category — what to watch out for while using a
# tool of that kind. Whether and how to reach for the tool at all is stated by
# :mod:`effgen.prompts.tool_contract`, which every tool-calling path carries;
# saying it twice, in two different wordings, produced a prompt that told a
# model both to work a calculation out itself and never to compute in its head.
# A category with nothing operational to add has no entry, and
# ``_build_category_tips`` leaves it out.
CATEGORY_INSTRUCTIONS: dict[ToolCategory, str] = {
    ToolCategory.INFORMATION_RETRIEVAL: (
        "- If your first search returns no results, try rephrasing with different keywords."
    ),
    ToolCategory.CODE_EXECUTION: (
        "- Always check the output/return value of executed code."
    ),
    ToolCategory.FILE_OPERATIONS: (
        "- Check that files exist before trying to read them."
    ),
    ToolCategory.SYSTEM: (
        "- Be careful with system operations — only run safe commands."
    ),
    ToolCategory.EXTERNAL_API: (
        "- Some tools call external APIs. Results may take a moment.\n"
        "- If an API call fails, inform the user and try an alternative approach."
    ),
    ToolCategory.DATA_PROCESSING: (
        "- Break complex data transformations into steps."
    ),
}


class AgentSystemPromptBuilder:
    """
    Generates optimized system prompts based on agent configuration.

    The generated prompt includes:
    1. Role definition
    2. Tool descriptions with examples
    3. ReAct format instructions
    4. Tool-specific tips based on tool categories
    5. Common mistakes to avoid
    6. Fallback instructions
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the prompt builder.

        Args:
            model_name: Optional model name for model-specific optimizations.
        """
        self.model_name = model_name or ""

    def build(
        self,
        tools: list[BaseTool],
        agent_name: str = "assistant",
        base_system_prompt: str | None = None,
        enable_fallback: bool = True,
        verbose: bool = True,
    ) -> str:
        """
        Build a complete system prompt for the agent.

        Args:
            tools: List of tools available to the agent.
            agent_name: Name of the agent.
            base_system_prompt: Optional base system prompt to prepend.
            enable_fallback: Whether fallback chains are enabled.
            verbose: Include verbose tool descriptions.

        Returns:
            Complete system prompt string.
        """
        sections = []

        # 1. Role definition
        sections.append(self._build_role_section(agent_name, base_system_prompt))

        # 2. Tool-category-specific tips
        categories = self._get_tool_categories(tools)
        if categories:
            sections.append(self._build_category_tips(categories))

        # 3. Common mistakes to avoid
        sections.append(self._build_mistakes_section(tools))

        # 4. Fallback instructions
        if enable_fallback and len(tools) > 1:
            sections.append(self._build_fallback_section())

        return "\n\n".join(s for s in sections if s)

    def _build_role_section(
        self, agent_name: str, base_prompt: str | None
    ) -> str:
        """Build the role definition section."""
        if base_prompt:
            return base_prompt

        return (
            f"You are {agent_name}, a helpful AI assistant. "
            "You can reason step-by-step and use tools to help answer questions. "
            "Always think carefully before acting, and use the most appropriate tool for each task."
        )

    def _get_tool_categories(self, tools: list[BaseTool]) -> set[ToolCategory]:
        """Get unique tool categories from the tool list.

        A tool whose category is missing or is not a
        :class:`~effgen.tools.base_tool.ToolCategory` — one built from a plain
        string, or from ``None`` — contributes nothing rather than reaching the
        sort below, which orders on ``.value``.
        """
        categories = set()
        for tool in tools:
            category = getattr(tool, 'category', None)
            if isinstance(category, ToolCategory):
                categories.add(category)
        return categories

    def _build_category_tips(self, categories: set[ToolCategory]) -> str:
        """Build category-specific tips."""
        tips = []
        for cat in sorted(categories, key=lambda c: c.value):
            instruction = CATEGORY_INSTRUCTIONS.get(cat)
            if instruction:
                tips.append(instruction)

        if not tips:
            return ""

        return "Tool Usage Tips:\n" + "\n".join(tips)

    def _build_mistakes_section(self, tools: list[BaseTool]) -> str:
        """Build common mistakes to avoid."""
        tool_names = [t.name for t in tools]

        lines = [
            "Common Mistakes to Avoid:",
            f"- Only use tools from this list: {', '.join(tool_names)}",
            "- Action Input must be valid JSON — use double quotes for strings",
            "- Do NOT invent or hallucinate tool names that are not listed",
            '- Always end with "Final Answer:" when you have the complete answer',
            "- Do NOT repeat the same failed action — try a different approach",
            "- Do NOT use tools for greetings, jokes, opinions, or recalling information from conversation",
        ]

        # Add tool-specific warnings
        for tool in tools:
            if hasattr(tool, 'metadata') and tool.metadata.requires_api_key:
                lines.append(
                    f"- {tool.name} requires an API key — check if it's configured"
                )

        return "\n".join(lines)

    def _build_fallback_section(self) -> str:
        """Build fallback instructions."""
        return (
            "If a tool fails:\n"
            "1. Read the error message carefully\n"
            "2. Try a different tool or approach\n"
            "3. If no tool can help, provide your best answer directly with a note about the limitation"
        )
