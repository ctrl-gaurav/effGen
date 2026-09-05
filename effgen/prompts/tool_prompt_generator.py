"""
Tool prompt generator for enhanced ReAct prompts.

Generates dynamic system prompts with tool calling examples,
parameter format examples, and common mistakes guidance,
optimized for Small Language Models (1B-7B parameters).
"""

from __future__ import annotations

import logging
from typing import Any

from ..tools.base_tool import BaseTool, ParameterType

logger = logging.getLogger(__name__)


class ToolPromptGenerator:
    """
    Generate enhanced tool descriptions with usage examples for SLM agents.

    Takes a list of BaseTool instances and produces formatted prompt sections
    including parameter details, usage examples, and common-mistake warnings.
    """

    # Model family detection patterns
    MODEL_FAMILIES = {
        "qwen": ["qwen", "qwen2"],
        "llama": ["llama", "meta-llama"],
        "phi": ["phi", "microsoft/phi"],
        "mistral": ["mistral"],
        "gemma": ["gemma"],
    }

    def __init__(self, tools: list[BaseTool], model_name: str | None = None) -> None:
        """
        Initialize the generator.

        Args:
            tools: List of BaseTool instances to generate prompts for.
            model_name: Optional model name for model-specific optimizations.
        """
        self.tools = tools
        self.model_name = model_name or ""
        self._model_family = self._detect_model_family()

    def _detect_model_family(self) -> str:
        """Detect model family from model name."""
        name_lower = self.model_name.lower()
        for family, patterns in self.MODEL_FAMILIES.items():
            for pattern in patterns:
                if pattern in name_lower:
                    return family
        return "generic"

    def generate_tools_section(self, verbose: bool = True) -> str:
        """
        Generate the complete tools description section for the prompt.

        Args:
            verbose: If True, include full parameter details and examples.

        Returns:
            Formatted tools section string.
        """
        if not self.tools:
            return "No tools available. Answer directly using your knowledge."

        sections = []
        for i, tool in enumerate(self.tools, 1):
            sections.append(self._format_tool(tool, i, verbose))

        tools_text = "\n\n".join(sections)
        rules = self._generate_rules_section()

        return f"{tools_text}\n\n{rules}"

    def _format_tool(self, tool: BaseTool, index: int, verbose: bool) -> str:
        """Format a single tool description."""
        lines = [f"{index}. {tool.name}"]
        lines.append(f"   Description: {tool.description}")

        if verbose and hasattr(tool, 'metadata') and tool.metadata.model_facing_parameters:
            lines.append("   Parameters:")
            for param in tool.metadata.model_facing_parameters:
                req = "required" if param.required else "optional"
                default_str = ""
                if param.default is not None:
                    default_str = f", default={param.default}"
                enum_str = ""
                if param.enum:
                    enum_str = f", options={param.enum}"
                lines.append(
                    f"     - {param.name} ({param.type.value}, {req}{default_str}{enum_str}): "
                    f"{param.description}"
                )

        # Add example
        example = self._generate_example(tool, verbose)
        if example:
            lines.append(f"   Example: {example}")

        return "\n".join(lines)

    def _generate_example(self, tool: BaseTool, verbose: bool) -> str:
        """Generate an example usage line for a tool."""
        # Use tool's own examples if available
        if hasattr(tool, 'metadata') and tool.metadata.examples:
            ex = tool.metadata.examples[0]
            # Build Action Input from example keys (exclude 'output')
            input_dict = {k: v for k, v in ex.items() if k != "output"}
            if input_dict:
                import json
                input_json = json.dumps(input_dict)
                return f"Action: {tool.name} | Action Input: {input_json}"

        # Auto-generate from parameter specs (model-facing only — never sample a
        # developer-only safety toggle into a model-facing example).
        if hasattr(tool, 'metadata') and tool.metadata.model_facing_parameters:
            sample = {}
            for param in tool.metadata.model_facing_parameters:
                if param.required:
                    sample[param.name] = self._sample_value(param)
            if sample:
                import json
                input_json = json.dumps(sample)
                return f"Action: {tool.name} | Action Input: {input_json}"

        return ""

    def _sample_value(self, param) -> Any:
        """Generate a sample value for a parameter spec."""
        if param.enum:
            return param.enum[0]
        if param.default is not None:
            return param.default
        type_samples = {
            ParameterType.STRING: f"<{param.name}>",
            ParameterType.INTEGER: 1,
            ParameterType.FLOAT: 1.0,
            ParameterType.BOOLEAN: True,
            ParameterType.ARRAY: [],
            ParameterType.OBJECT: {},
        }
        return type_samples.get(param.type, "<value>")

    def _generate_rules_section(self) -> str:
        """Generate the common-mistakes / rules section."""
        tool_names = [t.name for t in self.tools]
        tool_list = ", ".join(tool_names)
        return (
            "IMPORTANT RULES:\n"
            f"1. You can ONLY use these tools: {tool_list}\n"
            "2. Action Input MUST be valid JSON with the exact parameter names shown above\n"
            "3. Do NOT invent tools that are not listed\n"
            "4. Always use \"Final Answer:\" when you have the complete answer\n"
            "5. If a tool fails, try a different approach or provide your best answer directly"
        )

    def generate_react_prompt(
        self,
        task: str,
        scratchpad: str = "",
        conversation_history: str = "",
        system_prompt: str = "You are a helpful AI assistant.",
        verbose: bool = True,
        closing_instruction: str = "",
        answer_shape: str = "",
        tool_contract: str = "",
    ) -> str:
        """
        Generate a complete ReAct prompt with enhanced tool descriptions.

        Args:
            task: The user's question or task.
            scratchpad: Current scratchpad content from previous iterations.
            conversation_history: Formatted conversation history.
            system_prompt: System-level instructions.
            verbose: Whether to include verbose tool descriptions.
            closing_instruction: Text to place after the scratchpad, where it is
                the last thing the model reads. Used to say what to do with a
                trailing observation the model would otherwise return as-is
                (retrieved passages). Empty leaves the prompt unchanged.
            answer_shape: The shape the caller declared for the answer, placed
                ahead of *closing_instruction*. It carries no "answer now"
                label of its own, so it can be stated on a turn where the model
                may still call a tool. Empty leaves the prompt unchanged.
            tool_contract: What to do with the tools listed above it — see
                :mod:`effgen.prompts.tool_contract`. Placed with the tool list
                so the two are read together, and repeated on every turn
                because the whole scaffold is re-rendered every turn. Empty
                leaves the prompt unchanged.

        Returns:
            Complete formatted ReAct prompt string.
        """
        tools_section = self.generate_tools_section(verbose=verbose)

        # Apply model-specific formatting
        prompt = self._apply_model_format(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            tools_section=tools_section,
            task=task,
            scratchpad=scratchpad,
            tool_contract=tool_contract,
        )

        if answer_shape:
            # The ReAct formats parse an answer off the "Final Answer:" label,
            # so a statement about what the answer *is* has to carry how it is
            # marked -- otherwise a model that follows the shape emits the value
            # with no label, the turn reads as more reasoning, and the loop runs
            # to its cap on a run that was already answered. When a closing
            # instruction follows, it restates the label itself.
            prompt = f"{prompt}\n\n{answer_shape}"
            if not closing_instruction:
                prompt = (
                    f"{prompt}\nGive it after a 'Final Answer:' label once you "
                    "have it."
                )

        if closing_instruction:
            # The ReAct formats parse an answer off the "Final Answer:" label, so
            # restate the label alongside the instruction to keep that contract.
            prompt = (
                f"{prompt}\n\n{closing_instruction}\n"
                "Give that answer now, after a 'Final Answer:' label."
            )

        return prompt

    def _apply_model_format(
        self,
        system_prompt: str,
        conversation_history: str,
        tools_section: str,
        task: str,
        scratchpad: str,
        tool_contract: str = "",
    ) -> str:
        """
        Apply model-family-specific prompt formatting.

        Different SLM families respond better to different prompt structures.
        """
        # Build the core ReAct instruction (shared across all models)
        react_instructions = (
            "Use the following format:\n\n"
            "Question: the input question or task\n"
            "Thought: think step-by-step about what to do next\n"
            "Action: the tool to use (or \"Final Answer\" when ready to respond)\n"
            "Action Input: the input for the tool (must be valid JSON)\n"
            "Observation: the result of the tool\n"
            "... (repeat Thought/Action/Action Input/Observation as needed)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: the complete response to the original question"
        )

        # Model-specific wrapping
        if self._model_family == "qwen":
            return self._format_qwen(
                system_prompt, conversation_history, tools_section,
                react_instructions, task, scratchpad, tool_contract
            )
        elif self._model_family == "llama":
            return self._format_llama(
                system_prompt, conversation_history, tools_section,
                react_instructions, task, scratchpad, tool_contract
            )
        elif self._model_family == "phi":
            return self._format_phi(
                system_prompt, conversation_history, tools_section,
                react_instructions, task, scratchpad, tool_contract
            )
        else:
            return self._format_generic(
                system_prompt, conversation_history, tools_section,
                react_instructions, task, scratchpad, tool_contract
            )

    @staticmethod
    def _with_contract(tools_block: str, tool_contract: str) -> str:
        """Return *tools_block* with the contract stated under it.

        The contract is about the tools the block lists, so the two are read
        together, ahead of the format instructions. An empty contract leaves the
        block exactly as it was.
        """
        return f"{tools_block}\n\n{tool_contract}" if tool_contract else tools_block

    def _format_generic(
        self, system_prompt, conversation_history, tools_section,
        react_instructions, task, scratchpad, tool_contract=""
    ) -> str:
        """Default prompt format."""
        parts = [
            f"{system_prompt} You can reason step-by-step and use tools.",
            conversation_history,
            self._with_contract(f"Available tools:\n{tools_section}", tool_contract),
            "",
            "IMPORTANT: If there is previous conversation context above, use that information.",
            "",
            react_instructions,
            "",
            "Begin!",
            "",
            f"Question: {task}",
            scratchpad,
        ]
        return "\n".join(p for p in parts if p or p == "")

    def _format_qwen(
        self, system_prompt, conversation_history, tools_section,
        react_instructions, task, scratchpad, tool_contract=""
    ) -> str:
        # Qwen2.5 format: structured with clear section markers
        """Qwen-optimized prompt format with chat template hints."""
        parts = [
            f"{system_prompt} You can reason step-by-step and use tools.",
            conversation_history,
            self._with_contract(
                f"<|tools|>\nAvailable tools:\n{tools_section}\n<|/tools|>", tool_contract
            ),
            "",
            "IMPORTANT: If there is previous conversation context above, use that information.",
            "",
            react_instructions,
            "",
            "Begin!",
            "",
            f"Question: {task}",
            scratchpad,
        ]
        return "\n".join(p for p in parts if p or p == "")

    def _format_llama(
        self, system_prompt, conversation_history, tools_section,
        react_instructions, task, scratchpad, tool_contract=""
    ) -> str:
        # Llama-3 format: system-style instructions
        """Llama-optimized prompt format."""
        parts = [
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
            f"{system_prompt} You can reason step-by-step and use tools.",
            conversation_history,
            self._with_contract(f"Available tools:\n{tools_section}", tool_contract),
            "",
            "IMPORTANT: If there is previous conversation context above, use that information.",
            "",
            react_instructions,
            "<|eot_id|><|start_header_id|>user<|end_header_id|>",
            "",
            f"Question: {task}",
            scratchpad,
        ]
        return "\n".join(p for p in parts if p or p == "")

    def _format_phi(
        self, system_prompt, conversation_history, tools_section,
        react_instructions, task, scratchpad, tool_contract=""
    ) -> str:
        """Phi-optimized prompt: more concise instructions."""
        parts = [
            f"{system_prompt} You can reason step-by-step and use tools.",
            conversation_history,
            self._with_contract(f"Tools:\n{tools_section}", tool_contract),
            "",
            react_instructions,
            "",
            "Begin!",
            "",
            f"Question: {task}",
            scratchpad,
        ]
        return "\n".join(p for p in parts if p or p == "")
