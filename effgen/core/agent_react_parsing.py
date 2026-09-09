"""Reading a model turn for :class:`Agent`.

Turns the text or the tool calls a model returned into the action or the answer
the ReAct loop acts on, and probes what the loaded model can be told about
tools so a turn is read the way it was prompted. Mixed into :class:`Agent`
through :class:`~effgen.core.agent_react.AgentReActMixin`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from .agent_runtime import (
    _SCAFFOLD_LITERAL_RES,
    _TOOL_ECHO_RE,
    sanitize_final_answer,
)
from .tool_calling import (
    ToolCallResult,
    action_name,
    name_positional_arguments,
    parse_call_syntax,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .agent_response import PartialResult
    from .thread import AgentThread, ObservationStep

# The parsers log to the ReAct stream they serve.
logger = logging.getLogger("effgen.core.agent_react")

#: A turn that announces it has the answer, and then states it. What follows
#: the announcement is the answer; the announcement itself is not.
_KNOWS_THE_ANSWER_RE = re.compile(
    r"I (?:now )?know[^.]*\.\s*(.+)\Z", re.IGNORECASE | re.DOTALL
)

#: A thought shorter than this is bookkeeping, not something to report back.
_SUBSTANTIVE_THOUGHT_CHARS = 20

#: A weekday names an answer of its own, whatever the tool that produced it
#: went on to say.
_DAY_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


def _without_scaffolding(step: "ObservationStep") -> str | None:
    """What a tool returned, or ``None`` when the framework wrote the line.

    A call the loop declined to make is answered with an instruction rather
    than a result, and a transcript read back from an older run can carry one
    of the framework's own lines where an observation goes. Neither is progress
    the run made.
    """
    if getattr(step, "declined", None):
        return None
    text = step.text
    for pattern in _SCAFFOLD_LITERAL_RES:
        text = pattern.sub("", text)
    return text


class AgentReActParsingMixin:
    """Turn readers and model tool-calling probes for :class:`Agent`."""

    def _model_advertises_tool_calling(self) -> bool:
        """True when the loaded model reports native tool-calling support."""
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "supports_tool_calling"):
            return False
        try:
            return bool(model.supports_tool_calling())
        except Exception:  # noqa: BLE001 - a capability probe never breaks a run
            logger.debug("Tool-calling capability check failed", exc_info=True)
            return False

    def _model_tool_call_support(self) -> str:
        """How the loaded model receives tool definitions.

        One of ``"api"``, ``"template"`` or ``"none"`` — see
        :meth:`effgen.models.base.BaseModel.tool_call_support`. A model that
        predates the method, or one whose probe raises, is treated as ``"none"``
        so no prompt changes on its account.
        """
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "tool_call_support"):
            return "none"
        try:
            return str(model.tool_call_support())
        except Exception:  # noqa: BLE001 - a capability probe never breaks a run
            logger.debug("Tool-calling support check failed", exc_info=True)
            return "none"

    def _text_parse_strategy(self, used_native_prompt: bool) -> Any:
        """The strategy that reads this turn's text, matched to its prompt.

        ``native`` reads only the tool-call syntax a chat template teaches. When
        the model cannot be given the definitions at all, the turn above was
        prompted as ReAct instead, and the native reader finds neither a call
        nor an answer in it — the run then repeats the same turn to its
        iteration cap and ends with nothing. Reading such a turn with the
        hybrid strategy keeps the native syntax first and falls back to the
        ReAct text that actually arrived.

        Args:
            used_native_prompt: Whether this turn was prompted for native tool
                calling.

        Returns:
            The strategy to parse the turn's text with.
        """
        strategy = self._tool_calling_strategy
        if used_native_prompt or strategy.name != "native":
            return strategy
        if self._model_tool_call_support() != "none":
            return strategy
        cached = getattr(self, "_text_fallback_strategy", None)
        if cached is None:
            from .tool_calling import HybridStrategy

            cached = HybridStrategy()
            self._text_fallback_strategy = cached
        return cached

    def _extract_partial_answer(self, thread: "AgentThread") -> str | None:
        """What the run had reached when it stopped without writing an answer.

        Read from the run's steps rather than from the transcript they render:
        a thought is a thought because it is one, not because a line began with
        the word. A line the framework injected is not the run's progress and is
        never a candidate, and neither is a call the loop declined to make.

        Args:
            thread: The run's conversation.

        Returns:
            A partial answer string, or ``None`` when the run reached nothing
            worth reporting.
        """
        if thread is None or not len(thread):
            return None

        # A turn that says it has the answer and then states it. The statement
        # is what is wanted, not the announcement.
        for step in thread.thoughts():
            match = _KNOWS_THE_ANSWER_RE.match(step.text.strip())
            if match and match.group(1).strip():
                return match.group(1).strip()

        # What the tools returned. An observation the framework wrote — a
        # nudge, or a declined call answered with an instruction — is the
        # framework's words and is not progress.
        observations = [
            text
            for text in (_without_scaffolding(step) for step in thread.observations())
            if text is not None
        ]
        if observations:
            # Strip tool-echo prefixes ("[tool(args)] → result") so only the
            # results are joined, not the scaffolding.
            valid_obs = [
                self._humanize_observation(_TOOL_ECHO_RE.sub("", o.strip()))
                for o in observations
                if o.strip() and not o.strip().lower().startswith("error")
            ]
            # Deduplicate while preserving order — a model that loops on the same
            # retrieval/search tool produces the same passage repeatedly, and
            # joining the duplicates yields a messy answer.
            seen_obs: set[str] = set()
            deduped_obs: list[str] = []
            for o in valid_obs:
                if o and o not in seen_obs:
                    seen_obs.add(o)
                    deduped_obs.append(o)
            valid_obs = deduped_obs
            if len(valid_obs) > 1:
                return sanitize_final_answer(" | ".join(valid_obs))
            elif valid_obs:
                return sanitize_final_answer(valid_obs[-1])

            # A named day is an answer even when the tool that produced it
            # failed on the rest of the question.
            for obs in reversed(observations):
                obs_lower = obs.strip().lower()
                for day in _DAY_NAMES:
                    if day in obs_lower:
                        return obs.strip()

        # Failing that, the last thought the run wrote, when it wrote enough of
        # one to be worth reporting.
        thoughts = thread.thoughts()
        if thoughts:
            last_thought = thoughts[-1].text.strip()
            if len(last_thought) > _SUBSTANTIVE_THOUGHT_CHARS:
                return last_thought

        return None

    def _partial_result(
        self,
        thread: "AgentThread",
        *,
        text: str,
        calls: "Sequence[Any]" = (),
        iterations: int = 0,
        tool_calls: int = 0,
    ) -> "PartialResult":
        """Assemble what a run had reached when it stopped without an answer.

        *text* is the flattened one-line form the run would otherwise have
        returned; the observations come from the tool-call records, so they are
        the same calls :attr:`AgentResponse.tool_calls` reports, one for one,
        with each tool's own words rather than a joined rewrite of them.

        Args:
            thread: The run's conversation, read for the last thought.
            text: The flattened partial text.
            calls: The run's tool-call records, in call order.
            iterations: Iterations the run had made.
            tool_calls: Tool calls the run had made.

        Returns:
            The payload for :attr:`AgentResponse.partial`.
        """
        from .agent_response import PartialResult

        observations: list[str] = []
        for call in calls or ():
            result = getattr(call, "result", None)
            if result is None:
                result = getattr(call, "error", None)
            observations.append("" if result is None else str(result))
        thought = thread.last_thought() if thread is not None else None
        return PartialResult(
            observations=tuple(observations),
            last_observation=observations[-1] if observations else None,
            last_thought=thought.text.strip() if thought is not None else None,
            text=text,
            iterations=iterations,
            tool_calls=tool_calls,
        )

    def _should_return_direct_calculator_result(
        self,
        task: str,
        action: str,
        action_input: str,
    ) -> bool:
        """Return calculator output directly for simple arithmetic questions."""
        if action != "calculator" or not action_input:
            return False

        task_lower = task.lower()
        if any(
            marker in task_lower
            for marker in (
                "explain",
                "show your work",
                "step by step",
                "steps",
                "why",
                "reason",
            )
        ):
            return False

        try:
            parsed_input = json.loads(action_input)
        except (json.JSONDecodeError, TypeError):
            parsed_input = {"expression": action_input}
        if not isinstance(parsed_input, dict):
            return False

        operation = str(parsed_input.get("operation", "calculate")).lower()
        if operation != "calculate":
            return False

        expression = str(parsed_input.get("expression", "")).strip()
        if not expression:
            return False

        task_numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", task)
        expr_numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", expression)
        if not task_numbers or not all(num in expr_numbers for num in task_numbers):
            return False

        arithmetic_markers = (
            "+",
            "-",
            "*",
            "/",
            "^",
            "%",
            " plus ",
            " minus ",
            " times ",
            " multiplied ",
            " divide ",
            " divided ",
            " square root ",
            " sqrt",
            " squared",
            " cube",
            " cubed",
            " to the power",
            " power of ",
            " modulo ",
            " factorial",
        )
        return any(marker in f" {task_lower} " for marker in arithmetic_markers)

    def _parse_react_response(self, text: str) -> dict[str, Any]:
        """
        Parse a ReAct-formatted response into its components.

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
        parsed: dict[str, Any] = {
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
            # NOTE: "Answer:" must be at the start of a line to avoid greedy
            # mid-text matches (e.g. "The answer is 42" should NOT match here).
            final_patterns = [
                r"Final Answer:\s*(.+)",
                r"^Answer:\s*(.+)",
                r"^The answer is:\s*(.+)"
            ]

            for pattern in final_patterns:
                try:
                    final_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                    if final_match:
                        answer = final_match.group(1).strip()
                        # Stop at next section marker, observation, or human turn
                        answer = re.split(r'\n(?:Question|Thought|Action|Observation|Human):', answer, maxsplit=1)[0].strip()
                        # Strip trailing unrelated content (e.g. Phi-4 generating
                        # follow-up questions after the answer like "...is 42.What year...")
                        # Match sentence boundary (.!?) followed by a new sentence
                        # that itself contains a question mark — likely a hallucinated follow-up.
                        trailing = re.search(r'([.!?])[\s]*[A-Z][^.!?]*\?', answer)
                        if trailing:
                            answer = answer[:trailing.start() + 1].strip()
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
                        # Kept with its quoting intact for the call-syntax
                        # reader below; the name path still strips quotes.
                        action_raw = action_match.group(1).strip()
                        action = action_raw
                        # Clean up common artifacts
                        action = action.replace('"', '').replace("'", "")
                        # Drop a same-line "Action Input:"/"Args:" section so the
                        # name resolves against the registry.
                        action = action_name(action)

                        # Check if action is actually "Final Answer" - treat it as final answer, not tool
                        if action.lower() in ["final answer", "finalanswer", "answer"]:
                            logger.debug(f"Action '{action}' detected as Final Answer indicator")
                            # When model writes "Action: Final Answer", the answer may be:
                            # 1. On the same line: "Action: Final Answer: The answer is 42"
                            # 2. On the Action Input line: "Action Input: The answer is 42"
                            # But NOT if Action Input is JSON (model repeating tool input)

                            # Try same-line first — [: \t]+ excludes newlines
                            same_line = re.search(
                                r"Action:\s*Final\s*Answer[: \t]+([^\n]+)", text, re.IGNORECASE,
                            )
                            if same_line:
                                answer_text = same_line.group(1).strip()
                                if answer_text:
                                    parsed["final_answer"] = answer_text
                                    logger.debug("Extracted final answer from Action line")
                                    return parsed

                            # Try Action Input line (only if it's natural language, not JSON)
                            ai_match = re.search(
                                r"Action\s*Input:\s*(.+?)(?:\n|$)",
                                text, re.IGNORECASE,
                            )
                            if ai_match:
                                answer_text = ai_match.group(1).strip()
                                if answer_text and not answer_text.startswith(("{", "[")):
                                    parsed["final_answer"] = answer_text
                                    logger.debug("Extracted final answer from Action Input line")
                                    return parsed

                            # If we get here, the model wrote "Action: Final Answer"
                            # but didn't provide a proper answer text. Don't extract
                            # anything — let the loop continue for another iteration.
                            logger.debug("Action: Final Answer detected but no answer text found")
                            break

                        # A call written in Python call syntax, or a bare JSON
                        # object after the name. Read from the unmangled text:
                        # stripping every quote first leaves an argument list
                        # nothing downstream can name, and the tool then ran
                        # with the whole fragment as one value.
                        call = parse_call_syntax(action_raw)
                        if call is not None:
                            call_name, call_kwargs, call_positional = call
                            parsed["action"] = call_name
                            if call_kwargs:
                                raw_value = call_kwargs.get("__raw_input__")
                                parsed["action_input"] = (
                                    raw_value if raw_value is not None
                                    else json.dumps(call_kwargs)
                                )
                            elif call_positional:
                                parsed["action_input"] = json.dumps(
                                    name_positional_arguments(
                                        call_name, call_positional,
                                        getattr(self, "tools", None),
                                    )
                                )
                            break

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
                    r"Parameters?:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
                    # The name is trimmed at an `Args:`/`Arguments:` label too,
                    # so read the arguments from it rather than calling the tool
                    # with none.
                    r"Arg(?:ument)?s:\s*(.+?)(?=\n(?:Observation|Thought|Action|Question):|$)",
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

    @staticmethod
    def _parse_native_tool_calls(native_tool_calls: list[dict[str, Any]]) -> ToolCallResult:
        """Convert an adapter's reported tool_calls into a ToolCallResult.

        Adapters report::

            [{"id": "...", "type": "function",
              "function": {"name": "...", "arguments": "json-string"}}]

        Flat keys and an already-parsed ``arguments`` are still accepted, so a
        list that reached here from somewhere other than an adapter — a hosted
        model's own output passed through verbatim, say — is read the same way.
        """
        result = ToolCallResult(raw_text="")
        if not native_tool_calls:
            return result
        tc = native_tool_calls[0]
        fn = tc.get("function", tc)
        tool_name = fn.get("name")
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {"__raw_input__": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        if tool_name:
            result.tool_name = tool_name
            result.arguments = arguments
            result.is_tool_call = True
        return result

    @staticmethod
    def _tool_call_result_to_dict(result: ToolCallResult) -> dict[str, Any]:
        """Convert a ToolCallResult to the legacy dict format used by the ReAct loop.

        This bridges the new strategy-based parsing with the existing loop
        logic that expects ``{'thought', 'action', 'action_input', 'final_answer'}``.
        """
        parsed: dict[str, Any] = {
            "thought": result.thought,
            "action": None,
            "action_input": None,
            "final_answer": result.final_answer,
        }
        if result.is_tool_call and result.tool_name:
            parsed["action"] = result.tool_name
            # Convert arguments dict back to the string form the loop expects
            if result.arguments:
                raw = result.arguments.get("__raw_input__")
                if raw:
                    parsed["action_input"] = raw
                else:
                    parsed["action_input"] = json.dumps(result.arguments)
            else:
                parsed["action_input"] = "{}"
        return parsed
