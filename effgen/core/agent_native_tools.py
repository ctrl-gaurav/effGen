"""Provider-native tool loops for :class:`Agent`.

Runs a task through a provider's own tool-calling API — the OpenAI Responses
API and the Gemini built-in tools — instead of the ReAct scratchpad: the
provider executes its tools, and the answer, the sources and the usage come
back from one call. Mixed into :class:`Agent` through
:class:`~effgen.core.agent_react.AgentReActMixin`.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..models._adapter_utils import default_max_output_tokens
from ..models.base import GenerationConfig
from ..models.errors import generation_failure_text
from . import ledger as _ledger
from .agent_config import AgentMode
from .agent_response import AgentResponse
from .agent_runtime import (
    find_written_tool_call,
    sanitize_final_answer,
    written_call_only,
)
from .tool_call_record import ToolCall, ToolCallList, truncate_result

# The native paths log to the ReAct stream they stand in for.
logger = logging.getLogger("effgen.core.agent_react")


class AgentNativeToolsMixin:
    """Provider-native tool-calling run paths for :class:`Agent`."""

    def _has_native_tools(self) -> bool:
        """Return True if any OpenAI native tool is in the tools list and the model supports Responses API."""
        try:
            from ..models.openai_adapter import OpenAIAdapter
            from ..tools.builtin.openai_native import OpenAINativeTool
        except ImportError:
            return False
        has_native = any(isinstance(t, OpenAINativeTool) for t in self.tools.values())
        is_openai = isinstance(self.model, OpenAIAdapter)
        return has_native and is_openai

    def _run_with_native_tools(self, task: str, context: dict[str, Any], **kwargs) -> AgentResponse:
        """Execute a task using OpenAI native tools via the Responses API.

        Separates native tools (web_search, code_interpreter, file_search) from
        local effGen tools.  Native tools are passed directly to the Responses API
        spec; local tools are serialised as function-call specs alongside them.
        """
        from ..models.openai_adapter import OpenAIAdapter
        from ..tools.builtin.openai_native import OpenAINativeTool

        native_specs: list[dict] = []
        function_specs: list[dict] = []

        for tool in self.tools.values():
            if isinstance(tool, OpenAINativeTool):
                native_specs.append(tool.to_openai_tool_spec())
            else:
                # Regular effGen tool — include as function call spec
                schema = tool.metadata.to_json_schema()
                function_specs.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["parameters"],
                    },
                })

        adapter: OpenAIAdapter = self.model  # already validated in _has_native_tools

        system_prompt = self.config.system_prompt if self.config.stable_system_prompt else None

        # The native-tool call honours the same per-run settings every other
        # path does; a reasoning model in particular needs the budget the
        # caller asked for, or it can spend the default on reasoning alone.
        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get(
                "max_tokens",
                self.config.max_tokens or default_max_output_tokens(self.model, base=2048),
            ),
            top_p=kwargs.get("top_p", self.config.top_p),
            seed=kwargs.get("seed", self.config.seed),
        )

        try:
            result = _ledger.timed_model_call(
                adapter.generate_with_native_tools, str(adapter.model_name or ""),
                prompt=task,
                native_tool_specs=native_specs,
                function_tool_specs=function_specs if function_specs else None,
                system_prompt=system_prompt,
                config=gen_config,
            )
        except Exception as e:
            logger.debug("Native tool generation failed", exc_info=True)
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

        # Handle any function tool calls that came back (local effGen tools)
        native_results = result.metadata.get("native_tool_results", [])
        # "function_call" is what the Responses API names a call to a local
        # tool; "function" is the name every other adapter reports. Built-in
        # server-side tools (web search, file search) use their own types and
        # are not dispatched here.
        local_calls = [
            r for r in native_results
            if r.get("type") in ("function_call", "function")
        ]

        tool_calls_made = len(native_results)

        # Execute local tool calls and stitch their results into a follow-up
        executed_tools: set[str] = set()
        calls: list[ToolCall] = []
        if local_calls and function_specs:
            observations: list[str] = []
            for call in local_calls:
                fn = call.get("function", call)
                fn_name = call.get("name") or fn.get("name", "")
                fn_args_raw = call.get("arguments", fn.get("arguments", "{}"))
                try:
                    fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                if fn_name in self.tools:
                    call_start = time.time()
                    obs = self._execute_tool(fn_name, json.dumps(fn_args))
                    tool_calls_made += 1
                    executed_tools.add(fn_name)
                    calls.append(ToolCall(
                        name=fn_name, arguments=fn_args,
                        result=truncate_result(obs),
                        duration=time.time() - call_start, iteration=1,
                    ))
                    observations.append(f"[{fn_name}({fn_args})] → {obs}")

            if observations and not result.text:
                # Re-prompt with the observations to get a final answer
                obs_text = "\n".join(observations)
                followup = f"Tool results:\n{obs_text}\n\nBased on these results, answer the user's question: {task}"
                try:
                    followup_result = _ledger.timed_model_call(
                        adapter.generate, str(adapter.model_name or ""), followup,
                    )
                    return AgentResponse(
                        output=sanitize_final_answer(followup_result.text) or followup_result.text,
                        success=True,
                        mode=AgentMode.SINGLE,
                        iterations=2,
                        tool_calls=ToolCallList(calls, total=tool_calls_made),
                        tokens_used=result.tokens_used + followup_result.tokens_used,
                        metadata=result.metadata,
                    )
                except Exception:
                    logger.debug("Native tool follow-up assembly failed; using prior result", exc_info=True)

        if not result.text and (result.metadata or {}).get("reasoning_only"):
            return self._reasoning_only_native_response(result, tool_calls_made, calls)

        answer = sanitize_final_answer(result.text) or ""
        # Sanitizing a tagged call can leave its arguments behind as a bare JSON
        # fragment, so the text as the model wrote it is scanned as well.
        written = None
        if result.text:
            written = find_written_tool_call(
                answer, self.tools
            ) or find_written_tool_call(result.text, self.tools)
        if written and (
            written not in executed_tools or written_call_only(result.text, self.tools)
        ):
            return self._written_tool_call_response(
                written,
                result.text,
                iterations=1,
                tool_calls=ToolCallList(calls, total=tool_calls_made),
                tokens_used=result.tokens_used,
                tool_ran=written in executed_tools,
            )
        meta = dict(result.metadata or {})
        meta.setdefault("tool_calling_strategy", "openai_native")
        return AgentResponse(
            output=answer or "(no output from native tools call)",
            success=bool(result.text),
            mode=AgentMode.SINGLE,
            iterations=1,
            tool_calls=ToolCallList(calls, total=tool_calls_made),
            tokens_used=result.tokens_used,
            metadata=meta,
        )

    def _reasoning_only_native_response(
        self, result: Any, tool_calls_made: int,
        calls: "list[ToolCall] | None" = None,
    ) -> AgentResponse:
        """Failure response for a native-tool turn that produced only reasoning.

        The adapter already worked out why the answer is empty and named the cap
        and the reasoning budget; report that instead of stating only that there
        was no output.
        """
        meta = result.metadata or {}
        detail = {
            "type": "ReasoningOnlyResponse",
            "category": "reasoning_only",
            "provider": self._model_provider(self.model),
            "model": getattr(self.model, "model_name", None) or self.model_name or "unknown",
            "message": meta.get("empty_response_reason") or (
                "The model produced only internal reasoning and no answer."
            ),
            "reasoning_tokens": meta.get("reasoning_tokens", 0),
            "retryable": False,
        }
        return AgentResponse(
            output=generation_failure_text(detail),
            success=False,
            mode=AgentMode.SINGLE,
            iterations=1,
            tool_calls=ToolCallList(calls or [], total=tool_calls_made),
            tokens_used=result.tokens_used,
            metadata={"reason": "generation_failed", "error": detail},
        )

    def _has_gemini_native_tools(self) -> bool:
        """Return True if any Gemini native tool is in the tools list and model is Gemini."""
        try:
            from ..models.gemini_adapter import GeminiAdapter
            from ..tools.builtin.gemini_native import GeminiNativeTool
        except ImportError:
            return False
        has_native = any(isinstance(t, GeminiNativeTool) for t in self.tools.values())
        is_gemini = isinstance(self.model, GeminiAdapter)
        return has_native and is_gemini

    def _run_with_gemini_native_tools(self, task: str, context: dict[str, Any], **kwargs) -> AgentResponse:
        """Execute a task using Gemini server-side native tools.

        Passes ``GeminiNativeTool`` objects (google_search, url_context, code_execution)
        directly to the Gemini adapter alongside any local effGen tools serialised
        as function declarations.  The adapter routes them correctly to the SDK.
        """
        from ..tools.builtin.gemini_native import GeminiNativeTool

        # Separate native vs. local tools
        native_tools: list = []
        function_tool_specs: list[dict] = []
        for tool in self.tools.values():
            if isinstance(tool, GeminiNativeTool):
                native_tools.append(tool)
            else:
                schema = tool.metadata.to_json_schema()
                function_tool_specs.append({
                    "type": "function",
                    "function": {
                        "name": schema["name"],
                        "description": schema["description"],
                        "parameters": schema["parameters"],
                    },
                })

        # Combine: native tool objects first, then regular function specs
        all_tools = native_tools + function_tool_specs
        mixed = bool(native_tools and function_tool_specs)

        gen_config = GenerationConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get(
                "max_tokens",
                self.config.max_tokens or default_max_output_tokens(self.model, base=2048),
            ),
            top_p=kwargs.get("top_p", self.config.top_p),
            seed=kwargs.get("seed", self.config.seed),
        )

        system_note = ""
        if self.config.system_prompt:
            system_note = f"{self.config.system_prompt}\n\n"

        prompt = f"{system_note}{task}"

        try:
            result = _ledger.timed_model_call(
                self.model.generate, str(getattr(self.model, "model_name", "") or ""),
                prompt, config=gen_config, tools=all_tools,
            )
        except Exception as exc:
            logger.debug("Gemini native tool generation failed", exc_info=True)
            detail = self._build_error_detail(exc, self.model)
            # Most Gemini models reject combining a built-in tool (google_search,
            # url_context, code_execution) with custom function tools in one
            # request ("cannot be combined" / "context circulation is not
            # enabled"). Translate that raw 400 into an actionable remediation
            # instead of a bare provider error.
            low = detail.get("message", "").lower()
            if mixed and ("cannot be combined" in low or "circulation" in low
                          or "function calling" in low):
                native_names = ", ".join(t.name for t in native_tools)
                detail["message"] = (
                    f"Gemini model '{self.model_name}' cannot use its built-in tool(s) "
                    f"({native_names}) together with custom function tools in a single "
                    "request. Build the agent with EITHER the native tool(s) OR your "
                    "custom tools — not both — or use a model that supports tool-call "
                    "context circulation."
                )
                detail["category"] = "invalid_request"
                detail["retryable"] = False
            return AgentResponse(
                output=generation_failure_text(detail),
                success=False,
                mode=AgentMode.SINGLE,
                iterations=1,
                tool_calls=ToolCallList(),
                tokens_used=0,
                metadata={"reason": "generation_failed", "error": detail},
            )

        tool_calls_made = len(result.metadata.get("tool_calls") or [])

        # Execute any local effGen function calls that came back
        local_tc = result.metadata.get("tool_calls") or []
        observations: list[str] = []
        executed_tools: set[str] = set()
        calls: list[ToolCall] = []
        for tc in local_tc:
            # Adapters report a call as {"function": {"name", "arguments"}};
            # an adapter that also carries top-level keys is read the same way.
            fn = tc.get("function", tc)
            fn_name = tc.get("name") or fn.get("name", "")
            fn_args = tc.get("arguments", fn.get("arguments", {}))
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except (json.JSONDecodeError, TypeError):
                    fn_args = {"__raw_input__": fn_args}
            if fn_name in self.tools and not isinstance(self.tools[fn_name], GeminiNativeTool):
                call_start = time.time()
                obs = self._execute_tool(fn_name, json.dumps(fn_args) if isinstance(fn_args, dict) else fn_args)
                executed_tools.add(fn_name)
                calls.append(ToolCall(
                    name=fn_name, arguments=fn_args,
                    result=truncate_result(obs),
                    duration=time.time() - call_start, iteration=1,
                ))
                observations.append(f"[{fn_name}({fn_args})] → {obs}")

        # The adapter also encodes each call as a <tool_call> block in the
        # generated text, so a turn that did nothing but call tools is not
        # empty — it is empty once those blocks are stripped. Testing the raw
        # text would skip the follow-up and report the answer as missing while
        # the tool results sit unused.
        answer = sanitize_final_answer(result.text) or ""

        if observations and not answer.strip():
            obs_text = "\n".join(observations)
            followup = f"Tool results:\n{obs_text}\n\nBased on these results, answer: {task}"
            try:
                followup_result = _ledger.timed_model_call(
                    self.model.generate, str(getattr(self.model, "model_name", "") or ""),
                    followup, config=gen_config,
                )
                followup_answer = (
                    sanitize_final_answer(followup_result.text) or followup_result.text
                )
                # A follow-up that came back empty answers nothing; fall through
                # rather than report an empty output as a successful turn.
                if followup_answer and followup_answer.strip():
                    followup_meta = dict(result.metadata or {})
                    followup_meta.setdefault("tool_calling_strategy", "gemini_native")
                    return AgentResponse(
                        output=followup_answer,
                        success=True,
                        mode=AgentMode.SINGLE,
                        iterations=2,
                        tool_calls=ToolCallList(calls, total=tool_calls_made),
                        tokens_used=result.tokens_used + followup_result.tokens_used,
                        metadata=followup_meta,
                    )
            except Exception:
                logger.debug("Native tool follow-up assembly failed; using prior result", exc_info=True)

        if not result.text and (result.metadata or {}).get("reasoning_only"):
            return self._reasoning_only_native_response(result, tool_calls_made, calls)

        # Sanitizing a tagged call can leave its arguments behind as a bare JSON
        # fragment, so the text as the model wrote it is scanned as well.
        written = None
        if result.text:
            written = find_written_tool_call(
                answer, self.tools
            ) or find_written_tool_call(result.text, self.tools)
        if written and (
            written not in executed_tools or written_call_only(result.text, self.tools)
        ):
            return self._written_tool_call_response(
                written,
                result.text,
                iterations=1,
                tool_calls=ToolCallList(calls, total=tool_calls_made),
                tokens_used=result.tokens_used,
                tool_ran=written in executed_tools,
            )
        meta = dict(result.metadata or {})
        meta.setdefault("tool_calling_strategy", "gemini_native")
        return AgentResponse(
            output=answer or "(no output from Gemini native tools call)",
            # A turn whose whole text was a tool-call block leaves nothing to
            # report once it is stripped; that is a failed turn, not a success
            # carrying a placeholder.
            success=bool(answer),
            mode=AgentMode.SINGLE,
            iterations=1,
            tool_calls=ToolCallList(calls, total=tool_calls_made),
            tokens_used=result.tokens_used,
            metadata=meta,
        )
