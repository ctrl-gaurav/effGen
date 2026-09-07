"""Tool dispatch for :class:`Agent`.

Executes ReAct tool calls with input sanitization, circuit-breaker and
fallback handling, parameter mapping, and result normalization. Mixed into
:class:`Agent` through :class:`~effgen.core.agent_react.AgentReActMixin`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .execution_tracker import EventType, ExecutionEvent

logger = logging.getLogger(__name__)


class AgentToolExecutionMixin:
    """Tool-execution and parameter-mapping methods for :class:`Agent`."""

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool, with any configured middleware wrapped around it.

        A ``before_tool_call`` hook may rewrite the input, or answer the call
        itself — an approval gate refusing, a cache hit — in which case the tool
        does not run. ``after_tool_call`` sees whichever output was produced.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool (JSON string or plain text)

        Returns:
            Tool output as string
        """
        chain = self._middleware_chain()
        if not chain:
            return self._execute_tool_guarded(tool_name, tool_input)

        from .middleware import ToolCallContext

        ctx = ToolCallContext(
            tool_name=tool_name,
            tool_input=tool_input,
            run=getattr(self, "_active_run_context", None),
        )
        short_circuit = chain.before_tool_call(ctx)
        if short_circuit is not None:
            answered: str = chain.after_tool_call(ctx, short_circuit)
            return answered
        result = self._execute_tool_guarded(ctx.tool_name, ctx.tool_input)
        final: str = chain.after_tool_call(ctx, result)
        return final

    def _middleware_chain(self):
        """Return the middleware in force, per-call ones included.

        ``_active_middleware`` is set for the duration of one ``run()`` that was
        given ``middleware=``; otherwise the agent's configured chain applies.
        """
        return (
            getattr(self, "_active_middleware", None)
            or getattr(self, "_middleware", None)
        )

    def _execute_tool_guarded(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool with circuit breaker, fallback support, and input sanitization.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool (JSON string or plain text)

        Returns:
            Tool output as string
        """
        # Sanitize input
        tool_input = self._sanitize_tool_input(tool_input)

        # Pre-tool guardrail check (TOOL_INPUT)
        if self._guardrail_chain is not None:
            from ..guardrails.base import GuardrailPosition as _GP
            tool_obj = self.tools.get(tool_name)
            gr = self._guardrail_chain.check(
                tool_input, position=_GP.TOOL_INPUT,
                tool_name=tool_name, tool=tool_obj,
            )
            if not gr.passed:
                logger.info(f"Guardrail blocked tool '{tool_name}': {gr.reason}")
                return f"Error executing tool '{tool_name}': blocked by guardrail — {gr.reason}"
            if gr.modified_content is not None:
                tool_input = gr.modified_content

        # Human-in-the-loop approval check
        tool_obj = self.tools.get(tool_name)
        _requires_approval = getattr(
            getattr(tool_obj, '_metadata', None), 'requires_approval', False
        ) if tool_obj else False
        if self._approval_manager.should_request_approval(tool_name, _requires_approval):
            from .human_loop import ApprovalDecision
            decision = self._approval_manager.request_approval(tool_name, tool_input)
            if decision != ApprovalDecision.APPROVED:
                logger.info("Tool '%s' denied by human approval (%s)", tool_name, decision.value)
                return f"Error executing tool '{tool_name}': execution denied by human approval ({decision.value})"

        # Circuit breaker check
        if not self._circuit_breaker.is_available(tool_name):
            logger.info(f"Circuit breaker OPEN for '{tool_name}', skipping execution")
            return f"Error executing tool '{tool_name}': tool temporarily disabled due to repeated failures"

        result_str = self._execute_tool_once(tool_name, tool_input)

        # Update circuit breaker
        if result_str.startswith("Error executing tool"):
            self._circuit_breaker.record_failure(tool_name)
        else:
            self._circuit_breaker.record_success(tool_name)

        # Check if primary tool failed and fallback is enabled
        if (
            self._enable_fallback
            and result_str.startswith("Error executing tool")
            and self._fallback_chain.has_fallbacks(tool_name)
        ):
            fallbacks = self._fallback_chain.get_fallbacks(tool_name)
            for fb_name in fallbacks:
                if fb_name not in self.tools:
                    continue
                logger.info(f"Tool '{tool_name}' failed, trying fallback: {fb_name}")
                fb_result = self._execute_tool_once(fb_name, tool_input)
                if not fb_result.startswith("Error executing tool"):
                    logger.info(f"Fallback '{fb_name}' succeeded for '{tool_name}'")
                    return f"[Fallback: used {fb_name} instead of {tool_name}] {fb_result}"
            logger.info(f"All fallbacks exhausted for '{tool_name}'")

        # Post-tool guardrail check (TOOL_OUTPUT)
        if self._guardrail_chain is not None and not result_str.startswith("Error executing tool"):
            from ..guardrails.base import GuardrailPosition as _GP
            gr = self._guardrail_chain.check(
                result_str, position=_GP.TOOL_OUTPUT,
                tool_name=tool_name,
            )
            if not gr.passed:
                logger.info(f"Guardrail blocked output from '{tool_name}': {gr.reason}")
                return f"Error executing tool '{tool_name}': output blocked by guardrail — {gr.reason}"
            if gr.modified_content is not None:
                result_str = gr.modified_content

        return result_str

    def _execute_tool_once(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a single tool, without consulting the fallback chain.

        A failure is returned as an ``Error executing tool '<name>': ...``
        string rather than raised, so the loop can act on it.

        Args:
            tool_name: Name of tool to execute
            tool_input: Input for the tool (JSON string or plain text)

        Returns:
            Tool output as string
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

            # Provider-native tools (OpenAI web_search / Gemini google_search /
            # Anthropic computer-use, …) run server-side and cannot be executed
            # locally.  If we end up here it means the ReAct loop tried to call one
            # directly (e.g. "Action: openai_web_search") — either because the model
            # isn't the matching provider, or the native dispatch was bypassed.
            # Return a helpful note so the loop can recover instead of
            # surfacing a raw "cannot be executed locally" RuntimeError.
            native_hint = self._native_tool_loop_hint(tool, tool_name)
            if native_hint is not None:
                return native_hint

            # Parse input intelligently
            input_dict = {}
            if tool_input:
                try:
                    # Try parsing as JSON first (after cleaning SLM artifacts)
                    cleaned = self._clean_json_input(tool_input)
                    input_dict = json.loads(cleaned)
                    if not isinstance(input_dict, dict):
                        # JSON parsed but not a dict - need to intelligently map to tool parameters
                        input_dict = self._map_input_to_parameters(tool, input_dict)
                except json.JSONDecodeError:
                    # Not valid JSON — SLMs often produce Python-style dicts with
                    # single-quoted strings (e.g. {"data": '{"key": "val"}'}).
                    # Try ast.literal_eval as a fallback before plain-text mapping.
                    try:
                        import ast
                        parsed = ast.literal_eval(tool_input)
                        if isinstance(parsed, dict):
                            input_dict = parsed
                        else:
                            input_dict = self._map_input_to_parameters(tool, tool_input)
                    except (ValueError, SyntaxError):
                        # Not valid Python either — use plain text mapping
                        input_dict = self._map_input_to_parameters(tool, tool_input)
                except Exception as e:
                    logger.warning(f"Error parsing tool input, using as plain text: {e}")
                    input_dict = self._map_input_to_parameters(tool, tool_input)

            # Strip markdown code fences from 'code' param even after JSON parse
            if isinstance(input_dict, dict) and 'code' in input_dict and isinstance(input_dict['code'], str):
                code_val = input_dict['code']
                if '```' in code_val:
                    import re as _re
                    code_val = _re.sub(r'^```(?:python|py|javascript|js|bash|sh)?\n?', '', code_val, flags=_re.MULTILINE)
                    code_val = _re.sub(r'\n?```$', '', code_val, flags=_re.MULTILINE)
                    input_dict['code'] = code_val.strip()

            logger.debug(f"Executing tool '{tool_name}' with input: {input_dict}")

            # Execute tool (handle both sync and async)
            try:
                result = tool.execute(**input_dict)

                # Await a coroutine result before using it
                if asyncio.iscoroutine(result):
                    result = self._run_coroutine_sync(result)

            except TypeError as e:
                logger.error(f"Tool parameter error: {e}")
                raise ValueError(
                    f"Tool '{tool_name}' parameter error: {str(e)}. "
                    f"Input provided: {input_dict}"
                )

            # Capture retrieved evidence (RAG/search) so the final answer can
            # surface its sources and inline citations (AgentResponse.sources /
            # .citations). Best-effort: never let bookkeeping break tool output.
            mined_passages: list[dict[str, Any]] = []
            try:
                mined_passages = self._collect_citations(tool, tool_name, result) or []
            except Exception as _cite_err:  # pragma: no cover - defensive
                logger.debug("Citation capture skipped for %s: %s", tool_name, _cite_err)

            # Convert result to string safely
            if result is None:
                result_str = "No result returned"
            elif hasattr(result, 'output'):
                # ToolResult object - extract output
                if hasattr(result, 'success') and not result.success:
                    error_msg = str(getattr(result, 'error', 'Unknown error'))
                    # A tool that raised already carries this prefix on its
                    # error; adding a second one repeats it in the observation.
                    result_str = (
                        error_msg
                        if error_msg.startswith("Tool execution failed:")
                        else f"Tool execution failed: {error_msg}"
                    )
                else:
                    output = result.output
                    if isinstance(output, dict):
                        # Try common result keys: result, output, data, message
                        # PythonREPL returns {result: None, stdout: "..."}
                        # when code uses print(). Prefer stdout over a None result.
                        if 'result' in output and output['result'] is not None:
                            result_str = str(output['result'])
                        elif 'stdout' in output and output['stdout']:
                            # PythonREPL/CodeExecutor: stdout has the printed output
                            parts = []
                            parts.append(output['stdout'].rstrip())
                            if output.get('stderr'):
                                parts.append(f"stderr: {output['stderr'].rstrip()}")
                            if output.get('error'):
                                parts.append(f"Error: {output['error']}")
                            result_str = '\n'.join(parts)
                        elif 'stderr' in output and output['stderr']:
                            # CodeExecutor error: stdout empty but stderr has traceback
                            result_str = f"Error: {output['stderr'].rstrip()}"
                            if output.get('exit_code'):
                                result_str += f"\n(exit code: {output['exit_code']})"
                        elif 'error' in output and output['error']:
                            result_str = f"Error: {output['error']}"
                        elif 'exit_code' in output:
                            # CodeExecutor: ran successfully but no output
                            result_str = f"Code executed successfully (exit code {output['exit_code']})"
                        elif 'output' in output:
                            result_str = str(output['output'])
                        elif 'data' in output and 'success' in output:
                            # FileOperations-style: {success, data, message}
                            if output.get('success'):
                                result_str = str(output['data']) if output['data'] is not None else output.get('message', str(output))
                            else:
                                result_str = f"Operation failed: {output.get('message', str(output))}"
                        elif 'data' in output:
                            result_str = str(output['data'])
                        elif 'message' in output:
                            result_str = str(output['message'])
                        else:
                            result_str = str(output)
                    else:
                        result_str = str(output)
            elif hasattr(result, 'result'):
                result_str = str(result.result)
            else:
                result_str = str(result)

            # Record what this run retrieved. The two counters gate the answer
            # side: an answer only loses a trailing "[1]" when a context
            # observation was appended, and only for an index that observation
            # could have offered. Best-effort, like the citation capture above.
            try:
                if self._is_context_retrieval_tool(tool_name):
                    self._retrieval_observations += 1
                    self._retrieval_passages += len(mined_passages)
                    if mined_passages and self._cite_sources_requested():
                        # Show the passages the way the markers index them, so
                        # "numbered by its order above" is a real list and not
                        # something the model has to count for itself.
                        numbered = self._numbered_passage_block(mined_passages)
                        if numbered:
                            result_str = numbered
            except Exception as _ret_err:  # pragma: no cover - defensive
                logger.debug("Retrieval bookkeeping skipped for %s: %s", tool_name, _ret_err)

            # Check if result indicates a tool-level failure
            if result_str.startswith("Tool execution failed:"):
                raise ValueError(result_str)

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
            error_msg = str(e)
            logger.debug(f"Tool '{tool_name}' execution failed: {error_msg}")

            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_FAILED,
                agent_id=self.name,
                message=f"Tool {tool_name} failed: {error_msg}",
                data={"tool_name": tool_name, "error": error_msg, "input": tool_input}
            ))

            return f"Error executing tool '{tool_name}': {error_msg}"

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Tool '{tool_name}' execution failed: {error_msg}", exc_info=True)

            self.execution_tracker.track_event(ExecutionEvent(
                type=EventType.TOOL_CALL_FAILED,
                agent_id=self.name,
                message=f"Tool {tool_name} failed: {error_msg}",
                data={"tool_name": tool_name, "error": error_msg, "input": tool_input}
            ))

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

            # Clean up path - remove trailing whitespace but preserve absolute paths
            if result.get("path"):
                result["path"] = result["path"].strip()
                # Remove file:// prefix if still present after earlier cleanup
                if result["path"].startswith("file://"):
                    result["path"] = result["path"][7:]
                    # Ensure we keep at least one leading /
                    if not result["path"].startswith("/"):
                        result["path"] = "/" + result["path"]

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

    @staticmethod
    def _native_tool_loop_hint(tool: Any, tool_name: str) -> str | None:
        """Return a recovery note if *tool* is a provider-native server-side tool.

        Native tools (OpenAI web_search/code_interpreter/file_search, Gemini
        google_search/url_context/code_execution, Anthropic computer-use) execute
        inside the provider's infrastructure and cannot run locally.  When one is
        reached inside the ReAct loop we return an actionable hint instead of the
        raw ``RuntimeError`` from the tool's local ``_execute`` so the loop (and
        the user) get a clear next step.  Returns ``None`` for ordinary tools.
        """
        specs = (
            ("openai_native", "OpenAINativeTool", "OpenAI", "an OpenAI model"),
            ("gemini_native", "GeminiNativeTool", "Google", "a Gemini model"),
            ("anthropic_native", "AnthropicNativeTool", "Anthropic", "an Anthropic model"),
        )
        for module, cls_name, vendor, model_hint in specs:
            try:
                mod = __import__(
                    f"effgen.tools.builtin.{module}", fromlist=[cls_name]
                )
                native_cls = getattr(mod, cls_name)
            except (ImportError, AttributeError):
                continue
            if isinstance(tool, native_cls):
                return (
                    f"Tool '{tool_name}' is {vendor}'s server-side tool and cannot be "
                    f"executed locally in the ReAct loop. Pair it with {model_hint} and "
                    "use tool_calling_mode='native' so the adapter routes it to the "
                    f"provider; otherwise remove '{tool_name}' from this agent."
                )
        return None
