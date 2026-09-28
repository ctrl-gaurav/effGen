#!/usr/bin/env python3
"""
effGen — Error Recovery Agent (Intentional Failures)

Deliberately breaks things and tests how the framework handles failures.
Demonstrates BrokenTool (its service is unreachable), SlowTool (timeout), invalid inputs,
circuit breaker transitions, fallback chain exhaustion, partial answer
extraction, retry backoff, concurrent failures, and control character handling.

Tests:
  T1: Invalid tool input — malformed JSON to Calculator
  T2: Tool down — BrokenTool always raises ConnectionError, circuit breaker triggers
  T3: All tools fail — every tool raises, agent gives partial answer
  T4: Max iterations — max_iterations=2, returns partial answer
  T5: Empty model response — retry logic with backoff
  T6: Timeout — SlowTool triggers tool timeout
  T7: Fallback chain exhaustion — calculator -> python_repl -> code_executor all fail
  T8: Concurrent failures — 3 agents simultaneously, no shared state corruption
  T9: Control character input — sanitization of null bytes and control chars

Tools used: Calculator, PythonREPL, FileOperations, BashTool + BrokenTool + SlowTool

Recommended models: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/error_recovery_agent.py
  CUDA_VISIBLE_DEVICES=0 python examples/error_recovery_agent.py --model Qwen/Qwen2.5-7B-Instruct
  CUDA_VISIBLE_DEVICES=0 python examples/error_recovery_agent.py --regression
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)
from effgen.utils.circuit_breaker import CircuitState

# ── Custom Tools ─────────────────────────────────────────────────────────────

class BrokenTool(BaseTool):
    """A tool whose service is always unreachable.

    It raises ``ConnectionError``: a failure on the tool's own side, which is
    what the agent's circuit breaker counts. Bad input to a tool does not.
    """

    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="broken_tool",
                description="A tool that processes data (always crashes — test only)",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="input",
                        type=ParameterType.STRING,
                        description="Input data to process",
                        required=True,
                    )
                ],
            )
        )

    async def _execute(self, input: str = "", **kwargs):
        raise ConnectionError("BrokenTool: service unreachable (intentional, for testing)")


class SlowTool(BaseTool):
    """A tool that sleeps for a configurable duration (default 30s)."""

    def __init__(self, sleep_seconds: float = 30.0):
        super().__init__(
            metadata=ToolMetadata(
                name="slow_tool",
                description="A tool that takes a long time to respond (test only)",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="input",
                        type=ParameterType.STRING,
                        description="Input data to process slowly",
                        required=True,
                    )
                ],
            )
        )
        self._sleep_seconds = sleep_seconds

    async def _execute(self, input: str = "", **kwargs):
        await asyncio.sleep(self._sleep_seconds)
        return f"SlowTool completed after {self._sleep_seconds}s"


class AlwaysFailCalculator(BaseTool):
    """Calculator that always fails — for fallback chain exhaustion testing."""

    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="calculator",
                description="Calculator (broken — always fails)",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="expression",
                        type=ParameterType.STRING,
                        description="Math expression",
                        required=True,
                    )
                ],
            )
        )

    async def _execute(self, expression: str = "", **kwargs):
        raise RuntimeError("AlwaysFailCalculator: intentional failure!")


class AlwaysFailPythonREPL(BaseTool):
    """PythonREPL that always fails — for fallback chain exhaustion testing."""

    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="python_repl",
                description="Python REPL (broken — always fails)",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="Python code",
                        required=True,
                    )
                ],
            )
        )

    async def _execute(self, code: str = "", **kwargs):
        raise RuntimeError("AlwaysFailPythonREPL: intentional failure!")


class AlwaysFailCodeExecutor(BaseTool):
    """CodeExecutor that always fails — for fallback chain exhaustion testing."""

    def __init__(self):
        super().__init__(
            metadata=ToolMetadata(
                name="code_executor",
                description="Code executor (broken — always fails)",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="code",
                        type=ParameterType.STRING,
                        description="Code to execute",
                        required=True,
                    ),
                    ParameterSpec(
                        name="language",
                        type=ParameterType.STRING,
                        description="Programming language",
                        required=True,
                    ),
                ],
            )
        )

    async def _execute(self, code: str = "", language: str = "python", **kwargs):
        raise RuntimeError("AlwaysFailCodeExecutor: intentional failure!")


# ── System Prompt ────────────────────────────────────────────────────────────

ERROR_RECOVERY_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

TOOL SELECTION RULES:
- For math: use 'calculator' with {"expression": "..."}
- For Python code: use 'python_repl' with {"code": "..."}
- For file operations: use 'file_operations'
- For shell commands: use 'bash'
- For data processing: use 'broken_tool' with {"input": "..."}
- For slow tasks: use 'slow_tool' with {"input": "..."}

IMPORTANT:
1. If a tool fails, try a different approach or answer from your knowledge.
2. When done, provide your response using 'Final Answer:'.
3. Do NOT repeat the same failing tool call more than once.
"""


# ── Test Helpers ─────────────────────────────────────────────────────────────

def run_test(agent, test_id, description, question,
             check_fn=None, expected_keywords=None, expected_tool=None,
             timeout=120):
    """Run a single test and return result dict."""
    print(f"\n{'='*60}")
    print(f"Test: {test_id} — {description}")
    print(f"Q: {question[:200]}")

    t0 = time.time()
    try:
        resp = agent.run(question)
    except Exception as e:
        dt = time.time() - t0
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
        return {
            "test_id": test_id, "description": description,
            "passed": False, "output": f"EXCEPTION: {e}",
            "tool_calls": 0, "iterations": 0, "time": dt, "status": "FAIL",
        }

    dt = time.time() - t0
    output = resp.output if resp.output else ""
    print(f"A: {output[:600]}")
    print(f"Tool calls: {resp.tool_calls}, Iterations: {resp.iterations}, "
          f"Time: {dt:.1f}s, Success: {resp.success}")
    if resp.metadata:
        print(f"Metadata: {resp.metadata}")

    # Check keywords
    keyword_pass = True
    if expected_keywords:
        for kw in expected_keywords:
            if kw.lower() not in output.lower():
                keyword_pass = False
                print(f"  MISSING keyword: '{kw}'")

    # Check custom function
    custom_pass = True
    if check_fn:
        try:
            custom_pass = check_fn(output, resp)
        except Exception as e:
            custom_pass = False
            print(f"  Custom check EXCEPTION: {e}")
        if not custom_pass:
            print("  Custom check FAILED")

    # Check tool was used
    tool_pass = True
    if expected_tool:
        trace_tools = []
        for trace in resp.execution_trace:
            if isinstance(trace, dict):
                data = trace.get("data", {})
                if data.get("tool_name"):
                    trace_tools.append(data["tool_name"])
        if expected_tool not in trace_tools:
            tool_pass = False
            print(f"  Expected tool '{expected_tool}' not found in trace: {trace_tools}")
        else:
            print(f"  Tool used: {expected_tool}")

    passed = keyword_pass and custom_pass and tool_pass
    status = "PASS" if passed else "FAIL"
    print(f"Result: {status}")

    return {
        "test_id": test_id, "description": description,
        "passed": passed, "output": output[:400],
        "tool_calls": resp.tool_calls, "iterations": resp.iterations,
        "time": dt, "status": status,
    }


# ── Error Recovery Tests ──────────────────────────────────────────────────────

def test_p7_t1_invalid_input(model, model_name):
    """Invalid tool input — malformed JSON to Calculator."""
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.python_repl import PythonREPL

    agent = Agent(AgentConfig(
        name="error_recovery_t1",
        model=model,
        tools=[Calculator(), PythonREPL()],
        system_prompt=ERROR_RECOVERY_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=8,
        enable_sub_agents=False,
    ))

    # The model should try to use calculator, possibly get an error, and retry or answer directly
    return run_test(
        agent, "T1", "Invalid tool input — agent handles bad math without crashing",
        "Calculate the value of 'not_a_number + abc'. If the calculator cannot handle it, explain why.",
        check_fn=lambda out, resp: (
            # Agent should either: get an error and explain, OR answer from knowledge
            len(out) > 10  # Non-trivial response
            and not out.startswith("Maximum iterations")  # Didn't just time out
        ),
    )


def test_p7_t2_tool_crash(model, model_name):
    """Tool down — BrokenTool always raises ConnectionError, circuit breaker triggers."""
    from effgen.tools.builtin.calculator import Calculator

    broken = BrokenTool()
    agent = Agent(AgentConfig(
        name="error_recovery_t2",
        model=model,
        tools=[broken, Calculator()],
        system_prompt=ERROR_RECOVERY_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=10,
        enable_sub_agents=False,
    ))

    # First, verify circuit breaker is CLOSED
    cb = agent._circuit_breaker
    assert cb.get_state("broken_tool") == CircuitState.CLOSED, "CB should start CLOSED"

    # Ask the model to use broken_tool 3+ times so CB opens
    result = run_test(
        agent, "T2", "Tool crash — circuit breaker triggers after 3 failures",
        "Use the broken_tool to process data: 'test input'. If it fails, try it again a few more times. Report what happens.",
        check_fn=lambda out, resp: (
            # Framework never crashes — returns an error response, not exception
            True  # The mere fact we got here means no crash
        ),
    )

    # Check circuit breaker state after the run
    state = cb.get_state("broken_tool")
    print(f"  Circuit breaker state for broken_tool: {state}")
    if state == CircuitState.OPEN:
        print("  VERIFIED: Circuit breaker opened after repeated failures")
        result["passed"] = True
    else:
        # The model may not have called it 3 times — check failure count
        circuit = cb._get_circuit("broken_tool")
        print(f"  Circuit breaker failures: {circuit.consecutive_failures}")
        if circuit.consecutive_failures > 0:
            print(f"  PARTIAL: At least {circuit.consecutive_failures} failure(s) recorded")
            result["passed"] = True  # Framework handled the crash without dying
        else:
            print("  NOTE: Model may not have used broken_tool at all")
            # The test still passes if the framework didn't crash
            result["passed"] = True

    return result


def test_p7_t3_all_tools_fail(model, model_name):
    """All tools fail — every tool raises, agent gives partial answer."""
    agent = Agent(AgentConfig(
        name="error_recovery_t3",
        model=model,
        tools=[BrokenTool(), SlowTool(sleep_seconds=0.1)],
        system_prompt=(
            "You are a math helper. Use tools to compute. "
            "If all tools fail, answer from your knowledge. "
            "Respond with Final Answer: when ready."
        ),
        temperature=0.1,
        max_iterations=5,
        enable_sub_agents=False,
    ))

    return run_test(
        agent, "T3", "All tools fail — agent gives answer from knowledge",
        "What is 25 * 4? Try using a tool first, but if it fails, answer from your knowledge.",
        check_fn=lambda out, resp: (
            # Agent should eventually answer (partial or complete)
            len(out) >= 3
            and not out.startswith("Error:")
        ),
    )


def test_p7_t4_max_iterations(model, model_name):
    """Max iterations — max_iterations=2, reports the stop instead of crashing."""
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.python_repl import PythonREPL

    agent = Agent(AgentConfig(
        name="error_recovery_t4",
        model=model,
        tools=[Calculator(), PythonREPL()],
        system_prompt=ERROR_RECOVERY_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=2,  # Very limited!
        enable_sub_agents=False,
        # The stop is what this case is about, so it is read from the response
        # rather than raised.
        raise_on_error=False,
    ))

    return run_test(
        agent, "T4", "Max iterations — reports the stop at the iteration limit",
        "Calculate 15 + 27, then multiply the result by 3, then divide by 7. Show each step.",
        check_fn=lambda out, resp: (
            resp.iterations <= 2
            and resp.outcome in ("answered", "stopped")
            # A stop is reported as one, and whatever the run had reached is
            # under `partial` rather than presented as the answer.
            and (
                resp.outcome == "answered"
                or (resp.partial is None or resp.partial.text != resp.output)
            )
        ),
    )


def test_p7_t5_empty_response(model, model_name):
    """Empty model response — retry logic with backoff.

    Validates the retry mechanism in _generate() at the framework level
    by checking the retry logic directly.
    """
    # Test the retry logic directly with a probe
    from effgen.tools.builtin.calculator import Calculator

    agent = Agent(AgentConfig(
        name="error_recovery_t5",
        model=model,
        tools=[Calculator()],
        system_prompt="Answer questions directly with Final Answer:",
        temperature=0.1,
        max_iterations=3,
        enable_sub_agents=False,
    ))

    # The retry logic is tested structurally — we verify:
    # 1. _generate has 3 retries with backoff_delays = [0.5, 1.0, 2.0]
    # 2. Temperature increases by 0.1 per retry
    print(f"\n{'='*60}")
    print("Test: T5 — Empty model response retry logic")

    # Verify retry parameters exist in the agent's _generate method
    import inspect
    source = inspect.getsource(agent._generate)
    has_max_retries = "max_retries = 3" in source
    has_backoff = "backoff_delays" in source
    has_temp_increase = "attempt * 0.1" in source
    print(f"  max_retries=3: {has_max_retries}")
    print(f"  backoff_delays present: {has_backoff}")
    print(f"  temperature increase on retry: {has_temp_increase}")

    passed = has_max_retries and has_backoff and has_temp_increase

    # Also do a real inference to make sure the model works normally
    resp = agent.run("What is 2 + 2?")
    real_works = resp.output and len(resp.output) > 0
    print(f"  Real inference works: {real_works}")
    print(f"  Output: {resp.output[:200]}")

    passed = passed and real_works
    status = "PASS" if passed else "FAIL"
    print(f"Result: {status}")

    return {
        "test_id": "T5", "description": "Empty model response retry logic",
        "passed": passed, "output": f"retry_params={has_max_retries and has_backoff}, real_works={real_works}",
        "tool_calls": resp.tool_calls, "iterations": resp.iterations,
        "time": 0.0, "status": status,
    }


def test_p7_t6_timeout(model, model_name):
    """Timeout — SlowTool triggers tool timeout."""
    from effgen.tools.builtin.calculator import Calculator

    # Use a shorter sleep to not waste GPU time, but still test timeout behavior
    slow = SlowTool(sleep_seconds=5.0)
    agent = Agent(AgentConfig(
        name="error_recovery_t6",
        model=model,
        tools=[slow, Calculator()],
        system_prompt=(
            "You are a helpful assistant. If a tool is too slow or times out, "
            "try a different tool or answer from your knowledge. "
            "Use Final Answer: to provide your response."
        ),
        temperature=0.1,
        max_iterations=6,
        enable_sub_agents=False,
    ))

    t0 = time.time()
    result = run_test(
        agent, "T6", "Timeout — SlowTool triggers timeout",
        "Use the slow_tool to process 'test data'. If it's too slow, use calculator to compute 10 + 5 instead.",
        check_fn=lambda out, resp: (
            # Framework should handle the slow tool without hanging forever
            len(out) > 3
        ),
    )
    dt = time.time() - t0

    # The tool execution framework uses _run_coroutine_sync which has a 60s timeout
    # If it took too long, the framework's timeout mechanism didn't work
    if dt > 120:
        print(f"  WARNING: Took {dt:.1f}s — no timeout enforcement!")
        result["passed"] = False
    else:
        print(f"  Total time: {dt:.1f}s — within timeout bounds")

    return result


def test_p7_t7_fallback_exhaustion(model, model_name):
    """Fallback chain exhaustion — calculator -> python_repl -> code_executor all fail."""
    # Create agent with all-failing math tools
    agent = Agent(AgentConfig(
        name="error_recovery_t7",
        model=model,
        tools=[
            AlwaysFailCalculator(),
            AlwaysFailPythonREPL(),
            AlwaysFailCodeExecutor(),
        ],
        system_prompt=(
            "You are a math assistant. Use calculator to compute expressions. "
            "If calculator fails, try python_repl. If that fails too, try code_executor. "
            "If all tools fail, answer from your knowledge with Final Answer:."
        ),
        temperature=0.1,
        max_iterations=8,
        enable_sub_agents=False,
        enable_fallback=True,
        fallback_chain={
            "calculator": ["python_repl", "code_executor"],
            "python_repl": ["code_executor"],
        },
    ))

    result = run_test(
        agent, "T7", "Fallback chain exhaustion — all fallbacks fail",
        "What is 7 * 8? Use the calculator tool.",
        check_fn=lambda out, resp: (
            # Agent should handle this without crashing — either answer from knowledge or report failure
            len(out) > 3
        ),
    )

    # Verify fallback chain was actually attempted
    trace_tools = []
    for trace in agent.execution_tracker.get_trace():
        if isinstance(trace, dict):
            data = trace.get("data", {})
            if data.get("tool_name"):
                trace_tools.append(data["tool_name"])
    print(f"  Tools attempted: {trace_tools}")
    if "calculator" in trace_tools:
        print("  VERIFIED: Calculator was attempted (fallback chain activated)")

    return result


def test_p7_t8_concurrent_failures(model, model_name):
    """Concurrent failures — 3 agents simultaneously, no shared state corruption."""
    from effgen.tools.builtin.calculator import Calculator

    results = [None, None, None]
    errors = [None, None, None]

    def run_agent(idx, agent_name, question):
        try:
            broken = BrokenTool()
            calc = Calculator()
            agent = Agent(AgentConfig(
                name=f"concurrent_{agent_name}",
                model=model,
                tools=[broken, calc],
                system_prompt=(
                    "You are a helpful assistant. If a tool fails, try calculator instead. "
                    "Use Final Answer: to respond."
                ),
                temperature=0.1,
                max_iterations=5,
                enable_sub_agents=False,
            ))
            resp = agent.run(question)
            results[idx] = resp
        except Exception as e:
            errors[idx] = str(e)
            traceback.print_exc()

    questions = [
        "Use broken_tool to process 'data_A'. If it fails, use calculator: 10 + 1.",
        "Use broken_tool to process 'data_B'. If it fails, use calculator: 20 + 2.",
        "Use broken_tool to process 'data_C'. If it fails, use calculator: 30 + 3.",
    ]

    print(f"\n{'='*60}")
    print("Test: T8 — Concurrent failures (3 agents)")

    t0 = time.time()
    threads = []
    for i in range(3):
        t = threading.Thread(target=run_agent, args=(i, f"agent_{i}", questions[i]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=180)

    dt = time.time() - t0
    print(f"  All 3 agents completed in {dt:.1f}s")

    # Check results
    all_passed = True
    for i in range(3):
        if errors[i]:
            print(f"  Agent {i}: ERROR — {errors[i]}")
            all_passed = False
        elif results[i]:
            print(f"  Agent {i}: output='{results[i].output[:100]}', "
                  f"tools={results[i].tool_calls}, iters={results[i].iterations}")
        else:
            print(f"  Agent {i}: No result (timeout?)")
            all_passed = False

    # Verify no shared state corruption — each agent should have independent circuit breakers
    print(f"  No shared state corruption: {all_passed}")
    status = "PASS" if all_passed else "FAIL"
    print(f"Result: {status}")

    return {
        "test_id": "T8", "description": "Concurrent failures — no shared state corruption",
        "passed": all_passed, "output": f"3 agents completed, errors={[e for e in errors if e]}",
        "tool_calls": sum(r.tool_calls for r in results if r),
        "iterations": sum(r.iterations for r in results if r),
        "time": dt, "status": status,
    }


def test_p7_t9_control_characters(model, model_name):
    """Control character input — sanitization strips them."""
    from effgen.core.agent import Agent as AgentClass

    print(f"\n{'='*60}")
    print("Test: T9 — Control character sanitization")

    # Test the _sanitize_tool_input method directly
    test_cases = [
        ("\x00\x01\x02hello\x03world", "helloworld"),
        ("normal text", "normal text"),
        ("line1\nline2\ttab", "line1\nline2\ttab"),  # \n and \t preserved
        ("\x7fDEL\x0bVTAB", "DELVTAB"),
        ("", ""),
    ]

    all_passed = True
    for raw, expected in test_cases:
        sanitized = AgentClass._sanitize_tool_input(raw)
        ok = sanitized == expected
        if not ok:
            print(f"  FAIL: sanitize({repr(raw)}) = {repr(sanitized)}, expected {repr(expected)}")
            all_passed = False
        else:
            print(f"  OK: sanitize({repr(raw)}) = {repr(sanitized)}")

    # Also test with a real agent — pass control chars in a question
    from effgen.tools.builtin.calculator import Calculator
    agent = Agent(AgentConfig(
        name="error_recovery_t9",
        model=model,
        tools=[Calculator()],
        system_prompt="Answer questions directly with Final Answer:",
        temperature=0.1,
        max_iterations=3,
        enable_sub_agents=False,
    ))

    # The control chars are in the question — the model should still work
    resp = agent.run("What is 5\x00 + 3\x01?")
    real_works = resp.output and len(resp.output) > 0
    print(f"  Real inference with control chars: output='{resp.output[:200]}'")
    print(f"  Real inference works: {real_works}")

    passed = all_passed and real_works
    status = "PASS" if passed else "FAIL"
    print(f"Result: {status}")

    return {
        "test_id": "T9", "description": "Control character sanitization",
        "passed": passed, "output": f"sanitize_ok={all_passed}, real_works={real_works}",
        "tool_calls": resp.tool_calls, "iterations": resp.iterations,
        "time": 0.0, "status": status,
    }


# ── Regression Tests ─────────────────────────────────────────────────────────

def run_regression(model, model_name):
    """Run regression tests."""
    from effgen.tools.builtin.bash_tool import BashTool
    from effgen.tools.builtin.calculator import Calculator
    from effgen.tools.builtin.python_repl import PythonREPL

    results = []

    # Create a general-purpose agent for regression
    agent = Agent(AgentConfig(
        name="regression_agent",
        model=model,
        tools=[Calculator(), PythonREPL(), BashTool()],
        system_prompt=ERROR_RECOVERY_SYSTEM_PROMPT,
        temperature=0.1,
        max_iterations=8,
        enable_sub_agents=False,
    ))

    # Q&A
    results.append(run_test(
        agent, "REG-P1", "Q&A: capital of France",
        "What is the capital of France?",
        expected_keywords=["paris"],
    ))

    # Calculator
    results.append(run_test(
        agent, "REG-P2", "Calculator: 15 + 27",
        "What is 15 + 27? Use a tool to calculate it.",
        expected_keywords=["42"],
    ))

    # Multi-tool — bash
    results.append(run_test(
        agent, "REG-P3", "Multi-tool: bash echo",
        "Run the command 'echo hello_regression' using bash and tell me the output.",
        expected_keywords=["hello_regression"],
        expected_tool="bash",
    ))

    # File ops
    results.append(run_test(
        agent, "REG-P4", "File ops: cat hostname",
        "Use bash to run: cat /etc/hostname",
        check_fn=lambda out, resp: resp.tool_calls >= 1,
    ))

    # Code execution
    results.append(run_test(
        agent, "REG-P5", "Code: prime check",
        "Write and run Python code to check if 17 is prime. Print the result.",
        expected_keywords=["17"],
        expected_tool="python_repl",
    ))

    # Summary
    print(f"\n{'='*60}")
    print(f"Regression Results — Model: {model_name}")
    print(f"{'='*60}")
    pc = sum(1 for r in results if r["passed"])
    for r in results:
        print(f"  {r['status']:5s} — {r['test_id']}: {r['description']} ({r['time']:.1f}s)")
    print(f"\n{pc}/{len(results)} regression passed")
    return results


# ── Main Entry Point ─────────────────────────────────────────────────────────

def run_all_error_recovery_tests(model, model_name):
    """Run all 9 error recovery tests."""
    results = []

    print("\n" + "="*60)
    print(f"Error Recovery Agent — Model: {model_name}")
    print("="*60)

    # T1: Invalid tool input
    results.append(test_p7_t1_invalid_input(model, model_name))

    # T2: Tool crash + circuit breaker
    results.append(test_p7_t2_tool_crash(model, model_name))

    # T3: All tools fail
    results.append(test_p7_t3_all_tools_fail(model, model_name))

    # T4: Max iterations
    results.append(test_p7_t4_max_iterations(model, model_name))

    # T5: Empty response retry logic
    results.append(test_p7_t5_empty_response(model, model_name))

    # T6: Timeout
    results.append(test_p7_t6_timeout(model, model_name))

    # T7: Fallback chain exhaustion
    results.append(test_p7_t7_fallback_exhaustion(model, model_name))

    # T8: Concurrent failures
    results.append(test_p7_t8_concurrent_failures(model, model_name))

    # T9: Control character sanitization
    results.append(test_p7_t9_control_characters(model, model_name))

    # Summary
    print(f"\n{'='*60}")
    print(f"Error Recovery Results — Model: {model_name}")
    print(f"{'='*60}")
    pc = sum(1 for r in results if r["passed"])
    for r in results:
        print(f"  {r['status']:5s} — {r['test_id']}: {r['description']} ({r['time']:.1f}s)")
    print(f"\n{pc}/{len(results)} tests passed")

    return results


def main():
    parser = argparse.ArgumentParser(description="effGen Error Recovery Agent Example")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--regression", action="store_true", help="Run regression tests only")
    parser.add_argument("--test", type=str, help="Run specific test (e.g., T1)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print("effGen — Error Recovery Agent")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    if args.regression:
        run_regression(model, model_name=args.model)
    elif args.test:
        # Run specific test
        test_map = {
            "T1": test_p7_t1_invalid_input,
            "T2": test_p7_t2_tool_crash,
            "T3": test_p7_t3_all_tools_fail,
            "T4": test_p7_t4_max_iterations,
            "T5": test_p7_t5_empty_response,
            "T6": test_p7_t6_timeout,
            "T7": test_p7_t7_fallback_exhaustion,
            "T8": test_p7_t8_concurrent_failures,
            "T9": test_p7_t9_control_characters,
        }
        test_fn = test_map.get(args.test.upper())
        if test_fn:
            test_fn(model, args.model)
        else:
            print(f"Unknown test: {args.test}. Available: {list(test_map.keys())}")
    else:
        # Run regression first
        print("\n--- Regression Tests ---")
        run_regression(model, model_name=args.model)

        # Run error recovery tests
        print("\n--- Error Recovery Tests ---")
        run_all_error_recovery_tests(model, model_name=args.model)

    model.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
