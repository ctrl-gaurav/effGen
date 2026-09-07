#!/usr/bin/env python3
"""
effGen — Multi-Agent Pipeline (Orchestration)

Demonstrates two multi-agent patterns:
  Option A: Manual Pipeline — 3+ agents wired sequentially (A → B → C)
  Option B: Sub-Agent System — SubAgentRouter → SubAgentManager with real execution

Tests:
  - T1: Manual pipeline — Agent A generates question, B solves, C summarizes
  - T2: Sub-agent routing — simple vs complex task routing
  - T3: Parallel sub-agents — independent subtasks
  - T4: Sequential sub-agents — dependent subtasks, context passed
  - T5: Synthesis quality — results combined into coherent answer
  - T6: Error handling — pipeline handles failure without crashing
  - T7: User-explicit trigger — fuzzy matching for sub-agent requests
  - T8: Hard 4-stage pipeline (generate → solve → verify → report)
  - T9: Hard 3-agent sub-agent (shopping cart + tax + payment comparison)

Recommended models:
  - Qwen/Qwen2.5-3B-Instruct (default, fast)
  - Qwen/Qwen2.5-7B-Instruct (best quality)
  - microsoft/Phi-4-mini-instruct

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/multi_agent_pipeline.py
  CUDA_VISIBLE_DEVICES=0 python examples/multi_agent_pipeline.py --model Qwen/Qwen2.5-7B-Instruct
  CUDA_VISIBLE_DEVICES=0 python examples/multi_agent_pipeline.py --test T1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.core.agent import Agent, AgentConfig
from effgen.core.execution_tracker import ExecutionTracker
from effgen.core.router import RoutingStrategy, SubAgentRouter
from effgen.core.sub_agent_manager import SubAgentManager
from effgen.core.task import SubTask, TaskStatus
from effgen.presets import create_agent
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.python_repl import PythonREPL

# ANSI colors
BLUE = "\033[94m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(text):
    print(f"\n{BOLD}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{RESET}\n")


def print_stage(agent_name, role, text):
    print(f"{CYAN}[{agent_name}]{RESET} {YELLOW}({role}){RESET}")
    print(f"  {text[:500]}")
    print()


def print_result(test_id, passed, details=""):
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"\n{BOLD}[{test_id}]{RESET} {status} {details}")


# ── Option A: Manual Pipeline ────────────────────────────────────────────

def run_manual_pipeline(model, topic="compound interest"):
    """
    Manual 3-agent pipeline:
      Agent A (question generator) → Agent B (math solver) → Agent C (summarizer)
    """
    print_header("Option A: Manual Pipeline (A → B → C)")

    # ── Agent A: Generate a math/research question ──
    print(f"{MAGENTA}Stage 1: Agent A — Question Generator{RESET}")
    agent_a = Agent(AgentConfig(
        name="QuestionGenerator",
        model=model,
        tools=[],
        system_prompt=(
            "You are a math question generator. Given a topic, create ONE specific, "
            "solvable math word problem. Output ONLY the question, nothing else. "
            "The question must involve specific numbers so it can be solved with a calculator."
        ),
        max_iterations=1,
        enable_sub_agents=False,
        enable_memory=False,
    ))

    t0 = time.time()
    task_a = f"Create a math word problem about {topic}. Use specific numbers."
    resp_a = agent_a.run(task_a)
    dt_a = time.time() - t0
    question = resp_a.output.strip()
    print_stage("Agent A", "Question Generator", question)
    print(f"  Time: {dt_a:.1f}s")

    # ── Agent B: Solve the question using Calculator/PythonREPL ──
    print(f"{MAGENTA}Stage 2: Agent B — Math Solver{RESET}")
    agent_b = Agent(AgentConfig(
        name="MathSolver",
        model=model,
        tools=[Calculator(), PythonREPL()],
        system_prompt=(
            "You are a math solver. Solve the given math problem step by step. "
            "Use the calculator tool for arithmetic. Show your work and give the final numerical answer."
        ),
        max_iterations=8,
        enable_sub_agents=False,
        enable_memory=False,
    ))

    t1 = time.time()
    resp_b = agent_b.run(f"Solve this problem: {question}")
    dt_b = time.time() - t1
    solution = resp_b.output.strip()
    print_stage("Agent B", "Math Solver", solution)
    print(f"  Time: {dt_b:.1f}s | Tool calls: {resp_b.tool_calls}")

    # ── Agent C: Summarize the results ──
    print(f"{MAGENTA}Stage 3: Agent C — Summarizer{RESET}")
    agent_c = Agent(AgentConfig(
        name="Summarizer",
        model=model,
        tools=[],
        system_prompt=(
            "You are a concise summarizer. Given a question and its solution, "
            "produce a brief, clear summary of the problem and answer in 2-3 sentences."
        ),
        max_iterations=1,
        enable_sub_agents=False,
        enable_memory=False,
    ))

    t2 = time.time()
    task_c = f"Summarize this Q&A:\n\nQuestion: {question}\n\nSolution: {solution}"
    resp_c = agent_c.run(task_c)
    dt_c = time.time() - t2
    summary = resp_c.output.strip()
    print_stage("Agent C", "Summarizer", summary)
    print(f"  Time: {dt_c:.1f}s")

    total_time = dt_a + dt_b + dt_c
    print(f"\n{BOLD}Pipeline Total: {total_time:.1f}s{RESET}")

    # Validate pipeline
    pipeline_ok = (
        resp_a.success and len(question) > 10 and
        resp_b.success and len(solution) > 10 and
        resp_c.success and len(summary) > 10
    )

    return {
        "question": question,
        "solution": solution,
        "summary": summary,
        "pipeline_ok": pipeline_ok,
        "times": {"a": dt_a, "b": dt_b, "c": dt_c, "total": total_time},
        "tool_calls": resp_b.tool_calls,
    }


# ── Option B: Sub-Agent System ──────────────────────────────────────────

def test_sub_agent_routing(model):
    """T2: Test that the router correctly decides to use sub-agents for complex tasks."""
    print_header("T2: Sub-Agent Routing Decision")

    router = SubAgentRouter()

    # Simple task — should NOT trigger sub-agents
    simple_task = "What is 2 + 2?"
    simple_decision = router.route(simple_task)
    print(f"Simple task: '{simple_task}'")
    print(f"  Use sub-agents: {simple_decision.use_sub_agents}")
    print(f"  Strategy: {simple_decision.strategy.value}")
    print(f"  Complexity: {simple_decision.complexity_score.overall:.2f}")
    print()

    # Complex task — SHOULD trigger sub-agents
    complex_task = (
        "Research and analyze the impact of compound interest on savings. "
        "1) Calculate compound interest for $10,000 at 5% over 10 years. "
        "2) Compare this with simple interest for the same parameters. "
        "3) Create a summary report with key insights and recommendations."
    )
    complex_decision = router.route(complex_task)
    print(f"Complex task: '{complex_task[:80]}...'")
    print(f"  Use sub-agents: {complex_decision.use_sub_agents}")
    print(f"  Strategy: {complex_decision.strategy.value}")
    print(f"  Complexity: {complex_decision.complexity_score.overall:.2f}")
    print(f"  Num subtasks: {complex_decision.num_sub_agents}")
    print(f"  Specializations: {complex_decision.specializations}")
    if complex_decision.decomposition:
        print("  Decomposition:")
        for st in complex_decision.decomposition:
            print(f"    - [{st.id}] {st.description[:60]}... (depends: {st.depends_on})")
    print(f"  Reasoning: {complex_decision.reasoning}")

    # Validate
    routing_ok = (
        not simple_decision.use_sub_agents and
        complex_decision.use_sub_agents and
        complex_decision.num_sub_agents >= 2 and
        complex_decision.complexity_score.overall > simple_decision.complexity_score.overall
    )

    return {"routing_ok": routing_ok, "simple": simple_decision, "complex": complex_decision}


def test_sub_agent_execution(model, strategy="sequential"):
    """
    T3/T4: Test sub-agent execution with real model inference.

    The SubAgentManager currently uses _simulate_execution (placeholder).
    We'll wire it to actually use Agent.run() by monkey-patching or by
    directly building the pipeline.
    """
    print_header(f"T3/T4: Sub-Agent Execution ({strategy})")

    tracker = ExecutionTracker()
    manager = SubAgentManager(config={"max_parallel_agents": 3}, execution_tracker=tracker)

    if strategy == "parallel":
        subtasks = [
            SubTask(
                id="st_1",
                description="Calculate compound interest: $10,000 at 5% annual rate for 10 years, compounded annually.",
                expected_output="The compound interest amount",
                estimated_complexity=4.0,
                required_specialization="analysis",
                depends_on=[],
            ),
            SubTask(
                id="st_2",
                description="Calculate simple interest: $10,000 at 5% annual rate for 10 years.",
                expected_output="The simple interest amount",
                estimated_complexity=3.0,
                required_specialization="analysis",
                depends_on=[],
            ),
        ]
    else:
        subtasks = [
            SubTask(
                id="st_1",
                description="Calculate compound interest: $10,000 at 5% annual rate for 10 years, compounded annually.",
                expected_output="The compound interest amount",
                depends_on=[],
            ),
            SubTask(
                id="st_2",
                description="Based on the compound interest result, calculate the difference compared to simple interest ($10,000 * 0.05 * 10 = $5,000).",
                expected_output="The difference between compound and simple interest",
                depends_on=["st_1"],
            ),
        ]

    # Instead of using the placeholder _simulate_execution, we use real agents
    # by overriding the execution method

    def real_execute(agent_info, subtask, progress_callback=None):
        """Execute subtask using a real Agent with the shared model."""
        agent_id = agent_info["id"]
        start_time = time.time()

        try:
            subtask.status = TaskStatus.RUNNING
            print(f"  {CYAN}[{agent_id}]{RESET} Starting: {subtask.description[:60]}...")

            # Create a real agent for this subtask
            sub_agent = Agent(AgentConfig(
                name=agent_id,
                model=model,
                tools=[Calculator(), PythonREPL()],
                system_prompt=(
                    "You are a math specialist. Solve the given problem using the calculator tool. "
                    "Be precise and show the calculation."
                ),
                max_iterations=5,
                enable_sub_agents=False,
                enable_memory=False,
            ))

            resp = sub_agent.run(subtask.description)
            execution_time = time.time() - start_time

            subtask.status = TaskStatus.COMPLETED
            subtask.result = {"output": resp.output, "tokens_used": resp.tokens_used, "tool_calls": resp.tool_calls}

            print(f"  {GREEN}[{agent_id}]{RESET} Done in {execution_time:.1f}s: {resp.output[:80]}...")

            from effgen.core.sub_agent_manager import SubAgentResult
            return SubAgentResult(
                subtask_id=subtask.id,
                agent_id=agent_id,
                success=resp.success,
                result=subtask.result,
                execution_time=execution_time,
                tokens_used=resp.tokens_used,
                tool_calls=resp.tool_calls,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            subtask.status = TaskStatus.FAILED
            subtask.error = str(e)
            print(f"  {RED}[{agent_id}]{RESET} Failed: {e}")

            from effgen.core.sub_agent_manager import SubAgentResult
            return SubAgentResult(
                subtask_id=subtask.id,
                agent_id=agent_id,
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

    # Monkey-patch
    manager._execute_sub_agent = real_execute
    manager._simulate_execution = None  # Ensure placeholder isn't used

    t0 = time.time()
    if strategy == "parallel":
        # For parallel, we use sequential since we share the GPU
        # (True parallel would need separate GPU processes)
        results = manager.execute_sequential(subtasks)
    else:
        results = manager.execute_sequential(subtasks)
    total_time = time.time() - t0

    # Synthesize
    synthesis = manager.synthesize_results(
        results,
        "Compare compound vs simple interest on $10,000 at 5% for 10 years",
        RoutingStrategy.SEQUENTIAL_SUB_AGENTS if strategy == "sequential" else RoutingStrategy.PARALLEL_SUB_AGENTS,
    )

    print(f"\n{BOLD}Synthesis:{RESET}")
    print(synthesis["final_output"][:500])
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Successful: {synthesis['successful']}/{synthesis['total_subtasks']}")

    exec_ok = all(r.success for r in results) and synthesis["successful"] == len(subtasks)
    return {"exec_ok": exec_ok, "synthesis": synthesis, "results": results, "time": total_time}


def test_synthesis_quality(model):
    """T5: Test that synthesis produces a coherent final answer from multiple sub-agent results."""
    print_header("T5: Synthesis Quality")

    # Create a synthesizer agent that combines results
    agent = Agent(AgentConfig(
        name="Synthesizer",
        model=model,
        tools=[],
        system_prompt=(
            "You are an expert at synthesizing information. Given multiple pieces of data, "
            "combine them into a clear, coherent summary. Be concise."
        ),
        max_iterations=1,
        enable_sub_agents=False,
        enable_memory=False,
    ))

    # Simulate sub-agent results
    sub_results = {
        "Compound Interest": "A = 10000 * (1 + 0.05)^10 = $16,288.95. The compound interest earned is $6,288.95.",
        "Simple Interest": "I = 10000 * 0.05 * 10 = $5,000. Total with simple interest: $15,000.",
        "Comparison": "Compound interest earns $1,288.95 more than simple interest over 10 years."
    }

    synthesis_prompt = "Synthesize these financial analysis results into a clear summary:\n\n"
    for title, result in sub_results.items():
        synthesis_prompt += f"**{title}:** {result}\n\n"

    t0 = time.time()
    resp = agent.run(synthesis_prompt)
    dt = time.time() - t0

    print(f"Synthesis output:\n{resp.output[:500]}")
    print(f"\nTime: {dt:.1f}s")

    # Check quality: output should mention key numbers
    output_lower = resp.output.lower()
    has_compound = any(kw in output_lower for kw in ["16,288", "16288", "6,288", "6288", "compound"])
    has_simple = any(kw in output_lower for kw in ["15,000", "15000", "5,000", "5000", "simple"])
    has_comparison = any(kw in output_lower for kw in ["1,288", "1288", "more", "difference", "greater"])
    synthesis_ok = resp.success and has_compound and has_simple and len(resp.output) > 50

    print(f"\n  Has compound info: {has_compound}")
    print(f"  Has simple info: {has_simple}")
    print(f"  Has comparison: {has_comparison}")

    return {"synthesis_ok": synthesis_ok, "output": resp.output, "time": dt}


def test_error_handling(model):
    """T6: Test pipeline behavior when one agent fails."""
    print_header("T6: Error Handling in Pipeline")

    tracker = ExecutionTracker()
    manager = SubAgentManager(config={"stop_on_failure": False}, execution_tracker=tracker)

    subtasks = [
        SubTask(
            id="st_1",
            description="Calculate 25 * 4.",
            expected_output="100",
            depends_on=[],
        ),
        SubTask(
            id="st_2_broken",
            description="This will fail intentionally.",
            expected_output="Should fail",
            depends_on=[],
        ),
        SubTask(
            id="st_3",
            description="Calculate 7 * 8.",
            expected_output="56",
            depends_on=[],
        ),
    ]

    def real_execute_with_failure(agent_info, subtask, progress_callback=None):
        """Execute with intentional failure for st_2_broken."""
        agent_id = agent_info["id"]
        start_time = time.time()

        if "broken" in subtask.id:
            # Intentional failure
            subtask.status = TaskStatus.FAILED
            subtask.error = "Intentional failure for testing"
            print(f"  {RED}[{agent_id}]{RESET} INTENTIONAL FAILURE")
            from effgen.core.sub_agent_manager import SubAgentResult
            return SubAgentResult(
                subtask_id=subtask.id,
                agent_id=agent_id,
                success=False,
                error="Intentional failure for testing",
                execution_time=time.time() - start_time,
            )

        try:
            subtask.status = TaskStatus.RUNNING
            sub_agent = Agent(AgentConfig(
                name=agent_id,
                model=model,
                tools=[Calculator()],
                system_prompt="You are a calculator. Solve the math problem using the calculator tool.",
                max_iterations=3,
                enable_sub_agents=False,
                enable_memory=False,
            ))
            resp = sub_agent.run(subtask.description)
            execution_time = time.time() - start_time
            subtask.status = TaskStatus.COMPLETED
            subtask.result = {"output": resp.output, "tokens_used": resp.tokens_used, "tool_calls": resp.tool_calls}
            print(f"  {GREEN}[{agent_id}]{RESET} Done: {resp.output[:60]}...")
            from effgen.core.sub_agent_manager import SubAgentResult
            return SubAgentResult(
                subtask_id=subtask.id,
                agent_id=agent_id,
                success=True,
                result=subtask.result,
                execution_time=execution_time,
                tokens_used=resp.tokens_used,
                tool_calls=resp.tool_calls,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            subtask.status = TaskStatus.FAILED
            from effgen.core.sub_agent_manager import SubAgentResult
            return SubAgentResult(
                subtask_id=subtask.id, agent_id=agent_id, success=False,
                error=str(e), execution_time=execution_time,
            )

    manager._execute_sub_agent = real_execute_with_failure

    t0 = time.time()
    results = manager.execute_sequential(subtasks)
    total_time = time.time() - t0

    synthesis = manager.synthesize_results(
        results,
        "Calculate 25*4, handle failure, calculate 7*8",
        RoutingStrategy.SEQUENTIAL_SUB_AGENTS,
    )

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n  Total: {len(results)}, Successful: {len(successful)}, Failed: {len(failed)}")
    print(f"  Time: {total_time:.1f}s")
    print(f"\nSynthesis:\n{synthesis['final_output'][:300]}")

    # Pipeline should continue despite failure, partial results returned
    error_ok = (
        len(successful) == 2 and
        len(failed) == 1 and
        synthesis["successful"] == 2 and
        synthesis["failed"] == 1
    )

    return {"error_ok": error_ok, "synthesis": synthesis, "time": total_time}


# ── Harder Multi-Agent Tests ─────────────────────────────────────────────

def test_user_explicit_trigger(model):
    """T7: User explicitly asks to use sub-agents — router should honor it (fuzzy matching)."""
    print_header("T7: User-Explicit Sub-Agent Trigger (Fuzzy)")

    router = SubAgentRouter()
    passed = 0
    total = 0

    # Negative cases — should NOT trigger sub-agents
    negative_cases = [
        "What is 2 + 2?",
        "Tell me about agents in AI.",
        "How do submarines work?",
    ]
    for task in negative_cases:
        total += 1
        decision = router.route(task)
        ok = not decision.use_sub_agents
        status = f"{GREEN}OK{RESET}" if ok else f"{RED}WRONG{RESET}"
        print(f"  {status} NO-TRIGGER: '{task[:50]}' → sub_agents={decision.use_sub_agents}")
        if ok:
            passed += 1

    # Positive cases — SHOULD trigger sub-agents (various phrasings)
    positive_cases = [
        "Use sub-agents to calculate what is 2+2 and 3+3",
        "Use subagents for this task: compare prices",
        "Launch sub agents to handle this",
        "Launch 3 agents to parallelize the work",
        "Enable sub-agent mode for this complex task",
        "Spawn agents to research and compute",
        "Deploy multiple agents for analysis",
        "Split into sub-tasks and use agents for each",
        "Break this into subtasks with sub-agent routing",
        "Use 2 agents to handle calculation and research",
        "Run this with three agents in parallel",
        "Decompose into agents for parallel processing",
        "Can you use multiple agents for this?",
        "With 3 agents, solve these problems",
        "Activate sub-agent system for this request",
    ]
    for task in positive_cases:
        total += 1
        decision = router.route(task)
        ok = decision.use_sub_agents
        status = f"{GREEN}OK{RESET}" if ok else f"{RED}MISS{RESET}"
        print(f"  {status} TRIGGER: '{task[:55]}' → sub_agents={decision.use_sub_agents}")
        if ok:
            passed += 1

    print(f"\n  {BOLD}Trigger accuracy: {passed}/{total}{RESET}")
    # Allow at most 2 misses (some edge cases may be borderline)
    trigger_ok = passed >= total - 2

    return {"trigger_ok": trigger_ok, "passed": passed, "total": total}


def test_hard_pipeline(model, topic="physics"):
    """T8: Harder multi-agent pipeline with 4 stages and cross-domain reasoning."""
    print_header("T8: Hard Multi-Agent Pipeline (4 Stages)")

    # Stage 1: Generate a harder multi-step problem
    print(f"{MAGENTA}Stage 1: Problem Generator{RESET}")
    agent_gen = Agent(AgentConfig(
        name="ProblemGenerator",
        model=model,
        tools=[],
        system_prompt=(
            "You are a problem generator. Create a multi-step word problem that requires "
            "at least 2 different calculations. Output ONLY the problem, no solution. "
            "Use specific numbers. The problem should involve multiple steps."
        ),
        max_iterations=1,
        enable_sub_agents=False,
        enable_memory=False,
    ))
    t0 = time.time()
    resp_gen = agent_gen.run(
        "Create a multi-step math problem about a store selling items with tax and discount. "
        "Include at least 3 items with different prices, a percentage discount, and sales tax."
    )
    dt_gen = time.time() - t0
    problem = resp_gen.output.strip()
    print_stage("Generator", "Problem", problem)
    print(f"  Time: {dt_gen:.1f}s")

    # Stage 2: Solver A — calculate subtotals
    print(f"{MAGENTA}Stage 2: Calculator Agent{RESET}")
    agent_calc = Agent(AgentConfig(
        name="Calculator",
        model=model,
        tools=[Calculator(), PythonREPL()],
        system_prompt=(
            "You are a calculation specialist. Break down the problem step by step. "
            "Use the calculator or python_repl for each calculation. Show all intermediate results."
        ),
        max_iterations=10,
        enable_sub_agents=False,
        enable_memory=False,
    ))
    t1 = time.time()
    resp_calc = agent_calc.run(f"Solve step by step: {problem}")
    dt_calc = time.time() - t1
    solution = resp_calc.output.strip()
    print_stage("Calculator", "Solution", solution)
    print(f"  Time: {dt_calc:.1f}s | Tool calls: {resp_calc.tool_calls}")

    # Stage 3: Verifier — check the answer using python
    print(f"{MAGENTA}Stage 3: Verifier Agent{RESET}")
    agent_verify = Agent(AgentConfig(
        name="Verifier",
        model=model,
        tools=[PythonREPL()],
        system_prompt=(
            "You are a verification specialist. Given a problem and a proposed solution, "
            "write Python code to independently verify the answer. "
            "Print whether the solution is correct or not."
        ),
        max_iterations=5,
        enable_sub_agents=False,
        enable_memory=False,
    ))
    t2 = time.time()
    resp_verify = agent_verify.run(
        f"Verify this solution:\n\nProblem: {problem}\n\nProposed Solution: {solution}"
    )
    dt_verify = time.time() - t2
    verification = resp_verify.output.strip()
    print_stage("Verifier", "Verification", verification)
    print(f"  Time: {dt_verify:.1f}s | Tool calls: {resp_verify.tool_calls}")

    # Stage 4: Report Writer
    print(f"{MAGENTA}Stage 4: Report Writer{RESET}")
    agent_report = Agent(AgentConfig(
        name="ReportWriter",
        model=model,
        tools=[],
        system_prompt=(
            "You are a report writer. Combine the problem, solution, and verification "
            "into a clean, structured report with sections."
        ),
        max_iterations=1,
        enable_sub_agents=False,
        enable_memory=False,
    ))
    t3 = time.time()
    resp_report = agent_report.run(
        f"Write a brief report:\n\n"
        f"Problem: {problem}\n\n"
        f"Solution: {solution}\n\n"
        f"Verification: {verification}"
    )
    dt_report = time.time() - t3
    report = resp_report.output.strip()
    print_stage("ReportWriter", "Report", report)
    print(f"  Time: {dt_report:.1f}s")

    total_time = dt_gen + dt_calc + dt_verify + dt_report
    print(f"\n{BOLD}Hard Pipeline Total: {total_time:.1f}s across 4 stages{RESET}")

    pipeline_ok = (
        resp_gen.success and len(problem) > 20 and
        resp_calc.success and resp_calc.tool_calls > 0 and
        resp_verify.success and
        resp_report.success and len(report) > 50
    )

    return {
        "pipeline_ok": pipeline_ok,
        "times": {"gen": dt_gen, "calc": dt_calc, "verify": dt_verify, "report": dt_report, "total": total_time},
        "tool_calls": resp_calc.tool_calls + resp_verify.tool_calls,
    }


def test_hard_sub_agents(model):
    """T9: Harder sub-agent test — 3 real sub-agents with different specializations."""
    print_header("T9: Hard Sub-Agent Test (3 Specialized Agents)")

    tracker = ExecutionTracker()
    manager = SubAgentManager(config={"max_parallel_agents": 3}, execution_tracker=tracker)

    subtasks = [
        SubTask(
            id="st_1_research",
            description=(
                "Calculate the total cost of a shopping cart: "
                "3 books at $12.99 each, 2 pens at $3.50 each, and 1 notebook at $8.75. "
                "Then apply a 15% discount to the total."
            ),
            expected_output="Discounted total",
            estimated_complexity=5.0,
            required_specialization="analysis",
            depends_on=[],
        ),
        SubTask(
            id="st_2_compute",
            description=(
                "Calculate sales tax: take $54.28 as the subtotal and apply 8.5% sales tax. "
                "What is the final amount including tax?"
            ),
            expected_output="Final amount with tax",
            estimated_complexity=4.0,
            required_specialization="analysis",
            depends_on=[],
        ),
        SubTask(
            id="st_3_compare",
            description=(
                "Compare two payment plans: "
                "Plan A: Pay $58.89 now. "
                "Plan B: Pay $30 now and $32 in 30 days. "
                "Which plan costs less and by how much?"
            ),
            expected_output="Comparison result",
            estimated_complexity=4.0,
            required_specialization="analysis",
            depends_on=[],
        ),
    ]

    def real_execute(agent_info, subtask, progress_callback=None):
        agent_id = agent_info["id"]
        start_time = time.time()
        try:
            subtask.status = TaskStatus.RUNNING
            print(f"  {CYAN}[{agent_id}]{RESET} Starting: {subtask.description[:60]}...")
            sub_agent = Agent(AgentConfig(
                name=agent_id,
                model=model,
                tools=[Calculator(), PythonREPL()],
                system_prompt="You are a math specialist. Solve the problem using calculator or python_repl. Be precise.",
                max_iterations=6,
                enable_sub_agents=False,
                enable_memory=False,
            ))
            resp = sub_agent.run(subtask.description)
            execution_time = time.time() - start_time
            subtask.status = TaskStatus.COMPLETED
            subtask.result = {"output": resp.output, "tokens_used": resp.tokens_used, "tool_calls": resp.tool_calls}
            print(f"  {GREEN}[{agent_id}]{RESET} Done in {execution_time:.1f}s: {resp.output[:80]}...")
            from effgen.core.sub_agent_manager import SubAgentResult
            return SubAgentResult(
                subtask_id=subtask.id, agent_id=agent_id, success=resp.success,
                result=subtask.result, execution_time=execution_time,
                tokens_used=resp.tokens_used, tool_calls=resp.tool_calls,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            subtask.status = TaskStatus.FAILED
            from effgen.core.sub_agent_manager import SubAgentResult
            return SubAgentResult(
                subtask_id=subtask.id, agent_id=agent_id, success=False,
                error=str(e), execution_time=execution_time,
            )

    manager._execute_sub_agent = real_execute

    t0 = time.time()
    results = manager.execute_sequential(subtasks)
    total_time = time.time() - t0

    synthesis = manager.synthesize_results(
        results, "Complete shopping cart calculation with discount, tax, and payment comparison",
        RoutingStrategy.PARALLEL_SUB_AGENTS,
    )

    print(f"\n{BOLD}Synthesis:{RESET}")
    print(synthesis["final_output"][:500])
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Successful: {synthesis['successful']}/{synthesis['total_subtasks']}")

    # Execution trace
    print(f"\n{BOLD}Execution Trace:{RESET}")
    trace = tracker.format_for_display("plain")
    print(trace[:500])

    exec_ok = all(r.success for r in results) and synthesis["successful"] == 3
    return {"exec_ok": exec_ok, "synthesis": synthesis, "time": total_time}


# ── Regression Tests ─────────────────────────────────────────────────────

def run_regression(model, checks=None):
    """Re-run the single-agent behaviours the pipeline is built on."""
    if checks is None:
        checks = ["qa", "coding", "error_recovery", "streaming"]

    print_header("Regression Testing")
    results = {}

    # Q&A
    if "qa" in checks:
        print(f"{CYAN}Q&A Regression{RESET}")
        agent = create_agent("minimal", model)
        resp = agent.run("What is the capital of France?")
        qa_ok = resp.success and "paris" in resp.output.lower()
        print_result("QA-Reg", qa_ok, f"'{resp.output[:60]}...'")
        results["QA"] = qa_ok

    # Coding
    if "coding" in checks:
        print(f"\n{CYAN}Coding Regression{RESET}")
        agent = create_agent("coding", model)
        resp = agent.run("Use python_repl to calculate the sum of the first 10 natural numbers. Print the result.")
        coding_ok = resp.success and "55" in resp.output
        print_result("Coding-Reg", coding_ok, f"'{resp.output[:60]}...'")
        results["Coding"] = coding_ok

    # Error Recovery
    if "error_recovery" in checks:
        print(f"\n{CYAN}Error Recovery Regression{RESET}")
        agent = Agent(AgentConfig(
            name="ErrorRecovery",
            model=model,
            tools=[Calculator()],
            system_prompt="You are a helpful assistant. If a tool fails, answer from your knowledge.",
            max_iterations=3,
            enable_sub_agents=False,
            enable_memory=False,
        ))
        resp = agent.run("What is 7 * 8?")
        recovery_ok = resp.success and "56" in resp.output
        print_result("Recovery-Reg", recovery_ok, f"'{resp.output[:60]}...'")
        results["Recovery"] = recovery_ok

    # Streaming
    if "streaming" in checks:
        print(f"\n{CYAN}Streaming Regression{RESET}")
        agent = Agent(AgentConfig(
            name="StreamTest",
            model=model,
            tools=[Calculator()],
            system_prompt="You are a helpful assistant. Use the calculator for math.",
            max_iterations=3,
            enable_sub_agents=False,
            enable_memory=False,
            enable_streaming=True,
        ))
        tokens = []
        stream_gen = agent.stream("What is 15 + 27?")
        full_output = ""
        for token in stream_gen:
            tokens.append(token)
            full_output += token
        stream_ok = len(tokens) > 0 and "42" in full_output
        print_result("Streaming-Reg", stream_ok, f"tokens={len(tokens)}, '{full_output[:60]}...'")
        results["Streaming"] = stream_ok

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{BOLD}Regression: {passed}/{total} PASS{RESET}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="effGen Multi-Agent Pipeline Example")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--test", help="Run specific test (T1..T9, regression)")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--topic", default="compound interest",
                        help="Topic for manual pipeline question generation")
    args = parser.parse_args()

    print(f"\n{BOLD}effGen — Multi-Agent Pipeline{RESET}")
    print(f"Model: {args.model}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")

    # Load model
    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    all_results = {}

    if args.test:
        tests = [args.test.upper()]
    else:
        tests = ["T1", "T2", "T3", "T4", "T5", "T6",
                 "T7", "T8", "T9"]
        if not args.skip_regression:
            tests.append("REGRESSION")

    for test in tests:
        try:
            if test == "T1":
                r = run_manual_pipeline(model, args.topic)
                all_results["T1"] = r["pipeline_ok"]
                print_result("T1", r["pipeline_ok"],
                             f"Manual pipeline — {r['times']['total']:.1f}s, {r['tool_calls']} tool calls")

            elif test == "T2":
                r = test_sub_agent_routing(model)
                all_results["T2"] = r["routing_ok"]
                print_result("T2", r["routing_ok"], "Sub-agent routing decisions")

            elif test == "T3":
                r = test_sub_agent_execution(model, strategy="parallel")
                all_results["T3"] = r["exec_ok"]
                print_result("T3", r["exec_ok"],
                             f"Parallel sub-agents — {r['time']:.1f}s")

            elif test == "T4":
                r = test_sub_agent_execution(model, strategy="sequential")
                all_results["T4"] = r["exec_ok"]
                print_result("T4", r["exec_ok"],
                             f"Sequential sub-agents — {r['time']:.1f}s")

            elif test == "T5":
                r = test_synthesis_quality(model)
                all_results["T5"] = r["synthesis_ok"]
                print_result("T5", r["synthesis_ok"],
                             f"Synthesis quality — {r['time']:.1f}s")

            elif test == "T6":
                r = test_error_handling(model)
                all_results["T6"] = r["error_ok"]
                print_result("T6", r["error_ok"],
                             f"Error handling — {r['time']:.1f}s")

            elif test == "T7":
                r = test_user_explicit_trigger(model)
                all_results["T7"] = r["trigger_ok"]
                print_result("T7", r["trigger_ok"], "User-explicit sub-agent trigger")

            elif test == "T8":
                r = test_hard_pipeline(model)
                all_results["T8"] = r["pipeline_ok"]
                print_result("T8", r["pipeline_ok"],
                             f"Hard 4-stage pipeline — {r['times']['total']:.1f}s, {r['tool_calls']} tool calls")

            elif test == "T9":
                r = test_hard_sub_agents(model)
                all_results["T9"] = r["exec_ok"]
                print_result("T9", r["exec_ok"],
                             f"Hard 3-agent sub-agent — {r['time']:.1f}s")

            elif test == "REGRESSION":
                reg = run_regression(model)
                for k, v in reg.items():
                    all_results[f"Reg-{k}"] = v

        except Exception as e:
            print(f"{RED}ERROR in {test}: {e}{RESET}")
            traceback.print_exc()
            all_results[test] = False

    # ── Final Summary ──
    print_header("FINAL RESULTS")
    passed = sum(1 for v in all_results.values() if v)
    total = len(all_results)
    for test_id, ok in all_results.items():
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {test_id}: {status}")
    print(f"\n{BOLD}Total: {passed}/{total} PASS{RESET}")


if __name__ == "__main__":
    main()
