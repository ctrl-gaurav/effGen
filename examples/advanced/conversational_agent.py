#!/usr/bin/env python3
"""
effGen — Conversational Agent (Memory + Multi-Turn)

A conversational agent that maintains context across multiple turns using
ShortTermMemory and LongTermMemory (SQLite). Demonstrates conversation history
formatting, memory recall, summarization trigger, and long-term persistence.

Tools: Calculator, DateTimeTool (minimal — focus is on memory)
Memory: ShortTermMemory (max_tokens=4096, max_messages=50, auto_summarize=True)
        LongTermMemory (SQLite backend for persistence)

Recommended models: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct

Usage:
  CUDA_VISIBLE_DEVICES=0 python examples/conversational_agent.py
  CUDA_VISIBLE_DEVICES=0 python examples/conversational_agent.py --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effgen import load_model
from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin.calculator import Calculator
from effgen.tools.builtin.datetime_tool import DateTimeTool

CONVERSATIONAL_SYSTEM_PROMPT = """You are a conversational assistant with memory. Remember what the user tells you and use that context in future responses.

IMPORTANT:
1. Pay close attention to the earlier turns of this conversation — they contain facts the user shared already.
2. When the user asks you to recall something, look in those earlier turns for the answer.
3. If the user mentions a value and then asks you to convert or calculate with it, use the Calculator tool.
4. Always be specific — if the user said their name is Alice, say "Alice", not "you told me your name".

TOOL USAGE:
- calculator: Use for any math computation. Input: {"expression": "mathematical expression"}
- datetime_tool: Use for date/time queries. Input: {"operation": "now"}
"""


def run_multi_turn_test(agent, test_id, description, turns, check_fn=None):
    """
    Run a multi-turn conversation test.

    Args:
        agent: Agent instance (memory persists across turns)
        test_id: Test identifier
        description: Test description
        turns: List of (user_message, expected_keywords_or_None) tuples
        check_fn: Optional function(final_output, all_outputs) -> bool

    Returns:
        Result dict
    """
    print(f"\n{'='*60}")
    print(f"Test: {test_id} — {description}")

    all_outputs = []
    total_time = 0
    total_tool_calls = 0
    total_iterations = 0

    for i, (user_msg, expected_kws) in enumerate(turns):
        turn_num = i + 1
        print(f"\n  [Turn {turn_num}] User: {user_msg}")

        t0 = time.time()
        resp = agent.run(user_msg)
        dt = time.time() - t0
        total_time += dt
        total_tool_calls += resp.tool_calls
        total_iterations += resp.iterations

        output = resp.output if resp.output else ""
        all_outputs.append(output)
        print(f"  [Turn {turn_num}] Agent: {output[:400]}")
        print(f"  [Turn {turn_num}] (tools={resp.tool_calls}, iters={resp.iterations}, {dt:.1f}s)")

        # Check intermediate turn keywords if provided
        if expected_kws:
            for kw in expected_kws:
                if kw.lower() not in output.lower():
                    print(f"  [Turn {turn_num}] MISSING keyword: '{kw}'")

    # Final check
    final_output = all_outputs[-1] if all_outputs else ""
    passed = True

    # Check final turn keywords from last turn spec
    last_kws = turns[-1][1] if turns[-1][1] else []
    for kw in last_kws:
        if kw.lower() not in final_output.lower():
            passed = False
            print(f"  FINAL MISSING keyword: '{kw}'")

    if check_fn:
        custom_pass = check_fn(final_output, all_outputs)
        if not custom_pass:
            passed = False
            print("  Custom check FAILED")

    status = "PASS" if passed else "FAIL"
    print(f"\nResult: {status} — {test_id}: {description}")
    print(f"  Total: {len(turns)} turns, {total_tool_calls} tool calls, {total_time:.1f}s")

    # Show memory stats
    mem_stats = agent.short_term_memory.get_statistics()
    print(f"  Memory: {mem_stats['current_messages']} msgs, {mem_stats['current_tokens']} tokens, "
          f"{mem_stats['total_summarizations']} summarizations")

    return {
        "test_id": test_id,
        "description": description,
        "passed": passed,
        "status": status,
        "turns": len(turns),
        "tool_calls": total_tool_calls,
        "time": total_time,
        "final_output": final_output[:400],
        "memory_stats": mem_stats,
    }


def create_conversational_agent(model, memory_config=None, system_prompt=None):
    """Create a conversational agent with memory enabled."""
    mem_cfg = memory_config or {
        "short_term_max_tokens": 4096,
        "short_term_max_messages": 50,
        "auto_summarize": True,
    }

    tools = [Calculator(), DateTimeTool()]

    config = AgentConfig(
        name="conversational_agent",
        model=model,
        tools=tools,
        system_prompt=system_prompt or CONVERSATIONAL_SYSTEM_PROMPT,
        max_iterations=8,
        temperature=0.1,
        enable_sub_agents=False,
        enable_memory=True,
        memory_config=mem_cfg,
    )

    return Agent(config)


def run_all_tests(model, model_name="unknown"):
    """Run all conversational agent tests."""
    results = []

    # ── T1: Name recall ──
    agent = create_conversational_agent(model)
    results.append(run_multi_turn_test(
        agent, "T1", "Name recall across turns",
        turns=[
            ("My name is Alice and I'm a data scientist working on NLP projects.", None),
            ("What's my name and what do I do?", ["alice", "data scientist"]),
        ],
    ))

    # ── T2: Preference recall ──
    agent = create_conversational_agent(model)
    results.append(run_multi_turn_test(
        agent, "T2", "Preference recall",
        turns=[
            ("I prefer Python over JavaScript for backend development. Remember that.", None),
            ("I'm starting a new backend project. Which programming language should I use based on my preferences?",
             ["python"]),
        ],
    ))

    # ── T3: Contextual tool use ──
    agent = create_conversational_agent(model)
    results.append(run_multi_turn_test(
        agent, "T3", "Contextual tool use — recall + calculator",
        turns=[
            ("Remember that my office temperature is 22 degrees Celsius.", None),
            ("Convert my office temperature to Fahrenheit. Use the calculator to compute 22 * 9/5 + 32.",
             ["71.6"]),
        ],
    ))

    # ── T4: Many-turn stress test ──
    agent = create_conversational_agent(model)
    results.append(run_multi_turn_test(
        agent, "T4", "Many-turn stress test — 10 turns, recall Turn 1",
        turns=[
            ("My favorite color is blue and my lucky number is 42.", None),
            ("I work at a company called TechCorp.", None),
            ("My pet's name is Luna and she is a golden retriever.", None),
            ("I live in San Francisco.", None),
            ("My birthday is on March 15th.", None),
            ("I enjoy hiking and reading sci-fi novels.", None),
            ("I'm learning to play the guitar.", None),
            ("My favorite food is sushi.", None),
            ("I drive a Tesla Model 3.", None),
            ("Now tell me: what is my favorite color and what is my lucky number?",
             ["blue", "42"]),
        ],
    ))

    # ── T5: Memory summarization trigger ──
    # Use very small max_tokens to force summarization quickly
    agent = create_conversational_agent(model, memory_config={
        "short_term_max_tokens": 512,
        "short_term_max_messages": 50,
        "keep_recent_messages": 2,
        "auto_summarize": True,
    })
    results.append(run_multi_turn_test(
        agent, "T5", "Memory summarization trigger (max_tokens=512)",
        turns=[
            (("I am a software engineer named Bob who works at Google on the search team. "
             "I have 10 years of experience and I specialize in distributed systems. "
             "My favorite programming language is Go and I also know Python, Java, and C++."), None),
            (("Tell me about the latest trends in distributed systems and cloud computing. "
             "I'm particularly interested in how microservices architecture has evolved "
             "and what the best practices are for service mesh implementations."), None),
            (("What are the key differences between Kubernetes and Docker Swarm for "
             "container orchestration? Which one would you recommend for a large-scale "
             "production deployment?"), None),
            ("Now, can you remind me what my name is and where I work?", ["bob"]),
        ],
        check_fn=lambda final, alls: True,  # Main check: no crashes
    ))
    # Verify summarization actually triggered
    mem_stats = agent.short_term_memory.get_statistics()
    if mem_stats["total_summarizations"] > 0:
        print(f"  ✓ Summarization triggered {mem_stats['total_summarizations']} time(s)")
    else:
        print(f"  ⚠ Summarization did NOT trigger (tokens: {mem_stats['current_tokens']}/{512})")

    # ── T6: Long-term memory persistence (SQLite) ──
    tmpdir = tempfile.mkdtemp(prefix="effgen_ltm_")
    print(f"\n  [T6] Using temp dir: {tmpdir}")

    # First session: store facts
    agent1 = create_conversational_agent(model, memory_config={
        "short_term_max_tokens": 4096,
        "short_term_max_messages": 50,
        "auto_summarize": True,
        "long_term_backend": "sqlite",
        "long_term_persist_path": tmpdir,
    })

    # Check that long-term memory was initialized
    ltm_ok = agent1.long_term_memory is not None
    print(f"\n{'='*60}")
    print("Test: T6 — Long-term memory persistence (SQLite)")
    print(f"  LongTermMemory initialized: {ltm_ok}")

    if ltm_ok:
        # Store a conversation that uses tools (to trigger LTM storage)
        print("\n  [Session 1] Storing facts...")
        resp1 = agent1.run("What is 100 + 200? Use the calculator.")
        output1 = resp1.output if resp1.output else ""
        print(f"  [Session 1] Agent: {output1[:200]}")
        print(f"  [Session 1] Tool calls: {resp1.tool_calls}")

        # End session
        if agent1.long_term_memory:
            agent1.long_term_memory.end_session()

        # Check SQLite file exists
        db_path = os.path.join(tmpdir, "long_term.db")
        db_exists = os.path.exists(db_path)
        print(f"  SQLite DB exists: {db_exists} ({db_path})")

        # Query the DB directly to verify persistence
        if db_exists:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            mem_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM sessions")
            sess_count = cursor.fetchone()[0]
            conn.close()
            print(f"  Persisted: {mem_count} memories, {sess_count} sessions")

        p6_t6_passed = db_exists and mem_count > 0
    else:
        p6_t6_passed = False
        print("  FAIL: LongTermMemory not initialized")

    status = "PASS" if p6_t6_passed else "FAIL"
    print(f"\nResult: {status} — T6: Long-term memory persistence (SQLite)")
    results.append({
        "test_id": "T6",
        "description": "Long-term memory persistence (SQLite)",
        "passed": p6_t6_passed,
        "status": status,
        "turns": 1,
        "tool_calls": resp1.tool_calls if ltm_ok else 0,
        "time": 0,
        "final_output": "",
        "memory_stats": {},
    })

    # Cleanup SQLite temp dir
    try:
        shutil.rmtree(tmpdir)
        print(f"  Cleaned up: {tmpdir}")
    except Exception as e:
        print(f"  Failed to clean {tmpdir}: {e}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Conversational Agent Results — Model: {model_name}")
    print(f"{'='*60}")
    pass_count = 0
    for r in results:
        if r["passed"]:
            pass_count += 1
        print(f"  {r['status']:5s} — {r['test_id']}: {r['description']} ({r.get('time', 0):.1f}s)")
    print(f"\n{pass_count}/{len(results)} passed")
    return results


def run_regression(model, model_name="unknown"):
    """Run regression tests on the conversational agent."""
    from effgen.presets import create_agent as create_preset_agent

    results = []

    # Use coding preset for regression (has all needed tools)
    agent = create_preset_agent(
        "coding", model,
        system_prompt=("You are a helpful AI assistant with coding tools. "
                      "Answer questions directly when possible. "
                      "Use tools only when needed for computation or code execution."),
        temperature=0.1,
        max_iterations=8,
    )

    def run_test(test_id, description, question, expected_keywords=None,
                 check_fn=None, expected_tool=None):
        print(f"\n{'='*60}")
        print(f"Test: {test_id} — {description}")
        print(f"Q: {question}")

        t0 = time.time()
        resp = agent.run(question)
        dt = time.time() - t0

        output = resp.output if resp.output else ""
        print(f"A: {output[:400]}")
        print(f"Tool calls: {resp.tool_calls}, Iterations: {resp.iterations}, Time: {dt:.1f}s")

        keyword_pass = True
        if expected_keywords:
            for kw in expected_keywords:
                if kw.lower() not in output.lower():
                    keyword_pass = False
                    print(f"  MISSING keyword: '{kw}'")

        custom_pass = True
        if check_fn:
            custom_pass = check_fn(output, resp)

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
                print(f"  Expected tool '{expected_tool}' not in: {trace_tools}")

        passed = keyword_pass and custom_pass and tool_pass
        status = "PASS" if passed else "FAIL"
        print(f"Result: {status}")

        return {
            "test_id": test_id, "description": description, "passed": passed,
            "status": status, "time": dt, "output": output[:300],
            "tool_calls": resp.tool_calls,
        }

    # REG-P1: Q&A
    results.append(run_test("REG-P1", "Q&A: capital of France",
        "What is the capital of France?", ["paris"]))

    # REG-P2: Calculator
    results.append(run_test("REG-P2", "Calculator: 15 + 27",
        "What is 15 + 27? Use a tool to calculate.", ["42"]))

    # REG-P3: Multi-tool — bash
    results.append(run_test("REG-P3", "Bash: echo test",
        "Run 'echo hello_regression' using bash and tell me the output.",
        ["hello_regression"], expected_tool="bash"))

    # REG-P4: Code execution
    results.append(run_test("REG-P4", "Python: simple code",
        "Use python_repl to run: print('regression_ok')",
        ["regression_ok"], expected_tool="python_repl"))

    # REG-P5: Multi-turn memory (quick 2-turn)
    conv_agent = create_conversational_agent(model)
    resp1 = conv_agent.run("Remember: the password is 'sunshine123'.")
    resp2 = conv_agent.run("What was the password I told you?")
    output2 = resp2.output if resp2.output else ""
    reg_p5_pass = "sunshine123" in output2.lower()
    print(f"\n{'='*60}")
    print("Test: REG-P5 — Multi-turn memory recall")
    print(f"  Turn 1: Remember password -> {resp1.output[:100] if resp1.output else 'None'}")
    print(f"  Turn 2: Recall password -> {output2[:200]}")
    print(f"Result: {'PASS' if reg_p5_pass else 'FAIL'}")
    results.append({
        "test_id": "REG-P5", "description": "Multi-turn memory recall",
        "passed": reg_p5_pass, "status": "PASS" if reg_p5_pass else "FAIL",
        "time": 0, "output": output2[:200], "tool_calls": 0,
    })

    print(f"\n{'='*60}")
    print(f"Regression Results — Model: {model_name}")
    print(f"{'='*60}")
    pass_count = sum(1 for r in results if r["passed"])
    for r in results:
        print(f"  {r['status']:5s} — {r['test_id']}: {r['description']} ({r.get('time', 0):.1f}s)")
    print(f"\n{pass_count}/{len(results)} regression passed")
    return results


def interactive_mode(agent):
    """Interactive conversational chat."""
    print("\n--- Interactive Conversational Agent (type 'quit' to exit) ---")
    print("Available tools: calculator, datetime_tool")
    print("Memory is ON — the agent remembers your conversation.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "memory":
                stats = agent.short_term_memory.get_statistics()
                print(f"Memory stats: {stats}")
                continue
            if not user_input:
                continue
            t0 = time.time()
            resp = agent.run(user_input)
            dt = time.time() - t0
            print(f"Agent: {resp.output}")
            print(f"   (tools={resp.tool_calls}, iters={resp.iterations}, {dt:.1f}s)")
        except (KeyboardInterrupt, EOFError):
            break
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="effGen — Conversational Agent")
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--regression", action="store_true", help="Run regression tests only")
    parser.add_argument("--tests-only", action="store_true", help="Run conversational tests only (skip regression)")
    args = parser.parse_args()

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print("effGen — Conversational Agent (Memory + Multi-Turn)")
    print(f"Model: {args.model}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={gpu}")

    print("\nLoading model...")
    t0 = time.time()
    model = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    if args.interactive:
        agent = create_conversational_agent(model)
        interactive_mode(agent)
    elif args.regression:
        run_regression(model, model_name=args.model)
    else:
        if not args.tests_only:
            print("\n--- Regression Tests ---")
            run_regression(model, model_name=args.model)

        print("\n--- Conversational Agent Tests ---")
        run_all_tests(model, model_name=args.model)

    model.unload()
    print("\nDone.")


if __name__ == "__main__":
    main()
