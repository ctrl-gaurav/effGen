#!/usr/bin/env python3
"""
Agentic Search Agent Example - Demonstrates the AgenticSearch tool.

This example shows how to use the AgenticSearch tool to give an agent
access to a knowledge base using grep-based exact matching.

The AgenticSearch tool:
- Uses bash commands (grep) for exact string matching
- Provides context lines around each match
- Works well for technical queries, exact terms, and factual lookups
- Does not require embeddings - uses the file system directly
- Extracts keywords from queries and searches for them
"""

import logging
import asyncio
from pathlib import Path

from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import AgenticSearch, Calculator

# ============================================================================
# CONFIGURATION: Set to True for detailed logging, False for minimal logging
# ============================================================================
DETAILED_LOGGING = False
# ============================================================================

# Configure logging
if DETAILED_LOGGING:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
else:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )


async def main():
    print("=" * 80)
    print("AGENTIC SEARCH AGENT EXAMPLE")
    print("Using AgenticSearch tool for grep-based knowledge retrieval")
    print("=" * 80)

    # Check if we have the ARC dataset
    data_dir = Path(__file__).parent / "data"
    txt_path = data_dir / "arc_easy_test.txt"

    if not txt_path.exists():
        print("\nKnowledge base not found. Please run:")
        print("  python examples/data/download_arc.py --output-dir examples/data")
        print("\nOr you can point the AgenticSearch tool to your own text files.")
        return

    print("\n[1] Loading model...")
    try:
        model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")
        print("    Model loaded successfully!")
    except Exception as e:
        print(f"    Failed to load model: {e}")
        return

    print("\n[2] Initializing AgenticSearch tool...")
    agentic_search = AgenticSearch(
        data_path=str(txt_path),  # Path to search
        context_lines=5,           # Lines of context around matches
        max_results=3,             # Maximum results to return
    )
    await agentic_search.initialize()
    print(f"    AgenticSearch ready, searching in: {txt_path.name}")

    print("\n[3] Creating agent with AgenticSearch tool...")
    config = AgentConfig(
        name="agentic_search_agent",
        model=model,
        tools=[agentic_search, Calculator()],
        system_prompt=(
            "You are a helpful assistant with access to a knowledge base via the agentic_search tool. "
            "This tool uses grep to find exact matches in the knowledge base. "
            "When asked questions, ALWAYS use the agentic_search tool first to search "
            "for relevant information using keywords from the question. "
            "The search returns matching text with context. Use this to answer questions."
        ),
        max_iterations=5,
    )
    agent = Agent(config=config)

    # Test questions
    test_questions = [
        "What is photosynthesis?",
        "What do plants give off that animals need to breathe?",
        "What unit is used to measure astronomical distances?",
    ]

    print("\n[4] Running test questions...")
    print("-" * 80)

    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        result = agent.run(question)
        print(f"Answer: {result.output[:200]}..." if len(result.output) > 200 else f"Answer: {result.output}")
        print(f"Tool calls: {result.tool_calls}, Success: {result.success}")

    print("\n" + "=" * 80)
    print("AGENTIC SEARCH AGENT EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
