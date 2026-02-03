#!/usr/bin/env python3
"""
Retrieval Agent Example - Demonstrates the Retrieval (RAG) tool.

This example shows how to use the Retrieval tool to give an agent
access to a knowledge base for answering questions using semantic search.

The Retrieval tool:
- Uses embeddings to find semantically similar content
- Supports document chunking for long texts
- Falls back to TF-IDF if sentence-transformers is not available
- Can load documents from JSON, JSONL, or plain text files
"""

import logging
from pathlib import Path

from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Retrieval, Calculator

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


def main():
    print("=" * 80)
    print("RETRIEVAL AGENT EXAMPLE")
    print("Using Retrieval (RAG) tool for knowledge-base Q&A")
    print("=" * 80)

    # Check if we have the ARC dataset
    data_dir = Path(__file__).parent / "data"
    kb_path = data_dir / "arc_easy_test_kb.json"

    if not kb_path.exists():
        print("\nKnowledge base not found. Please run:")
        print("  python examples/data/download_arc.py --output-dir examples/data")
        print("\nOr you can add your own documents to the Retrieval tool.")
        return

    print("\n[1] Loading model...")
    try:
        model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")
        print("    Model loaded successfully!")
    except Exception as e:
        print(f"    Failed to load model: {e}")
        return

    print("\n[2] Initializing Retrieval tool...")
    retrieval = Retrieval(
        chunk_size=500,      # Characters per chunk
        chunk_overlap=100,   # Overlap between chunks
    )

    # Load knowledge base
    num_docs = retrieval.add_from_file(str(kb_path), file_type="json", chunk=False)
    print(f"    Loaded {num_docs} documents into the knowledge base")

    print("\n[3] Creating agent with Retrieval tool...")
    config = AgentConfig(
        name="retrieval_agent",
        model=model,
        tools=[retrieval, Calculator()],
        system_prompt=(
            "You are a helpful assistant with access to a knowledge base. "
            "When asked questions, ALWAYS use the retrieval tool first to search "
            "the knowledge base for relevant information. Use the search results "
            "to provide accurate answers."
        ),
        max_iterations=5,
    )
    agent = Agent(config=config)

    # Test questions
    test_questions = [
        "What is the structure of an atom?",
        "Where does meiosis occur?",
        "What unit is used to measure distances between galaxies?",
    ]

    print("\n[4] Running test questions...")
    print("-" * 80)

    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        result = agent.run(question)
        print(f"Answer: {result.output[:200]}..." if len(result.output) > 200 else f"Answer: {result.output}")
        print(f"Tool calls: {result.tool_calls}, Success: {result.success}")

    print("\n" + "=" * 80)
    print("RETRIEVAL AGENT EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
