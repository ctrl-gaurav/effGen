"""
Basic effGen agent using MLX on Apple Silicon.

This example demonstrates using the LiquidAI LFM2.5-1.2B-Instruct model
via the MLX backend for native Metal GPU acceleration on M-series Macs.

Requirements:
    pip install effgen[mlx]
    # or: pip install mlx mlx-lm

Usage:
    python examples/basic/basic_agent_mlx.py
"""

import logging
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.models.base import GenerationConfig
from effgen.tools.builtin import Calculator, PythonREPL, CodeExecutor

# ============================================================================
# CONFIGURATION
# ============================================================================
DETAILED_LOGGING = False  # Change to True for debug output
MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit"
# ============================================================================

if DETAILED_LOGGING:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
else:
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


def main():
    print("=" * 70)
    print("effGen Agent — MLX Backend (Apple Silicon)")
    print(f"Model: {MODEL_ID}")
    print("=" * 70)

    # Load model with MLX engine
    # The model is pre-quantized to 8-bit — no additional quantization needed.
    # MLX auto-detects on Apple Silicon; you can also pass engine="mlx" explicitly.
    model = load_model(
        MODEL_ID,
        engine="mlx",
        apply_chat_template=True,
    )

    # LFM2.5 recommended generation parameters
    gen_config = GenerationConfig(
        temperature=0.1,       # Near-greedy (model recommendation)
        top_p=0.1,             # Tight nucleus sampling
        max_tokens=512,
        repetition_penalty=1.05,
    )

    # --- Example 1: Simple question ---
    print("\n[Example 1] Direct generation (no agent)")
    print("-" * 50)
    result = model.generate(
        "Explain what unified memory means on Apple Silicon in 2 sentences.",
        config=gen_config,
    )
    print(f"Response: {result.text}")
    print(f"Tokens used: {result.tokens_used}")

    # --- Example 2: Streaming ---
    print("\n[Example 2] Streaming generation")
    print("-" * 50)
    print("Response: ", end="", flush=True)
    for chunk in model.generate_stream(
        "What are 3 benefits of small language models?",
        config=gen_config,
    ):
        print(chunk, end="", flush=True)
    print()

    # --- Example 3: Agent with tools ---
    print("\n[Example 3] Agent with tools")
    print("-" * 50)
    config = AgentConfig(
        name="mlx_calculator_agent",
        model=model,
        tools=[Calculator(), PythonREPL(), CodeExecutor()],
        system_prompt="You are a helpful assistant. Use tools when needed.",
    )
    agent = Agent(config=config)
    result = agent.run("What is 24344 * 334?")

    print(f"Output: {result.output}")
    print(f"Success: {result.success}")
    if hasattr(result, 'execution_history') and result.execution_history:
        print(f"Steps: {len(result.execution_history)}")

    # --- Example 4: Batch generation ---
    print("\n[Example 4] Batch generation")
    print("-" * 50)
    prompts = [
        "What is the capital of Japan?",
        "Name one benefit of edge AI.",
        "What does RAG stand for in AI?",
    ]
    results = model.generate_batch(prompts, config=gen_config)
    for prompt, res in zip(prompts, results):
        print(f"  Q: {prompt}")
        print(f"  A: {res.text[:150]}")
        print()

    # Cleanup
    model.unload()
    print("=" * 70)
    print("Done. Model unloaded.")


if __name__ == '__main__':
    main()
