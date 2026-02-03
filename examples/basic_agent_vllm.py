import logging
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL, CodeExecutor

# ============================================================================
# CONFIGURATION: Set to True for detailed logging, False for minimal logging
# ============================================================================
DETAILED_LOGGING = False  # ← Change this to True/False
# ============================================================================

# Configure logging based on DETAILED_LOGGING setting
if DETAILED_LOGGING:
    logging.basicConfig(
        level=logging.DEBUG,  # Show all logs including DEBUG
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
else:
    # Minimal logging - only show warnings and errors
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s'
    )


def main():
    """Main function - required for vLLM multiprocessing."""
    if DETAILED_LOGGING:
        print("=" * 80)
        print("STARTING AGENT TEST WITH DETAILED LOGGING (vLLM)")
        print("=" * 80)
    else:
        print("=" * 80)
        print("STARTING AGENT TEST (Minimal Logging) - vLLM Backend")
        print("=" * 80)

    # Load model with vLLM engine for faster inference
    # Key parameters:
    #   - engine="vllm": Use vLLM backend (5-10x faster than Transformers)
    #   - gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0), lower if OOM
    #   - apply_chat_template: Automatically format prompts for instruction-tuned models
    #   - trust_remote_code: Required for some models like Qwen
    model = load_model(
        "Qwen/Qwen2.5-3B-Instruct",
        engine="vllm",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    # Create agent configuration
    config = AgentConfig(
        name="calculator_agent",
        model=model,
        tools=[CodeExecutor(), Calculator(), PythonREPL()],
        system_prompt="You are a helpful assistant."
    )

    # Create agent
    agent = Agent(config=config)

    # Run a task
    print("\n" + "=" * 80)
    print("RUNNING AGENT TASK")
    print("=" * 80)

    result = agent.run("What is 24344*334?")

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"Output: {result.output}")
    print(f"Success: {result.success}")
    print(f"Steps taken: {len(result.execution_history) if hasattr(result, 'execution_history') else 'N/A'}")

    # Print detailed execution history only if DETAILED_LOGGING is enabled
    if DETAILED_LOGGING and hasattr(result, 'execution_history') and result.execution_history:
        print("\n" + "=" * 80)
        print("EXECUTION HISTORY")
        print("=" * 80)
        for i, step in enumerate(result.execution_history, 1):
            print(f"\n[Step {i}]")
            print(f"  Type: {step.get('type', 'unknown')}")
            print(f"  Content: {step.get('content', '')[:200]}...")  # First 200 chars

    # Unload model to free GPU memory
    model.unload()


# IMPORTANT: This guard is required for vLLM's multiprocessing to work correctly
if __name__ == '__main__':
    main()
