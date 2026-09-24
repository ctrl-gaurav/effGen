"""
Cerebras CostTracker example.

Prerequisites:
    pip install "effgen[cerebras]"
    export CEREBRAS_API_KEY="your-key"   # or place in ~/.effgen/.env

What this demonstrates:
  - CostTracker accumulating token usage across multiple calls
  - Cerebras free tier = $0 for all models
  - Comparing per-model token counts
  - Resetting stats and starting fresh
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".effgen" / ".env", override=False)

if not os.getenv("CEREBRAS_API_KEY"):
    raise SystemExit("Set CEREBRAS_API_KEY in ~/.effgen/.env or the environment.")

from effgen.models._cost import CostTracker  # noqa: E402
from effgen.models.cerebras_adapter import CerebrasAdapter  # noqa: E402


def _cost(row: dict) -> str:
    """Render a row's spend. ``None`` means the model publishes no rate."""
    cost = row["cost_usd"]
    return "unpriced" if cost is None else f"${cost:.6f}"


def print_summary(label: str) -> None:
    tracker = CostTracker.get()
    print(f"\n--- {label} ---")
    summary = tracker.summary()
    if not summary:
        print("  (no calls recorded yet)")
        return
    for row in summary:
        print(
            f"  {row['provider']}/{row['model']}\n"
            f"    requests:    {row['requests']}\n"
            f"    prompt:      {row['prompt_tokens']} tokens\n"
            f"    completion:  {row['completion_tokens']} tokens\n"
            f"    total:       {row['total_tokens']} tokens\n"
            f"    cost:        {_cost(row)} (free tier = $0)"
        )
    total = tracker.total_cost()
    # None when every call recorded so far was on a model with no published price.
    print(f"\n  Grand total cost: {'unpriced' if total is None else f'${total:.6f}'}")


def main():
    print("CostTracker demo — accumulating token usage across Cerebras calls")
    print_summary("Initial state (empty)")

    # Make some calls with gpt-oss-120b
    llama = CerebrasAdapter("gpt-oss-120b", enable_rate_limiting=False, enable_cost_tracking=True)
    llama.load()

    print("\nMaking 3 calls with gpt-oss-120b...")
    llama.generate("What is the capital of France?")
    llama.generate("Name three programming languages.")
    llama.generate("Explain what a neural network is in one sentence.")
    llama.unload()

    print_summary("After 3 gpt-oss-120b calls")

    # Make a call with qwen
    qwen = CerebrasAdapter(
        "zai-glm-4.7",
        enable_rate_limiting=False,
        enable_cost_tracking=True,
    )
    qwen.load()
    print("\nMaking 1 call with zai-glm-4.7...")
    qwen.generate("What is machine learning?")
    qwen.unload()

    print_summary("After adding 1 qwen call")

    # Show per-model filtering
    tracker = CostTracker.get()
    llama_tokens = tracker.total_tokens("cerebras", "gpt-oss-120b")
    print(f"\ngpt-oss-120b total tokens: {llama_tokens['total']}")

    # Reset stats
    print("\nResetting CostTracker stats...")
    tracker.reset_stats()
    print_summary("After reset")


if __name__ == "__main__":
    main()
