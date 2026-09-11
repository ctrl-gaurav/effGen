"""
Debug inspector for effGen agents.

Provides:
    - DebugTrace / DebugIteration: structured data for every ReAct step
    - DebugAgent: thin wrapper around Agent that injects debug=True
    - run_debug_cli: interactive TUI powered by rich

Usage:
    from effgen.debug import DebugAgent
    agent = DebugAgent(config)
    result = agent.run("What is 2+2?")
    trace = result.metadata["debug_trace"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trace data structures
# ---------------------------------------------------------------------------


def _conversation_text(iteration: "DebugIteration") -> str:
    """The iteration's conversation as text, from its steps where it has them.

    Rendering the steps is what the run itself renders, so what the panel shows
    is the transcript the model saw. An iteration recorded before the steps were
    kept falls back to the snapshot it stored.
    """
    saved = getattr(iteration, "thread_snapshot", None)
    if saved:
        try:
            from ..core.thread import AgentThread

            return AgentThread.from_dict(saved).to_text()
        except (ValueError, TypeError, KeyError):
            logger.debug("Debug trace: stored steps could not be read", exc_info=True)
    return iteration.scratchpad_snapshot


@dataclass
class DebugIteration:
    """Captured state of a single ReAct iteration."""
    iteration: int
    raw_prompt: str = ""
    raw_response: str = ""
    thought: str = ""
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    final_answer: str | None = None
    tokens_used: int = 0
    latency: float = 0.0
    scratchpad_snapshot: str = ""
    #: The run's conversation at the end of this iteration, as
    #: :meth:`effgen.core.thread.AgentThread.to_dict` data. ``scratchpad_snapshot``
    #: is the text those steps render to and is kept beside it.
    thread_snapshot: dict[str, Any] = field(default_factory=dict)
    memory_snapshot: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "iteration": self.iteration,
            "raw_prompt": self.raw_prompt,
            "raw_response": self.raw_response,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "final_answer": self.final_answer,
            "tokens_used": self.tokens_used,
            "latency": self.latency,
            "thread": self.thread_snapshot,
            "metadata": self.metadata,
        }


@dataclass
class DebugTrace:
    """Full trace of an agent run — attached to AgentResponse.metadata["debug_trace"]."""
    task: str = ""
    agent_name: str = ""
    run_id: str = ""
    iterations: list[DebugIteration] = field(default_factory=list)
    total_tokens: int = 0
    total_latency: float = 0.0
    final_answer: str | None = None
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "task": self.task,
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "iterations": [it.to_dict() for it in self.iterations],
            "total_tokens": self.total_tokens,
            "total_latency": self.total_latency,
            "final_answer": self.final_answer,
            "success": self.success,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """One-line summary."""
        return (
            f"DebugTrace({self.agent_name}, {len(self.iterations)} iters, "
            f"{self.total_tokens} tokens, {self.total_latency:.2f}s, "
            f"success={self.success})"
        )

    def print_rich(self) -> None:
        """Pretty-print the trace using rich (if available)."""
        _print_trace_rich(self)


# ---------------------------------------------------------------------------
# DebugAgent — wraps Agent and always returns a DebugTrace
# ---------------------------------------------------------------------------

class DebugAgent:
    """Wrapper around :class:`effgen.core.agent.Agent` that records a DebugTrace per run.

    Usage::

        from effgen.debug import DebugAgent
        agent = DebugAgent(config)
        result = agent.run("What is 2+2?")
        trace = result.metadata["debug_trace"]
    """

    def __init__(self, config: Any) -> None:
        """
        Args:
            config: AgentConfig instance (same as Agent.__init__)
        """
        from effgen.core.agent import Agent
        self._agent = Agent(config)

    def run(self, task: str, **kwargs: Any) -> Any:
        """
        Run the agent with debug=True.

        Returns an AgentResponse with ``metadata["debug_trace"]`` populated.
        """
        kwargs["debug"] = True
        return self._agent.run(task, **kwargs)

    @property
    def agent(self) -> Any:
        """Access the underlying Agent instance."""
        return self._agent

    def close(self) -> None:
        """Release the underlying agent's resources (avoids GC-close warnings)."""
        close = getattr(self._agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "DebugAgent":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Rich TUI for interactive debugging
# ---------------------------------------------------------------------------

def run_debug_cli(
    task: str,
    config: Any = None,
    preset: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    step: bool = False,
) -> int:
    """
    Run an agent in interactive debug mode with a rich TUI.

    Args:
        task: Task to execute
        config: AgentConfig (if None, built from preset/model)
        preset: Preset name (e.g. "math", "research")
        model: Model name/path
        provider: Optional provider for a bare model id (e.g. "groq").
            Equivalent to the ``provider:model`` prefix form.
        step: If True, pause after each iteration for user input

    Returns:
        Process exit code: ``0`` when the run succeeded, ``1`` when the agent
        run failed, and ``2`` for a user/config error (no model/preset, missing
        dependency). Lets ``effgen debug`` report accurate exit codes.
    """
    try:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text  # noqa: F401

        from effgen.ui.theme import get_console
    except ImportError:
        print("rich library required for debug CLI. Install with: pip install rich")
        return 2

    console = get_console()

    # Build agent config if not provided
    if config is None:
        config = _build_debug_config(preset=preset, model=model, provider=provider)
        if config is None:
            console.print(
                "[red]Could not create agent config.[/red] Provide -m/--model "
                "(e.g. `-m gpt-5-nano`) or --preset. Run `effgen models list` "
                "for options or `effgen doctor` to check usable providers.",
                highlight=False,
            )
            return 2

    console.print(Panel(f"[bold]effGen Debug Mode[/bold]\nTask: {task}", style="blue"))

    agent = DebugAgent(config)
    try:
        result = agent.run(task)
    finally:
        agent.close()
    trace: DebugTrace | None = result.metadata.get("debug_trace")

    if trace is None:
        console.print("[yellow]No debug trace captured.[/yellow]", highlight=False)
        console.print(f"Output: {result.output}", highlight=False)
        return 0 if getattr(result, "success", False) else 1

    # Display each iteration
    for it in trace.iterations:
        table = Table(title=f"Iteration {it.iteration}", show_lines=True)
        table.add_column("Field", style="cyan", width=16)
        table.add_column("Value", style="white")

        table.add_row("Thought", it.thought or "(none)")
        table.add_row("Action", it.action or "(none)")
        table.add_row("Action Input", _truncate(it.action_input or "", 200))
        table.add_row("Observation", _truncate(it.observation or "", 300))
        table.add_row("Tokens", str(it.tokens_used))
        table.add_row("Latency", f"{it.latency:.3f}s")

        if it.final_answer:
            table.add_row("Final Answer", it.final_answer)

        console.print(table)

        if step and not it.final_answer:
            action = console.input("[dim]Press Enter to continue, 's' for scratchpad, 'q' to quit: [/dim]")
            if action.strip().lower() == "q":
                break
            if action.strip().lower() == "s":
                console.print(Panel(_conversation_text(it) or "(empty)", title="Conversation"))

    # Summary
    summary = Table(title="Run Summary", show_lines=True)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Agent", trace.agent_name)
    summary.add_row("Iterations", str(len(trace.iterations)))
    summary.add_row("Total Tokens", str(trace.total_tokens))
    summary.add_row("Total Latency", f"{trace.total_latency:.3f}s")
    summary.add_row("Success", str(trace.success))
    summary.add_row("Output", _truncate(result.output, 300))
    console.print(summary)

    return 0 if getattr(result, "success", trace.success) else 1


# ---------------------------------------------------------------------------
# Rich trace printer (non-interactive)
# ---------------------------------------------------------------------------

def _print_trace_rich(trace: DebugTrace) -> None:
    """Pretty-print a DebugTrace using rich."""
    try:
        from rich.panel import Panel
        from rich.table import Table

        from effgen.ui.theme import get_console
    except ImportError:
        # Fallback to plain text
        print(trace.summary())
        for it in trace.iterations:
            print(f"  [{it.iteration}] action={it.action} obs={_truncate(it.observation or '', 80)}")
        return

    console = get_console()
    console.print(Panel(trace.summary(), title="Debug Trace", style="blue"))

    for it in trace.iterations:
        table = Table(title=f"Iteration {it.iteration}", show_lines=True)
        table.add_column("Field", style="cyan", width=16)
        table.add_column("Value")
        table.add_row("Thought", it.thought or "(none)")
        table.add_row("Action", it.action or "(none)")
        table.add_row("Observation", _truncate(it.observation or "", 200))
        if it.final_answer:
            table.add_row("Final Answer", it.final_answer)
        table.add_row("Tokens / Latency", f"{it.tokens_used} / {it.latency:.3f}s")
        console.print(table)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s


def _build_debug_config(
    preset: str | None, model: str | None, provider: str | None = None
) -> Any:
    """Build an AgentConfig from preset/model args."""
    if preset:
        try:
            from effgen.presets import create_agent
            # create_agent returns an Agent, we need its config
            kwargs = {"provider": provider} if provider else {}
            agent = create_agent(
                preset, model or "Qwen/Qwen2.5-1.5B-Instruct", **kwargs
            )
            return agent.config
        except Exception:
            pass

    if model:
        from effgen.core.agent import AgentConfig
        from effgen.tools.builtin import Calculator
        return AgentConfig(
            name="debug_agent",
            model=model,
            provider=provider,
            tools=[Calculator()],
        )

    return None
