"""The ``effgen run`` command and its interactive wizard.

:mod:`effgen.cli._main` parses the arguments and exposes these through the
matching ``CLIInterface`` methods. Names that a caller may replace on the
``effgen.cli._main`` module — the agent classes and the Rich renderables behind
its single availability shim — are read from that module at call time rather
than imported here, so an override on ``effgen.cli._main`` still takes effect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Live status / progress presentation (TTY-aware; degrades to plain text).
from effgen.cli import progress as _progress
from effgen.cli.commands._shared import (
    _checkpoint_run_kwargs,
    _invoked_command,
    _preflight_model_hint,
    _quickstart_suggest_model,
    _warn_unapplied_config_keys,
    filter_incompatible_tools,
    resolve_provider_name,
)
from effgen.ui.render import ascii_fold as _ascii_fold
from effgen.ui.render import json_ensure_ascii

if TYPE_CHECKING:
    from effgen.cli._main import CLIInterface

logger = logging.getLogger(__name__)


def _failure_text(response: Any) -> str:
    """What to show in the error panel of a failed run.

    Most failures put the classified message in the answer, but a run refused
    before any model call — an empty task, a blocked input — has no answer
    text and carries its message on the typed error instead. An error panel
    with nothing in it tells the reader nothing, so fall back to that message.
    """
    text = (getattr(response, "output", "") or "").strip()
    if text:
        return text
    metadata = getattr(response, "metadata", None) or {}
    error = metadata.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or "").strip()
        if message:
            return message
    reason = str(metadata.get("reason") or "").strip()
    return reason or "The run failed and reported no message."


def run_document(response: Any) -> dict[str, Any]:
    """The run as JSON data, with anything key-shaped in it redacted.

    Every document ``effgen run`` writes — ``--json`` on stdout, ``-o`` to a
    file, ``--card`` to an HTML page — is meant to be piped, saved and shared,
    so a provider key that reached a tool's output, an instruction or the
    answer is replaced by a labelled placeholder on the way out. The library's
    own :meth:`~effgen.core.agent_response.AgentResponse.to_dict` is untouched
    and still returns exactly what the run produced.
    """
    from effgen.observability.redact import get_redactor

    return get_redactor().scrub_dict(response.to_dict())


def print_thread(cli: "CLIInterface", response: Any) -> None:
    """Print the run's conversation, step by step.

    Shows what the model was actually sent on every turn of a completed run —
    the persona and the question it was framed by, each thought, each tool call
    with the input it was given, each result and how the run ended — with no
    debug logging turned on. Secrets are redacted and nothing that differs
    between two runs of the same path is printed, so two of these diff cleanly.
    """
    from effgen.core.thread_render import render_thread

    cli.print_header("Conversation")
    steps = render_thread(getattr(response, "thread", None))
    if not steps:
        cli.print("(no conversation recorded for this run)")
        return
    for step in steps:
        text = _ascii_fold(step.to_text(), cli._human_stream())
        if cli.console:
            cli.console.print(text, highlight=False)
        else:
            print(text)


#: Heading over the text a run had reached when its iteration cap stopped it.
PARTIAL_PROGRESS_TITLE = "Partial progress (not an answer)"


def present_response(cli: "CLIInterface", response: Any) -> None:
    """Print a finished run: its answer, or what stopped it.

    Three outcomes read differently. A success is framed as the answer. A run
    the loop stopped — at its iteration cap, on a repeated call, on a repeated
    result, or on an empty final answer — has no answer, so the panel carries
    what stopped it and what to do, and the tool output and reasoning it had
    reached follows in its own panel under :data:`PARTIAL_PROGRESS_TITLE` —
    labelled as progress, so tool output is not read as a result. Any other
    failure reads as a red "Error" panel, the same as a model-load failure.
    """
    metadata = getattr(response, "metadata", None) or {}
    partial = bool(metadata.get("partial"))
    stopped = getattr(response, "outcome", None) == "stopped" or (
        not response.success and partial
    )
    progress = str(metadata.get("partial_output") or "") if stopped else ""

    if not response.success and not stopped:
        cli.print_error_panel(_failure_text(response), title="Error")
        return
    if cli.console:
        from effgen.ui.render import answer_surface

        answer_surface(
            response.output,
            success=response.success,
            partial=partial,
            framed=True,
            title="Stopped" if stopped else "Agent Response",
            console=cli.console,
        )
        if progress:
            answer_surface(
                progress,
                success=False,
                partial=True,
                framed=True,
                title=PARTIAL_PROGRESS_TITLE,
                console=cli.console,
            )
        return
    print(response.output)
    if progress:
        print()
        print(f"{PARTIAL_PROGRESS_TITLE}:")
        print(progress)


#: What a caller is told when the wizard is reached without a terminal to
#: prompt on. Written to stderr as a single block so a pipeline sees nothing on
#: stdout; pinned byte-for-byte by ``tests/cli/test_brand.py``.
NO_TERMINAL_HINT = (
    "No interactive terminal detected, so the setup wizard can't prompt for "
    "input. Pass your task directly, e.g.:\n"
    '    effgen run "What is the capital of France?"\n'
    "  (add --provider/--model to pick a backend; see 'effgen run --help')."
)


def interactive_wizard(cli: "CLIInterface", args: Any) -> int | None:
    """
    Interactive setup wizard for configuring and running agents.

    Walks the caller through a reasoning style, a tool selection, a model, the
    advanced options and the task prompt, then runs the agent it built.

    Every step reads from stdin, so without a terminal the wizard cannot run at
    all. In that case it writes :data:`NO_TERMINAL_HINT` to stderr, prints
    nothing to stdout and returns 2 without rendering any of the steps.

    Args:
        cli: The command-line interface the wizard prints and runs through.
        args: Parsed command-line arguments (may have partial values).

    Returns:
        Exit code.
    """
    from effgen.cli import _main

    if not sys.stdin.isatty():
        print(NO_TERMINAL_HINT, file=sys.stderr)
        return 2

    Agent, AgentConfig = _main.Agent, _main.AgentConfig
    # Rich renderables are only reached when ``cli.console`` exists, which is
    # exactly when rich imported; read them defensively so a rich-less install
    # enters the wizard and takes the plain-text branches.
    Panel = getattr(_main, "Panel", None)
    Table = getattr(_main, "Table", None)
    __version__ = _main.__version__

    cli.print_header(f"effGen v{__version__} - Interactive Setup Wizard")
    cli.print()

    if cli.console:
        cli.console.print(Panel(
            "[bold cyan]Welcome to effGen Interactive Mode![/bold cyan]\n\n"
            "This wizard will guide you through setting up and running an agent.\n"
            "Press Ctrl+C at any time to exit.",
            title="Interactive Setup",
            border_style="cyan"
        ))
    else:
        print("=" * 60)
        print("Welcome to effGen Interactive Mode!")
        print("This wizard will guide you through setting up an agent.")
        print("Press Ctrl+C at any time to exit.")
        print("=" * 60)

    try:
        # Step 1: Reasoning style. effGen has ONE Agent class whose behavior
        # adapts to the task; this picks the tool-using strategy (not a
        # separate agent class).
        cli.print_header("Step 1: Select Reasoning Style")
        agent_types = [
            ("1", "auto", ("Let effGen choose: native tool-calling when the model "
                          "supports it, else ReAct (recommended)")),
            ("2", "react", "Explicit Reason → Act → Observe loop with tools"),
            ("3", "single", "One model call, no tool loop (plain Q&A)"),
        ]

        if cli.console:
            table = Table(title="Reasoning Styles (one Agent, different strategies)")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Style", style="magenta")
            table.add_column("Description", style="white")
            for num, name, desc in agent_types:
                table.add_row(num, name, desc)
            cli.console.print(table)
        else:
            for num, name, desc in agent_types:
                print(f"  [{num}] {name}: {desc}")

        agent_type_input = input("\nSelect reasoning style [1]: ").strip() or "1"
        agent_type_map = {"1": "auto", "2": "react", "3": "single"}
        agent_type = agent_type_map.get(agent_type_input, "auto")
        cli.print_success(f"Selected: {agent_type}")

        # Step 2: Tool Selection
        cli.print_header("Step 2: Select Tools")

        # Discover and list available tools
        cli.tool_registry.discover_builtin_tools()
        available_tools = cli.tool_registry.list_tools()

        if cli.console:
            table = Table(title=f"Available Tools ({len(available_tools)})")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Name", style="magenta")
            table.add_column("Description", style="white")

            for i, tool_name in enumerate(available_tools, 1):
                try:
                    metadata = cli.tool_registry.get_metadata(tool_name)
                    desc = metadata.description[:40] + "..." if len(metadata.description) > 40 else metadata.description
                    table.add_row(str(i), tool_name, desc)
                except Exception:
                    table.add_row(str(i), tool_name, "No description")
            cli.console.print(table)
        else:
            for i, tool_name in enumerate(available_tools, 1):
                print(f"  [{i}] {tool_name}")

        cli.print("\nEnter tool numbers separated by commas (e.g., 1,2,3)")
        cli.print("Or press Enter to use all tools, 'none' for no tools")
        tool_input = input("Tools [all]: ").strip().lower()

        selected_tools = []
        if tool_input == "none":
            pass
        elif tool_input == "" or tool_input == "all":
            for name in available_tools:
                try:
                    tool = asyncio.run(cli.tool_registry.get_tool(name))
                    selected_tools.append(tool)
                except Exception:
                    pass
        else:
            try:
                indices = [int(x.strip()) - 1 for x in tool_input.split(",")]
                for idx in indices:
                    if 0 <= idx < len(available_tools):
                        tool_name = available_tools[idx]
                        try:
                            tool = asyncio.run(cli.tool_registry.get_tool(tool_name))
                            selected_tools.append(tool)
                        except Exception as e:
                            cli.print_warning(f"Failed to load {tool_name}: {e}")
            except ValueError:
                cli.print_warning("Invalid input, using all tools")
                for name in available_tools:
                    try:
                        tool = asyncio.run(cli.tool_registry.get_tool(name))
                        selected_tools.append(tool)
                    except Exception:
                        pass

        cli.print_success(f"Selected {len(selected_tools)} tool(s)")

        # Step 3: Model Configuration
        cli.print_header("Step 3: Configure Model")

        model_types = [
            ("1", "TransformersModel", "Local Hugging Face model (e.g., Qwen/Qwen2.5-1.5B-Instruct)"),
            ("2", "OpenAIModel", "OpenAI API (requires OPENAI_API_KEY)"),
            ("3", "AnthropicModel", "Anthropic API (requires ANTHROPIC_API_KEY)"),
            ("4", "vLLMModel", "vLLM server (requires running vLLM instance)"),
            ("5", "LiteLLMModel", "LiteLLM proxy (supports multiple backends)")
        ]

        if cli.console:
            table = Table(title="Model Types")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Type", style="magenta")
            table.add_column("Description", style="white")
            for num, name, desc in model_types:
                table.add_row(num, name, desc)
            cli.console.print(table)
        else:
            for num, name, desc in model_types:
                print(f"  [{num}] {name}: {desc}")

        model_type_input = input("\nSelect model type [1]: ").strip() or "1"

        # Get model ID based on type
        default_models = {
            "1": "Qwen/Qwen2.5-1.5B-Instruct",
            "2": "gpt-4o-mini",
            "3": "claude-3-haiku-20240307",
            "4": "Qwen/Qwen2.5-7B-Instruct",
            "5": "openai/gpt-4o-mini"
        }

        default_model = default_models.get(model_type_input, "Qwen/Qwen2.5-1.5B-Instruct")
        model_id = input(f"Model ID [{default_model}]: ").strip() or default_model
        cli.print_success(f"Model: {model_id}")

        # Step 4: Advanced Options
        cli.print_header("Step 4: Advanced Options")

        temp_input = input("Temperature [0.7]: ").strip()
        temperature = float(temp_input) if temp_input else 0.7

        max_iter_input = input("Max iterations [10]: ").strip()
        max_iterations = int(max_iter_input) if max_iter_input else 10

        sub_agents_input = input("Enable sub-agents? [Y/n]: ").strip().lower()
        enable_sub_agents = sub_agents_input != "n"

        stream_input = input("Stream output? [y/N]: ").strip().lower()
        enable_streaming = stream_input == "y"

        # Step 5: Task Input
        cli.print_header("Step 5: Enter Task")

        if cli.console:
            cli.console.print("[italic]Enter your task or question for the agent.[/italic]", highlight=False)
            cli.console.print("[dim]For multi-line input, end with an empty line.[/dim]\n", highlight=False)
        else:
            print("Enter your task or question for the agent.")
            print("For multi-line input, end with an empty line.\n")

        lines = []
        while True:
            try:
                line = input("> " if not lines else "  ")
                if line == "" and lines:
                    break
                lines.append(line)
            except EOFError:
                break

        task = "\n".join(lines).strip()

        if not task:
            cli.print_error("No task provided")
            return 1

        # Confirm and Run
        cli.print_header("Configuration Summary")

        summary = {
            "Reasoning Style": agent_type,
            "Model": model_id,
            "Tools": len(selected_tools),
            "Temperature": temperature,
            "Max Iterations": max_iterations,
            "Sub-agents": "enabled" if enable_sub_agents else "disabled",
            "Streaming": "enabled" if enable_streaming else "disabled",
            "Task": task[:50] + "..." if len(task) > 50 else task
        }

        if cli.console:
            table = Table(title="Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="magenta")
            for key, value in summary.items():
                table.add_row(key, str(value))
            cli.console.print(table)
        else:
            for key, value in summary.items():
                print(f"  {key}: {value}")

        confirm = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
        if confirm == "n":
            cli.print_warning("Cancelled by user")
            return 0

        # Create agent and run task
        cli.print_header("Running Agent")

        # Filter provider-specific native tools that are incompatible with
        # the selected model so the agent doesn't reject them at startup.
        selected_tools, _skipped = filter_incompatible_tools(
            selected_tools, model_id, warn=cli.print_warning
        )

        agent_config = AgentConfig(
            name="interactive-agent",
            raise_on_error=False,
            model=model_id,
            tools=selected_tools,
            temperature=temperature,
            max_iterations=max_iterations,
            enable_sub_agents=enable_sub_agents,
            enable_streaming=enable_streaming
        )

        agent = Agent(agent_config)

        if enable_streaming:
            if cli.console:
                cli.console.print("\n[effgen.success]Agent:[/effgen.success] ", end="", highlight=False)
            else:
                print("\nAgent: ", end="", flush=True)

            _progress.stream_answer(
                cli.console,
                agent.stream(task),
                animate=_progress.animation_enabled(),
                interactive=True,
                render_plain=True,
                trailing_newline=True,
            )
        else:
            if cli.console:
                with _progress.thinking_status(cli.console, animate=True):
                    response = agent.run(task)
            else:
                print("Thinking...")
                response = agent.run(task)

            # Display response
            cli.print_header("Response")

            if cli.console:
                from effgen.ui.render import answer_surface
                answer_surface(
                    response.output,
                    success=response.success,
                    framed=True,
                    title="Agent Response",
                    console=cli.console,
                )
            else:
                print(response.output)

            # Display statistics
            cli.print_header("Execution Statistics")
            stats = {
                "Success": "Yes" if response.success else "No",
                "Iterations": response.iterations,
                "Tool Calls": response.tool_calls,
                "Tokens Used": response.tokens_used,
                "Execution Time": f"{response.execution_time:.2f}s"
            }

            if cli.console:
                stats_table = Table()
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="magenta")
                for key, value in stats.items():
                    stats_table.add_row(key, str(value))
                cli.console.print(stats_table)
            else:
                for key, value in stats.items():
                    print(f"  {key}: {value}")

        # Ask if user wants to continue
        continue_input = input("\nRun another task? [y/N]: ").strip().lower()
        if continue_input == "y":
            return cli.interactive_wizard(args)

        return 0

    except KeyboardInterrupt:
        cli.print("\n\nWizard cancelled")
        return 130
    except EOFError:
        # stdin was a terminal at the guard above but reached end-of-input part
        # way through — a closed terminal or a harness that hands over a tty and
        # then stops writing. Expected, not a crash: same hint, same exit code.
        print(NO_TERMINAL_HINT, file=sys.stderr)
        return 2
    except Exception as e:
        cli.print_error(f"Error in interactive wizard: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        return 1


def run_agent(cli: "CLIInterface", args: argparse.Namespace) -> int | None:
    """
    Run an agent with a task.

    Args:
        args: Parsed command-line arguments
    """
    from effgen.cli import _main

    Agent, AgentConfig, AgentMode = _main.Agent, _main.AgentConfig, _main.AgentMode
    # See the note in ``interactive_wizard``: this is reached only behind
    # ``cli.console``.
    Syntax = getattr(_main, "Syntax", None)
    __version__ = _main.__version__

    input_files = getattr(args, 'input_files', None)

    # Check if we need to launch interactive wizard
    if args.task is None and not input_files:
        return cli.interactive_wizard(args)
    if args.task is None:
        args.task = ""

    # Attach --file/--input content: an image becomes multimodal `inputs=`;
    # a document is read with the same loaders RAG ingestion uses and
    # prepended to the task as text context. With ``--preset rag`` the
    # documents are instead indexed into the agent's retrieval tool, so a
    # CLI user can point the rag preset at a knowledge base without dropping
    # into Python — the agent retrieves and cites instead of seeing the raw
    # text as context.
    extra_inputs: list[Any] = []
    rag_docs: list[str] = []
    _preset_is_rag = getattr(args, 'preset', None) == 'rag'
    if input_files:
        from effgen.core.multimodal import _media_kind_from_name, image_from
        from effgen.rag.ingest import DocumentIngester

        doc_sections = []
        for file_path in input_files:
            p = Path(file_path)
            if not p.exists():
                cli.print_error(f"--file: file not found: {file_path}")
                return 1
            if _media_kind_from_name(str(p)) == "image":
                extra_inputs.append(image_from(p))
                continue
            if _preset_is_rag:
                rag_docs.append(str(p))
                continue
            ingester = DocumentIngester(
                show_progress=False, chunk_size=10_000_000,
                chunk_overlap=0, dedupe=False,
            )
            doc_chunks = ingester.ingest(p)
            if doc_chunks:
                doc_text = "\n\n".join(c.content for c in doc_chunks)
                doc_sections.append(f"--- {p.name} ---\n{doc_text}")
                continue
            reason = (
                ingester.last_skipped[0][1] if ingester.last_skipped
                else "no extractable text"
            )
            # A type without a dedicated loader may still be plain text the
            # user wants read as-is (an uncommon source extension, a config
            # file). Read it as UTF-8 text when it decodes; only a genuinely
            # binary or unreadable file is refused.
            if "unsupported extension" in reason:
                try:
                    text = p.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    text = None
                if text and text.strip():
                    cli.print(f"Reading {p.name} as plain text.")
                    doc_sections.append(f"--- {p.name} ---\n{text}")
                    continue
            cli.print_error(f"--file: could not read {file_path}: {reason}")
            return 1
        if doc_sections:
            args.task = "\n\n".join(doc_sections) + "\n\n---\n\n" + args.task
        if extra_inputs and getattr(args, 'stream', False):
            cli.print_error(
                "--file with an image is not supported together with "
                "--stream; drop --stream or attach a document instead."
            )
            return 1

    # Headless JSON contract: keep stdout pure (only the JSON result object)
    # by routing all human chatter to stderr and never streaming. `-q --json`
    # gives a clean document to pipe straight into `jq`.
    json_mode = getattr(args, 'output_json', False)
    if json_mode:
        cli._human_to_stderr = True
        args.stream = False

    # `-q/--quiet` (global or on the subcommand) suppresses the run banner
    # and the setup status lines so only the final answer/result is printed.
    quiet = getattr(args, 'quiet', False)

    # Streaming prints tokens as they arrive and never assembles the result
    # object the file outputs are written from, so a request for both is
    # refused up front rather than leaving the file unwritten.
    if getattr(args, 'stream', False):
        for flag, value in (
            ("-o/--output", getattr(args, 'output', None)),
            ("--card", getattr(args, 'card', None)),
        ):
            if value:
                cli.print_error(
                    f"{flag} cannot be combined with --stream: a streamed run "
                    f"produces no result document to write. Drop --stream to "
                    f"get {value}."
                )
                return 1

    # Validate an explicit --provider before doing any work, so a typo
    # (e.g. "grok") fails fast with a suggestion instead of falling through
    # to a multi-gigabyte local model download.
    provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
    if prov_err:
        cli.print_error(prov_err)
        return 1

    if not quiet:
        cli.print_header(f"effGen v{__version__} - Running Task")

    # Resolve the model once so the preset and plain paths agree. An explicit
    # -m/--model wins; the resolution for a run without one needs the config
    # file, so it happens after the file is loaded, below.
    run_model: str | None = None
    if args.model:
        run_model = args.model
        _preflight_model_hint(cli, run_model, provider)

    agent = None
    try:
        # Load configuration if provided
        config = {}
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                loaded_config = cli.config_loader.load_config(config_path)
                config = loaded_config.to_dict()
                if not quiet:
                    cli.print_success(f"Loaded configuration from {config_path}")
                _warn_unapplied_config_keys(config, cli)
            else:
                cli.print_error(f"Configuration file not found: {config_path}")
                return 1

        # With no -m/--model: take the model the config file names, else mirror
        # `quickstart` — prefer a detected cheap cloud model, else a small local
        # one — and say which and why, so the choice is never a silent surprise
        # (a paid cloud call or a multi-GB local download).
        if run_model is None:
            config_model = config.get("model")
            if isinstance(config_model, str) and config_model.strip():
                run_model = config_model.strip()
                config_provider = config.get("provider")
                if provider is None and isinstance(config_provider, str) and config_provider:
                    provider, prov_err = resolve_provider_name(config_provider)
                    if prov_err:
                        cli.print_error(prov_err)
                        return 1
                if not quiet:
                    cli.print(f"Using model {run_model} (from {args.config}); "
                              "override with -m/--model.")
            else:
                run_model, _sugg_provider, _sugg_reason = _quickstart_suggest_model()
                if provider is None and _sugg_provider:
                    provider = _sugg_provider
                if not quiet:
                    cli.print(f"Using model {run_model} ({_sugg_reason}); "
                              "override with -m/--model.")

        guardrails = getattr(args, 'guardrails', None) or config.get("guardrails")

        # Use preset if specified
        if getattr(args, 'preset', None):
            from effgen.presets import create_agent as _create_preset_agent
            model_id = run_model
            if not quiet:
                cli.print(f"Using preset: {args.preset}")
            _preset_overrides = {"provider": provider} if provider else {}
            if rag_docs:
                _preset_overrides["knowledge_base"] = rag_docs
            agent = _create_preset_agent(
                args.preset,
                model_id,
                agent_name=args.name,
                system_prompt=args.system_prompt or config.get("system_prompt"),
                max_iterations=args.max_iterations,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                enable_streaming=args.stream,
                session_id=getattr(args, 'session_id', None),
                guardrails=guardrails,
                **_preset_overrides,
            )
            if not quiet:
                cli.print_success(f"Created {args.preset} preset agent")
                cli.print(f"Model: {model_id}")
            tools = agent.config.tools if hasattr(agent, 'config') else []
        else:
            # Initialize tools
            tools = []
            if args.tools:
                if not quiet:
                    cli.print(f"Loading tools: {', '.join(args.tools)}")
                for tool_name in args.tools:
                    try:
                        tool = cli.tool_registry.get_tool_sync(tool_name)
                        tools.append(tool)
                        if not quiet:
                            cli.print_success(f"Loaded tool: {tool_name}")
                    except KeyError:
                        cli._suggest_tool(tool_name)
                        return 1
            else:
                # Conservative default tool set: a single deterministic
                # utility tool that rarely fires on general questions.
                # (web_search/weather used to be defaults and triggered bogus
                # calls like weather("Paris") for "capital of France?".) Users
                # who want more tools pass --tools explicitly.
                _default_safe_tools = ["calculator"]
                cli.tool_registry.discover_builtin_tools()
                all_tool_names = cli.tool_registry.list_tools()
                for name in _default_safe_tools:
                    if name in all_tool_names:
                        try:
                            tools.append(cli.tool_registry.get_tool_sync(name))
                        except Exception as e:
                            logger.debug(f"Failed to load default tool {name}: {e}")

            # Create agent configuration
            agent_config = AgentConfig(
                name=args.name or "cli-agent",
                raise_on_error=False,
                model=run_model,
                provider=provider,
                tools=tools,
                system_prompt=args.system_prompt or config.get("system_prompt",
                    "You are a helpful AI assistant."),
                temperature=args.temperature or config.get("temperature", 0.7),
                max_iterations=args.max_iterations or config.get("max_iterations", 10),
                max_tokens=args.max_tokens or config.get("max_tokens"),
                enable_sub_agents=not args.no_sub_agents,
                enable_streaming=args.stream,
                guardrails=guardrails,
            )

            # Create agent
            if not quiet:
                cli.print(f"\nInitializing agent: {agent_config.name}")
                cli.print(f"Model: {agent_config.model}")
                cli.print(f"Tools: {len(tools)} available")
                cli.print(f"Sub-agents: {'enabled' if agent_config.enable_sub_agents else 'disabled'}")
                if guardrails:
                    cli.print(f"Guardrails: {guardrails}")

            agent = Agent(agent_config, session_id=getattr(args, 'session_id', None))

        # Determine execution mode. Default to the agent's own
        # config.mode (SINGLE unless set otherwise) instead of forcing
        # AUTO, so a plain `effgen run "<task>"` doesn't switch to
        # sub-agent decomposition on its own — pass --mode auto to
        # opt in explicitly for a call.
        mode = None
        if args.mode == "single":
            mode = AgentMode.SINGLE
        elif args.mode == "sub_agents":
            mode = AgentMode.SUB_AGENTS
        elif args.mode == "auto":
            mode = AgentMode.AUTO

        # Run task
        if not quiet:
            cli.print(f"\n[bold]Task:[/bold] {args.task}" if cli.console else f"\nTask: {args.task}")
            cli.print()

        exit_code = 0
        # --json emits a single JSON document to stdout: no live spinner,
        # which would otherwise render there on an interactive terminal.
        animate = cli._animate(args) and not json_mode
        model_label = _progress.short_model_label(
            getattr(agent.config, "model", None) if hasattr(agent, "config") else None
        )
        if args.stream:
            # Streamed answer: a live markdown region on an interactive colour
            # terminal, raw token passthrough when piped/redirected/non-TTY.
            if not quiet:
                cli.print(
                    "[italic]Streaming response...[/italic]\n"
                    if cli.console else "Streaming response...\n"
                )
            try:
                _progress.stream_answer(
                    cli.console,
                    agent.stream(args.task, mode=mode),
                    animate=animate,
                    interactive=False,
                    quiet=quiet,
                    trailing_newline=False,
                )
            except KeyboardInterrupt:
                cli._handle_interrupt(agent)
                return 130
            print()  # New line after streaming
        else:
            # Regular output with a live, accurate status line.
            run_kwargs = _checkpoint_run_kwargs(args)
            if extra_inputs:
                run_kwargs['inputs'] = extra_inputs
            try:
                if animate:
                    reasoning = _progress.is_reasoning_agent(agent)
                    with _progress.LiveStatus(
                        cli.console,
                        model_label=model_label,
                        reasoning=reasoning,
                        tracker=agent.execution_tracker,
                    ):
                        response = agent.run(args.task, mode=mode, **run_kwargs)
                else:
                    if not quiet:
                        cli.print("Thinking...")
                    response = agent.run(args.task, mode=mode, **run_kwargs)
            except KeyboardInterrupt:
                cli._handle_interrupt(agent)
                return 130

            # Surface failure in the process exit code.
            if not response.success:
                exit_code = 1

            # Human presentation is skipped in --json mode so stdout stays a
            # single clean JSON document; the JSON is emitted below.
            if not json_mode:
                # Display response
                cli.print_header("Response")
                present_response(cli, response)

                # Frozen one-glance summary: ✓ Done in 3.2s · 2 tools · 1,204 tokens · $…
                if not quiet:
                    _progress.print_summary(cli, response)

                _explain = getattr(args, 'explain', False)
                _trace = getattr(args, 'trace', False)

                # The run's conversation, as the steps it was made of.
                if getattr(args, 'show_thread', False):
                    print_thread(cli, response)

                # A per-step timeline (bars + durations) shows where the
                # wall-clock went across the run's steps.
                if _trace and response.execution_trace:
                    cli.print_header("Timeline")
                    _tl = _progress.execution_timeline_lines(
                        response.execution_trace, stream=cli._human_stream()
                    )
                    if not _tl:
                        cli.print("(no timed steps recorded for this run)")
                    for _style, _text in _tl:
                        _text = _ascii_fold(_text, cli._human_stream())
                        if cli.console:
                            cli.console.print(f"[{_style}]{_text}[/{_style}]", highlight=False)
                        else:
                            print(_text)

                # Display the step trace (tool reasoning + per-step timing).
                if (_explain or _trace) and response.execution_trace:
                    cli.print_header("Execution Trace")
                    _lines = _progress.execution_trace_lines(
                        response.execution_trace, stream=cli._human_stream()
                    )
                    if not _lines:
                        cli.print("(no detailed steps recorded for this run)")
                    for _style, _text in _lines:
                        _text = _ascii_fold(_text, cli._human_stream())
                        if cli.console:
                            cli.console.print(f"[{_style}]{_text}[/{_style}]", highlight=False)
                        else:
                            print(_text)

                # On a multi-step run without an explicit trace flag, point
                # the user at the timeline rather than leaving it hidden.
                elif not quiet and not _explain and int(getattr(response, "tool_calls", 0) or 0) >= 1:
                    _steps = int(getattr(response, "tool_calls", 0) or 0)
                    _hint = _ascii_fold(
                        f"{_steps} tool step{'s' if _steps != 1 else ''} — run with --trace to see the timeline",
                        cli._human_stream(),
                    )
                    if cli.console:
                        cli.console.print(f"[effgen.muted]{_hint}[/effgen.muted]", highlight=False)
                    else:
                        print(_hint)

                # Display execution statistics
                if getattr(args, 'verbose', False) or _explain or _trace:
                    cli.print_header("Execution Statistics")
                    stats_table = cli._create_stats_table({
                        "Mode": response.mode.value,
                        "Success": "Yes" if response.success else "No",
                        "Iterations": response.iterations,
                        "Tool Calls": response.tool_calls,
                        "Tokens Used": response.tokens_used,
                        "Execution Time": f"{response.execution_time:.2f}s"
                    })

                    if cli.console:
                        cli.console.print(stats_table)
                    else:
                        for key, value in stats_table.items():
                            print(f"{key}: {value}")

                    # Full verbose trace
                    if getattr(args, 'verbose', False) and response.execution_trace:
                        cli.print_header("Full ReAct Trace")
                        trace_json = json.dumps(response.execution_trace, indent=2, default=str)
                        if cli.console:
                            cli.console.print(Syntax(trace_json, "json", line_numbers=True))
                        else:
                            print(trace_json)

            # Save response if output file specified
            if args.output:
                output_path = Path(args.output)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(run_document(response), f, indent=2, ensure_ascii=False)
                cli.print_success(f"Response saved to {output_path}")

            # Write the shareable HTML card when asked. Composes with -o:
            # the JSON document and the card are written from the same
            # result, and neither touches stdout.
            card_path = getattr(args, 'card', None)
            if card_path:
                from effgen.ui.report_html import ReportError, write_html_report
                try:
                    written = write_html_report(
                        card_path,
                        run_document(response),
                        kind="run",
                        command=_invoked_command(),
                    )
                except (ReportError, OSError) as exc:
                    cli.print_error(f"--card: could not write {card_path}: {exc}")
                    exit_code = exit_code or 1
                else:
                    cli.print_success(f"Run card written to {written}")

            # Emit the result object to stdout for piping (same document the
            # -o file carries). Goes to real stdout regardless of the
            # stderr-routed human output above.
            if json_mode:
                print(json.dumps(
                    run_document(response), indent=2, ensure_ascii=json_ensure_ascii()
                ))

        return exit_code

    except Exception as e:
        # Same presentation as a generation failure above — a red "Error"
        # panel — so a run that fails to load its model reads the same as
        # one that fails mid-generation.
        cli.print_error_panel(str(e), title="Error")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        # A failure here happens before any AgentResponse exists (e.g. an
        # unknown model id) — --json still gets one clean JSON object on
        # stdout, matching the envelope a successful/runtime-failure run
        # would have emitted, instead of empty stdout with a nonzero exit.
        if json_mode:
            print(json.dumps({
                "success": False,
                "error": {"type": type(e).__name__, "message": str(e)},
            }, indent=2, ensure_ascii=json_ensure_ascii()))
        return 1
    finally:
        # Release the agent explicitly so the CLI never emits the
        # "Agent was garbage-collected without calling close()" warning.
        if agent is not None:
            try:
                agent.close()
            except Exception as e:
                logger.debug(f"Agent close failed: {e}")


def handle_interrupt(cli: "CLIInterface", agent: Any) -> None:
    """Render a friendly Ctrl-C stop (partial trace + 'Stopped.'), no traceback."""
    # Move to a clean line after any in-flight spinner/stream output.
    try:
        sys.stdout.write("\n")
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    # Surface whatever partial progress the tracker captured.
    try:
        tracker = getattr(agent, "execution_tracker", None)
        tools = sorted(getattr(tracker, "active_tools", set()) or set())
        done = [
            n.name for n in getattr(tracker, "nodes", {}).values()
            if getattr(n, "node_type", "") == "tool" and getattr(n, "status", "") == "completed"
        ]
        if done:
            cli.print(f"Partial progress: {len(done)} tool call(s) completed "
                       f"({', '.join(done[:5])}).")
        if tools:
            cli.print(f"Cancelled in-flight: {', '.join(tools)}.")
    except Exception:  # noqa: BLE001 - never let cleanup add noise
        pass
    if cli.console:
        cli.console.print("[yellow]Stopped.[/yellow]", highlight=False)
    else:
        print("Stopped.")
