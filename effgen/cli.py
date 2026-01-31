"""
effGen CLI - Command-line interface for the effGen framework.

This module provides a comprehensive CLI for interacting with effGen:
- Run agents with tasks (with interactive wizard support)
- Interactive chat mode
- API server mode
- Configuration management
- Tool listing and testing
- Model management
- Example runner

Usage:
    # Direct task execution
    effgen run "What is 2+2?" --model Qwen/Qwen2.5-1.5B-Instruct

    # Interactive wizard (launches when no task provided)
    effgen run
    effgen  # Same as above

    # effgen-agent command (similar to smolagent)
    effgen-agent "Plan a trip to Tokyo" --model Qwen/Qwen2.5-1.5B-Instruct --tools web_search
    effgen-agent  # Interactive mode

    # Chat mode
    effgen chat --model phi3-mini

    # Other commands
    effgen serve --port 8000
    effgen config show
    effgen tools list
    effgen models list
    effgen examples run basic_agent

Interactive mode guides you through:
    - Agent type selection (CodeAgent vs ToolCallingAgent vs ReActAgent)
    - Tool selection from available toolbox
    - Model configuration (type, ID, API settings)
    - Advanced options (temperature, max iterations, etc.)
    - Task prompt input
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import importlib.util
from datetime import datetime

# Rich terminal output (fallback to basic if not available)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.layout import Layout
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


# Import effGen components
try:
    from effgen import (
        Agent,
        AgentConfig,
        load_model,
        ConfigLoader,
        get_tool_registry,
        __version__
    )
    from effgen.core.agent import AgentMode
    from effgen.tools.builtin import *
except ImportError:
    print("Error: effGen package not found. Please install it first.")
    sys.exit(1)


# Configure logging
def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """
    Configure logging for CLI.

    Args:
        verbose: Enable verbose logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class CLIInterface:
    """Main CLI interface for effGen."""

    def __init__(self):
        """Initialize CLI interface."""
        self.console = Console() if RICH_AVAILABLE else None
        self.config_loader = ConfigLoader()
        self.tool_registry = get_tool_registry()

    def print(self, *args, **kwargs):
        """Print with rich formatting if available."""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)

    def print_header(self, text: str):
        """Print a header."""
        if self.console:
            self.console.print(f"\n[bold cyan]{text}[/bold cyan]")
        else:
            print(f"\n=== {text} ===")

    def print_success(self, text: str):
        """Print success message."""
        if self.console:
            self.console.print(f"[green]✓[/green] {text}")
        else:
            print(f"✓ {text}")

    def print_error(self, text: str):
        """Print error message."""
        if self.console:
            self.console.print(f"[red]✗[/red] {text}")
        else:
            print(f"✗ {text}")

    def print_warning(self, text: str):
        """Print warning message."""
        if self.console:
            self.console.print(f"[yellow]⚠[/yellow] {text}")
        else:
            print(f"⚠ {text}")

    def interactive_wizard(self, args):
        """
        Interactive setup wizard for configuring and running agents.

        Similar to smolagents CLI, guides users through:
        - Agent type selection
        - Tool selection from available toolbox
        - Model configuration (type, ID, API settings)
        - Advanced options like additional imports
        - Task prompt input

        Args:
            args: Parsed command-line arguments (may have partial values)

        Returns:
            Exit code
        """
        self.print_header(f"effGen v{__version__} - Interactive Setup Wizard")
        self.print()

        if self.console:
            self.console.print(Panel(
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
            # Step 1: Agent Type Selection
            self.print_header("Step 1: Select Agent Type")
            agent_types = [
                ("1", "CodeAgent", "Agent that generates and executes code (recommended)"),
                ("2", "ToolCallingAgent", "Agent that calls tools via structured outputs"),
                ("3", "ReActAgent", "Agent using Reason+Act pattern (default)")
            ]

            if self.console:
                table = Table(title="Available Agent Types")
                table.add_column("#", style="cyan", width=3)
                table.add_column("Type", style="magenta")
                table.add_column("Description", style="white")
                for num, name, desc in agent_types:
                    table.add_row(num, name, desc)
                self.console.print(table)
            else:
                for num, name, desc in agent_types:
                    print(f"  [{num}] {name}: {desc}")

            agent_type_input = input("\nSelect agent type [3]: ").strip() or "3"
            agent_type_map = {"1": "code", "2": "tool_calling", "3": "react"}
            agent_type = agent_type_map.get(agent_type_input, "react")
            self.print_success(f"Selected: {agent_type}")

            # Step 2: Tool Selection
            self.print_header("Step 2: Select Tools")

            # Discover and list available tools
            self.tool_registry.discover_builtin_tools()
            available_tools = self.tool_registry.list_tools()

            if self.console:
                table = Table(title=f"Available Tools ({len(available_tools)})")
                table.add_column("#", style="cyan", width=3)
                table.add_column("Name", style="magenta")
                table.add_column("Description", style="white")

                for i, tool_name in enumerate(available_tools, 1):
                    try:
                        metadata = self.tool_registry.get_metadata(tool_name)
                        desc = metadata.description[:40] + "..." if len(metadata.description) > 40 else metadata.description
                        table.add_row(str(i), tool_name, desc)
                    except Exception:
                        table.add_row(str(i), tool_name, "No description")
                self.console.print(table)
            else:
                for i, tool_name in enumerate(available_tools, 1):
                    print(f"  [{i}] {tool_name}")

            self.print("\nEnter tool numbers separated by commas (e.g., 1,2,3)")
            self.print("Or press Enter to use all tools, 'none' for no tools")
            tool_input = input("Tools [all]: ").strip().lower()

            selected_tools = []
            if tool_input == "none":
                pass
            elif tool_input == "" or tool_input == "all":
                for name in available_tools:
                    try:
                        tool = asyncio.run(self.tool_registry.get_tool(name))
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
                                tool = asyncio.run(self.tool_registry.get_tool(tool_name))
                                selected_tools.append(tool)
                            except Exception as e:
                                self.print_warning(f"Failed to load {tool_name}: {e}")
                except ValueError:
                    self.print_warning("Invalid input, using all tools")
                    for name in available_tools:
                        try:
                            tool = asyncio.run(self.tool_registry.get_tool(name))
                            selected_tools.append(tool)
                        except Exception:
                            pass

            self.print_success(f"Selected {len(selected_tools)} tool(s)")

            # Step 3: Model Configuration
            self.print_header("Step 3: Configure Model")

            model_types = [
                ("1", "TransformersModel", "Local Hugging Face model (e.g., Qwen/Qwen2.5-1.5B-Instruct)"),
                ("2", "OpenAIModel", "OpenAI API (requires OPENAI_API_KEY)"),
                ("3", "AnthropicModel", "Anthropic API (requires ANTHROPIC_API_KEY)"),
                ("4", "vLLMModel", "vLLM server (requires running vLLM instance)"),
                ("5", "LiteLLMModel", "LiteLLM proxy (supports multiple backends)")
            ]

            if self.console:
                table = Table(title="Model Types")
                table.add_column("#", style="cyan", width=3)
                table.add_column("Type", style="magenta")
                table.add_column("Description", style="white")
                for num, name, desc in model_types:
                    table.add_row(num, name, desc)
                self.console.print(table)
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
            self.print_success(f"Model: {model_id}")

            # Step 4: Advanced Options
            self.print_header("Step 4: Advanced Options")

            temp_input = input("Temperature [0.7]: ").strip()
            temperature = float(temp_input) if temp_input else 0.7

            max_iter_input = input("Max iterations [10]: ").strip()
            max_iterations = int(max_iter_input) if max_iter_input else 10

            sub_agents_input = input("Enable sub-agents? [Y/n]: ").strip().lower()
            enable_sub_agents = sub_agents_input != "n"

            stream_input = input("Stream output? [y/N]: ").strip().lower()
            enable_streaming = stream_input == "y"

            # Step 5: Task Input
            self.print_header("Step 5: Enter Task")

            if self.console:
                self.console.print("[italic]Enter your task or question for the agent.[/italic]")
                self.console.print("[dim]For multi-line input, end with an empty line.[/dim]\n")
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
                self.print_error("No task provided")
                return 1

            # Confirm and Run
            self.print_header("Configuration Summary")

            summary = {
                "Agent Type": agent_type,
                "Model": model_id,
                "Tools": len(selected_tools),
                "Temperature": temperature,
                "Max Iterations": max_iterations,
                "Sub-agents": "enabled" if enable_sub_agents else "disabled",
                "Streaming": "enabled" if enable_streaming else "disabled",
                "Task": task[:50] + "..." if len(task) > 50 else task
            }

            if self.console:
                table = Table(title="Configuration")
                table.add_column("Setting", style="cyan")
                table.add_column("Value", style="magenta")
                for key, value in summary.items():
                    table.add_row(key, str(value))
                self.console.print(table)
            else:
                for key, value in summary.items():
                    print(f"  {key}: {value}")

            confirm = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
            if confirm == "n":
                self.print_warning("Cancelled by user")
                return 0

            # Create agent and run task
            self.print_header("Running Agent")

            agent_config = AgentConfig(
                name="interactive-agent",
                model=model_id,
                tools=selected_tools,
                temperature=temperature,
                max_iterations=max_iterations,
                enable_sub_agents=enable_sub_agents,
                enable_streaming=enable_streaming
            )

            agent = Agent(agent_config)

            if enable_streaming:
                if self.console:
                    self.console.print("\n[bold green]Agent:[/bold green] ", end="")
                else:
                    print("\nAgent: ", end="", flush=True)

                for token in agent.stream(task):
                    print(token, end='', flush=True)
                print()
            else:
                if self.console:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=self.console
                    ) as progress:
                        progress.add_task("Thinking...", total=None)
                        response = agent.run(task)
                else:
                    print("Thinking...")
                    response = agent.run(task)

                # Display response
                self.print_header("Response")

                if self.console:
                    self.console.print(Panel(
                        Markdown(response.output),
                        title="Agent Response",
                        border_style="green" if response.success else "red"
                    ))
                else:
                    print(response.output)

                # Display statistics
                self.print_header("Execution Statistics")
                stats = {
                    "Success": "Yes" if response.success else "No",
                    "Iterations": response.iterations,
                    "Tool Calls": response.tool_calls,
                    "Tokens Used": response.tokens_used,
                    "Execution Time": f"{response.execution_time:.2f}s"
                }

                if self.console:
                    stats_table = Table()
                    stats_table.add_column("Metric", style="cyan")
                    stats_table.add_column("Value", style="magenta")
                    for key, value in stats.items():
                        stats_table.add_row(key, str(value))
                    self.console.print(stats_table)
                else:
                    for key, value in stats.items():
                        print(f"  {key}: {value}")

            # Ask if user wants to continue
            continue_input = input("\nRun another task? [y/N]: ").strip().lower()
            if continue_input == "y":
                return self.interactive_wizard(args)

            return 0

        except KeyboardInterrupt:
            self.print("\n\nWizard cancelled")
            return 130
        except Exception as e:
            self.print_error(f"Error in interactive wizard: {e}")
            import traceback
            traceback.print_exc()
            return 1

    def run_agent(self, args):
        """
        Run an agent with a task.

        Args:
            args: Parsed command-line arguments
        """
        # Check if we need to launch interactive wizard
        if args.task is None:
            return self.interactive_wizard(args)

        self.print_header(f"effGen v{__version__} - Running Task")

        try:
            # Load configuration if provided
            config = {}
            if args.config:
                config_path = Path(args.config)
                if config_path.exists():
                    loaded_config = self.config_loader.load_config(config_path)
                    config = loaded_config.to_dict()
                    self.print_success(f"Loaded configuration from {config_path}")
                else:
                    self.print_error(f"Configuration file not found: {config_path}")
                    return 1

            # Initialize tools
            tools = []
            if args.tools:
                self.print(f"Loading tools: {', '.join(args.tools)}")
                for tool_name in args.tools:
                    try:
                        tool = asyncio.run(self.tool_registry.get_tool(tool_name))
                        tools.append(tool)
                        self.print_success(f"Loaded tool: {tool_name}")
                    except KeyError:
                        self.print_error(f"Tool not found: {tool_name}")
                        return 1
            else:
                # Load all builtin tools by default
                self.tool_registry.discover_builtin_tools()
                tool_names = self.tool_registry.list_tools()
                for name in tool_names[:5]:  # Limit to first 5 tools
                    try:
                        tool = asyncio.run(self.tool_registry.get_tool(name))
                        tools.append(tool)
                    except Exception as e:
                        logging.debug(f"Failed to load tool {name}: {e}")

            # Create agent configuration
            agent_config = AgentConfig(
                name=args.name or "cli-agent",
                model=args.model or "phi3-mini",
                tools=tools,
                system_prompt=args.system_prompt or config.get("system_prompt",
                    "You are a helpful AI assistant."),
                temperature=args.temperature or config.get("temperature", 0.7),
                max_iterations=args.max_iterations or config.get("max_iterations", 10),
                enable_sub_agents=not args.no_sub_agents,
                enable_streaming=args.stream
            )

            # Create agent
            self.print(f"\nInitializing agent: {agent_config.name}")
            self.print(f"Model: {agent_config.model}")
            self.print(f"Tools: {len(tools)} available")
            self.print(f"Sub-agents: {'enabled' if agent_config.enable_sub_agents else 'disabled'}")

            agent = Agent(agent_config)

            # Determine execution mode
            mode = AgentMode.AUTO
            if args.mode:
                if args.mode == "single":
                    mode = AgentMode.SINGLE
                elif args.mode == "sub_agents":
                    mode = AgentMode.SUB_AGENTS

            # Run task
            self.print(f"\n[bold]Task:[/bold] {args.task}" if self.console else f"\nTask: {args.task}")
            self.print()

            if args.stream:
                # Streaming output
                self.print("[italic]Streaming response...[/italic]\n" if self.console else "Streaming response...\n")
                for token in agent.stream(args.task, mode=mode):
                    print(token, end='', flush=True)
                print()  # New line after streaming
            else:
                # Regular output with spinner
                if self.console:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=self.console
                    ) as progress:
                        task_progress = progress.add_task("Thinking...", total=None)
                        response = agent.run(args.task, mode=mode)
                else:
                    self.print("Thinking...")
                    response = agent.run(args.task, mode=mode)

                # Display response
                self.print_header("Response")

                if self.console:
                    # Rich markdown formatting
                    self.console.print(Panel(
                        Markdown(response.output),
                        title="Agent Response",
                        border_style="green" if response.success else "red"
                    ))
                else:
                    print(response.output)

                # Display execution statistics
                if args.verbose:
                    self.print_header("Execution Statistics")
                    stats_table = self._create_stats_table({
                        "Mode": response.mode.value,
                        "Success": "Yes" if response.success else "No",
                        "Iterations": response.iterations,
                        "Tool Calls": response.tool_calls,
                        "Tokens Used": response.tokens_used,
                        "Execution Time": f"{response.execution_time:.2f}s"
                    })

                    if self.console:
                        self.console.print(stats_table)
                    else:
                        for key, value in stats_table.items():
                            print(f"{key}: {value}")

                # Save response if output file specified
                if args.output:
                    output_path = Path(args.output)
                    with open(output_path, 'w') as f:
                        json.dump(response.to_dict(), f, indent=2)
                    self.print_success(f"Response saved to {output_path}")

            return 0

        except Exception as e:
            self.print_error(f"Error running agent: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _create_stats_table(self, stats: Dict[str, Any]) -> Any:
        """Create statistics table."""
        if not self.console:
            return stats

        table = Table(title="Execution Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in stats.items():
            table.add_row(key, str(value))

        return table

    def chat_mode(self, args):
        """
        Interactive chat mode.

        Args:
            args: Parsed command-line arguments
        """
        self.print_header(f"effGen v{__version__} - Chat Mode")
        self.print("Type 'exit' or 'quit' to end the conversation")
        self.print("Type 'clear' to clear conversation history")
        self.print("Type 'help' for available commands\n")

        try:
            # Initialize agent (similar to run_agent)
            tools = []
            self.tool_registry.discover_builtin_tools()
            tool_names = self.tool_registry.list_tools()[:5]
            for name in tool_names:
                try:
                    tool = asyncio.run(self.tool_registry.get_tool(name))
                    tools.append(tool)
                except Exception:
                    pass

            agent_config = AgentConfig(
                name="chat-agent",
                model=args.model or "phi3-mini",
                tools=tools,
                temperature=args.temperature or 0.7,
                enable_sub_agents=not args.no_sub_agents,
                enable_streaming=True
            )

            agent = Agent(agent_config)
            conversation_history = []

            while True:
                try:
                    # Get user input
                    if self.console:
                        user_input = self.console.input("\n[bold cyan]You:[/bold cyan] ")
                    else:
                        user_input = input("\nYou: ")

                    if not user_input.strip():
                        continue

                    # Handle commands
                    if user_input.lower() in ['exit', 'quit']:
                        self.print("\nGoodbye!")
                        break
                    elif user_input.lower() == 'clear':
                        agent.reset_memory()
                        conversation_history = []
                        self.print_success("Conversation history cleared")
                        continue
                    elif user_input.lower() == 'help':
                        self._print_chat_help()
                        continue
                    elif user_input.lower() == 'save':
                        self._save_conversation(conversation_history)
                        continue

                    # Add to history
                    conversation_history.append({
                        "role": "user",
                        "content": user_input,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Get agent response
                    if self.console:
                        self.console.print("\n[bold green]Agent:[/bold green] ", end="")
                    else:
                        print("\nAgent: ", end="", flush=True)

                    response_text = ""
                    for token in agent.stream(user_input):
                        print(token, end='', flush=True)
                        response_text += token
                    print()  # New line

                    # Add to history
                    conversation_history.append({
                        "role": "agent",
                        "content": response_text,
                        "timestamp": datetime.now().isoformat()
                    })

                except KeyboardInterrupt:
                    self.print("\n\nInterrupted. Type 'exit' to quit.")
                    continue
                except Exception as e:
                    self.print_error(f"Error: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

            return 0

        except Exception as e:
            self.print_error(f"Error in chat mode: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def _print_chat_help(self):
        """Print chat mode help."""
        help_text = """
        [bold]Available Commands:[/bold]
        - exit, quit: Exit chat mode
        - clear: Clear conversation history
        - save: Save conversation to file
        - help: Show this help message
        """ if self.console else """
        Available Commands:
        - exit, quit: Exit chat mode
        - clear: Clear conversation history
        - save: Save conversation to file
        - help: Show this help message
        """
        self.print(help_text)

    def _save_conversation(self, history: List[Dict]):
        """Save conversation history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(history, f, indent=2)

        self.print_success(f"Conversation saved to {filename}")

    def serve_api(self, args):
        """
        Start API server.

        Args:
            args: Parsed command-line arguments
        """
        self.print_header(f"effGen v{__version__} - API Server")

        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel
            import uvicorn

            app = FastAPI(
                title="effGen API",
                description="API server for effGen framework",
                version=__version__
            )

            # Add CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # Request/Response models
            class TaskRequest(BaseModel):
                task: str
                model: Optional[str] = "phi3-mini"
                tools: Optional[List[str]] = None
                temperature: Optional[float] = 0.7
                max_iterations: Optional[int] = 10
                stream: bool = False

            class TaskResponse(BaseModel):
                output: str
                success: bool
                metadata: Dict[str, Any]

            # Initialize agent (reusable)
            agent_instance = None

            @app.post("/run", response_model=TaskResponse)
            async def run_task(request: TaskRequest):
                """Run a task with an agent."""
                nonlocal agent_instance

                try:
                    # Create agent if not exists
                    if not agent_instance:
                        tools = []
                        self.tool_registry.discover_builtin_tools()
                        tool_names = self.tool_registry.list_tools()[:5]
                        for name in tool_names:
                            try:
                                tool = await self.tool_registry.get_tool(name)
                                tools.append(tool)
                            except Exception:
                                pass

                        agent_config = AgentConfig(
                            name="api-agent",
                            model=request.model,
                            tools=tools,
                            temperature=request.temperature,
                            max_iterations=request.max_iterations
                        )
                        agent_instance = Agent(agent_config)

                    # Run task
                    response = agent_instance.run(request.task)

                    return TaskResponse(
                        output=response.output,
                        success=response.success,
                        metadata={
                            "mode": response.mode.value,
                            "iterations": response.iterations,
                            "tool_calls": response.tool_calls,
                            "execution_time": response.execution_time
                        }
                    )

                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @app.get("/health")
            async def health():
                """Health check endpoint."""
                return {"status": "healthy", "version": __version__}

            @app.get("/tools")
            async def list_tools():
                """List available tools."""
                tools = self.tool_registry.list_tools()
                return {"tools": tools}

            # Start server
            self.print(f"Starting server on {args.host}:{args.port}")
            self.print(f"API docs available at http://{args.host}:{args.port}/docs")
            self.print()

            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                log_level="info" if args.verbose else "warning"
            )

            return 0

        except ImportError:
            self.print_error("FastAPI and uvicorn are required for server mode.")
            self.print("Install with: pip install fastapi uvicorn")
            return 1
        except Exception as e:
            self.print_error(f"Error starting server: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    def config_commands(self, args):
        """
        Configuration management commands.

        Args:
            args: Parsed command-line arguments
        """
        if args.config_command == 'show':
            self._config_show(args)
        elif args.config_command == 'validate':
            self._config_validate(args)
        elif args.config_command == 'init':
            self._config_init(args)
        else:
            self.print_error(f"Unknown config command: {args.config_command}")
            return 1

        return 0

    def _config_show(self, args):
        """Show current configuration."""
        self.print_header("Configuration")

        if args.file:
            try:
                config = self.config_loader.load_config(args.file)

                if self.console:
                    syntax = Syntax(
                        json.dumps(config.to_dict(), indent=2),
                        "json",
                        theme="monokai",
                        line_numbers=True
                    )
                    self.console.print(syntax)
                else:
                    print(json.dumps(config.to_dict(), indent=2))

            except Exception as e:
                self.print_error(f"Error loading config: {e}")
        else:
            self.print_warning("No configuration file specified")
            self.print("Use: effgen config show --file <path>")

    def _config_validate(self, args):
        """Validate configuration file."""
        if not args.file:
            self.print_error("Configuration file required")
            return

        try:
            config = self.config_loader.load_config(args.file, validate=True)
            self.print_success(f"Configuration is valid: {args.file}")
        except Exception as e:
            self.print_error(f"Configuration validation failed: {e}")

    def _config_init(self, args):
        """Initialize a new configuration file."""
        output_path = Path(args.output or "config.yaml")

        if output_path.exists() and not args.force:
            self.print_error(f"File already exists: {output_path}")
            self.print("Use --force to overwrite")
            return

        # Create default configuration
        default_config = {
            "models": {
                "default": "phi3-mini",
                "phi3_mini": {
                    "model_path": "microsoft/Phi-3-mini-4k-instruct",
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            },
            "tools": {
                "enabled": ["calculator", "web_search", "file_ops"]
            },
            "system_prompt": "You are a helpful AI assistant.",
            "max_iterations": 10
        }

        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        self.print_success(f"Configuration initialized: {output_path}")

    def tools_commands(self, args):
        """
        Tool management commands.

        Args:
            args: Parsed command-line arguments
        """
        if args.tool_command == 'list':
            self._tools_list(args)
        elif args.tool_command == 'info':
            self._tools_info(args)
        elif args.tool_command == 'test':
            self._tools_test(args)
        else:
            self.print_error(f"Unknown tools command: {args.tool_command}")
            return 1

        return 0

    def _tools_list(self, args):
        """List available tools."""
        self.print_header("Available Tools")

        # Discover builtin tools
        self.tool_registry.discover_builtin_tools()

        # Get tools
        tools = self.tool_registry.list_tools()

        if not tools:
            self.print_warning("No tools registered")
            return

        if self.console:
            table = Table(title=f"Registered Tools ({len(tools)})")
            table.add_column("Name", style="cyan")
            table.add_column("Category", style="magenta")
            table.add_column("Description", style="white")

            for tool_name in tools:
                try:
                    metadata = self.tool_registry.get_metadata(tool_name)
                    table.add_row(
                        metadata.name,
                        metadata.category.value,
                        metadata.description[:50] + "..." if len(metadata.description) > 50 else metadata.description
                    )
                except Exception as e:
                    logging.debug(f"Error getting metadata for {tool_name}: {e}")

            self.console.print(table)
        else:
            for tool_name in tools:
                print(f"- {tool_name}")

    def _tools_info(self, args):
        """Show detailed tool information."""
        if not args.name:
            self.print_error("Tool name required")
            return

        try:
            metadata = self.tool_registry.get_metadata(args.name)

            self.print_header(f"Tool: {metadata.name}")
            self.print(f"\n[bold]Description:[/bold] {metadata.description}" if self.console else f"\nDescription: {metadata.description}")
            self.print(f"[bold]Category:[/bold] {metadata.category.value}" if self.console else f"Category: {metadata.category.value}")
            self.print(f"[bold]Version:[/bold] {metadata.version}" if self.console else f"Version: {metadata.version}")

            if metadata.tags:
                self.print(f"[bold]Tags:[/bold] {', '.join(metadata.tags)}" if self.console else f"Tags: {', '.join(metadata.tags)}")

            # Show parameters
            if metadata.input_schema:
                self.print("\n[bold]Parameters:[/bold]" if self.console else "\nParameters:")
                if self.console:
                    syntax = Syntax(
                        json.dumps(metadata.input_schema, indent=2),
                        "json",
                        theme="monokai"
                    )
                    self.console.print(syntax)
                else:
                    print(json.dumps(metadata.input_schema, indent=2))

        except KeyError:
            self.print_error(f"Tool not found: {args.name}")
        except Exception as e:
            self.print_error(f"Error getting tool info: {e}")

    def _tools_test(self, args):
        """Test a tool with sample input."""
        if not args.name:
            self.print_error("Tool name required")
            return

        try:
            tool = asyncio.run(self.tool_registry.get_tool(args.name))

            self.print_header(f"Testing Tool: {args.name}")

            # Parse input
            if args.input:
                try:
                    input_data = json.loads(args.input)
                except json.JSONDecodeError:
                    input_data = {"input": args.input}
            else:
                input_data = {}

            # Execute tool
            self.print(f"Input: {input_data}\n")
            result = asyncio.run(tool.execute(**input_data))

            self.print("[bold]Result:[/bold]" if self.console else "Result:")
            if self.console:
                self.console.print(Panel(str(result), border_style="green"))
            else:
                print(result)

        except KeyError:
            self.print_error(f"Tool not found: {args.name}")
        except Exception as e:
            self.print_error(f"Error testing tool: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    def models_commands(self, args):
        """
        Model management commands.

        Args:
            args: Parsed command-line arguments
        """
        if args.model_command == 'list':
            self._models_list(args)
        elif args.model_command == 'info':
            self._models_info(args)
        else:
            self.print_error(f"Unknown models command: {args.model_command}")
            return 1

        return 0

    def _models_list(self, args):
        """List available models."""
        self.print_header("Available Models")

        # Load models from config if available
        config_dir = Path("configs")
        models_config = config_dir / "models.yaml"

        if models_config.exists():
            config = self.config_loader.load_config(models_config)
            models = config.get("models", {})

            if self.console:
                table = Table(title="Configured Models")
                table.add_column("Name", style="cyan")
                table.add_column("Path/API", style="magenta")
                table.add_column("Type", style="white")

                for name, model_config in models.items():
                    if isinstance(model_config, dict):
                        table.add_row(
                            name,
                            model_config.get("model_path", model_config.get("api", "N/A")),
                            model_config.get("type", "unknown")
                        )

                self.console.print(table)
            else:
                for name in models.keys():
                    print(f"- {name}")
        else:
            self.print_warning("No models configuration found")
            self.print("Common models:")
            common_models = [
                "phi3-mini",
                "mistral-7b",
                "llama-2-7b",
                "gemma-7b"
            ]
            for model in common_models:
                print(f"- {model}")

    def _models_info(self, args):
        """Show model information."""
        if not args.name:
            self.print_error("Model name required")
            return

        self.print_header(f"Model: {args.name}")
        self.print("Model information coming soon...")

    def examples_commands(self, args):
        """
        Run example scripts.

        Args:
            args: Parsed command-line arguments
        """
        if args.example_command == 'list':
            self._examples_list(args)
        elif args.example_command == 'run':
            self._examples_run(args)
        else:
            self.print_error(f"Unknown examples command: {args.example_command}")
            return 1

        return 0

    def _examples_list(self, args):
        """List available examples."""
        self.print_header("Available Examples")

        examples_dir = Path(__file__).parent.parent / "examples"

        if not examples_dir.exists():
            self.print_warning("Examples directory not found")
            return

        examples = []
        for file in examples_dir.glob("*.py"):
            if not file.name.startswith("_"):
                examples.append(file.stem)

        # Also check agents subdirectory
        agents_dir = examples_dir / "agents"
        if agents_dir.exists():
            for file in agents_dir.glob("*.py"):
                if not file.name.startswith("_"):
                    examples.append(f"agents/{file.stem}")

        if self.console:
            table = Table(title="Example Scripts")
            table.add_column("Name", style="cyan")
            table.add_column("Command", style="magenta")

            for example in sorted(examples):
                table.add_row(example, f"effgen examples run {example}")

            self.console.print(table)
        else:
            for example in sorted(examples):
                print(f"- {example}")

    def _examples_run(self, args):
        """Run an example script."""
        if not args.name:
            self.print_error("Example name required")
            return

        examples_dir = Path(__file__).parent.parent / "examples"
        example_path = examples_dir / f"{args.name}.py"

        if not example_path.exists():
            self.print_error(f"Example not found: {args.name}")
            return

        self.print_header(f"Running Example: {args.name}")
        self.print()

        # Load and run example
        try:
            spec = importlib.util.spec_from_file_location("example", example_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Run main function if exists
            if hasattr(module, 'main'):
                module.main()
            else:
                self.print_warning("Example does not have a main() function")

        except Exception as e:
            self.print_error(f"Error running example: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


def create_parser():
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description=f"effGen v{__version__} - CLI for agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  effgen run "What is the weather in Paris?" --model phi3-mini
  effgen chat --model phi3-mini --temperature 0.8
  effgen serve --port 8000
  effgen config show --file configs/models.yaml
  effgen tools list
  effgen examples run basic_agent
        """
    )

    parser.add_argument('--version', action='version', version=f'effGen {__version__}')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--log-file', help='Log file path')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run an agent with a task')
    run_parser.add_argument('task', nargs='?', default=None, help='Task description (launches interactive wizard if not provided)')
    run_parser.add_argument('-m', '--model', help='Model to use')
    run_parser.add_argument('-n', '--name', help='Agent name')
    run_parser.add_argument('-t', '--tools', nargs='+', help='Tools to enable')
    run_parser.add_argument('-c', '--config', help='Configuration file')
    run_parser.add_argument('--system-prompt', help='System prompt')
    run_parser.add_argument('--temperature', type=float, help='Temperature')
    run_parser.add_argument('--max-iterations', type=int, help='Max iterations')
    run_parser.add_argument('--mode', choices=['auto', 'single', 'sub_agents'], help='Execution mode')
    run_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')
    run_parser.add_argument('--stream', action='store_true', help='Stream output')
    run_parser.add_argument('-o', '--output', help='Output file for response')

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('-m', '--model', help='Model to use')
    chat_parser.add_argument('--temperature', type=float, help='Temperature')
    chat_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('-p', '--port', type=int, default=8000, help='Port to bind to')

    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config command')

    config_show = config_subparsers.add_parser('show', help='Show configuration')
    config_show.add_argument('-f', '--file', help='Configuration file')

    config_validate = config_subparsers.add_parser('validate', help='Validate configuration')
    config_validate.add_argument('-f', '--file', required=True, help='Configuration file')

    config_init = config_subparsers.add_parser('init', help='Initialize new configuration')
    config_init.add_argument('-o', '--output', help='Output file')
    config_init.add_argument('--force', action='store_true', help='Overwrite existing file')

    # Tools commands
    tools_parser = subparsers.add_parser('tools', help='Tool management')
    tools_subparsers = tools_parser.add_subparsers(dest='tool_command', help='Tools command')

    tools_list = tools_subparsers.add_parser('list', help='List tools')

    tools_info = tools_subparsers.add_parser('info', help='Show tool information')
    tools_info.add_argument('name', help='Tool name')

    tools_test = tools_subparsers.add_parser('test', help='Test a tool')
    tools_test.add_argument('name', help='Tool name')
    tools_test.add_argument('-i', '--input', help='Tool input (JSON or string)')

    # Models commands
    models_parser = subparsers.add_parser('models', help='Model management')
    models_subparsers = models_parser.add_subparsers(dest='model_command', help='Models command')

    models_list = models_subparsers.add_parser('list', help='List models')

    models_info = models_subparsers.add_parser('info', help='Show model information')
    models_info.add_argument('name', help='Model name')

    # Examples commands
    examples_parser = subparsers.add_parser('examples', help='Run example scripts')
    examples_subparsers = examples_parser.add_subparsers(dest='example_command', help='Examples command')

    examples_list = examples_subparsers.add_parser('list', help='List examples')

    examples_run = examples_subparsers.add_parser('run', help='Run an example')
    examples_run.add_argument('name', help='Example name')

    return parser


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(getattr(args, 'verbose', False), getattr(args, 'log_file', None))

    # Create CLI interface
    cli = CLIInterface()

    # Route to appropriate handler
    try:
        if args.command == 'run':
            exit_code = cli.run_agent(args)
        elif args.command == 'chat':
            exit_code = cli.chat_mode(args)
        elif args.command == 'serve':
            exit_code = cli.serve_api(args)
        elif args.command == 'config':
            exit_code = cli.config_commands(args)
        elif args.command == 'tools':
            exit_code = cli.tools_commands(args)
        elif args.command == 'models':
            exit_code = cli.models_commands(args)
        elif args.command == 'examples':
            exit_code = cli.examples_commands(args)
        elif args.command is None:
            # No command - launch interactive wizard
            # Create a namespace with default values for run command
            class WizardArgs:
                task = None
                model = None
                name = None
                tools = None
                config = None
                system_prompt = None
                temperature = None
                max_iterations = None
                mode = None
                no_sub_agents = False
                stream = False
                output = None
                verbose = getattr(args, 'verbose', False)
            exit_code = cli.interactive_wizard(WizardArgs())
        else:
            parser.print_help()
            exit_code = 0

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def agent_main():
    """
    Entry point for effgen-agent CLI (similar to smolagent).

    A generalist command to run a multi-step agent that can be equipped with various tools.

    Usage:
        # Run with direct prompt and options
        effgen-agent "Plan a trip to Tokyo" --model Qwen/Qwen2.5-1.5B-Instruct --tools web_search calculator

        # Run in interactive mode (launches setup wizard when no prompt provided)
        effgen-agent

    Interactive mode guides you through:
        - Agent type selection (CodeAgent vs ToolCallingAgent)
        - Tool selection from available toolbox
        - Model configuration (type, ID, API settings)
        - Advanced options like additional imports
        - Task prompt input
    """
    import sys

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Direct task mode - pass to run command
        task = sys.argv[1]
        remaining_args = sys.argv[2:]

        # Build new argv for main()
        new_argv = [sys.argv[0], 'run', task] + remaining_args
        sys.argv = new_argv
    elif len(sys.argv) == 1:
        # No arguments - launch interactive wizard
        sys.argv = [sys.argv[0], 'run']  # run without task triggers wizard
    # else: arguments starting with '-' will be handled by argparse

    main()


def web_agent_main():
    """
    Entry point for web agent CLI (effgen-web).

    A specialized agent for web browsing tasks.

    Usage:
        effgen-web "go to example.com and get the page title"
        effgen-web  # Interactive mode
    """
    import sys

    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Direct task mode
        task = sys.argv[1]
        sys.argv = [sys.argv[0], 'run', task, '--tools', 'web_search'] + sys.argv[2:]
    else:
        # Interactive mode - show help
        print(f"effGen Web Agent v{__version__}")
        print()
        print("Usage:")
        print("  effgen-web \"<task>\"           Run a web task")
        print("  effgen-web --model <model>    Specify model")
        print()
        print("Example:")
        print("  effgen-web \"Search for the latest Python release\"")
        print()
        return

    main()


if __name__ == "__main__":
    main()
