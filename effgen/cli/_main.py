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
    effgen chat --model Qwen/Qwen2.5-3B-Instruct

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

from __future__ import annotations

import argparse
import asyncio  # noqa: F401 - module attribute for command modules that import it from here
import importlib.util  # noqa: F401 - module attribute for command modules
import json  # noqa: F401 - module attribute for command modules
import logging
import os
import sys
from datetime import datetime  # noqa: F401 - module attribute for command modules
from pathlib import Path
from typing import Any

# Rich terminal output (fallback to basic if not available)
try:
    from rich import print as rprint  # noqa: F401
    from rich.console import Console
    from rich.layout import Layout  # noqa: F401
    from rich.live import Live  # noqa: F401
    from rich.markdown import Markdown  # noqa: F401 - module attribute for command modules
    from rich.panel import Panel  # noqa: F401 - module attribute for command modules
    from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: F401 - module attrs
    from rich.syntax import Syntax  # noqa: F401 - availability shim / module attribute
    from rich.table import Table  # noqa: F401 - module attribute for command modules
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None


# Import effGen components
try:
    from effgen import (  # noqa: F401
        Agent,
        AgentConfig,
        ConfigLoader,
        __version__,
        get_tool_registry,
        load_model,
    )
    from effgen.core.agent import AgentMode  # noqa: F401 - read as a module attribute
except ImportError:
    print("Error: effGen package not found. Please install it first.")
    sys.exit(1)

# .env discovery + loading is shared with the library-facing ``effgen.load_env()``
# so the CLI and a script/notebook resolve keys the same way. These aliases keep
# the historical CLI-internal names (used by tests and callers) working.
from effgen._env import dotenv_disabled as _dotenv_disabled  # noqa: F401 - re-export
from effgen._env import env_search_paths as _env_search_paths  # noqa: F401 - re-export
from effgen._env import load_env as load_env_files

# Branded bare-command landing (TTY-only; falls through to the wizard otherwise).
from effgen.cli import landing as _landing

# Tips, first-run welcome, "did you mean?" and teaching-error helpers.
from effgen.cli import onboarding as _onboarding

# Live status / progress presentation (TTY-aware; degrades to plain text).
from effgen.cli import progress as _progress  # noqa: F401 - module attribute

# The methods ``CLIInterface`` prints through. Mixed into the class below so a
# caller keeps reaching them as ``CLIInterface.print`` / ``.print_error`` / ….
from effgen.cli._console import CLIConsoleMixin

# Logging setup and the echoed-error console filter, re-exported so
# ``effgen.cli._main.setup_logging`` keeps resolving.
from effgen.cli._logging import (  # noqa: F401 - re-export
    _SELF_RENDERING_ERROR_COMMANDS,
    _CLIEchoedErrorFilter,
    _suppress_echoed_error_logs,
    setup_logging,
)

# Helpers shared by several commands. They live in ``effgen.cli.commands._shared``
# and are re-exported here so ``effgen.cli._main`` stays the import path callers
# and tests use.
from effgen.cli.commands._shared import (  # noqa: F401 - re-export
    _QUICKSTART_CLOUD_MODELS,
    _QUICKSTART_LOCAL_MODEL,
    _RUN_CONFIG_APPLIED_KEYS,
    KNOWN_PROVIDERS,
    PROVIDER_ALIASES,
    _checkpoint_run_kwargs,
    _invoked_command,
    _preflight_model_hint,
    _print_group_help,
    _quickstart_suggest_model,
    _warn_unapplied_config_keys,
    filter_incompatible_tools,
    resolve_provider_name,
)

# The ``batch`` command. Imported at module scope so the handler is an attribute
# of this module as soon as it finishes importing.
from effgen.cli.commands.batch import (  # noqa: F401 - re-export
    _batch_structured_kwargs,
    _handle_batch_command,
    _read_done_indices,
)

# The ``compare`` command handler and its per-metric terminal tables.
# Module-level free functions; imported at module scope + re-exported (tests
# reach ``_main._handle_compare_command`` / ``_render_comparison_tables``).
from effgen.cli.commands.compare import (  # noqa: F401 - re-export
    _handle_compare_command,
    _render_comparison_tables,
)

# The ``cost`` command handler (spend dashboard + budget). Imported at module
# scope + re-exported; ``effgen.cli.__init__`` re-exports it from here, so
# ``from effgen.cli import _handle_cost_command`` keeps resolving.
from effgen.cli.commands.cost import _handle_cost_command  # noqa: F401 - re-export

# The ``doctor`` command handler and its report builders. Imported at module
# scope + re-exported: ``effgen chat`` and ``effgen code``'s ``/doctor`` import
# ``_handle_doctor_command`` from here at call time, and the report builders are
# read back through this module so replacing one changes what doctor renders.
from effgen.cli.commands.doctor import (  # noqa: F401 - re-export
    _doctor_code_report,
    _doctor_exit_code,
    _doctor_live_probe,
    _doctor_reliability_report,
    _doctor_system_report,
    _env_template_hint,
    _handle_doctor_command,
)

# The ``eval`` command handler plus ``--suite`` resolution. Module-level free
# functions taking ``(args, cli)``; imported at module scope + re-exported (tests
# reach ``_main._handle_eval_command`` / ``_resolve_eval_suite``).
from effgen.cli.commands.eval import (  # noqa: F401 - re-export
    _handle_eval_command,
    _resolve_eval_suite,
)

# The ``create-plugin`` scaffold generator and its template renderer. Imported
# at module scope + re-exported (tests reach ``_main._create_plugin_scaffold``).
from effgen.cli.commands.plugin import (  # noqa: F401 - re-export
    _create_plugin_scaffold,
    _render_plugin_template,
)

# The ``prompts`` command handler. Imported at module scope + re-exported;
# ``effgen.cli.__init__`` re-exports it from here, so
# ``from effgen.cli import _handle_prompts_command`` keeps resolving.
from effgen.cli.commands.prompts import _handle_prompts_command  # noqa: F401 - re-export

# The ``quickstart``/``tutorial`` guided run, its project scaffold and its
# coding step. Imported at module scope + re-exported; the scaffold and the
# coding step are called back through this module so replacing either here
# changes what the guided run does.
from effgen.cli.commands.quickstart import (  # noqa: F401 - re-export
    _QUICKSTART_CODE_TASK,
    _handle_quickstart_command,
    _quickstart_code_step,
    _quickstart_code_wanted,
    _quickstart_init_step,
)

# The ``report`` command handler and the ``-o``/``--report`` artifact writers
# shared by ``eval``/``compare``/``cost``. Imported at module scope so the names
# are attributes of this module (tests reach ``_main._artifact_format`` /
# ``_write_result_artifact`` and the ``eval``/``compare``/``cost`` handlers call
# the bare names).
from effgen.cli.commands.report import (  # noqa: F401 - re-export
    _artifact_format,
    _handle_report_command,
    _write_html_report_arg,
    _write_result_artifact,
)

# The ``resume`` command handler (continue a run from a checkpoint snapshot).
# Imported at module scope + re-exported.
from effgen.cli.commands.resume import _handle_resume_command  # noqa: F401 - re-export

# The ``serve`` command launcher and its ``/run``, ``/tools``, ``/``, ``/slo``,
# ``/ws`` convenience routes. ``CLIInterface.serve_api`` /
# ``._register_convenience_routes`` delegate there. The request models,
# ``WebSocket`` type, and default-tool helper are re-exported so FastAPI's
# annotation resolution and any ``_main.TaskRequest``/``_main.WebSocket``
# reference keep working (the models must live in the handler's own module).
from effgen.cli.commands.serve import (  # noqa: F401 - re-export
    TaskRequest,
    TaskResponse,
    WebSocket,
    _general_purpose_tool_names,
    _PydanticBaseModel,
    _PydanticConfigDict,
)

# The ``sessions`` and ``runs`` command handlers plus their formatting helpers.
# Module-level free functions taking ``(args, cli)``; imported at module scope so
# every name is an attribute of this module as soon as it finishes importing
# (tests reach them as ``_main._handle_sessions_command`` / ``_main._fmt_cost``).
from effgen.cli.commands.sessions import (  # noqa: F401 - re-export
    _browse_sessions,
    _fmt_cost,
    _handle_runs_command,
    _handle_sessions_command,
    _one_line,
    _render_session,
    _session_matches,
    _short_ts,
)

# The ``workflow`` command handler. A module-level free function taking
# ``(args, cli)``; imported at module scope so it is an attribute of this module
# as soon as it finishes importing (tests reach it as ``_main._handle_workflow_command``).
from effgen.cli.commands.workflow import (  # noqa: F401 - re-export
    _handle_workflow_command,
)
from effgen.ui.palette import glyph as _glyph  # noqa: F401 - module attribute

# JSON written to a stream that cannot encode it is escaped, not transliterated.
from effgen.ui.render import json_ensure_ascii  # noqa: F401 - module attribute

# Shared Rich theme + console factory (one palette across the whole CLI).
# ``console_is_interactive``/``render_table`` are re-exported so callers and
# tests keep reaching them as ``effgen.cli._main`` attributes.
from effgen.ui.tables import (  # noqa: F401 - re-export
    console_is_interactive,
    render_table,
)
from effgen.ui.theme import CODE_THEME  # noqa: F401 - module attribute for command modules
from effgen.ui.theme import get_console as _get_console

logger = logging.getLogger(__name__)


class CLIInterface(CLIConsoleMixin):
    """Main CLI interface for effGen."""

    def __init__(self) -> None:
        """Initialize CLI interface."""
        self.console = _get_console() if RICH_AVAILABLE else None
        self.config_loader = ConfigLoader()
        self.tool_registry = get_tool_registry()
        # When True, all human-facing chatter is routed to stderr so stdout
        # carries only machine-readable output (e.g. `effgen run --json`). A
        # stderr-bound rich console is created lazily on first use.
        self._human_to_stderr = False
        self._err_console = None

    def interactive_wizard(self, args: Any) -> int | None:
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
        from effgen.cli.commands.run import interactive_wizard

        return interactive_wizard(self, args)

    def run_agent(self, args: argparse.Namespace) -> int | None:
        """
        Run an agent with a task.

        Args:
            args: Parsed command-line arguments
        """
        from effgen.cli.commands.run import run_agent

        return run_agent(self, args)

    def _handle_interrupt(self, agent) -> None:
        """Render a friendly Ctrl-C stop (partial trace + 'Stopped.'), no traceback."""
        from effgen.cli.commands.run import handle_interrupt

        return handle_interrupt(self, agent)

    def chat_mode(self, args: argparse.Namespace) -> int | None:
        """Interactive chat REPL.

        Delegates to :class:`effgen.cli.chat.ChatREPL`, which provides streaming
        answers with a thinking spinner, a model/tool-aware prompt, slash
        commands (``/model``, ``/tools``, ``/cost``, ``/trace``, …), persistent
        ↑/↓ history, multiline input, and a per-turn Ctrl-C that cancels the
        current turn without exiting the session.
        """
        from effgen.cli.commands.chat import chat_mode

        return chat_mode(self, args)

    def serve_api(self, args: argparse.Namespace) -> int | None:
        """Start the effGen API server.

        Serves the OpenAI-compatible ``/v1`` app plus the ``/run``, ``/tools``,
        ``/``, ``/slo``, ``/ws`` convenience routes.
        """
        from effgen.cli.commands.serve import serve_api
        return serve_api(self, args)

    def _register_convenience_routes(self, app: Any) -> None:
        """Attach the legacy ``/run``, ``/tools``, ``/``, ``/slo``, ``/ws``
        routes onto the secure app from ``create_app``."""
        from effgen.cli.commands.serve import register_convenience_routes
        return register_convenience_routes(self, app)

    def config_commands(self, args: argparse.Namespace) -> int | None:
        """Configuration management commands."""
        from effgen.cli.commands.config import config_commands
        return config_commands(self, args)

    def _config_set(self, args):
        """Handle 'effgen config set <key> <value>'."""
        from effgen.cli.commands.config import config_set
        return config_set(self, args)

    def _config_show(self, args):
        """Show current configuration."""
        from effgen.cli.commands.config import config_show
        return config_show(self, args)

    def _config_validate(self, args):
        """Validate configuration file."""
        from effgen.cli.commands.config import config_validate
        return config_validate(self, args)

    def _config_init(self, args):
        """Initialize a new configuration file."""
        from effgen.cli.commands.config import config_init
        return config_init(self, args)

    def tools_commands(self, args: argparse.Namespace) -> int | None:
        """Tool management commands."""
        from effgen.cli.commands.tools import tools_commands
        return tools_commands(self, args)

    def _suggest_tool(self, name: str) -> None:
        """Print a 'tool not found' error with close-match suggestions."""
        from effgen.cli.commands.tools import suggest_tool
        return suggest_tool(self, name)

    def _tools_list(self, args):
        """List available tools."""
        from effgen.cli.commands.tools import tools_list
        return tools_list(self, args)

    def _example_input(self, metadata, tool=None) -> dict:
        """Build a runnable example input for a tool from its metadata."""
        from effgen.cli.commands.tools import example_input
        return example_input(self, metadata, tool)

    def _print_tool_usage(self, metadata, tool=None) -> None:
        """Print a tool's input schema and a copy-paste runnable example."""
        from effgen.cli.commands.tools import print_tool_usage
        return print_tool_usage(self, metadata, tool)

    def _tools_info(self, args: argparse.Namespace) -> int | None:
        """Show detailed tool information."""
        from effgen.cli.commands.tools import tools_info
        return tools_info(self, args)

    def _tools_test(self, args: argparse.Namespace) -> int | None:
        """Test a tool with sample input."""
        from effgen.cli.commands.tools import tools_test
        return tools_test(self, args)

    def models_commands(self, args: argparse.Namespace) -> int | None:
        """Model management commands."""
        from effgen.cli.commands.models import models_commands
        return models_commands(self, args)

    @staticmethod
    def _price_cell(rec) -> str:
        """Format a model's input/output price per 1M tokens for a table cell."""
        from effgen.cli.commands.models import price_cell
        return price_cell(rec)

    # File extensions that count as actual model weights (an ".index.json" is a
    # shard manifest, not weights — a repo with only a manifest is still partial).
    _WEIGHT_SUFFIXES = (
        ".safetensors", ".bin", ".gguf", ".pt", ".pth",
        ".onnx", ".msgpack", ".h5", ".tflite", ".ot",
    )

    def _local_cached_models(self) -> list[dict]:
        """Models actually downloaded in the local HuggingFace cache (on disk)."""
        from effgen.cli.commands.models import local_cached_models
        return local_cached_models(self)

    def _local_model_context_window(self, path: str) -> int | None:
        """Read the model's max context length from its on-disk ``config.json``."""
        from effgen.cli.commands.models import local_model_context_window
        return local_model_context_window(self, path)

    def _models_list(self, args):
        """List models from the drift-aware registry (not a static yaml)."""
        from effgen.cli.commands.models import models_list
        return models_list(self, args)

    def _rich_tables(self) -> bool:
        """True when rich table rendering fits the destination (a real terminal)."""
        from effgen.cli.commands.models import rich_tables
        return rich_tables(self)

    @staticmethod
    def _browse_filter_sort(recs, args):
        """Apply the browse filters/sort to a list of catalog records."""
        from effgen.cli.commands.models import browse_filter_sort
        return browse_filter_sort(recs, args)

    def _models_browse(self, args):
        """Browse the full cross-provider catalog with search/filter/sort/paging."""
        from effgen.cli.commands.models import models_browse
        return models_browse(self, args)

    @classmethod
    def _price_in_out_cells(cls, rec) -> tuple[str, str]:
        """Return (input, output) price cells for the split-column browse table."""
        from effgen.cli.commands.models import price_in_out_cells
        return price_in_out_cells(rec)

    def _local_model_payload(self, entry: dict) -> dict:
        """Build the local-cache facts for one model: engines, size, ctx, status."""
        from effgen.cli.commands.models import local_model_payload
        return local_model_payload(self, entry)

    def _render_local_model_info(self, payload: dict) -> None:
        """Render the 'this model is in your local cache' block for `models info`."""
        from effgen.cli.commands.models import render_local_model_info
        return render_local_model_info(self, payload)

    def _models_info(self, args):
        """Show detailed information for one model from the registry."""
        from effgen.cli.commands.models import models_info
        return models_info(self, args)

    def _models_load(self, args):
        """Pre-load a model into the model pool."""
        from effgen.cli.commands.models import models_load
        return models_load(self, args)

    def _models_unload(self, args):
        """Unload a model from memory."""
        from effgen.cli.commands.models import models_unload
        return models_unload(self, args)

    def _models_status(self, args):
        """Show loaded models and GPU memory status."""
        from effgen.cli.commands.models import models_status
        return models_status(self, args)

    def _models_status_json(self) -> int:
        """Emit the GPU table + loaded models as JSON for ops/edge tooling."""
        from effgen.cli.commands.models import models_status_json
        return models_status_json(self)

    def _models_refresh(self, args):
        """Refresh the bundled model catalog from each provider's live API."""
        from effgen.cli.commands.models import models_refresh
        return models_refresh(self, args)

    def examples_commands(self, args: argparse.Namespace) -> int | None:
        """Run example scripts."""
        from effgen.cli.commands.examples import examples_commands
        return examples_commands(self, args)

    @staticmethod
    def _find_examples_dir() -> "Path | None":
        """Locate the bundled `examples/` directory."""
        from effgen.cli.commands.examples import find_examples_dir
        return find_examples_dir()

    def _examples_list(self, args):
        """List available examples."""
        from effgen.cli.commands.examples import examples_list
        return examples_list(self, args)

    def _examples_run(self, args):
        """Run an example script."""
        from effgen.cli.commands.examples import examples_run
        return examples_run(self, args)


class _EffgenArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that turns an "invalid choice" into a teaching error.

    A mistyped subcommand (``effgen rnu``) or a mistyped ``choices=`` option
    value (``--preset codng``) gets a fuzzy "did you mean 'run'?" /
    "did you mean 'coding'?" suggestion instead of just dumping the usage
    banner, so the CLI is self-correcting like the model/tool suggesters
    elsewhere. Because ``add_subparsers`` inherits this parser class, the
    handler fires for both the top-level command and every sub-command.
    """

    def _print_message(self, message: str, file=None):  # noqa: D401 - argparse hook
        """Write argparse's own output, folded to ASCII when the stream needs it.

        Usage banners, ``--help`` and parser errors are written by argparse
        straight to the stream, so the em-dashes and arrows in the command
        descriptions reach a ``PYTHONIOENCODING=ascii`` terminal unfolded and the
        write raises. Folding here is a no-op on a UTF-8 stream, so the bytes a
        normal terminal receives are unchanged; anything the glyph table does not
        cover is replaced rather than allowed to raise.
        """
        if not message:
            return
        stream = file or sys.stderr
        from effgen.ui.render import ascii_fold

        text = ascii_fold(message, stream)
        try:
            stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(stream, "encoding", None) or "ascii"
            stream.write(text.encode(encoding, "replace").decode(encoding))

    def error(self, message: str):  # noqa: D401 - argparse hook
        if "invalid choice:" in message:
            import re

            m_bad = re.search(r"invalid choice: '([^']*)'", message)
            m_arg = re.search(r"argument ([^:]+): invalid choice", message)
            # The offending argument's valid choices are spelled out in the
            # message ("(choose from 'a', 'b', …)"); fall back to the
            # subparsers action so an older argparse phrasing still works.
            choices: list[str] = []
            if "choose from" in message:
                tail = message.split("choose from", 1)[1]
                choices = re.findall(r"'([^']*)'", tail)
            sub_action = next(
                (a for a in self._actions
                 if isinstance(a, argparse._SubParsersAction)), None)
            if not choices and sub_action is not None:
                choices = list(sub_action.choices.keys())
            if m_bad and choices:
                bad = m_bad.group(1)
                arg_name = m_arg.group(1) if m_arg else None
                hint = _onboarding.did_you_mean(bad, choices, n=1, cutoff=0.5)
                self.print_usage(sys.stderr)
                # A positional sub-command (dest matches the subparsers action)
                # reads best as "unknown command"; a flag value as "invalid value".
                is_subcommand = sub_action is not None and (
                    arg_name is None or arg_name in (sub_action.dest, sub_action.metavar)
                )
                if is_subcommand:
                    line = f"{self.prog}: error: unknown command '{bad}'."
                    if hint:
                        line += f" {hint}"
                    line += f"\nAvailable commands: {', '.join(choices)}."
                else:
                    label = f" for {arg_name}" if arg_name else ""
                    line = f"{self.prog}: error: invalid value '{bad}'{label}."
                    if hint:
                        line += f" {hint}"
                    line += f"\nValid choices: {', '.join(choices)}."
                print(line, file=sys.stderr)
                self.exit(2)
        super().error(message)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    # Drive --preset choices from the preset registry so all 9 presets are
    # accepted (rag/media/multimodal/notify were previously rejected).
    try:
        from effgen.presets import list_presets as _list_presets
        _preset_choices = sorted(_list_presets().keys())
    except Exception:
        _preset_choices = ['math', 'research', 'coding', 'general', 'minimal']

    parser = _EffgenArgumentParser(
        description=f"effGen v{__version__} - CLI for agent framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  effgen run "What is 25 * 17?" --model Qwen/Qwen2.5-1.5B-Instruct
  effgen run "Summarize quantum computing" -m gpt-5-nano --provider openai
  effgen run "Tell me a joke" -m groq:openai/gpt-oss-20b
  effgen chat -m gpt-5-nano --provider openai
  effgen models list                 # provider registry overview
  effgen models list --provider groq # full per-model detail
  effgen doctor --live --cheap       # confirm providers are actually usable
  effgen tools list

Model id formats:
  - Local HuggingFace repo:   Qwen/Qwen2.5-1.5B-Instruct
  - Provider-prefixed:        openai:gpt-5-nano   groq:openai/gpt-oss-20b
  - Bare id + --provider:     -m gpt-5-nano --provider openai
  Providers: openai, anthropic, gemini, cerebras, groq, together,
             fireworks, replicate, hf
        """
    )

    parser.add_argument('--version', action='version', version=f'effGen {__version__}')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output (show DEBUG/INFO logs)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Quiet output (errors only)')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--no-animation', action='store_true',
                        help='Disable live spinners/progress animation '
                             '(also via NO_COLOR or EFFGEN_NO_ANIM=1)')
    parser.add_argument('--theme', choices=['default', 'high-contrast', 'monochrome', 'light'],
                        help='Color theme for terminal output (also via EFFGEN_THEME). '
                             'high-contrast targets low-vision readers; monochrome keeps '
                             'structure without hue; NO_COLOR still turns color off entirely')
    parser.add_argument('--completion', choices=['bash', 'zsh', 'fish'],
                        help='Print shell completion script and exit')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Each command's flags are declared in ``effgen.cli.parsers``, next to the
    # other flags of its family. The calls below are in the order the commands
    # appear in ``effgen --help``, which is the order argparse reports them in.
    from effgen.cli.parsers import agent, catalog, history, jobs, ops  # noqa: PLC0415

    agent.add_run_parser(subparsers, preset_choices=_preset_choices)
    agent.add_resume_parser(subparsers, preset_choices=_preset_choices)
    history.add_sessions_parser(subparsers)
    history.add_runs_parser(subparsers)
    agent.add_chat_parser(subparsers, preset_choices=_preset_choices)
    agent.add_code_parser(subparsers)
    ops.add_serve_parser(subparsers)
    ops.add_top_parsers(subparsers)
    catalog.add_config_parser(subparsers)
    catalog.add_tools_parser(subparsers)
    catalog.add_models_parser(subparsers)
    catalog.add_examples_parser(subparsers)
    ops.add_health_parser(subparsers)
    ops.add_doctor_parser(subparsers)
    ops.add_plugin_parser(subparsers)
    catalog.add_presets_parser(subparsers)
    agent.add_quickstart_parsers(subparsers)
    jobs.add_workflow_parser(subparsers)
    jobs.add_batch_parser(subparsers, preset_choices=_preset_choices)
    jobs.add_eval_parser(subparsers, preset_choices=_preset_choices)
    jobs.add_compare_parser(subparsers, preset_choices=_preset_choices)
    jobs.add_battle_parser(subparsers)
    agent.add_debug_parser(subparsers, preset_choices=_preset_choices)
    jobs.add_cost_parser(subparsers)
    jobs.add_report_parser(subparsers)
    catalog.add_prompts_parser(subparsers)

    # Load test command
    from effgen.cli.loadtest import add_loadtest_subparser  # noqa: PLC0415
    add_loadtest_subparser(subparsers)

    return parser


def _extract_theme_arg(argv: list[str]) -> tuple[list[str], str | None]:
    """Pull a ``--theme <name>`` (any position) out of *argv*.

    ``--theme`` selects the terminal color theme for every command's output, so
    it is accepted before or after the subcommand. Returns the remaining
    arguments and the requested theme name (or ``None``).
    """
    remaining: list[str] = []
    theme: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--theme":
            if i + 1 < len(argv):
                theme = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        if arg.startswith("--theme="):
            theme = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return remaining, theme


def _dispatch(args: argparse.Namespace, cli: "CLIInterface", parser: argparse.ArgumentParser) -> int:
    """Route parsed *args* to the matching command handler and return its exit code.

    One routing table shared by ``main()`` and the bare-command landing, so a
    quick action chosen on the landing reaches the same handler as typing the
    command directly.
    """
    if args.command == 'run':
        exit_code = cli.run_agent(args)
    elif args.command == 'chat':
        exit_code = cli.chat_mode(args)
    elif args.command == 'code':
        from effgen.cli.commands.code import run_code_command
        exit_code = run_code_command(cli, args)
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
    elif args.command == 'health':
        from effgen.cli.commands.health import handle_health_command
        exit_code = handle_health_command(args, cli)
    elif args.command == 'doctor':
        exit_code = _handle_doctor_command(args)
    elif args.command == 'resume':
        exit_code = _handle_resume_command(args, cli)
    elif args.command == 'sessions':
        exit_code = _handle_sessions_command(args, cli)
    elif args.command == 'runs':
        exit_code = _handle_runs_command(args, cli)
    elif args.command == 'create-plugin':
        exit_code = _create_plugin_scaffold(args.plugin_name, args.output_dir)
    elif args.command == 'presets':
        from effgen.cli.commands.presets import handle_presets_command
        exit_code = handle_presets_command(args, cli)
    elif args.command in ('quickstart', 'tutorial'):
        exit_code = _handle_quickstart_command(args, cli)
    elif args.command == 'workflow':
        exit_code = _handle_workflow_command(args, cli)
    elif args.command == 'batch':
        exit_code = _handle_batch_command(args, cli)
    elif args.command == 'eval':
        exit_code = _handle_eval_command(args, cli)
    elif args.command == 'compare':
        exit_code = _handle_compare_command(args, cli)
    elif args.command == 'battle':
        from effgen.cli.battle import run_battle_command
        exit_code = run_battle_command(args)
    elif args.command in ('top', 'monitor'):
        from effgen.cli.monitor import run_monitor_command
        exit_code = run_monitor_command(args)
    elif args.command == 'cost':
        exit_code = _handle_cost_command(args, cli)
    elif args.command == 'report':
        exit_code = _handle_report_command(args, cli)
    elif args.command == 'prompts':
        exit_code = _handle_prompts_command(args, cli)
    elif args.command == 'loadtest':
        func = getattr(args, 'func', None)
        if func:
            exit_code = func(args)
        else:
            from effgen.cli.loadtest import run_loadtest_command
            exit_code = run_loadtest_command(args)
    elif args.command == 'debug':
        from effgen.debug.inspector import run_debug_cli
        # Validate an explicit --provider up front (a typo like "grok"
        # should fail fast with a suggestion), mirroring `run`/`chat`.
        provider, prov_err = resolve_provider_name(getattr(args, 'provider', None))
        if prov_err:
            cli.print_error(prov_err)
            exit_code = 1
        else:
            exit_code = run_debug_cli(
                task=args.task,
                preset=getattr(args, 'preset', None),
                model=getattr(args, 'model', None),
                provider=provider,
                step=getattr(args, 'step', False),
            )
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
    return exit_code if exit_code is not None else 0


def _silence_broken_pipe() -> None:
    """Point stdout at the null device after a reader closed the pipe.

    ``effgen models list --json | head`` leaves the process writing into a pipe
    nobody reads. Without this the interpreter's exit-time flush raises again
    and prints ``Exception ignored ... BrokenPipeError`` after the command has
    already done its job.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError):  # pragma: no cover - stdout already gone
        pass


def _fold_output_streams():
    """Route ``stdout``/``stderr`` through an ASCII fold when they need one.

    A console forced to a non-UTF-8 encoding cannot write the em-dashes, table
    rules and glyphs the listing and summary commands print, and the failed
    write turns a command that did its work into one that exits non-zero. This
    substitutes ASCII stand-ins at the single point where text becomes bytes,
    so every command -- including one added later -- reports its real status.

    A stream that can already encode those characters is left exactly as it is,
    so a normal terminal receives the same bytes as before. Returns a callable
    that puts the original streams back.
    """
    from effgen.ui.render import ascii_folding_stream

    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = ascii_folding_stream(sys.stdout)
    sys.stderr = ascii_folding_stream(sys.stderr)

    def restore() -> None:
        sys.stdout, sys.stderr = original_stdout, original_stderr

    return restore


def main() -> None:
    """Main entry point for CLI."""
    restore_streams = _fold_output_streams()
    try:
        _run_cli()
    finally:
        restore_streams()


def _run_cli() -> None:
    """Parse the command line and dispatch to the selected command."""
    # Before anything can warn: render this package's own warnings as one line
    # instead of Python's file/line/source-echo block, which lands mid-answer
    # and points at internals the reader did not write.
    from effgen.cli import _warnings as _warning_render

    _warning_render.install()

    # Load .env early so all subcommands see API keys (see load_env_files).
    load_env_files()

    # A --theme selects the color theme for every console built below; accept it
    # in any position (EFFGEN_THEME serves the same purpose without a flag).
    argv, theme_arg = _extract_theme_arg(sys.argv[1:])
    if theme_arg:
        os.environ["EFFGEN_THEME"] = theme_arg

    parser = create_parser()
    args = parser.parse_args(argv)

    # Handle completion script generation
    if getattr(args, 'completion', None):
        from effgen.completion import get_completion
        print(get_completion(args.completion))
        sys.exit(0)

    # Setup logging. --verbose / --quiet may appear either before the
    # subcommand (global) or after it (per-command); honor whichever is set.
    _verbose = getattr(args, 'verbose', False)
    setup_logging(
        verbose=_verbose,
        log_file=getattr(args, 'log_file', None),
        quiet=getattr(args, 'quiet', False),
    )
    # For a command that renders its own failures, keep the console free of the
    # duplicate raw library ERROR line at default verbosity (--verbose / a
    # --log-file still carry the full diagnostic stream).
    if not _verbose and getattr(args, 'command', None) in _SELF_RENDERING_ERROR_COMMANDS:
        _suppress_echoed_error_logs()

    # Create CLI interface
    cli = CLIInterface()

    # On a bare `effgen` at an interactive terminal, show the branded landing
    # (logo + quick actions) instead of dropping straight into the wizard; the
    # landing absorbs the first-run welcome. Every other path — a subcommand,
    # piped/redirected output, --quiet, CI, no rich — keeps its current behavior.
    show_landing = _landing.should_show(cli, args)

    # One-time friendly welcome on first interactive use (records a flag so it
    # only ever shows once). Silent under --quiet / non-interactive / CI. Skipped
    # when the landing will render it inline.
    if not show_landing:
        _onboarding.maybe_show_first_run_welcome(quiet=getattr(args, 'quiet', False))

    # Route to appropriate handler
    try:
        if show_landing:
            exit_code = _landing.run(cli, parser, args, _dispatch)
        else:
            exit_code = _dispatch(args, cli, parser)

        # A gentle, rotating tip at a natural moment — only after the commands a
        # human watches finish cleanly, never under --quiet / non-interactive /
        # EFFGEN_TIPS=0, and only every few runs (see onboarding.maybe_print_tip).
        _TIP_COMMANDS = {'run', 'chat', 'code', 'quickstart', 'tutorial', 'presets', 'doctor'}
        if (
            exit_code == 0
            and args.command in _TIP_COMMANDS
            and not getattr(args, 'output_json', False)
        ):
            _onboarding.maybe_print_tip(quiet=getattr(args, 'quiet', False))

        # Flush inside the guard: a pipe closed by the reader must be reported
        # here, not by the interpreter's exit-time flush where it becomes an
        # "Exception ignored" line no one can act on.
        sys.stdout.flush()
        sys.exit(exit_code)

    except BrokenPipeError:
        # The reader closed the pipe first (`effgen … | head`). The command did
        # its work; report the conventional SIGPIPE status without a traceback.
        _silence_broken_pipe()
        sys.exit(141)
    except KeyboardInterrupt:
        # Both of these are for the human, so they go to stderr: a command
        # interrupted or failed mid-pipeline must not put prose on stdout where
        # a `--json` reader or a downstream `jq` expects the result alone.
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


def agent_main() -> None:
    """
    Entry point for effgen-agent CLI (similar to smolagent).

    A generalist command to run a multi-step agent that can be equipped with various tools.

    Usage:
        # Run with direct prompt and options
        effgen-agent "Plan a trip to Tokyo" --model Qwen/Qwen2.5-1.5B-Instruct --tools web_search calculator

        # Run in interactive mode (launches setup wizard when no prompt provided)
        effgen-agent

    Interactive mode guides you through:
        - Reasoning style (auto / react / single — one Agent, different strategies)
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


def web_agent_main() -> None:
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
