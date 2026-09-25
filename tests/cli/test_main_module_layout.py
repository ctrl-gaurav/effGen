"""Layout of the ``effgen.cli._main`` split.

``_main`` keeps five responsibilities and nothing else:

* ``CLIInterface`` — its constructor, the output methods it inherits from
  ``_console.CLIConsoleMixin``, and the thin delegates to ``commands/*``;
* parsing — ``_EffgenArgumentParser`` and ``create_parser``, the latter reduced
  to the top-level parser, the global flags and the ordered builder calls;
* dispatch and the entry points — ``_dispatch``, ``main``, ``_run_cli`` and the
  ``effgen-agent`` / ``effgen-web`` entry points;
* the availability shims — the ``rich`` and ``effgen`` import blocks other
  command modules read back as module attributes;
* the re-export block that keeps ``effgen.cli._main`` the one import path.

Four invariants keep the split safe:

* every moved name is defined in exactly one module and re-exported from
  ``_main`` as the same object, so an importer or a patch target never moves;
* late binding through ``_main`` is load-bearing — replacing ``Agent``,
  ``_quickstart_code_step``, ``_doctor_code_report`` or ``_dispatch`` on that
  module must still change what the real code path does. A moved function that
  resolved its neighbour in its own module would break this without failing any
  assertion of its own, so each one is exercised rather than read;
* the split modules only import ``_main`` inside a function, so the import graph
  stays acyclic;
* the parser builders register exactly the commands they claim, in the recorded
  order, and every ``CLIInterface`` method is defined once across the MRO.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import io
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from effgen.cli import _main

#: The modules the split is made of, ``_main`` excluded.
SPLIT_MODULES = (
    "effgen.cli._console",
    "effgen.cli._logging",
    "effgen.cli.commands.doctor",
    "effgen.cli.commands.health",
    "effgen.cli.commands.plugin",
    "effgen.cli.commands.presets",
    "effgen.cli.commands.prompts",
    "effgen.cli.commands.quickstart",
    "effgen.cli.commands.resume",
    "effgen.cli.parsers.agent",
    "effgen.cli.parsers.catalog",
    "effgen.cli.parsers.history",
    "effgen.cli.parsers.jobs",
    "effgen.cli.parsers.ops",
)

#: Every name that left ``_main``, and the module that now defines it. A name
#: drifting back — or being copied rather than moved — is caught here.
OWNERS = {
    # the output methods, mixed into CLIInterface
    "CLIConsoleMixin": "effgen.cli._console",
    # logging setup and the echoed-error console filter
    "_SELF_RENDERING_ERROR_COMMANDS": "effgen.cli._logging",
    "_CLIEchoedErrorFilter": "effgen.cli._logging",
    "_suppress_echoed_error_logs": "effgen.cli._logging",
    "setup_logging": "effgen.cli._logging",
    # doctor and its report builders
    "_handle_doctor_command": "effgen.cli.commands.doctor",
    "_doctor_exit_code": "effgen.cli.commands.doctor",
    "_doctor_reliability_report": "effgen.cli.commands.doctor",
    "_doctor_code_report": "effgen.cli.commands.doctor",
    "_doctor_system_report": "effgen.cli.commands.doctor",
    "_env_template_hint": "effgen.cli.commands.doctor",
    "_doctor_live_probe": "effgen.cli.commands.doctor",
    # the guided first run
    "_QUICKSTART_CODE_TASK": "effgen.cli.commands.quickstart",
    "_quickstart_code_wanted": "effgen.cli.commands.quickstart",
    "_quickstart_code_step": "effgen.cli.commands.quickstart",
    "_quickstart_init_step": "effgen.cli.commands.quickstart",
    "_handle_quickstart_command": "effgen.cli.commands.quickstart",
    # the remaining stranded handlers
    "_handle_prompts_command": "effgen.cli.commands.prompts",
    "_handle_resume_command": "effgen.cli.commands.resume",
    "_render_plugin_template": "effgen.cli.commands.plugin",
    "_create_plugin_scaffold": "effgen.cli.commands.plugin",
}

#: The output methods ``CLIInterface`` inherits rather than defines.
CONSOLE_METHODS = {
    "_human",
    "_animate",
    "print",
    "print_data",
    "print_line",
    "print_header",
    "_human_stream",
    "print_success",
    "print_error",
    "print_warning",
    "print_error_panel",
    "_create_stats_table",
}

#: ``_main``'s own top-level definitions, imports excluded: parsing, dispatch,
#: the entry points and the class. Anything else belongs in a split module.
MAIN_DEFINITIONS = {
    "logger",
    "CLIInterface",
    "_EffgenArgumentParser",
    "create_parser",
    "_extract_theme_arg",
    "_dispatch",
    "_silence_broken_pipe",
    "_fold_output_streams",
    "main",
    "_run_cli",
    "agent_main",
    "web_agent_main",
}

#: The commands each parser builder registers, in declaration order.
PARSER_BUILDERS = {
    ("effgen.cli.parsers.agent", "add_run_parser"): ["run"],
    ("effgen.cli.parsers.agent", "add_resume_parser"): ["resume"],
    ("effgen.cli.parsers.history", "add_sessions_parser"): ["sessions"],
    ("effgen.cli.parsers.history", "add_runs_parser"): ["runs"],
    ("effgen.cli.parsers.agent", "add_chat_parser"): ["chat"],
    ("effgen.cli.parsers.agent", "add_code_parser"): ["code"],
    ("effgen.cli.parsers.ops", "add_serve_parser"): ["serve"],
    ("effgen.cli.parsers.ops", "add_top_parsers"): ["top", "monitor"],
    ("effgen.cli.parsers.catalog", "add_config_parser"): ["config"],
    ("effgen.cli.parsers.catalog", "add_tools_parser"): ["tools"],
    ("effgen.cli.parsers.catalog", "add_models_parser"): ["models"],
    ("effgen.cli.parsers.catalog", "add_examples_parser"): ["examples"],
    ("effgen.cli.parsers.ops", "add_health_parser"): ["health"],
    ("effgen.cli.parsers.ops", "add_doctor_parser"): ["doctor"],
    ("effgen.cli.parsers.ops", "add_plugin_parser"): ["create-plugin"],
    ("effgen.cli.parsers.catalog", "add_presets_parser"): ["presets"],
    ("effgen.cli.parsers.agent", "add_quickstart_parsers"): ["quickstart", "tutorial"],
    ("effgen.cli.parsers.jobs", "add_workflow_parser"): ["workflow"],
    ("effgen.cli.parsers.jobs", "add_batch_parser"): ["batch"],
    ("effgen.cli.parsers.jobs", "add_eval_parser"): ["eval"],
    ("effgen.cli.parsers.jobs", "add_compare_parser"): ["compare"],
    ("effgen.cli.parsers.jobs", "add_battle_parser"): ["battle"],
    ("effgen.cli.parsers.jobs", "add_bench_parser"): ["bench"],
    ("effgen.cli.parsers.agent", "add_debug_parser"): ["debug"],
    ("effgen.cli.parsers.jobs", "add_cost_parser"): ["cost"],
    ("effgen.cli.parsers.jobs", "add_report_parser"): ["report"],
    ("effgen.cli.parsers.catalog", "add_prompts_parser"): ["prompts"],
}

#: The commands ``effgen --help`` lists, in order. ``loadtest`` is declared by
#: ``effgen.cli.loadtest`` and closes the list.
COMMAND_ORDER = [
    "run", "resume", "sessions", "runs", "chat", "code", "serve", "top",
    "monitor", "config", "tools", "models", "examples", "health", "doctor",
    "create-plugin", "presets", "quickstart", "tutorial", "workflow", "batch",
    "eval", "compare", "battle", "bench", "debug", "cost", "report", "prompts",
    "loadtest",
]

#: Names ``dir(effgen.cli._main)`` carried before the split, less the dunders
#: the interpreter stamps on every module. None of them may stop resolving.
RECORDED_SURFACE = {
    "Agent", "AgentConfig", "AgentMode", "Any", "CLIInterface", "CODE_THEME",
    "ConfigLoader", "Console", "KNOWN_PROVIDERS", "Layout", "Live", "Markdown",
    "PROVIDER_ALIASES", "Panel", "Path", "Progress", "RICH_AVAILABLE",
    "SpinnerColumn", "Syntax", "Table", "TaskRequest", "TaskResponse",
    "TextColumn", "WebSocket", "_CLIEchoedErrorFilter", "_EffgenArgumentParser",
    "_PydanticBaseModel", "_PydanticConfigDict", "_QUICKSTART_CLOUD_MODELS",
    "_QUICKSTART_CODE_TASK", "_QUICKSTART_LOCAL_MODEL",
    "_RUN_CONFIG_APPLIED_KEYS", "_SELF_RENDERING_ERROR_COMMANDS", "__version__",
    "_artifact_format", "_batch_structured_kwargs", "_browse_sessions",
    "_checkpoint_run_kwargs", "_create_plugin_scaffold", "_dispatch",
    "_doctor_code_report", "_doctor_exit_code", "_doctor_live_probe",
    "_doctor_reliability_report", "_doctor_system_report", "_dotenv_disabled",
    "_env_search_paths", "_extract_theme_arg", "_fmt_cost",
    "_fold_output_streams", "_general_purpose_tool_names", "_get_console",
    "_glyph", "_handle_batch_command", "_handle_compare_command",
    "_handle_cost_command", "_handle_doctor_command", "_handle_eval_command",
    "_handle_prompts_command", "_handle_quickstart_command",
    "_handle_report_command", "_handle_resume_command", "_handle_runs_command",
    "_handle_sessions_command", "_handle_workflow_command", "_invoked_command",
    "_landing", "_onboarding", "_one_line", "_preflight_model_hint",
    "_print_group_help", "_progress", "_quickstart_code_step",
    "_quickstart_code_wanted", "_quickstart_suggest_model",
    "_read_done_indices", "_render_comparison_tables", "_render_plugin_template",
    "_render_session", "_resolve_eval_suite", "_run_cli", "_session_matches",
    "_short_ts", "_silence_broken_pipe", "_suppress_echoed_error_logs",
    "_warn_unapplied_config_keys", "_write_html_report_arg",
    "_write_result_artifact", "agent_main", "annotations", "argparse",
    "asyncio", "console_is_interactive", "create_parser", "datetime",
    "filter_incompatible_tools", "get_tool_registry", "importlib", "json",
    "json_ensure_ascii", "load_env_files", "load_model", "logger", "logging",
    "main", "os", "render_table", "resolve_provider_name", "rprint",
    "setup_logging", "sys", "web_agent_main",
}


def module(name: str):
    return importlib.import_module(name)


def top_level_names(name: str) -> set[str]:
    """Every name the module binds at module scope, imports excluded."""
    tree = ast.parse(Path(inspect.getsourcefile(module(name))).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _is_type_checking_guard(node: ast.AST) -> bool:
    """Whether *node* is an ``if TYPE_CHECKING:`` block, which never runs."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
        or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")
    )


def module_scope_imports(name: str) -> set[str]:
    """The dotted modules *name* imports at import time.

    A ``try:``/``if:`` at module scope still runs, so its body counts; an
    ``if TYPE_CHECKING:`` block never does, so it does not.
    """
    tree = ast.parse(Path(inspect.getsourcefile(module(name))).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        stack = [node]
        while stack:
            current = stack.pop()
            if _is_type_checking_guard(current):
                continue
            if isinstance(current, ast.Import):
                found.update(alias.name for alias in current.names)
            elif isinstance(current, ast.ImportFrom):
                if current.module:
                    found.add(current.module)
                    found.update(f"{current.module}.{a.name}" for a in current.names)
            elif isinstance(current, ast.Try | ast.If):
                stack.extend(
                    child
                    for attr in ("body", "orelse", "finalbody", "handlers")
                    for child in getattr(current, attr, [])
                )
            elif isinstance(current, ast.ExceptHandler):
                stack.extend(current.body)
    return found


def plain_cli():
    """A ``CLIInterface`` with no rich console, so output stays plain text."""
    cli = _main.CLIInterface()
    cli.console = None
    return cli


def capture(fn, *args, **kwargs):
    """Run *fn*, returning ``(exit code, everything it wrote)``."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = fn(*args, **kwargs)
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue() + err.getvalue()


class _StubAgent:
    """Stands in for ``_main.Agent`` and records the configs it was built with."""

    built: list = []

    def __init__(self, config, session_id=None):
        type(self).built.append(config)
        self.config = config
        self.execution_tracker = None

    def run(self, task, **kwargs):
        return SimpleNamespace(
            output="STUBBED-ANSWER", success=True, metadata={"cost_usd": 0.0},
            execution_trace=[], iterations=1, tool_calls=0, tokens_used=3,
            execution_time=0.01, citations=[], sources=[], error=None,
        )

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Ownership: one definition, in one module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "owner"), sorted(OWNERS.items()))
def test_moved_name_is_defined_in_its_owner_alone(name, owner):
    assert name in top_level_names(owner), f"{name} is not defined in {owner}"
    for other in (*SPLIT_MODULES, "effgen.cli._main"):
        if other != owner:
            assert name not in top_level_names(other), (
                f"{name} is also defined in {other}; it must live in {owner} alone"
            )


def test_main_defines_only_parsing_dispatch_and_the_entry_points():
    assert top_level_names("effgen.cli._main") == MAIN_DEFINITIONS


def test_no_name_is_defined_in_two_modules_of_the_split():
    seen: dict[str, str] = {}
    clashes: list[str] = []
    for name in SPLIT_MODULES:
        for defined in top_level_names(name):
            # Each module keeps its own logger, exactly as every other command
            # module does; that is the one name they are meant to share.
            if defined == "logger":
                continue
            if defined in seen:
                clashes.append(f"{defined}: {seen[defined]} and {name}")
            else:
                seen[defined] = name
    assert clashes == [], f"defined more than once: {clashes}"


# ---------------------------------------------------------------------------
# ``_main`` stays the one import path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "owner"), sorted(OWNERS.items()))
def test_main_re_exports_every_moved_name_as_the_same_object(name, owner):
    assert hasattr(_main, name), f"{name} is no longer reachable on effgen.cli._main"
    assert getattr(_main, name) is getattr(module(owner), name)


def test_every_recorded_name_still_resolves():
    missing = sorted(RECORDED_SURFACE - set(dir(_main)))
    assert missing == [], f"names dropped from effgen.cli._main: {missing}"


def test_main_stays_at_its_recorded_path():
    # A test derives the repository root from this file's location.
    assert Path(_main.__file__).name == "_main.py"
    assert Path(_main.__file__).parent.name == "cli"


# ---------------------------------------------------------------------------
# Import direction: the split modules reach ``_main`` only at call time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_a_split_module_never_imports_main_at_module_scope(name):
    imported = module_scope_imports(name)
    assert not any(i == "effgen.cli._main" or i.startswith("effgen.cli._main.")
                   for i in imported), (
        f"{name} imports effgen.cli._main at module scope; it must do so inside a function"
    )


def test_main_imports_the_console_and_logging_modules_at_import_time():
    imported = module_scope_imports("effgen.cli._main")
    assert "effgen.cli._console" in imported
    assert "effgen.cli._logging" in imported


def test_the_parser_builders_are_imported_inside_create_parser_only():
    imported = module_scope_imports("effgen.cli._main")
    assert not any(i.startswith("effgen.cli.parsers") for i in imported), (
        "the parser builders must stay function-local so importing effgen.cli "
        "does not pull them in"
    )
    assert "from effgen.cli.parsers import" in inspect.getsource(_main.create_parser)


# ---------------------------------------------------------------------------
# The four late-binding seams, exercised rather than read
# ---------------------------------------------------------------------------

def test_a_stub_on_main_agent_intercepts_a_real_run(monkeypatch):
    _StubAgent.built = []
    monkeypatch.setattr(_main, "Agent", _StubAgent)
    args = _main.create_parser().parse_args(
        ["run", "what is 2+2?", "-m", "does-not-matter", "-q"]
    )
    code, output = capture(plain_cli().run_agent, args)
    assert code in (0, None)
    assert len(_StubAgent.built) == 1
    assert "STUBBED-ANSWER" in output


def test_a_stub_on_main_quickstart_code_step_intercepts_the_guided_run(monkeypatch):
    _StubAgent.built = []
    calls: list = []
    monkeypatch.setattr(_main, "Agent", _StubAgent)
    monkeypatch.setattr(
        _main, "_quickstart_code_step", lambda *a, **k: (calls.append(a) or 0)
    )
    args = _main.create_parser().parse_args(
        ["quickstart", "-m", "does-not-matter", "-y", "--task", "hi"]
    )
    code, output = capture(_main._handle_quickstart_command, args, plain_cli())
    assert code == 0
    assert len(calls) == 1, "the guided run resolved its coding step somewhere else"
    assert "You're set!" in output


def test_a_stub_on_main_doctor_code_report_intercepts_the_coding_repl(monkeypatch):
    from effgen.cli.code.repl import CodeREPL

    seen: list = []
    real = _main._doctor_code_report
    monkeypatch.setattr(
        _main, "_doctor_code_report", lambda w: (seen.append(w) or real(w))
    )
    args = SimpleNamespace(
        workspace="/tmp", model="fake:model", provider=None, plan_only=False,
        auto_edit=False, assume_yes=False, temperature=None, max_tokens=None,
        max_iterations=None, quiet=False, verbose=False,
    )
    repl = CodeREPL(plain_cli(), args)
    _, output = capture(repl._cmd_doctor)
    assert seen == ["/tmp"], "doctor resolved its coding report somewhere else"
    assert "Coding" in output


def test_a_stub_on_main_dispatch_intercepts_the_entry_point(monkeypatch):
    hits: list = []
    monkeypatch.setattr(
        _main, "_dispatch", lambda args, cli, parser: (hits.append(args.command) or 0)
    )
    monkeypatch.setattr(sys, "argv", ["effgen", "tools", "list"])
    code, _ = capture(_main.main)
    assert hits == ["tools"]
    assert code == 0


def test_the_moved_modules_read_the_rich_shims_off_main():
    # A rich-less install has no Panel/Markdown/Table to import; the moved code
    # must read them off ``_main`` behind the existing guard rather than
    # importing them, or such an install would fail at import time.
    for name in ("effgen.cli.commands.quickstart", "effgen.cli.commands.doctor",
                 "effgen.cli._console"):
        imported = module_scope_imports(name)
        assert not any(i.startswith("rich") for i in imported), (
            f"{name} imports rich at module scope"
        )


# ---------------------------------------------------------------------------
# ``CLIInterface``: the same methods, each defined once
# ---------------------------------------------------------------------------

def test_the_output_methods_are_inherited_from_the_console_module():
    from effgen.cli._console import CLIConsoleMixin

    assert CLIConsoleMixin in _main.CLIInterface.__mro__
    own = set(vars(_main.CLIInterface))
    assert not (CONSOLE_METHODS & own), (
        f"{sorted(CONSOLE_METHODS & own)} are defined on CLIInterface as well as the mixin"
    )
    for name in CONSOLE_METHODS:
        assert name in vars(CLIConsoleMixin)
        assert hasattr(_main.CLIInterface, name)


def test_no_cliinterface_method_is_defined_twice_across_the_mro():
    defined: Counter[str] = Counter()
    for klass in _main.CLIInterface.__mro__:
        for name, value in vars(klass).items():
            if name.startswith("__"):
                continue
            if callable(value) or isinstance(value, staticmethod | classmethod | property):
                defined[name] += 1
    duplicated = {n: c for n, c in defined.items() if c > 1}
    assert duplicated == {}, f"defined more than once across the MRO: {duplicated}"
    assert len(defined) == 51


# ---------------------------------------------------------------------------
# Parser assembly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("where", "commands"), sorted(PARSER_BUILDERS.items()))
def test_a_parser_builder_registers_exactly_the_commands_it_claims(where, commands):
    import argparse

    module_name, builder_name = where
    builder = getattr(module(module_name), builder_name)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    if "preset_choices" in inspect.signature(builder).parameters:
        builder(subparsers, preset_choices=["math", "coding"])
    else:
        builder(subparsers)
    assert list(subparsers.choices) == commands


def test_the_top_level_command_order_is_unchanged():
    import argparse

    parser = _main.create_parser()
    action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    assert list(action.choices) == COMMAND_ORDER


def test_create_parser_still_lives_in_main_and_returns_the_top_level_parser():
    # ``effgen.completion`` and the public-surface snapshot import it from here.
    assert _main.create_parser.__module__ == "effgen.cli._main"
    assert "create_parser" in top_level_names("effgen.cli._main")
    assert _main.create_parser().prog


def test_every_preset_taking_builder_is_given_the_same_choices():
    # Recomputing the preset list per module could disagree with the one
    # ``create_parser`` resolved, so the choices are passed down instead.
    source = inspect.getsource(_main.create_parser)
    for (_, builder_name), _ in PARSER_BUILDERS.items():
        builder = None
        for module_name, name in PARSER_BUILDERS:
            if name == builder_name:
                builder = getattr(module(module_name), name)
                break
        if "preset_choices" in inspect.signature(builder).parameters:
            assert f"{builder_name}(subparsers, preset_choices=_preset_choices)" in source
