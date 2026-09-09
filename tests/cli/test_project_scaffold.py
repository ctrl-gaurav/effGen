"""``effgen quickstart --init`` — an empty directory to a project that runs.

The command writes four files, puts a daily spend cap in force when none is
configured, and prints the next three commands. What is pinned here is what a
newcomer depends on:

* the files land inside the named directory and nowhere else;
* the ``.env`` template names every provider variable and carries no value, so
  copying it unchanged cannot shadow a key the environment already has;
* the configuration carries only keys ``effgen run`` applies, and ``effgen run``
  with no ``-m`` takes the model from it;
* the model id is written provider-prefixed, since no ``--provider`` flag
  travels with a file;
* nothing prompts, so the same call works piped and in CI;
* an existing file, and an existing spend cap, are left alone without ``--force``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from effgen.cli import _main
from effgen.cli import scaffold as _scaffold


@pytest.fixture
def state(tmp_path, monkeypatch):
    """Point every piece of user state at *tmp_path* and hide the real keys."""
    home = tmp_path / "state"
    monkeypatch.setenv("EFFGEN_HOME", str(home))
    monkeypatch.setenv("EFFGEN_BUDGET_CONFIG", str(home / "budget.json"))
    monkeypatch.setenv("EFFGEN_NO_DOTENV", "1")
    for key in (
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY",
        "CEREBRAS_API_KEY", "TOGETHER_API_KEY", "FIREWORKS_API_KEY",
        "REPLICATE_API_TOKEN", "HF_TOKEN", "HUGGINGFACE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    return home


def _cli():
    cli = _main.CLIInterface()
    cli.console = None  # plain text, so the assertions read what a pipe gets
    return cli


def _init(*argv: str) -> tuple[int, str]:
    """Run ``effgen quickstart --init ...`` through the real parser."""
    args = _main.create_parser().parse_args(["quickstart", *argv])
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        code = _main._handle_quickstart_command(args, _cli())
    return code, out.getvalue()


# ── the command exists and does the whole job ────────────────────────────────

def test_init_writes_the_four_files_and_reports_them(state, tmp_path):
    project = tmp_path / "proj"
    code, output = _init("--init", str(project))

    assert code == 0
    for name, _description in _scaffold.SCAFFOLD_FILES:
        assert (project / name).is_file(), f"{name} was not written"
        assert name in output
    assert "Next three commands" in output


def test_init_prints_the_next_three_commands(state, tmp_path):
    _code, output = _init("--init", str(tmp_path / "proj"))
    for index, command in enumerate(_scaffold.next_commands(), 1):
        assert f"{index}. {command}" in output


def test_init_makes_no_model_call(state, tmp_path, monkeypatch):
    """The one step that must work before any key is set."""
    def _explode(*a, **k):  # pragma: no cover - only runs on a regression
        raise AssertionError("--init built an agent")

    monkeypatch.setattr(_main, "Agent", _explode)
    code, _output = _init("--init", str(tmp_path / "proj"))
    assert code == 0


def test_init_writes_nothing_outside_the_named_directory(state, tmp_path):
    project = tmp_path / "nested" / "proj"
    _init("--init", str(project))

    written = {p for p in tmp_path.rglob("*") if p.is_file()}
    outside = sorted(
        str(p) for p in written
        if project not in p.parents and p.parent != state and state not in p.parents
    )
    assert outside == [], f"wrote outside the project and the state directory: {outside}"


def test_init_creates_a_missing_directory(state, tmp_path):
    project = tmp_path / "a" / "b" / "c"
    assert _init("--init", str(project))[0] == 0
    assert (project / _scaffold.PROJECT_CONFIG_NAME).is_file()


def test_init_defaults_to_the_current_directory(state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _init("--init")[0] == 0
    assert (tmp_path / _scaffold.PROJECT_CONFIG_NAME).is_file()


# ── existing files ───────────────────────────────────────────────────────────

def test_init_keeps_an_existing_file_and_says_so(state, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    config = project / _scaffold.PROJECT_CONFIG_NAME
    config.write_text("model: mine\n", encoding="utf-8")

    code, output = _init("--init", str(project))
    assert code == 0
    assert config.read_text(encoding="utf-8") == "model: mine\n"
    assert "--force" in output


def test_force_replaces_an_existing_file(state, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    config = project / _scaffold.PROJECT_CONFIG_NAME
    config.write_text("model: mine\n", encoding="utf-8")

    assert _init("--init", str(project), "--force")[0] == 0
    assert "system_prompt:" in config.read_text(encoding="utf-8")


def test_a_directory_that_cannot_be_written_is_reported(state, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        code, output = _init("--init", str(blocked / "proj"))
    finally:
        blocked.chmod(0o700)
    assert code == 1
    assert "Fix:" in output


# ── the configuration ────────────────────────────────────────────────────────

def test_the_config_only_carries_keys_a_run_applies(state, tmp_path):
    import yaml

    from effgen.cli.commands._shared import _RUN_CONFIG_APPLIED_KEYS

    project = tmp_path / "proj"
    _init("--init", str(project))
    document = yaml.safe_load(
        (project / _scaffold.PROJECT_CONFIG_NAME).read_text(encoding="utf-8")
    )
    assert set(document) <= set(_RUN_CONFIG_APPLIED_KEYS), (
        f"{sorted(set(document) - set(_RUN_CONFIG_APPLIED_KEYS))} would be loaded "
        "and then ignored"
    )
    assert document["max_tokens"] > 0
    assert document["max_iterations"] > 0


def test_the_config_loads_and_reports_nothing_unapplied(state, tmp_path):
    project = tmp_path / "proj"
    _init("--init", str(project))
    cli = _cli()
    loaded = cli.config_loader.load_config(project / _scaffold.PROJECT_CONFIG_NAME)

    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        _main._warn_unapplied_config_keys(loaded.to_dict(), cli)
    assert out.getvalue().strip() == ""


def test_the_written_model_id_names_its_provider(state, tmp_path):
    """A file travels without a --provider flag, so the id has to carry one."""
    import yaml

    monkeyed = tmp_path / "proj"
    _init("--init", str(monkeyed))
    model = yaml.safe_load(
        (monkeyed / _scaffold.PROJECT_CONFIG_NAME).read_text(encoding="utf-8")
    )["model"]
    assert ":" in model, f"{model!r} names no provider or engine"


@pytest.mark.parametrize(
    ("suggested", "expected"),
    [
        (("gpt-5-nano", "openai", "openai key detected"), "openai:gpt-5-nano"),
        (("groq:openai/gpt-oss-20b", "groq", "groq key detected"),
         "groq:openai/gpt-oss-20b"),
        (("transformers:Qwen/Qwen2.5-1.5B-Instruct", None, "no cloud key found"),
         "transformers:Qwen/Qwen2.5-1.5B-Instruct"),
    ],
)
def test_a_bare_cloud_suggestion_is_prefixed(monkeypatch, suggested, expected):
    monkeypatch.setattr(
        "effgen.cli.commands._shared._quickstart_suggest_model", lambda: suggested
    )
    assert _scaffold.resolve_model()[0] == expected


# ── the .env template ────────────────────────────────────────────────────────

def test_the_env_template_names_every_provider_variable_with_no_value(state, tmp_path):
    project = tmp_path / "proj"
    _init("--init", str(project))
    body = (project / _scaffold.ENV_TEMPLATE_NAME).read_text(encoding="utf-8")

    assigned, commented = set(), set()
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # A commented-out entry, not prose that happens to name a variable:
            # the value is empty once any trailing comment is taken off.
            key, sep, rest = line.lstrip("# ").partition("=")
            if sep == "=" and key.isupper() and not rest.split("#", 1)[0].strip():
                commented.add(key)
            continue
        key, sep, value = line.partition("=")
        assert sep == "=", f"{line!r} is not a variable assignment"
        assert value == "", f"{key} was given a value; the template invents none"
        assigned.add(key)

    expected = {key for _p, keys in _scaffold.provider_env_keys() for key in keys}
    assert expected, "no provider is registered"
    assert assigned | commented == expected


def test_the_template_leaves_no_bare_endpoint_assignment(state, tmp_path):
    """An endpoint variable names an address, and a blank one is not "absent".

    Every OpenAI-protocol client on the machine reads ``OPENAI_BASE_URL``, and
    the OpenAI SDK treats a variable that is present but empty as a real
    address — it sends the request to ``''`` and the failure reads as a
    provider outage. Copying this template unchanged must not arm that.
    """
    from effgen.models._base_url import BASE_URL_ENV_VARS

    project = tmp_path / "proj"
    _init("--init", str(project))
    body = (project / _scaffold.ENV_TEMPLATE_NAME).read_text(encoding="utf-8")

    live = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for var in BASE_URL_ENV_VARS:
        assert var in body, f"{var} is no longer named in the template"
        assert not any(ln.startswith(f"{var}=") for ln in live), (
            f"{var} is assigned rather than commented out"
        )


def test_copying_the_template_unchanged_does_not_shadow_a_real_key(state, tmp_path):
    """An empty assignment must read as "no key", not as a key set to ''."""
    pytest.importorskip("dotenv")
    from dotenv import load_dotenv

    project = tmp_path / "proj"
    _init("--init", str(project))
    env_file = project / ".env"
    env_file.write_text(
        (project / _scaffold.ENV_TEMPLATE_NAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # load_dotenv writes into the real os.environ, and override=False still sets
    # every name the file lists that is not already there — so this test loads
    # the whole template into the session unless the mapping is put back.
    # Leaking it once cost 45 later live cells, which failed as provider
    # outages because their endpoint variable had been left set to "".
    before = dict(os.environ)
    try:
        os.environ["GROQ_API_KEY"] = "real-key-from-the-environment"
        load_dotenv(env_file, override=False)
        assert os.environ["GROQ_API_KEY"] == "real-key-from-the-environment"

        from effgen.models.auth import check_keys
        assert check_keys(["groq"])["groq"]["available"] is True
        # A variable the file set to empty still reads as absent.
        assert check_keys(["together"])["together"]["available"] is False
    finally:
        for name in set(os.environ) - set(before):
            del os.environ[name]
        os.environ.update(before)


def test_the_gitignore_excludes_the_env_file(state, tmp_path):
    project = tmp_path / "proj"
    _init("--init", str(project))
    lines = (project / _scaffold.GITIGNORE_NAME).read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in lines]


# ── the example script ───────────────────────────────────────────────────────

def test_the_example_is_valid_python_that_imports_and_runs_offline(state, tmp_path):
    project = tmp_path / "proj"
    _init("--init", str(project))
    example = project / _scaffold.EXAMPLE_NAME

    compile(example.read_text(encoding="utf-8"), str(example), "exec")
    # Importing it must not call a model: everything happens under main().
    probe = (
        "import importlib.util,sys;"
        f"spec=importlib.util.spec_from_file_location('scaffolded_example', {str(example)!r});"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "print(m.MODEL)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert ":" in proc.stdout.strip()


def test_a_reasoning_model_gets_the_headroom_it_needs(state, tmp_path):
    """512 output tokens can be spent entirely on hidden reasoning.

    The answer is then empty and still billed, so the budget written for such a
    model is the one the framework asks for, not the ordinary default.
    """
    import yaml

    from effgen.models._adapter_utils import default_max_output_tokens

    project = tmp_path / "proj"
    _init("--init", str(project), "-m", "openai:gpt-5-nano")
    document = yaml.safe_load(
        (project / _scaffold.PROJECT_CONFIG_NAME).read_text(encoding="utf-8")
    )
    from types import SimpleNamespace
    wanted = default_max_output_tokens(SimpleNamespace(model_name="openai:gpt-5-nano"))
    assert document["max_tokens"] == wanted > _scaffold.DEFAULT_MAX_TOKENS
    assert f"max_tokens={wanted}," in (
        project / _scaffold.EXAMPLE_NAME
    ).read_text(encoding="utf-8")


def test_an_ordinary_model_keeps_the_ordinary_budget(state, tmp_path):
    import yaml

    project = tmp_path / "proj"
    _init("--init", str(project), "-m", "groq:openai/gpt-oss-20b")
    document = yaml.safe_load(
        (project / _scaffold.PROJECT_CONFIG_NAME).read_text(encoding="utf-8")
    )
    assert document["max_tokens"] == _scaffold.DEFAULT_MAX_TOKENS
    assert "hidden reasoning" not in (
        project / _scaffold.PROJECT_CONFIG_NAME
    ).read_text(encoding="utf-8")


def test_the_example_and_the_config_agree_on_the_budget(state, tmp_path):
    import re

    import yaml

    for model in ("openai:gpt-5-nano", "groq:openai/gpt-oss-20b"):
        project = tmp_path / model.replace(":", "-").replace("/", "-")
        _init("--init", str(project), "-m", model)
        document = yaml.safe_load(
            (project / _scaffold.PROJECT_CONFIG_NAME).read_text(encoding="utf-8")
        )
        example = (project / _scaffold.EXAMPLE_NAME).read_text(encoding="utf-8")
        in_example = int(re.search(r"max_tokens=(\d+),", example).group(1))
        assert in_example == document["max_tokens"], model


def test_the_example_and_the_config_name_the_same_model(state, tmp_path):
    import yaml

    project = tmp_path / "proj"
    _init("--init", str(project))
    model = yaml.safe_load(
        (project / _scaffold.PROJECT_CONFIG_NAME).read_text(encoding="utf-8")
    )["model"]
    assert f'MODEL = "{model}"' in (
        project / _scaffold.EXAMPLE_NAME
    ).read_text(encoding="utf-8")


# ── the spend cap ────────────────────────────────────────────────────────────

def test_a_first_scaffold_puts_a_daily_cap_in_force(state, tmp_path):
    _code, output = _init("--init", str(tmp_path / "proj"))
    budget = json.loads(Path(os.environ["EFFGEN_BUDGET_CONFIG"]).read_text())
    assert budget["daily"] == _scaffold.DEFAULT_DAILY_BUDGET_USD
    assert "Spend cap" in output


def test_an_existing_cap_is_never_overwritten(state, tmp_path):
    from effgen.cli.commands.cost import set_daily_budget

    set_daily_budget(7.5)
    _code, output = _init("--init", str(tmp_path / "proj"), "--budget", "1")
    budget = json.loads(Path(os.environ["EFFGEN_BUDGET_CONFIG"]).read_text())
    assert budget["daily"] == 7.5
    assert "already configured" in output


def test_budget_zero_sets_no_cap(state, tmp_path):
    _code, output = _init("--init", str(tmp_path / "proj"), "--budget", "0")
    assert not Path(os.environ["EFFGEN_BUDGET_CONFIG"]).exists()
    assert "none in force" in output


def test_a_monthly_cap_survives_a_daily_one_being_set(state, tmp_path):
    from effgen.cli.commands.cost import configured_daily_budget, set_daily_budget

    path = Path(os.environ["EFFGEN_BUDGET_CONFIG"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"monthly": 20.0}))

    set_daily_budget(2.0)
    written = json.loads(path.read_text())
    assert written == {"monthly": 20.0, "daily": 2.0}
    assert configured_daily_budget() == 2.0


# ── `effgen run -c` reads the model the config names ─────────────────────────

def test_run_takes_the_model_from_the_config_file(state, tmp_path, monkeypatch):
    """Without this the scaffolded config names a model no run would use."""
    built: list = []

    class _StubAgent:
        def __init__(self, config, session_id=None):
            built.append(config)
            self.config = config
            self.execution_tracker = None

        def run(self, task, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(
                output="425", success=True, metadata={"cost_usd": 0.0},
                execution_trace=[], iterations=1, tool_calls=0, tokens_used=3,
                execution_time=0.01, citations=[], sources=[], error=None,
            )

        def close(self):
            pass

    monkeypatch.setattr(_main, "Agent", _StubAgent)
    project = tmp_path / "proj"
    _init("--init", str(project), "-m", "groq:openai/gpt-oss-20b")

    args = _main.create_parser().parse_args(
        ["run", "what is 25*17?", "-c", str(project / _scaffold.PROJECT_CONFIG_NAME)]
    )
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        code = _cli().run_agent(args)

    assert code in (0, None)
    assert len(built) == 1
    assert built[0].model == "groq:openai/gpt-oss-20b"
    assert "from" in out.getvalue()


def test_an_unreadable_config_is_refused_before_a_model_is_chosen(state, tmp_path, monkeypatch):
    """No model is announced for a run that cannot start.

    Model resolution reads the config file, so it happens after the file is
    loaded — which also means a run about to fail on a missing ``-c`` no longer
    names a model it will never call.
    """
    monkeypatch.setattr(_main, "Agent", lambda *a, **k: pytest.fail("built an agent"))
    args = _main.create_parser().parse_args(
        ["run", "hi", "-c", str(tmp_path / "absent.yaml")]
    )
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(out):
        code = _cli().run_agent(args)
    text = out.getvalue()
    assert code == 1
    assert "Configuration file not found" in text
    assert "Using model" not in text


def test_an_explicit_model_flag_still_wins_over_the_config(state, tmp_path, monkeypatch):
    built: list = []

    class _StubAgent:
        def __init__(self, config, session_id=None):
            built.append(config)
            self.config = config
            self.execution_tracker = None

        def run(self, task, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(
                output="ok", success=True, metadata={}, execution_trace=[],
                iterations=1, tool_calls=0, tokens_used=1, execution_time=0.0,
                citations=[], sources=[], error=None,
            )

        def close(self):
            pass

    monkeypatch.setattr(_main, "Agent", _StubAgent)
    project = tmp_path / "proj"
    _init("--init", str(project), "-m", "groq:openai/gpt-oss-20b")

    args = _main.create_parser().parse_args([
        "run", "hi",
        "-c", str(project / _scaffold.PROJECT_CONFIG_NAME),
        "-m", "gemini:gemini-3.1-flash-lite",
    ])
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        _cli().run_agent(args)
    assert built[0].model == "gemini:gemini-3.1-flash-lite"


# ── `effgen doctor` points at the template ───────────────────────────────────

def test_doctor_offers_the_template_when_no_env_file_exists(state, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    _init("--init", str(project))
    monkeypatch.chdir(project)
    assert "cp .env.example .env" in (_main._env_template_hint() or "")


def test_doctor_says_nothing_once_the_env_file_exists(state, tmp_path, monkeypatch):
    project = tmp_path / "proj"
    _init("--init", str(project))
    (project / ".env").write_text("", encoding="utf-8")
    monkeypatch.chdir(project)
    assert _main._env_template_hint() is None


def test_doctor_says_nothing_in_a_directory_with_no_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _main._env_template_hint() is None


# ── the late-binding seam, exercised ─────────────────────────────────────────

def test_a_stub_on_main_quickstart_init_step_intercepts_the_command(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        _main, "_quickstart_init_step", lambda *a, **k: (calls.append(a) or 0)
    )
    args = _main.create_parser().parse_args(["quickstart", "--init", "somewhere"])
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        code = _main._handle_quickstart_command(args, _cli())
    assert code == 0
    assert len(calls) == 1, "the guided run resolved its scaffold somewhere else"


# ── `effgen config init` writes the same working document ────────────────────

def test_config_init_writes_a_document_a_run_applies(state, tmp_path, monkeypatch):
    import yaml

    from effgen.cli.commands._shared import _RUN_CONFIG_APPLIED_KEYS

    monkeypatch.chdir(tmp_path)
    args = _main.create_parser().parse_args(["config", "init"])
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        code = _cli()._config_init(args)
    assert code == 0

    document = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert set(document) <= set(_RUN_CONFIG_APPLIED_KEYS)
    assert ":" in document["model"]
