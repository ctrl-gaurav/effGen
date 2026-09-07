"""Onboarding surfaces that lead a first-run user to the coding agent.

Covers the readiness checks ``effgen doctor`` reports (workspace, sandbox
backend, git), the guided run's coding step and the flags that select it, and
the bare-command landing action. The readiness checks run against real
directories and the real git installation — they read the machine, so nothing
about them is simulated. The guided coding step is driven with a stub agent so
the step's own reporting is what is under test, not a model's wording.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from effgen.cli import _main, landing
from effgen.cli import onboarding as ob
from effgen.cli.code import build_code_tools
from effgen.cli.code.readiness import ReadinessCheck, code_readiness
from effgen.tools.builtin._fs import WORKSPACE_ENV_VAR


def _capture(fn, *a, **k):
    """Run *fn* capturing stdout/stderr, returning ``(stdout, stderr)``."""
    out, err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        result = fn(*a, **k)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return result, out.getvalue(), err.getvalue()


def _named(report, name: str) -> ReadinessCheck:
    return next(c for c in report.checks if c.name == name)


# --------------------------------------------------------------------------- #
# readiness — workspace
# --------------------------------------------------------------------------- #

def test_explicit_workspace_is_reported_ready_and_not_created(tmp_path: Path):
    target = tmp_path / "ws"
    target.mkdir()
    report = code_readiness(str(target))
    check = _named(report, "workspace")
    assert check.ok and check.status == "ready"
    assert str(target) in check.detail and "-w/--workspace" in check.detail
    assert report.workspace == str(target)


def test_a_missing_workspace_is_ready_and_left_uncreated(tmp_path: Path):
    target = tmp_path / "not-yet"
    report = code_readiness(str(target))
    check = _named(report, "workspace")
    assert check.ok and "created on first use" in check.detail
    # A check reads the machine; it never leaves a directory behind.
    assert not target.exists()


def test_workspace_env_var_is_the_source_when_no_flag(tmp_path, monkeypatch):
    ws = tmp_path / "env-ws"
    ws.mkdir()
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(ws))
    check = _named(code_readiness(), "workspace")
    assert check.ok and WORKSPACE_ENV_VAR in check.detail


def test_current_directory_is_the_source_when_nothing_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    check = _named(code_readiness(), "workspace")
    assert check.ok and "current directory" in check.detail


def test_a_file_where_the_workspace_should_be_is_not_ready(tmp_path: Path):
    target = tmp_path / "afile"
    target.write_text("x", encoding="utf-8")
    check = _named(code_readiness(str(target)), "workspace")
    assert check.ok is False
    assert check.status == "not a directory" and check.fix


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes to a read-only directory")
def test_an_unwritable_workspace_is_reported_with_a_fix(tmp_path: Path):
    target = tmp_path / "locked"
    target.mkdir(mode=0o500)
    try:
        check = _named(code_readiness(str(target)), "workspace")
        assert check.ok is False
        assert check.status == "read-only" and "-w/--workspace" in check.fix
    finally:
        target.chmod(0o700)


# --------------------------------------------------------------------------- #
# readiness — sandbox
# --------------------------------------------------------------------------- #

def test_sandbox_off_is_the_one_blocking_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "off")
    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert check.ok is False and check.status == "disabled"
    assert "EFFGEN_SANDBOX_BACKEND" in check.fix
    assert code_readiness(str(tmp_path)).ready is False


def test_subprocess_backend_is_usable_but_reports_its_limit(tmp_path, monkeypatch):
    """The backend runs, and the line names the isolation this host gives it.

    Which isolation that is depends on the host — a runner without unprivileged
    user namespaces gets none — so the expected wording is taken from the same
    probe the check reads rather than assumed. What holds everywhere is that the
    backend is usable, is reported as limited, and points at Docker.
    """
    from effgen.cli.code import readiness as readiness_mod

    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")
    network_isolated, writes_confined = readiness_mod._subprocess_isolation()

    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert check.ok and check.status == "limited"
    assert "Docker" in check.fix
    if writes_confined:
        assert "network isolated, writes confined to the workspace" in check.detail
    elif network_isolated:
        assert "network isolated" in check.detail
        assert "cannot confine writes" in check.detail
    else:
        assert "no user namespaces" in check.detail
        assert "network isolated" not in check.detail


def test_sandbox_check_does_not_claim_confinement_the_host_cannot_deliver(
    tmp_path, monkeypatch,
):
    """The readiness line states what this host enforces, never more.

    A host without the namespaces the subprocess backend needs still runs
    code — it just confines nothing. Reporting "writes confined to the
    workspace" there would send the user into a coding session trusting a
    boundary that is not present.
    """
    from effgen.cli.code import readiness as readiness_mod

    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "subprocess")

    monkeypatch.setattr(readiness_mod, "_subprocess_isolation", lambda: (True, False))
    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert "confined to the workspace" not in check.detail
    assert "cannot confine" in check.detail
    assert "network isolated" in check.detail

    monkeypatch.setattr(readiness_mod, "_subprocess_isolation", lambda: (False, False))
    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert "confined to the workspace" not in check.detail
    assert "no user namespaces" in check.detail

    monkeypatch.setattr(readiness_mod, "_subprocess_isolation", lambda: (True, True))
    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert "writes confined to the workspace" in check.detail


def test_a_known_but_unimplemented_backend_says_so(tmp_path, monkeypatch):
    """firecracker resolves as a name but cannot run code, so say that."""
    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "firecracker")
    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert check.ok is False and check.status == "unavailable"
    assert "not implemented" in check.detail
    assert "docker or subprocess" in check.fix


def test_an_unknown_backend_name_is_reported_not_guessed(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "nitro")
    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert check.ok is False and check.status == "unknown"
    assert "docker or subprocess" in check.fix


def test_docker_backend_reports_unavailable_when_the_daemon_does_not_answer(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "docker")
    monkeypatch.setattr("effgen.cli.code.readiness._probe_docker", lambda: False)
    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert check.ok is False and check.status == "unavailable"
    assert "Start Docker" in check.fix


def test_auto_prefers_docker_when_it_answers(tmp_path, monkeypatch):
    monkeypatch.delenv("EFFGEN_SANDBOX_BACKEND", raising=False)
    monkeypatch.setattr("effgen.cli.code.readiness._probe_docker", lambda: True)
    check = _named(code_readiness(str(tmp_path)), "sandbox")
    assert check.ok and check.status == "ready" and "docker" in check.detail


def test_the_check_never_resolves_or_caches_a_sandbox_backend(tmp_path, monkeypatch):
    """The readiness probe must not claim the process-wide backend a run picks."""
    from effgen.security import sandbox as sandbox_mod

    sandbox_mod.reset_sandbox_cache()
    monkeypatch.delenv("EFFGEN_SANDBOX_BACKEND", raising=False)
    code_readiness(str(tmp_path))
    assert sandbox_mod._resolved_backend is None


# --------------------------------------------------------------------------- #
# readiness — git
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_check_names_the_repository_the_workspace_is_in(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "o@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Owner"], cwd=root, check=True)
    check = _named(code_readiness(str(root)), "git")
    assert check.ok and check.status == "ready"
    assert str(root) in check.detail


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_outside_a_repository_git_is_still_ok_with_a_fix(tmp_path: Path):
    outside = tmp_path / "plain"
    outside.mkdir()
    check = _named(code_readiness(str(outside)), "git")
    # A workspace outside a repository is a supported way to run, so it never
    # blocks; the fix explains what repository context would add.
    assert check.ok and check.status == "no repository"
    assert "git init" in check.fix


def test_a_machine_without_git_is_still_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "effgen.cli.code.readiness.shutil.which",
        lambda name: None if name == "git" else "/usr/bin/" + name,
    )
    report = code_readiness(str(tmp_path))
    check = _named(report, "git")
    assert check.ok and check.status == "missing" and "Install git" in check.fix
    assert report.ready is True


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #

def _doctor_args(**kw):
    ns = SimpleNamespace(
        output_json=False, doctor_provider=None, live=False, cheap=False, workspace=None
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_doctor_json_carries_the_code_section_beside_the_existing_ones(tmp_path):
    rc, out, _ = _capture(
        _main._handle_doctor_command, _doctor_args(output_json=True, workspace=str(tmp_path))
    )
    assert rc == 0
    doc = json.loads(out)
    # The keys that were already there are untouched; "code" is added.
    for key in ("providers", "system", "reliability"):
        assert key in doc
    code = doc["code"]
    assert code["workspace"] == str(tmp_path)
    assert {c["name"] for c in code["checks"]} == {"workspace", "sandbox", "git"}
    assert isinstance(code["ready"], bool)


def test_doctor_json_stays_one_parseable_document(tmp_path):
    _, out, _ = _capture(
        _main._handle_doctor_command, _doctor_args(output_json=True, workspace=str(tmp_path))
    )
    json.loads(out)  # raises if anything printed alongside it


def test_doctor_human_output_names_the_coding_checks(tmp_path):
    rc, out, _ = _capture(_main._handle_doctor_command, _doctor_args(workspace=str(tmp_path)))
    assert rc == 0
    assert "Coding" in out
    for name in ("workspace", "sandbox", "git"):
        assert name in out


def test_doctor_exit_code_ignores_coding_readiness(tmp_path, monkeypatch):
    # A disabled sandbox is reported, but `doctor` without --live still exits 0:
    # the exit code stays a statement about provider usability.
    monkeypatch.setenv("EFFGEN_SANDBOX_BACKEND", "off")
    rc, out, _ = _capture(
        _main._handle_doctor_command, _doctor_args(output_json=True, workspace=str(tmp_path))
    )
    assert rc == 0
    assert json.loads(out)["code"]["ready"] is False


def test_doctor_code_report_reports_a_failure_instead_of_raising(monkeypatch):
    def _boom(_ws):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr("effgen.cli.code.readiness.code_readiness", _boom)
    report = _main._doctor_code_report(None)
    assert report["ready"] is False and "probe exploded" in report["error"]


def test_doctor_names_a_failed_coding_check_instead_of_an_empty_section(monkeypatch):
    """A section that could not be read says why, rather than printing nothing."""
    monkeypatch.setattr(
        _main, "_doctor_code_report",
        lambda ws: {"ready": False, "workspace": "", "checks": [], "error": "probe exploded"},
    )
    _, out, _ = _capture(_main._handle_doctor_command, _doctor_args())
    assert "could not run" in out and "probe exploded" in out


# --------------------------------------------------------------------------- #
# the guided coding step
# --------------------------------------------------------------------------- #

class _StubAgent:
    """An agent that writes a file (and optionally runs it) through the gated tools."""

    def __init__(
        self,
        tools,
        *,
        answer="It printed 55.",
        write=("fib.py", "print(55)\n"),
        run_code="print(55)",
    ):
        self._tools = {t.name: t for t in tools}
        self._answer = answer
        self._write = write
        self._run_code = run_code
        self.closed = False

    def run(self, task):
        import asyncio

        if self._write is not None:
            path, content = self._write
            asyncio.run(
                self._tools["file_operations"].execute(
                    operation="write", path=path, content=content
                )
            )
        if self._run_code is not None:
            asyncio.run(
                self._tools["code_executor"].execute(
                    code=self._run_code, language="python"
                )
            )
        return SimpleNamespace(
            output=self._answer, success=True, metadata={"cost_usd": 0.0},
            iterations=1, tool_calls=1, tokens_used=10, execution_time=0.1,
        )

    def close(self):
        self.closed = True


def _quickstart_args(**kw):
    ns = SimpleNamespace(
        model=None, provider=None, task=None, yes=True, code=False, no_code=False,
        quiet=False, verbose=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _plain_cli():
    cli = _main.CLIInterface()
    cli.console = None
    return cli


@pytest.fixture()
def stub_code_engine(monkeypatch):
    """Make the guided coding step build a stub agent instead of a real one."""
    built: list = []

    def _build(self):
        self._agent = _StubAgent(build_code_tools(self.gate))
        built.append(self)
        return self._agent

    monkeypatch.setattr("effgen.cli.code.engine.CodeEngine.build_agent", _build)
    monkeypatch.setattr(
        "effgen.cli.code.engine.workspace_execution_note", lambda ws: None
    )
    return built


def test_yes_alone_skips_the_coding_step_and_points_at_it(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    rc, out, _ = _capture(
        _main._quickstart_code_step,
        _quickstart_args(), _plain_cli(), "gpt-5-nano", "openai", interactive=False,
    )
    assert rc == 0
    assert "effgen code" in out
    assert not (tmp_path / "quickstart-code").exists()


def test_no_code_skips_the_step_even_when_interactive(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    rc, out, _ = _capture(
        _main._quickstart_code_step,
        _quickstart_args(no_code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=True,
    )
    assert rc == 0 and "Skipping the coding step" in out


def test_code_flag_runs_the_step_and_confines_the_write(
    monkeypatch, tmp_path, stub_code_engine
):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    rc, out, _ = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    assert rc == 0
    workspace = tmp_path / "quickstart-code"
    # The file the step wrote is inside the named directory, nowhere else.
    assert (workspace / "fib.py").read_text(encoding="utf-8") == "print(55)\n"
    assert "It printed 55." in out
    assert str(workspace) in out
    assert "effgen code --undo" in out


def test_the_step_announces_its_workspace_permissions_and_sandbox(
    monkeypatch, tmp_path, stub_code_engine
):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    _, out, _ = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    assert "Workspace:" in out and "Permissions:" in out and "Sandbox:" in out
    assert "the files it changes stay there" in out


def test_an_answer_prompt_decides_the_step_on_a_terminal(
    monkeypatch, tmp_path, stub_code_engine
):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    rc, out, _ = _capture(
        _main._quickstart_code_step,
        _quickstart_args(), _plain_cli(), "gpt-5-nano", "openai", interactive=True,
    )
    assert rc == 0 and "Skipping the coding step" in out

    monkeypatch.setattr("builtins.input", lambda *a: "")  # bare Enter accepts
    rc, out, _ = _capture(
        _main._quickstart_code_step,
        _quickstart_args(), _plain_cli(), "gpt-5-nano", "openai", interactive=True,
    )
    assert rc == 0 and (tmp_path / "quickstart-code" / "fib.py").exists()


def test_a_workspace_the_sandbox_cannot_read_is_named_before_the_run(
    monkeypatch, tmp_path, stub_code_engine
):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    monkeypatch.setattr(
        "effgen.cli.code.engine.workspace_execution_note",
        lambda ws: "the sandbox cannot see it",
    )
    _, out, err = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    text = out + err
    # The fix names EFFGEN_HOME, which is what decides this directory — not the
    # -w/--workspace flag the guided run does not take.
    assert "EFFGEN_HOME" in text and "-w/--workspace" not in text


def test_a_step_that_wrote_but_never_ran_says_so(monkeypatch, tmp_path):
    """A write-only turn must not let the model's answer stand in for a result."""
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))

    def _build(self):
        # The stub writes fib.py and answers, but never runs anything.
        self._agent = _StubAgent(
            build_code_tools(self.gate), answer="fib.py", run_code=None
        )
        return self._agent

    monkeypatch.setattr("effgen.cli.code.engine.CodeEngine.build_agent", _build)
    monkeypatch.setattr("effgen.cli.code.engine.workspace_execution_note", lambda ws: None)
    rc, out, err = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    text = out + err
    assert rc == 0
    assert "never ran it" in text and "nothing above was computed" in text


def test_a_failed_coding_step_fails_the_guided_run(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))

    def _build(self):
        raise RuntimeError("model unreachable")

    monkeypatch.setattr("effgen.cli.code.engine.CodeEngine.build_agent", _build)
    monkeypatch.setattr("effgen.cli.code.engine.workspace_execution_note", lambda ws: None)
    rc, out, err = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    assert rc == 1
    assert "model unreachable" in (out + err)


def _canned_result(**kw):
    """A CodeRunResult the step reports on, standing in for a real run."""
    from effgen.cli.code.engine import CodeRunResult

    fields = {
        "task": "t", "answer": "done", "success": True, "reason": "", "model": "m",
        "provider": None, "workspace": "/w", "permission_mode": "yes",
        "iterations": 1, "tool_calls": 1,
    }
    fields.update(kw)
    return CodeRunResult(**fields)


@pytest.fixture()
def canned_run(monkeypatch):
    """Drive the step with a prepared result so its reporting is what is tested."""
    def _install(result):
        monkeypatch.setattr(
            "effgen.cli.code.engine.CodeEngine.build_agent", lambda self: _StubAgent([])
        )
        monkeypatch.setattr(
            "effgen.cli.code.engine.CodeEngine.run", lambda self, task: result
        )
        monkeypatch.setattr(
            "effgen.cli.code.engine.workspace_execution_note", lambda ws: None
        )
    return _install


def test_the_iteration_cap_is_reported_with_a_way_forward(monkeypatch, tmp_path, canned_run):
    from effgen.cli.code.permissions import ActionRecord

    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    canned_run(_canned_result(
        reason="max_iterations_partial", iterations=12,
        actions=[ActionRecord(kind="run", summary="python", decision="allowed", outcome="ok")],
    ))
    rc, out, err = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    text = out + err
    assert rc == 0
    assert "all 12 steps" in text and "--max-iterations" in text


def test_withheld_actions_qualify_the_report(monkeypatch, tmp_path, canned_run):
    from effgen.cli.code.permissions import ActionRecord

    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    canned_run(_canned_result(actions=[
        ActionRecord(kind="run", summary="python", decision="allowed", outcome="ok"),
        ActionRecord(kind="shell", summary="pytest", decision="withheld", reason="no terminal"),
    ]))
    _, out, err = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    text = out + err
    assert "1 action(s) (shell) were not carried out" in text


def test_a_refused_action_is_named(monkeypatch, tmp_path, canned_run):
    from effgen.cli.code.permissions import ActionRecord

    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    canned_run(_canned_result(actions=[
        ActionRecord(kind="run", summary="python", decision="allowed", outcome="ok"),
        ActionRecord(
            kind="write", summary="write /etc/passwd", decision="refused",
            reason="/etc/passwd is outside the workspace",
        ),
    ]))
    _, out, err = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    assert "outside the workspace" in (out + err)


def test_an_unsuccessful_run_fails_the_step_with_a_larger_model_hint(
    monkeypatch, tmp_path, canned_run
):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    canned_run(_canned_result(success=False, answer="I gave up."))
    rc, out, err = _capture(
        _main._quickstart_code_step,
        _quickstart_args(code=True), _plain_cli(), "gpt-5-nano", "openai",
        interactive=False,
    )
    text = out + err
    assert rc == 1
    assert "did not complete" in text and "larger model" in text
    assert "Docs: docs/cli/code.md" in text


class _AnsweringAgent:
    """Stands in for the guided run's sample-task agent, with no model behind it.

    Patched over ``_main.Agent`` — the name the guided run actually calls — so
    the test needs no provider key and makes no request.
    """

    def __init__(self, config):
        self.config = config
        self.execution_tracker = None

    def run(self, task):
        return SimpleNamespace(
            output="425", success=True, metadata={"cost_usd": 0.0}, execution_trace=[],
            iterations=1, tool_calls=0, tokens_used=10, execution_time=0.1,
        )

    def close(self):
        pass


@pytest.mark.parametrize(
    ("code_rc", "expected", "absent"),
    [(0, "You're set!", "no-such-text"), (1, "Next steps", "You're set!")],
)
def test_the_guided_run_only_claims_success_when_the_coding_step_worked(
    monkeypatch, tmp_path, code_rc, expected, absent
):
    built: list = []
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    monkeypatch.setattr(
        _main, "Agent", lambda config: built.append(config) or _AnsweringAgent(config)
    )
    monkeypatch.setattr(_main, "_quickstart_code_step", lambda *a, **k: code_rc)
    rc, out, err = _capture(
        _main._handle_quickstart_command,
        _quickstart_args(model="gpt-5-nano", provider="openai", task="What is 25 * 17?"),
        _plain_cli(),
    )
    text = out + err
    # The guided run built the stand-in, so no key was needed and no request made.
    assert len(built) == 1
    assert rc == code_rc
    assert expected in text and absent not in text


def test_code_and_no_code_together_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    rc, out, err = _capture(
        _main._handle_quickstart_command,
        _quickstart_args(code=True, no_code=True), _plain_cli(),
    )
    assert rc == 1
    assert "cannot be combined" in (out + err)


def test_quickstart_and_tutorial_both_accept_the_coding_flags():
    parser = _main.create_parser()
    for command in ("quickstart", "tutorial"):
        ns = parser.parse_args([command, "--code"])
        assert ns.code is True and ns.no_code is False
        ns = parser.parse_args([command, "--no-code"])
        assert ns.no_code is True


# --------------------------------------------------------------------------- #
# landing + welcome + tips
# --------------------------------------------------------------------------- #

def test_every_doc_pointer_the_cli_prints_names_a_shipped_page():
    """A teaching error is only teaching if the page it names exists."""
    import re

    repo_root = Path(_main.__file__).resolve().parents[2]
    cli_dir = repo_root / "effgen" / "cli"
    missing = []
    for source in cli_dir.rglob("*.py"):
        for ref in re.findall(r'"(docs/[\w./-]+\.md)"', source.read_text(encoding="utf-8")):
            if not (repo_root / ref).is_file():
                missing.append(f"{source.relative_to(repo_root)} -> {ref}")
    assert not missing, "doc pointers with no page: " + ", ".join(missing)


def test_the_landing_lists_code_as_an_action():
    targets = [target for _key, _desc, target in landing._ACTIONS]
    assert "code" in targets
    descriptions = [desc for _key, desc, target in landing._ACTIONS if target == "code"]
    assert descriptions and descriptions[0].startswith("code —")


def test_every_landing_key_is_unique_and_resolves():
    keys = [key for key, _desc, _target in landing._ACTIONS]
    assert len(keys) == len(set(keys))
    for key, _desc, target in landing._ACTIONS:
        if key == "Enter":
            continue
        assert landing._CHOICES[key] == target


def test_the_code_choice_parses_into_the_code_command():
    parser = _main.create_parser()
    ns = parser.parse_args([landing._CHOICES["e"]])
    assert ns.command == "code"


def test_the_first_run_welcome_points_at_the_coding_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
    monkeypatch.setattr(ob, "is_interactive", lambda: True)
    buf = io.StringIO()
    assert ob.maybe_show_first_run_welcome(quiet=False, stream=buf) is True
    text = buf.getvalue()
    assert "effgen doctor" in text and "effgen code" in text


def test_a_tip_covers_the_coding_agent():
    assert any("effgen code" in tip for tip in ob.TIPS)


# --------------------------------------------------------------------------- #
# the session's own /doctor
# --------------------------------------------------------------------------- #

def test_session_doctor_checks_the_session_workspace(tmp_path, monkeypatch):
    from effgen.cli.code.repl import CodeREPL

    ws = tmp_path / "ws"
    ws.mkdir()
    seen: list = []
    real = _main._doctor_code_report
    monkeypatch.setattr(
        _main, "_doctor_code_report", lambda w: seen.append(w) or real(w)
    )
    args = SimpleNamespace(
        workspace=str(ws), model="fake:model", provider=None,
        plan_only=False, auto_edit=False, assume_yes=False,
        temperature=None, max_tokens=None, max_iterations=None,
        quiet=False, verbose=False,
    )
    cli = _main.CLIInterface()
    cli.console = None
    repl = CodeREPL(cli, args)

    _, out, err = _capture(repl._cmd_doctor)
    text = out + err
    # The coding checks describe the session's workspace, not the process's cwd.
    assert seen == [str(ws)]
    assert "Coding" in text
    for name in ("workspace", "sandbox", "git"):
        assert name in text
    # The provider table it always showed is still there.
    assert "Provider" in text or "provider" in text
