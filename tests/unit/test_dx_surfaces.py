"""DX-surface regression tests: shell completion, debug CLI, VSCode metadata.

These guard the developer-experience surfaces (Jupyter magics, shell completion,
`effgen debug`, the VSCode extension) against silent drift — stale model ids,
hand-listed subcommands that fall out of sync, and exit codes that always say 0.

The deeper live behaviour (magics in a real kernel, debug/compare against real
models) is exercised by separate evidence scripts; these are
the fast, offline guards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
VSCODE_DIR = REPO_ROOT / "tools" / "vscode-effgen"

# Model ids that providers have retired — must never reappear in shipped DX
# surfaces (see the model-registry phases; Cerebras live = gpt-oss-120b/zai-glm).
STALE_MODEL_IDS = ("cerebras:llama3.1-8b", "cerebras:qwen-3-235b-a22b")


# ---------------------------------------------------------------------------
# Shell completion — generated, not hand-listed
# ---------------------------------------------------------------------------


class TestCompletionGenerated:
    def test_real_subcommands_present(self):
        """The completion must list the *current* subcommands, including ones a
        hand-maintained list had drifted away from (debug, compare, doctor)."""
        from effgen.completion import get_completion

        script = get_completion("bash")
        for cmd in ("run", "chat", "doctor", "compare", "debug", "cost", "eval"):
            assert cmd in script, f"completion missing subcommand: {cmd}"

    def test_subcommands_match_parser(self):
        """Completion subcommands are derived from the live parser, not static."""
        import argparse

        from effgen.cli._main import create_parser
        from effgen.completion import get_completion

        parser = create_parser()
        real = set()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                real = set(action.choices.keys())
                break

        script = get_completion("bash")
        # Every real subcommand appears in the generated completion.
        for cmd in real:
            assert cmd in script, f"parser command {cmd!r} missing from completion"

    def test_presets_generated_from_registry(self):
        from effgen.completion import get_completion
        from effgen.presets import list_presets

        script = get_completion("bash")
        for preset in list_presets():
            assert preset in script, f"completion missing preset: {preset}"

    def test_no_stale_model_ids(self):
        from effgen.completion import get_completion

        for shell in ("bash", "zsh", "fish"):
            script = get_completion(shell)
            for stale in STALE_MODEL_IDS:
                assert stale not in script

    def test_unsupported_shell_raises(self):
        from effgen.completion import get_completion

        with pytest.raises(ValueError):
            get_completion("powershell")

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
    def test_bash_script_parses(self, tmp_path):
        from effgen.completion import get_completion

        path = tmp_path / "comp.bash"
        path.write_text(get_completion("bash"))
        result = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh not available")
    def test_zsh_script_parses(self, tmp_path):
        from effgen.completion import get_completion

        path = tmp_path / "comp.zsh"
        path.write_text(get_completion("zsh"))
        result = subprocess.run(
            ["zsh", "-n", str(path)], capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(shutil.which("fish") is None, reason="fish not available")
    def test_fish_script_parses(self, tmp_path):
        from effgen.completion import get_completion

        path = tmp_path / "comp.fish"
        path.write_text(get_completion("fish"))
        result = subprocess.run(
            ["fish", "--no-execute", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# effgen debug — accurate exit codes
# ---------------------------------------------------------------------------


class TestDebugExitCodes:
    def test_no_model_or_preset_returns_config_error(self):
        """With neither --model nor --preset, debug exits non-zero (config error)
        and never tries to run a model."""
        from effgen.debug.inspector import run_debug_cli

        code = run_debug_cli(task="what is 2+2?", model=None, preset=None)
        assert code == 2

    def test_run_debug_cli_returns_int(self):
        from effgen.debug.inspector import run_debug_cli

        code = run_debug_cli(task="x", model=None, preset=None)
        assert isinstance(code, int)

    def test_parser_accepts_provider_flag(self):
        """`effgen debug -m <id> --provider <name>` parses — used to
        argparse-error 2 with 'unrecognized arguments: --provider'."""
        from effgen.cli._main import create_parser

        args = create_parser().parse_args(
            ["debug", "do x", "-m", "openai/gpt-oss-20b", "--provider", "groq"]
        )
        assert args.command == "debug"
        assert args.provider == "groq"

    def test_build_config_threads_provider(self):
        """_build_debug_config wires --provider into the AgentConfig so a bare
        id resolves to the chosen provider (like `run`/`chat`)."""
        from effgen.debug.inspector import _build_debug_config

        cfg = _build_debug_config(
            preset=None, model="openai/gpt-oss-20b", provider="groq"
        )
        assert cfg is not None
        assert cfg.provider == "groq"


# ---------------------------------------------------------------------------
# Jupyter magics — registry-driven default (no hardcoded paid model)
# ---------------------------------------------------------------------------


class TestMagicsDefaultModel:
    def test_no_hardcoded_default(self, monkeypatch):
        # Importing the magics module requires IPython; skip cleanly where the
        # unit lane runs without it (same pattern as the matplotlib/faiss tests).
        pytest.importorskip("IPython")
        monkeypatch.delenv("EFFGEN_JUPYTER_MODEL", raising=False)
        monkeypatch.delenv("EFFGEN_DEFAULT_MODEL", raising=False)
        from effgen.jupyter.magics import _default_model

        assert _default_model() is None

    def test_magics_source_has_no_stale_default(self):
        src = (REPO_ROOT / "effgen" / "jupyter" / "magics.py").read_text()
        for stale in STALE_MODEL_IDS:
            assert stale not in src


# ---------------------------------------------------------------------------
# VSCode extension — no stale ids, marked experimental
# ---------------------------------------------------------------------------


class TestVscodeMetadata:
    def test_no_stale_model_id_in_source(self):
        src = (VSCODE_DIR / "src" / "extension.ts").read_text()
        pkg = (VSCODE_DIR / "package.json").read_text()
        for stale in STALE_MODEL_IDS:
            assert stale not in src
            assert stale not in pkg

    def test_default_model_is_current(self):
        pkg = json.loads((VSCODE_DIR / "package.json").read_text())
        default = (
            pkg["contributes"]["configuration"]["properties"]["effgen.defaultModel"][
                "default"
            ]
        )
        assert default not in STALE_MODEL_IDS
        assert default  # non-empty

    def test_readme_marks_experimental(self):
        readme = VSCODE_DIR / "README.md"
        assert readme.exists()
        assert "experimental" in readme.read_text().lower()
