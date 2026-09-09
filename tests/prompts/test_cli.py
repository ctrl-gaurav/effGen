"""Tests for 'effgen prompts' CLI subcommand."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "effgen.cli"] + list(args),
        capture_output=True,
        text=True,
    )


class TestPromptsListCLI:
    def test_prompts_list_exits_zero(self):
        result = run_cli("prompts", "list")
        assert result.returncode == 0

    def test_prompts_list_empty_registry(self):
        result = run_cli("prompts", "list")
        assert result.returncode == 0
        # Should mention 'Total' even if empty
        assert "Total" in result.stdout or "No prompts" in result.stdout

    def test_prompts_list_json_format(self):
        result = run_cli("prompts", "list", "--format", "json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)

    def test_prompts_list_markdown_format(self):
        result = run_cli("prompts", "list", "--format", "markdown")
        assert result.returncode == 0
        assert "| Name |" in result.stdout

    def test_prompts_list_domain_filter(self):
        result = run_cli("prompts", "list", "--domain", "research")
        assert result.returncode == 0

    def test_prompts_list_variant_filter(self):
        result = run_cli("prompts", "list", "--variant", "cot")
        assert result.returncode == 0


class TestPromptsShowCLI:
    def test_prompts_show_missing_exits_nonzero(self):
        result = run_cli("prompts", "show", "does.not.exist.v1")
        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()

    def test_prompts_show_missing_suggests_close_match(self):
        # A near-miss typo of a real registered name gets a "did you mean".
        result = run_cli("prompts", "show", "business.elevater_pitch.v1")
        assert result.returncode != 0
        combined = (result.stdout + result.stderr).lower()
        assert "did you mean" in combined
        assert "business.elevator_pitch.v1" in combined

    def test_prompts_show_base_name_resolves_default_variant(self):
        # Omitting the trailing ".v1" (the id truncated in the table view)
        # still resolves when the base name is unambiguous.
        result = run_cli("prompts", "show", "business.elevator_pitch")
        assert result.returncode == 0
        assert "business.elevator_pitch.v1" in result.stdout

    def test_prompts_list_narrow_terminal_does_not_ellipsis_truncate_names(self):
        # At a narrow terminal a long, space-free name (e.g.
        # "business.elevator_pitch.v1") would previously be cut mid-word
        # with an ellipsis ("business.elevator_pitch…"), hiding the ".v1"
        # a user has to type back into `show`/`run`/`render`. It must now
        # wrap onto an extra line instead.
        env = {**os.environ, "COLUMNS": "80"}
        result = subprocess.run(
            [sys.executable, "-m", "effgen.cli", "prompts", "list"],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0
        assert "pitch…" not in result.stdout
        assert "summar…" not in result.stdout

    def test_prompts_table_name_column_uses_fold_overflow(self):
        # Regression guard tied to the actual fix: the Name column must use
        # "fold" (wraps a space-free id onto extra lines) rather than the
        # default ellipsis truncation.
        import inspect

        from effgen.cli.commands import prompts as prompts_command

        src = inspect.getsource(prompts_command)
        assert 'add_column("Name", style="effgen.model", overflow="fold")' in src


class TestPromptsEvalCLI:
    def test_eval_empty_registry_exits_zero(self, tmp_path):
        result = run_cli("prompts", "eval")
        assert result.returncode == 0
        assert "empty" in result.stdout.lower() or "Total" in result.stdout

    def test_eval_with_output_file(self, tmp_path):
        out = str(tmp_path / "eval.txt")
        result = run_cli("prompts", "eval", "--output", out)
        assert result.returncode == 0
        assert Path(out).exists()
        content = Path(out).read_text()
        assert "Golden Eval" in content

    def test_eval_live_requires_model(self):
        result = run_cli("prompts", "eval", "--live")
        assert result.returncode != 0
        assert "--model" in result.stdout or "--model" in result.stderr


class TestPromptsEvalExitCode:
    def test_all_pass_exits_zero(self):
        result = run_cli("prompts", "eval", "--domain", "business")
        assert result.returncode == 0

    def test_fail_under_met_exits_zero(self):
        result = run_cli("prompts", "eval", "--domain", "business", "--fail-under", "1.0")
        assert result.returncode == 0

    def test_fail_under_unreachable_exits_nonzero(self):
        # An all-pass golden run has a 100% rate; a 110% threshold cannot be met.
        result = run_cli("prompts", "eval", "--domain", "business", "--fail-under", "1.1")
        assert result.returncode == 1
        assert "below" in (result.stdout + result.stderr).lower()

    def test_golden_failure_exits_nonzero(self, tmp_path):
        # A user template whose stored golden no longer matches its render must
        # make the eval exit non-zero so it can gate CI.
        from effgen.prompts.library.eval import PromptEval

        (tmp_path / "zz.py").write_text(
            'from effgen.prompts.library import LibraryPrompt\n'
            'PROMPTS = [LibraryPrompt(name="zzeval.stale.v1", domain="zzeval",\n'
            '    variant="zero_shot", description="d",\n'
            '    template=lambda **_: "FRESH RENDER",\n'
            '    input_schema={"type": "object", "properties": {}},\n'
            '    fixture={}, expected_shape=None, tags=[])]\n'
        )
        golden = PromptEval().goldens_dir / "zzeval.stale.v1.txt"
        golden.write_text("STALE CONTENT")
        try:
            env = dict(os.environ, EFFGEN_PROMPTS_DIR=str(tmp_path))
            result = subprocess.run(
                [sys.executable, "-m", "effgen.cli", "prompts", "eval", "--domain", "zzeval"],
                capture_output=True, text=True, env=env,
            )
            assert result.returncode == 1
            assert "fail" in result.stdout.lower()
        finally:
            golden.unlink(missing_ok=True)


class TestPromptsListJsonEncoding:
    def test_list_json_emits_raw_unicode(self):
        result = run_cli("prompts", "list", "--format", "json")
        assert result.returncode == 0
        # Raw UTF-8, not \\u-escaped, consistent with other --json surfaces.
        assert "\\u2264" not in result.stdout


class TestPromptsRunFlags:
    def test_run_help_lists_max_tokens_and_temperature(self):
        result = run_cli("prompts", "run", "--help")
        assert result.returncode == 0
        assert "--max-tokens" in result.stdout
        assert "--temperature" in result.stdout

    def test_run_unknown_key_exits_nonzero(self, tmp_path):
        inp = tmp_path / "bad.json"
        inp.write_text(json.dumps({
            "product_name": "X", "target_audience": "devs",
            "problem": "a problem here", "solution": "a solution here",
            "differentiator": "a differentiator here", "TYPO_KEY": "oops",
        }))
        result = run_cli(
            "prompts", "run", "business.elevator_pitch.v1",
            "--input", str(inp), "-m", "groq:openai/gpt-oss-20b",
        )
        assert result.returncode != 0
        assert "unknown input key" in (result.stdout + result.stderr)
        # No private render function name leaks.
        assert "_elevator_pitch" not in (result.stdout + result.stderr)
