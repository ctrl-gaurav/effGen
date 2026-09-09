"""
Tests for the prompt playground — both unit tests and pexpect REPL walk-throughs.

pexpect tests drive the REPL via a subprocess so they exercise the full CLI
stack including arg parsing, REPL dispatch, render, session save/load, and
exit.

Non-pexpect tests cover the session serialization and the cmd_render /
cmd_run helpers directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _effgen_bin() -> str:
    """Return the effgen executable in the currently active Python env."""
    scripts = Path(sys.executable).parent
    for name in ("effgen", "effgen.exe"):
        candidate = scripts / name
        if candidate.exists():
            return str(candidate)
    return "effgen"


EFFGEN = _effgen_bin()


# -------------------------------------------------------------------------
# Session unit tests
# -------------------------------------------------------------------------

class TestPlaygroundSession:
    def test_set_unset(self):
        from effgen.prompts.library.session import PlaygroundSession

        s = PlaygroundSession(prompt_name="x.v1")
        s.set_var("topic", "diffusion models")
        assert s.variables["topic"] == "diffusion models"
        removed = s.unset_var("topic")
        assert removed
        assert "topic" not in s.variables
        assert not s.unset_var("topic")  # idempotent

    def test_add_render_run(self):
        from effgen.prompts.library.session import PlaygroundSession

        s = PlaygroundSession(prompt_name="x.v1")
        s.add_render("Hello, world")
        assert len(s.render_history) == 1
        s.add_run("gpt-4", "Hello, world", "Hi there!")
        assert len(s.run_history) == 1
        assert s.run_history[0].model == "gpt-4"

    def test_save_load_roundtrip(self, tmp_path):
        from effgen.prompts.library.session import PlaygroundSession

        s = PlaygroundSession(prompt_name="research.v1", variables={"topic": "AI"})
        s.add_render("Sample rendered text")
        s.add_run("llama3", "Sample rendered text", "Model response")
        path = tmp_path / "session.json"
        saved = s.save(path)
        assert saved == path
        s2 = PlaygroundSession.load(path)
        assert s2.prompt_name == "research.v1"
        assert s2.variables["topic"] == "AI"
        assert len(s2.render_history) == 1
        assert len(s2.run_history) == 1
        assert s2.run_history[0].model == "llama3"

    def test_auto_save_path(self, tmp_path, monkeypatch):
        from effgen.prompts.library import session as sess_mod
        from effgen.prompts.library.session import PlaygroundSession

        monkeypatch.setattr(sess_mod, "_SESSIONS_DIR", tmp_path)
        s = PlaygroundSession(prompt_name="business.elevator_pitch.v1")
        path = s.save()
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["prompt_name"] == "business.elevator_pitch.v1"

    def test_summary(self):
        from effgen.prompts.library.session import PlaygroundSession

        s = PlaygroundSession(prompt_name="research.v1", variables={"a": 1, "b": 2})
        s.add_render("x")
        summary = s.summary()
        assert "research.v1" in summary
        assert "vars=2" in summary
        assert "renders=1" in summary


# -------------------------------------------------------------------------
# cmd_render / cmd_run unit tests
# -------------------------------------------------------------------------

class TestNonInteractive:
    def test_cmd_render_known_prompt(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render("research.literature_review.v1.zero_shot", {})
        assert rc == 0
        captured = capsys.readouterr()
        assert "literature review" in captured.out.lower() or captured.out  # some output

    def test_cmd_render_unknown_prompt(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render("no.such.prompt.v99", {})
        assert rc == 1

    def test_cmd_render_typo_suggests_close_match(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render("research.literature_reviw.v1.zero_shot", {})
        assert rc == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "Did you mean" in combined
        assert "research.literature_review.v1.zero_shot" in combined

    def test_cmd_render_unrelated_name_gets_no_suggestion(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render("no.such.prompt.v99", {})
        assert rc == 1
        captured = capsys.readouterr()
        assert "Did you mean" not in (captured.out + captured.err)

    def test_cmd_render_with_inputs(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render(
            "research.literature_review.v1.zero_shot",
            {"topic": "quantum computing", "years_range": "2022-2024", "max_papers": 3},
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "quantum computing" in captured.out

    def test_cmd_render_base_name_resolves_default_variant(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render(
            "research.literature_review.v1",
            {"topic": "small language models", "years_range": "2023-2026", "max_papers": 3},
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "small" in captured.out
        assert "language" in captured.out
        assert "models" in captured.out

    def test_cmd_render_business_prompt(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render("business.elevator_pitch.v1", {})
        assert rc == 0


# -------------------------------------------------------------------------
# --input validation against input_schema
# -------------------------------------------------------------------------

class TestInputValidation:
    def test_array_field_passed_as_string_is_rejected(self, capsys):
        """A list-valued field passed as a string must fail closed, not
        silently render one item per character."""
        from effgen.cli.playground import cmd_render

        rc = cmd_render(
            "business.email_draft.v1",
            {
                "purpose": "announce a maintenance window",
                "recipient": "all staff",
                "key_points": "point one; point two; point three",
                "tone": "formal",
            },
        )
        assert rc == 1
        captured = capsys.readouterr()
        combined = " ".join((captured.out + captured.err).split())
        assert "key_points" in combined
        assert "not of type 'array'" in combined

    def test_missing_required_field_is_rejected(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render(
            "business.email_draft.v1",
            {
                "purpose": "announce a maintenance window",
                "recipient": "all staff",
                "key_points": ["point one", "point two"],
                # "tone" omitted
            },
        )
        assert rc == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "tone" in combined
        assert "required" in combined

    def test_out_of_enum_value_is_rejected(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render(
            "business.email_draft.v1",
            {
                "purpose": "announce a maintenance window",
                "recipient": "all staff",
                "key_points": ["point one", "point two"],
                "tone": "urgent",
            },
        )
        assert rc == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "tone" in combined

    def test_correct_array_input_renders(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render(
            "business.email_draft.v1",
            {
                "purpose": "announce a maintenance window",
                "recipient": "all staff",
                "key_points": ["point one", "point two"],
                "tone": "formal",
            },
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "point one" in captured.out
        assert "point two" in captured.out

    def test_no_input_still_renders_fixture(self, capsys):
        """Omitting --input entirely (empty dict) is not validated — it
        renders the prompt's own fixture demo."""
        from effgen.cli.playground import cmd_render

        rc = cmd_render("business.email_draft.v1", {})
        assert rc == 0

    def test_cli_rejects_char_split_array_input(self, tmp_path):
        import subprocess

        bad_input = tmp_path / "bad.json"
        bad_input.write_text(json.dumps({
            "purpose": "announce a maintenance window",
            "recipient": "all staff",
            "key_points": "point one; point two",
            "tone": "formal",
        }))
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "business.email_draft.v1",
             "--input", str(bad_input)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "key_points" in result.stdout or "key_points" in result.stderr


# -------------------------------------------------------------------------
# CLI non-interactive tests (subprocess)
# -------------------------------------------------------------------------

class TestCLINonInteractive:
    def test_prompts_list(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "list"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "research" in result.stdout.lower() or "business" in result.stdout.lower()

    def test_prompts_render_stdout(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "research.literature_review.v1.zero_shot"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_prompts_render_with_input_file(self, tmp_path):
        import subprocess
        inp = tmp_path / "input.json"
        inp.write_text(json.dumps({
            "topic": "neural radiance fields",
            "years_range": "2021-2024",
            "max_papers": 5,
        }))
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "research.literature_review.v1.zero_shot",
             "--input", str(inp)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        # Rich may wrap "neural radiance fields" across panel border lines;
        # strip all non-alphanumeric separators to do a substring check.
        import re as _re
        cleaned = _re.sub(r"[│╭╰╮╯─\s]+", " ", result.stdout)
        assert "neural" in cleaned and "radiance fields" in cleaned

    def test_prompts_render_unknown(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "no.such.prompt.v99"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0

    def test_prompts_render_coding(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "coding.docstring_fill.v1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0

    def test_prompts_render_legal(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "legal.clause_classify.v1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_prompts_render_medical(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "medical.symptom_triage.v1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_prompts_render_creative(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "creative.story_continuation.v1.zero_shot"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0

    def test_prompts_render_data(self):
        import subprocess
        result = subprocess.run(
            [EFFGEN, "prompts", "render", "data.sql_explain.v1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0


# -------------------------------------------------------------------------
# pexpect REPL tests
# -------------------------------------------------------------------------

pexpect = pytest.importorskip("pexpect", reason="pexpect not installed")


class TestPlaygroundREPL:
    """Scripted REPL walk-through via pexpect."""

    TIMEOUT = 30

    def _spawn(self):
        child = pexpect.spawn(
            EFFGEN,
            ["prompts", "playground"],
            encoding="utf-8",
            timeout=self.TIMEOUT,
        )
        child.expect(r"effGen Prompt Playground", timeout=self.TIMEOUT)
        return child

    def test_help_command(self):
        child = self._spawn()
        child.sendline("help")
        child.expect("select", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_select_and_render(self):
        child = self._spawn()
        child.sendline("select research.literature_review.v1.zero_shot")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline("render")
        child.expect("literature review", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_select_base_name_resolves_default_variant(self):
        child = self._spawn()
        child.sendline("select research.literature_review.v1")
        child.expect("Selected: research.literature_review.v1.zero_shot", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_set_then_render(self):
        child = self._spawn()
        child.sendline("select research.literature_review.v1.zero_shot")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline('set topic "protein folding"')
        child.expect("Set topic", timeout=self.TIMEOUT)
        child.sendline("render")
        child.expect("protein folding", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_unset(self):
        child = self._spawn()
        child.sendline("select business.elevator_pitch.v1")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline("set product_name TestProduct")
        child.expect("Set product_name", timeout=self.TIMEOUT)
        child.sendline("unset product_name")
        child.expect("Unset product_name", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_select_unknown_prompt(self):
        child = self._spawn()
        child.sendline("select no.such.prompt.v99")
        child.expect("not found", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_render_without_select(self):
        child = self._spawn()
        child.sendline("render")
        child.expect("No prompt selected", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_save_and_load_session(self, tmp_path):
        session_file = str(tmp_path / "session.json")
        child = self._spawn()
        child.sendline("select research.literature_review.v1.zero_shot")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline('set topic "deep learning"')
        child.expect("Set topic", timeout=self.TIMEOUT)
        child.sendline(f"save {session_file}")
        child.expect("Session saved", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

        # Verify the file was written
        assert Path(session_file).exists()
        data = json.loads(Path(session_file).read_text())
        assert data["prompt_name"] == "research.literature_review.v1.zero_shot"
        assert data["variables"]["topic"] == "deep learning"

        # Load in a new session
        child2 = self._spawn()
        child2.sendline(f"load {session_file}")
        child2.expect("Session loaded", timeout=self.TIMEOUT)
        child2.sendline("exit")
        child2.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_list_command_in_repl(self):
        child = self._spawn()
        child.sendline("list")
        child.expect("research|business|coding", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_list_with_domain_filter(self):
        child = self._spawn()
        child.sendline("list --domain coding")
        child.expect("coding", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_show_prompt(self):
        child = self._spawn()
        child.sendline("show coding.code_review.v1")
        child.expect("Domain", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_unknown_command(self):
        child = self._spawn()
        child.sendline("foobar")
        child.expect("Unknown command", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_quit_alias(self):
        child = self._spawn()
        child.sendline("quit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_reload_without_select(self):
        child = self._spawn()
        child.sendline("reload")
        child.expect("No prompt selected", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_reload_with_selected_prompt(self):
        child = self._spawn()
        child.sendline("select coding.docstring_fill.v1")
        child.expect("Selected", timeout=self.TIMEOUT)
        child.sendline("reload")
        # Either reloaded or warned, but no crash
        child.expect(r"Reloaded|No matching|Warning", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)

    def test_full_round_trip_multiple_domains(self):
        """REPL round-trip on prompts from research, coding, business, legal, data."""
        child = self._spawn()
        for name in [
            "research.literature_review.v1.zero_shot",
            "coding.docstring_fill.v1",
            "business.elevator_pitch.v1",
            "legal.clause_classify.v1",
            "data.sql_explain.v1",
            "medical.symptom_triage.v1",
            "creative.story_continuation.v1.zero_shot",
        ]:
            child.sendline(f"select {name}")
            child.expect("Selected", timeout=self.TIMEOUT)
            child.sendline("render")
            child.expect(r"Rendered Prompt|---", timeout=self.TIMEOUT)
        child.sendline("exit")
        child.expect(pexpect.EOF, timeout=self.TIMEOUT)


# -------------------------------------------------------------------------
# Unknown-input-key handling
# -------------------------------------------------------------------------

class TestUnknownInputKeys:
    def test_extra_key_reports_clean_error_not_typeerror(self, capsys):
        from effgen.cli.playground import cmd_render

        rc = cmd_render(
            "business.elevator_pitch.v1",
            {
                "product_name": "X",
                "target_audience": "devs",
                "problem": "a problem statement here",
                "solution": "a solution statement here",
                "differentiator": "a differentiator here",
                "EXTRA_TYPO_KEY": "oops",
            },
        )
        assert rc == 1
        out = capsys.readouterr().out
        assert "unknown input key 'EXTRA_TYPO_KEY'" in out
        # The private render function name must never leak.
        assert "_elevator_pitch" not in out
        assert "unexpected keyword argument" not in out

    def test_valid_keys_are_listed(self, capsys):
        from effgen.cli.playground import cmd_render

        cmd_render("business.elevator_pitch.v1", {"nope": 1})
        out = capsys.readouterr().out
        assert "valid keys:" in out
        assert "product_name" in out

    def test_kwargs_template_allows_extra_keys(self):
        from effgen.cli.playground import _unknown_input_keys
        from effgen.prompts.library.base import LibraryPrompt

        def _render(topic="a", **_):
            return f"About {topic}"

        p = LibraryPrompt(
            name="t.kw.v1",
            domain="t",
            variant="zero_shot",
            description="d",
            template=_render,
            input_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
            fixture={"topic": "a"},
            expected_shape=None,
            tags=[],
        )
        # The render signature accepts **kwargs, so no key is "unknown".
        assert _unknown_input_keys(p, {"topic": "a", "extra": "b"}) == []


# -------------------------------------------------------------------------
# Empty / truncated result handling (fail-closed) + footer + verdict
# -------------------------------------------------------------------------

class TestRunFailClosed:
    def _stub(self, monkeypatch, result):
        import effgen.cli.playground as pg

        def fake_run(rendered, model, *, max_tokens=None, temperature=None):
            return result

        monkeypatch.setattr(pg, "_run_prompt", fake_run)

    def test_truncated_result_exits_nonzero_with_marker(self, monkeypatch, capsys):
        from effgen.cli.playground import cmd_run
        from effgen.prompts.library.eval import RunOutput

        self._stub(
            monkeypatch,
            RunOutput(text="", finish_reason="length", truncated=True, max_tokens=16),
        )
        rc = cmd_run("business.elevator_pitch.v1", {}, "openai:gpt-5-nano")
        assert rc == 1
        out = capsys.readouterr().out
        assert "no usable output" in out
        assert "--max-tokens" in out

    def test_empty_text_exits_nonzero(self, monkeypatch, capsys):
        from effgen.cli.playground import cmd_run
        from effgen.prompts.library.eval import RunOutput

        self._stub(monkeypatch, RunOutput(text="   ", finish_reason="stop"))
        rc = cmd_run("business.elevator_pitch.v1", {}, "groq:openai/gpt-oss-20b")
        assert rc == 1

    def test_nonempty_result_prints_footer(self, monkeypatch, capsys):
        from effgen.cli.playground import cmd_run
        from effgen.prompts.library.eval import RunOutput

        self._stub(
            monkeypatch,
            RunOutput(
                text="a real answer",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=5,
                cost_usd=0.00012,
                latency_ms=321.0,
            ),
        )
        rc = cmd_run("business.elevator_pitch.v1", {}, "groq:openai/gpt-oss-20b")
        assert rc == 0
        out = capsys.readouterr().out
        assert "a real answer" in out
        assert "tokens: 15 (10 in / 5 out)" in out
        assert "cost: $0.000120" in out

    def test_truncated_nonempty_result_exits_nonzero(self, monkeypatch, capsys):
        from effgen.cli.playground import cmd_run
        from effgen.prompts.library.eval import RunOutput

        self._stub(
            monkeypatch,
            RunOutput(
                text='{"issues": [{"severity": "hi',
                finish_reason="length",
                truncated=True,
                max_tokens=16,
            ),
        )
        rc = cmd_run("coding.code_review.v1", {}, "groq:openai/gpt-oss-20b")
        assert rc == 1
        out = capsys.readouterr().out
        # The partial answer is still shown, alongside a named reason.
        assert '"issues"' in out
        assert "Output is incomplete" in out
        assert "--max-tokens (currently 16)" in out

    def test_structured_verdict_reported(self, monkeypatch, capsys):
        from effgen.cli.playground import cmd_run
        from effgen.prompts.library.eval import RunOutput

        # coding.code_review.v1 declares a json expected_shape.
        self._stub(
            monkeypatch,
            RunOutput(text='{"issues": []}', finish_reason="stop"),
        )
        rc = cmd_run("coding.code_review.v1", {}, "groq:openai/gpt-oss-20b")
        assert rc == 0
        out = capsys.readouterr().out
        assert "expected_shape" in out


class TestBracketedTextIsPrintedVerbatim:
    """Model output and error details keep their square brackets on a rich console."""

    def test_output_with_closing_tag_does_not_abort(self, capsys):
        from effgen.cli.playground import _print_output

        _print_output("See [ref] and [/close] markers.", "groq:openai/gpt-oss-20b")
        out = capsys.readouterr().out
        assert "[ref]" in out
        assert "[/close]" in out

    def test_empty_result_marker_is_visible(self, monkeypatch, capsys):
        from effgen.cli.playground import cmd_run
        from effgen.prompts.library.eval import RunOutput

        def fake_run(rendered, model, *, max_tokens=None, temperature=None):
            return RunOutput(text="", finish_reason="length", truncated=True, max_tokens=16)

        monkeypatch.setattr("effgen.cli.playground._run_prompt", fake_run)
        assert cmd_run("business.elevator_pitch.v1", {}, "openai:gpt-5-nano") == 1
        assert "[empty result]" in capsys.readouterr().out

    def test_rendered_prompt_with_brackets_survives(self, capsys):
        from effgen.cli.playground import _print_prompt

        _print_prompt("Return [/json] verbatim")
        assert "[/json]" in capsys.readouterr().out


class TestRunOutput:
    def test_is_empty(self):
        from effgen.prompts.library.eval import RunOutput

        assert RunOutput(text="").is_empty
        assert RunOutput(text="   \n ").is_empty
        assert not RunOutput(text="x").is_empty
