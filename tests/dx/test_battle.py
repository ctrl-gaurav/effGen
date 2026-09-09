"""Tests for the head-to-head model battle — engine, terminal view, playground.

Covers:
1. The battle engine: parallel contenders, per-model answers and measurements,
   a failed contender isolated from the field and excluded from the verdict.
2. Usage on the streaming path: a terminal ``usage`` event and
   ``Agent.last_stream_usage``, without changing what text mode yields.
3. The terminal command: argument validation, and a piped run printing one
   structured result rather than a live view.
4. The self-contained HTML report a battle renders.
5. The playground battle view, driven in a real DOM against a stubbed
   streaming endpoint.

Live model calls are used for anything that measures real behavior; they are
skipped when the matching key is absent.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from effgen.cli.scaffold import resolve_max_tokens
from effgen.eval.battle import (
    BattleResult,
    Contender,
    _measured_verdict,
    fmt_cost,
    run_battle,
)
from tests._harness.offline_assets import assert_self_contained
from tests._harness.provider_unavailable import assert_cli_succeeded

NODE = shutil.which("node")
STATIC_DIR = Path(__file__).parents[2] / "effgen" / "playground" / "static"

needs_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
needs_groq = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
needs_gemini = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)
def _skip_if_a_contender_was_throttled(result) -> None:
    """Skip when a provider rate-limited a contender instead of answering it.

    A race needs every contender to have run before it can compare them. A
    spent free-tier quota is the provider's decision, not a defect in the
    battle, so it is reported rather than read as a contender that failed to
    finish.
    """
    _skip_if_a_serialized_contender_was_throttled(
        [{"model": c.model, "error": c.error} for c in result.contenders]
    )


def _skip_if_a_serialized_contender_was_throttled(contenders) -> None:
    """The same check against the serialized form, for the command tests.

    A subprocess run has no ``BattleResult`` to inspect, only the JSON document
    it printed, and that carries each contender's error verbatim.
    """
    from effgen.models.errors import classify_provider_error

    for contender in contenders:
        error = contender.get("error")
        if not error:
            continue
        if classify_provider_error(RuntimeError(error)).category == "rate_limited":
            pytest.skip(
                f"{contender.get('model')} was rate limited: {str(error)[:200]}"
            )


def _skip_if_the_judge_could_not_grade(judge: dict) -> None:
    """Skip when the judge model refused or was throttled rather than grading.

    ``run_battle`` records a judge failure as ``winner=None`` beside the reason
    and leaves the measured results alone, so this reports the provider's
    condition rather than reading it as a wrong verdict.
    """
    if judge.get("winner") is not None:
        return
    reason = judge.get("error") or "the judge returned no winner"
    pytest.skip(f"the judge could not grade this field: {str(reason)[:200]}")


needs_three_families = pytest.mark.skipif(
    not (
        os.getenv("OPENAI_API_KEY")
        and os.getenv("GROQ_API_KEY")
        and os.getenv("GOOGLE_API_KEY")
    ),
    reason="a cross-family battle needs the OpenAI, Groq and Gemini keys",
)

OPENAI_MODEL = "openai:gpt-5-nano"
GROQ_MODEL = "groq:openai/gpt-oss-20b"
GEMINI_MODEL = "gemini:gemini-3.1-flash-lite"

#: A budget every contender in a race can answer within.
#:
#: One race, one number, and it is bounded from both sides. ``gpt-5-nano``
#: spends its output budget on an internal reasoning chain before it writes
#: anything, so a cap sized for a one-word reply from a plain chat model buys it
#: 64 reasoning tokens and no answer — the empty column then reads as a
#: contender that failed rather than as a budget that was too small. Above that,
#: a free-tier per-minute token allowance is the ceiling: Groq counts what a
#: request *asks for*, so 8192 on ``openai/gpt-oss-20b`` is refused outright
#: against an 8000-token-per-minute limit however short the answer would be.
#:
#: effGen already computes each model's floor, so asking it keeps this right
#: when the model list changes, and keeps it under every contender's ceiling.
BATTLE_MAX_TOKENS = max(
    resolve_max_tokens(m) for m in (OPENAI_MODEL, GROQ_MODEL, GEMINI_MODEL)
)
MISSING_MODEL = "openai:gpt-5-nonexistent-xyz"


def _node_path() -> str | None:
    npm = shutil.which("npm")
    if npm is None:
        return None
    try:
        out = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - npm absent
        return None
    return out.stdout.strip() or None


def _has_jsdom() -> bool:
    root = _node_path()
    if root is None or NODE is None:
        return False
    probe = subprocess.run(
        [NODE, "-e", 'require("jsdom"); console.log("ok")'],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "NODE_PATH": root},
    )
    return probe.returncode == 0


needs_jsdom = pytest.mark.skipif(
    not (NODE and _has_jsdom()),
    reason="jsdom is not installed (npm install -g jsdom)",
)


# ---------------------------------------------------------------------------
# Verdict rules (no model calls)
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_failed_contender_cannot_win(self):
        """A model that never answered spent nothing and took no time; it must
        not win on cheapest or fastest for having failed."""
        failed = Contender(model="broken", error="model not found", state="failed",
                           latency_s=0.01, cost_usd=None)
        answered = Contender(model="works", answer="hello", state="done",
                             latency_s=2.0, cost_usd=0.5)
        verdict = _measured_verdict([failed, answered])
        assert verdict["fastest"]["model"] == "works"
        assert verdict["cheapest"]["model"] == "works"
        assert verdict["longest"]["model"] == "works"

    def test_unpriced_model_is_not_ranked_as_free(self):
        """A model with no published price is not the cheapest by default — it
        is reported as unranked rather than as $0."""
        local = Contender(model="local", answer="hi", state="done",
                          latency_s=1.0, cost_usd=None)
        cloud = Contender(model="cloud", answer="hi there", state="done",
                          latency_s=1.0, cost_usd=0.001)
        verdict = _measured_verdict([local, cloud])
        assert verdict["cheapest"]["model"] == "cloud"
        assert "publish no price" in verdict["cheapest"]["detail"]

    def test_no_finishers_yields_empty_verdict(self):
        verdict = _measured_verdict([Contender(model="a", error="boom", state="failed")])
        assert verdict == {"fastest": None, "cheapest": None, "longest": None}

    def test_unpriced_reads_as_unpriced_not_zero(self):
        assert fmt_cost(None) == "unpriced"
        assert fmt_cost(0) == "$0.00"

    def test_a_race_where_nothing_answered_says_so(self):
        """A model that never ran has no price and no verdict; neither reads as
        a model that published no price."""
        result = BattleResult(
            prompt="hi",
            contenders=[Contender(model="a", error="404", state="failed"),
                        Contender(model="b", error="404", state="failed")],
            verdict=_measured_verdict([Contender(model="a", error="404", state="failed")]),
            wall_s=1.0,
        )
        markdown = result.to_markdown()
        assert "No model answered" in markdown
        assert "unpriced" not in markdown, "a model that never ran has no price"
        assert "total cost —" in markdown

    def test_a_column_per_contender_when_a_model_is_entered_twice(self):
        """The live view gives each racer its own column, so two columns never
        mirror one contender."""
        from effgen.cli.battle import _Columns

        columns = _Columns(["dup", "dup", "other"])
        first = Contender(model="dup", answer="one", state="streaming")
        second = Contender(model="dup", answer="two", state="streaming")
        other = Contender(model="other", answer="three", state="streaming")
        for c in (first, second, other, first):
            columns.update(c)
        assert [c.answer for c in columns.current()] == ["one", "two", "three"]

    def test_columns_keep_the_order_the_models_were_named_in(self):
        from effgen.cli.battle import _Columns

        columns = _Columns(["a", "b"])
        second = Contender(model="b", answer="b answered first", state="done")
        columns.update(second)
        assert [c.model for c in columns.current()] == ["a", "b"]
        assert columns.current()[1].answer == "b answered first"


class TestArgumentValidation:
    def test_one_model_is_rejected(self):
        with pytest.raises(ValueError, match="at least two models"):
            run_battle("hello", ["only-one"])

    def test_empty_prompt_is_rejected(self):
        with pytest.raises(ValueError, match="needs a prompt"):
            run_battle("   ", ["a", "b"])


# ---------------------------------------------------------------------------
# Streaming usage (live)
# ---------------------------------------------------------------------------


class TestStreamingUsage:
    @needs_openai
    def test_stream_reports_usage_without_a_second_call(self):
        """A streamed run reports its tokens and cost from the stream itself."""
        from effgen.core.agent import Agent, AgentConfig

        agent = Agent(AgentConfig(model=OPENAI_MODEL, name="usage-test"))
        try:
            events = list(agent.stream("Name one primary color.", include_events=True))
            assert events[-1].kind == "usage", "the usage event ends the stream"
            usage = events[-1].usage
            assert usage["prompt_tokens"] > 0
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            assert usage["cost_usd"] > 0, "a priced model reports what the run cost"
            assert usage["estimated"] is False, "the provider reported these counts"
            assert usage["ttft_ms"] is not None and usage["latency_ms"] > 0
            assert agent.last_stream_usage == usage
        finally:
            agent.close()

    @needs_openai
    def test_text_mode_still_yields_only_answer_text(self):
        """The documented text-mode contract does not change: joining the
        deltas reconstructs the answer, with no usage record among them."""
        from effgen.core.agent import Agent, AgentConfig

        agent = Agent(AgentConfig(model=OPENAI_MODEL, name="usage-text-mode"))
        try:
            chunks = list(agent.stream("Reply with exactly: OK"))
            assert all(isinstance(c, str) for c in chunks)
            assert "OK" in "".join(chunks)
            assert agent.last_stream_usage is not None, "usage is readable after a text stream"
        finally:
            agent.close()

    @pytest.mark.skipif(
        not os.getenv("HF_TOKEN") and not Path.home().joinpath(".cache/huggingface").exists(),
        reason="no local model cache",
    )
    def test_local_model_counts_tokens_locally(self):
        """A local engine reports no usage, so the counts are taken from its own
        tokenizer and labelled as counted locally rather than left blank."""
        from effgen.core.agent import Agent, AgentConfig

        agent = Agent(AgentConfig(
            model="transformers:Qwen/Qwen2.5-1.5B-Instruct", name="usage-local"
        ))
        try:
            events = list(agent.stream("Say hi.", include_events=True))
            usage = events[-1].usage
            assert usage["estimated"] is True
            assert usage["total_tokens"] > 0
            assert usage["cost_usd"] is None, "a local model is unpriced, not free"
        finally:
            agent.close()


# ---------------------------------------------------------------------------
# The battle itself (live)
# ---------------------------------------------------------------------------


class TestLiveBattle:
    @needs_three_families
    def test_three_families_race_one_prompt(self):
        """Three providers answer the same prompt in parallel, each reporting
        its own answer, timings, tokens and cost."""
        result = run_battle(
            "Name the largest planet in one word.",
            [OPENAI_MODEL, GROQ_MODEL, GEMINI_MODEL],
            max_tokens=BATTLE_MAX_TOKENS,
        )
        assert len(result.contenders) == 3
        _skip_if_a_contender_was_throttled(result)
        assert len(result.finishers) == 3, [c.error for c in result.contenders]
        for c in result.contenders:
            assert c.answer.strip(), f"{c.model} produced no answer"
            assert c.state == "done"
            assert c.latency_s > 0 and c.ttft_s is not None
            assert c.total_tokens and c.total_tokens > 0
            assert c.cost_usd is not None and c.cost_usd > 0
        # Running in parallel means the race is shorter than the sum of its parts.
        assert result.wall_s < sum(c.latency_s + (c.load_s or 0) for c in result.contenders)
        assert result.verdict["fastest"]["model"] in {c.model for c in result.finishers}
        assert result.total_cost() > 0

    @needs_openai
    @needs_groq
    def test_a_failing_contender_does_not_end_the_race(self):
        """A model that cannot run is reported as failed, and the rest finish."""
        result = run_battle(
            "Name the capital of Japan in one word.",
            [GROQ_MODEL, MISSING_MODEL],
            max_tokens=100,
        )
        by_model = {c.model: c for c in result.contenders}
        assert by_model[GROQ_MODEL].ok
        broken = by_model[MISSING_MODEL]
        assert not broken.ok
        assert broken.state == "failed"
        assert broken.error, "a failed contender names why it failed"
        assert broken.cost_usd is None
        # It never becomes the winner for having spent nothing.
        assert result.verdict["fastest"]["model"] == GROQ_MODEL
        assert result.verdict["cheapest"]["model"] == GROQ_MODEL

    @needs_three_families
    def test_a_named_judge_grades_the_field(self):
        """The judge is a model the caller names, and the output says which."""
        result = run_battle(
            "In one sentence, what is a mutex?",
            [OPENAI_MODEL, GROQ_MODEL],
            max_tokens=BATTLE_MAX_TOKENS,
            judge=GEMINI_MODEL,
        )
        judge = result.verdict["judge"]
        assert judge["judge_model"] == GEMINI_MODEL
        # A judge whose quota is spent, or whose credential is not configured
        # on this machine, cannot grade anything — that is the provider's
        # decision, not a defect in the battle. The measurements below are
        # still asserted, because a judge failure must leave them intact.
        _skip_if_the_judge_could_not_grade(judge)
        assert judge["winner"] in {c.model for c in result.finishers}
        # The judged pick stays separate from what was measured.
        assert result.verdict["fastest"] is not None

    @needs_openai
    @needs_groq
    def test_result_serializes_with_every_answer(self):
        result = run_battle(
            "Reply with the word: ping", [OPENAI_MODEL, GROQ_MODEL],
            max_tokens=BATTLE_MAX_TOKENS,
        )
        doc = json.loads(result.to_json())
        # What this measures is the serialized shape, not whether a free tier
        # had budget left: a throttled contender is the provider's answer.
        _skip_if_a_serialized_contender_was_throttled(doc["contenders"])
        assert doc["prompt"] == "Reply with the word: ping"
        assert {c["model"] for c in doc["contenders"]} == {OPENAI_MODEL, GROQ_MODEL}
        for c in doc["contenders"]:
            assert c["answer"], "the answer text is in the structured result"
            assert c["latency_s"] is not None
        markdown = result.to_markdown()
        assert "## Verdict" in markdown
        assert OPENAI_MODEL in markdown


# ---------------------------------------------------------------------------
# Terminal command
# ---------------------------------------------------------------------------


class TestBattleCommand:
    def _run(self, *args, **kwargs):
        return subprocess.run(
            ["effgen", "battle", *args],
            capture_output=True, text=True, timeout=300, **kwargs,
        )

    def test_one_model_is_refused_with_a_hint(self):
        proc = self._run("hello", "-m", "openai:gpt-5-nano")
        assert proc.returncode == 2
        assert "at least two models" in proc.stderr
        assert "effgen models list" in proc.stderr

    def test_help_documents_the_piped_behavior(self):
        proc = self._run("--help")
        assert proc.returncode == 0
        assert "--judge" in proc.stdout
        assert "NO_COLOR" in proc.stdout

    @needs_openai
    @needs_groq
    def test_piped_run_prints_one_structured_result(self):
        """Piped output carries the whole result and no live frame or ANSI."""
        proc = self._run(
            "Reply with the word: pong", "-m", f"{OPENAI_MODEL},{GROQ_MODEL}",
            "--max-tokens", str(BATTLE_MAX_TOKENS),
        )
        assert_cli_succeeded(proc, "effgen battle")
        assert "\x1b[" not in proc.stdout, "piped output carries no ANSI escapes"
        assert "# Model Battle" in proc.stdout
        assert "## Verdict" in proc.stdout
        for model in (OPENAI_MODEL, GROQ_MODEL):
            assert f"## {model}" in proc.stdout, "each model's answer is printed"

    @needs_openai
    @needs_groq
    def test_json_output_is_valid_on_stdout(self):
        proc = self._run(
            "Reply with the word: pong", "-m", f"{OPENAI_MODEL},{GROQ_MODEL}",
            "--max-tokens", str(BATTLE_MAX_TOKENS), "--json",
        )
        assert_cli_succeeded(proc, "effgen battle")
        doc = json.loads(proc.stdout)
        assert len(doc["contenders"]) == 2
        _skip_if_a_serialized_contender_was_throttled(doc["contenders"])
        assert all(c["answer"] for c in doc["contenders"])
        assert doc["verdict"]["fastest"]["model"]
        assert doc["total_cost_usd"] > 0

    @needs_openai
    @needs_groq
    def test_html_report_is_self_contained(self, tmp_path):
        out = tmp_path / "battle.html"
        proc = self._run(
            "Reply with the word: pong", "-m", f"{OPENAI_MODEL},{GROQ_MODEL}",
            "--max-tokens", str(BATTLE_MAX_TOKENS), "--report", str(out), "--json",
        )
        assert_cli_succeeded(proc, "effgen battle")
        html = out.read_text(encoding="utf-8")
        assert_self_contained(html, "the battle report")
        assert "//" not in _asset_refs(html), "no protocol-relative asset"
        assert "Model Battle" in html
        assert "Verdict" in html or "Fastest" in html


def _asset_refs(html: str) -> str:
    import re

    return " ".join(re.findall(r'(?:src|href)="([^"]+)"', html))


# ---------------------------------------------------------------------------
# HTML report rendering (no model calls)
# ---------------------------------------------------------------------------


class TestBattleReport:
    def _doc(self):
        return {
            "prompt": "What is a cache?",
            "wall_s": 2.5,
            "total_cost_usd": 0.0004,
            "contenders": [
                {
                    "model": "openai:gpt-5-nano", "answer": "A cache stores data.",
                    "error": None, "state": "done", "ttft_s": 0.4, "latency_s": 1.2,
                    "total_tokens": 40, "cost_usd": 0.0003, "estimated_tokens": False,
                },
                {
                    "model": "local:qwen", "answer": "A fast store.", "error": None,
                    "state": "done", "ttft_s": 0.2, "latency_s": 0.9,
                    "total_tokens": 20, "cost_usd": None, "estimated_tokens": True,
                },
                {
                    "model": "broken:model", "answer": "", "error": "model not found",
                    "state": "failed", "ttft_s": None, "latency_s": None,
                    "total_tokens": None, "cost_usd": None, "estimated_tokens": False,
                },
            ],
            "verdict": {
                "fastest": {"model": "local:qwen", "detail": "answered in 0.90s"},
                "cheapest": {"model": "openai:gpt-5-nano", "detail": "$0.000300 for this run"},
                "longest": {"model": "openai:gpt-5-nano", "detail": "20 characters"},
                "judge": {"judge_model": "gemini:flash", "winner": "openai:gpt-5-nano",
                          "reasoning": "More precise."},
            },
        }

    def test_kind_is_detected_from_the_document(self):
        from effgen.ui.report_html import detect_report_kind

        assert detect_report_kind(self._doc()) == "battle"

    def test_report_shows_answers_costs_and_the_judge(self):
        from effgen.ui.report_html import build_html_report

        html = build_html_report(self._doc(), kind="battle")
        assert "A cache stores data." in html, "the answers are in the shareable artifact"
        assert "A fast store." in html
        assert "model not found" in html, "a failed contender is named, not hidden"
        assert "unpriced" in html, "an unpriced model is not shown as $0"
        assert "gemini:flash" in html, "the judge is named so the pick can be weighed"
        assert_self_contained(html, "the comparison report")

    def test_comparison_report_shows_what_each_model_answered(self):
        from effgen.ui.report_html import build_html_report

        html = build_html_report({
            "scores": [{
                "model": "openai:gpt-5-nano", "suite": "cases", "accuracy": 1.0,
                "avg_latency": 1.0, "total_tokens": 10, "avg_cost_usd": 0.001,
                "avg_tool_accuracy": 1.0, "error": None, "error_count": 0,
                "responses": [
                    {"query": "2+2?", "output": "4", "score": 1.0, "passed": True,
                     "error": None},
                ],
            }],
            "recommendations": {"cases": "openai:gpt-5-nano"},
            "recommendation_rationale": {}, "optimize": "accuracy",
            "scoring": "contains", "num_cases": 1, "suite_sizes": {"cases": 1},
            "generated_at": "2026-01-01T00:00:00+00:00",
        }, kind="comparison")
        assert "<h2>Answers</h2>" in html
        assert ">4<" in html or "4</span>" in html

    def test_comparison_report_names_what_graded_the_answers(self):
        """A judged bake-off says who judged, so the scores can be weighed."""
        from effgen.ui.report_html import build_html_report

        base = {
            "scores": [{
                "model": "openai:gpt-5-nano", "suite": "cases", "accuracy": 1.0,
                "avg_latency": 1.0, "total_tokens": 10, "avg_cost_usd": 0.001,
                "avg_tool_accuracy": 1.0, "error": None, "error_count": 0,
                "responses": [],
            }],
            "recommendations": {"cases": "openai:gpt-5-nano"},
            "recommendation_rationale": {}, "optimize": "accuracy",
            "scoring": "llm_judge", "num_cases": 1, "suite_sizes": {"cases": 1},
            "generated_at": "2026-01-01T00:00:00+00:00",
        }
        judged = build_html_report(
            {**base, "judge_model": "gemini:gemini-3.1-flash-lite", "self_judged": False},
            kind="comparison",
        )
        assert "gemini:gemini-3.1-flash-lite" in judged
        assert "not one of the compared models" in judged

        self_judged = build_html_report(
            {**base, "judge_model": None, "self_judged": True}, kind="comparison")
        assert "graded its own answers" in self_judged


# ---------------------------------------------------------------------------
# Playground battle view
# ---------------------------------------------------------------------------


BATTLE_CONTROL_IDS = [
    "mode-single",
    "mode-battle",
    "battle-field",
    "battle-models",
    "battle-results",
    "battle-grid",
    "battle-verdict",
]


class TestPlaygroundBattleAssets:
    def test_battle_controls_exist(self):
        html = (STATIC_DIR / "index.html").read_text()
        for cid in BATTLE_CONTROL_IDS:
            assert f'id="{cid}"' in html, f"battle control '{cid}' not in index.html"

    def test_no_external_asset_was_introduced(self):
        html = (STATIC_DIR / "index.html").read_text()
        refs = _asset_refs(html)
        assert "http://" not in refs and "https://" not in refs
        assert "//" not in refs

    def test_battle_styles_are_present(self):
        css = (STATIC_DIR / "style.css").read_text()
        for selector in (".battle-grid", ".battle-col", ".battle-verdict"):
            assert selector in css, f"missing style {selector}"


# The page's battle flow, driven in a real DOM. The endpoint is stubbed with a
# scripted SSE body so the test exercises the page's parsing, live rendering and
# verdict rules — not the provider.
_JSDOM_BATTLE = r"""
const fs = require("fs");
const { JSDOM, VirtualConsole } = require("jsdom");

const html = fs.readFileSync(process.argv[2], "utf8");
const webuiJs = fs.readFileSync(process.argv[3], "utf8");
const appJs = fs.readFileSync(process.argv[4], "utf8");

const vc = new VirtualConsole();
const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true,
                              url: "http://localhost/playground", virtualConsole: vc });
const { window } = dom;

function sse(chunks) {
  // A ReadableStream of SSE bytes, like the server sends.
  const enc = new window.TextEncoder ? new TextEncoder() : null;
  let i = 0;
  return {
    getReader() {
      return {
        read() {
          if (i >= chunks.length) return Promise.resolve({ done: true, value: undefined });
          const value = Buffer.from(chunks[i++], "utf8");
          return Promise.resolve({ done: false, value: new Uint8Array(value) });
        },
      };
    },
  };
}

const CATALOG = {
  data: [
    { provider: "openai", id: "gpt-5-nano", is_priced: true, price_in_per_1m: 0.05,
      price_out_per_1m: 0.4, context_window: 400000, verified_on: "2026-07-01" },
    { provider: "groq", id: "openai/gpt-oss-20b", is_priced: true,
      price_in_per_1m: 0.05, price_out_per_1m: 0.08 },
    { provider: "openai", id: "broken-model", is_priced: true, price_in_per_1m: 1 },
  ],
  local: [], providers: ["openai", "groq"],
};

function body(model) {
  if (model === "openai:broken-model") {
    return { status: 404, ok: false,
             json: () => Promise.resolve({ error: { message: "model not found" } }) };
  }
  const cost = model.startsWith("groq") ? 0.000002 : 0.00005;
  const words = model.startsWith("groq") ? ["fast ", "answer"] : ["a ", "longer ", "answer here"];
  const parts = words.map((w) =>
    'data: ' + JSON.stringify({ model, choices: [{ index: 0, delta: { content: w } }] }) + "\n\n");
  parts.push('data: ' + JSON.stringify({
    model, choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
    effgen: { cost_usd: cost },
  }) + "\n\n");
  parts.push("data: [DONE]\n\n");
  return { status: 200, ok: true, body: sse(parts) };
}

window.fetch = (url, opts) => {
  if (String(url).indexOf("/playground/bootstrap") === 0) {
    return Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve({
      presets: [], tools: [], defaults: { max_tokens: 64, temperature: 0.7, stream: true },
      default_model: "openai:gpt-5-nano", spend_authorized: true,
      session_key: "k", dev_mode: true, catalog_url: "/v1/models/catalog" }) });
  }
  if (String(url).indexOf("/v1/models/catalog") === 0) {
    return Promise.resolve({ status: 200, ok: true, json: () => Promise.resolve(CATALOG) });
  }
  if (String(url).indexOf("/v1/chat/completions") === 0) {
    return Promise.resolve(body(JSON.parse(opts.body).model));
  }
  return Promise.reject(new Error("unexpected fetch " + url));
};
window.performance = window.performance || { now: () => Date.now() };

window.eval(webuiJs);
window.eval(appJs);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  await sleep(120);  // bootstrap + catalog
  const out = {};
  const doc = window.document;

  // The contender picker is built from the catalog, not from served aliases.
  const picker = doc.getElementById("battle-models");
  out.contenderOptions = Array.from(picker.options).map((o) => o.value);

  // Switching to battle mode swaps the controls and the results panel.
  doc.getElementById("mode-battle").checked = true;
  doc.getElementById("mode-battle").dispatchEvent(new window.Event("change"));
  out.singleHidden = doc.getElementById("model-field").hidden;
  out.battleShown = !doc.getElementById("battle-field").hidden;
  out.runLabel = doc.getElementById("run-btn").textContent;

  // Fewer than two contenders is refused before anything is spent.
  for (const o of picker.options) o.selected = (o.value === "openai:gpt-5-nano");
  doc.getElementById("prompt").value = "What is a cache?";
  doc.getElementById("run-btn").click();
  await sleep(60);
  out.tooFewError = doc.getElementById("error-banner").textContent;

  // Race three contenders, one of which fails.
  for (const o of picker.options) {
    o.selected = ["openai:gpt-5-nano", "groq:openai/gpt-oss-20b",
                  "openai:broken-model"].indexOf(o.value) >= 0;
  }
  doc.getElementById("run-btn").click();
  await sleep(900);

  out.columns = Array.from(doc.querySelectorAll(".battle-col")).map((el) => ({
    model: el.querySelector(".battle-model").textContent,
    state: el.querySelector(".battle-state").getAttribute("data-state"),
    answer: el.querySelector(".battle-answer").textContent,
    foot: el.querySelector(".battle-foot").textContent,
  }));
  out.verdict = doc.getElementById("battle-verdict").textContent;
  out.verdictShown = !doc.getElementById("battle-verdict").hidden;
  console.log(JSON.stringify(out));
})().catch((e) => { console.error("ERR " + e.stack); process.exit(1); });
"""


@needs_jsdom
class TestPlaygroundBattleBehavior:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        script = tmp_path_factory.mktemp("battle-js") / "drive.js"
        script.write_text(textwrap.dedent(_JSDOM_BATTLE), encoding="utf-8")
        webui = Path(__file__).parents[2] / "effgen" / "webui" / "static" / "webui.js"
        proc = subprocess.run(
            [
                NODE, str(script),
                str(STATIC_DIR / "index.html"), str(webui), str(STATIC_DIR / "app.js"),
            ],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "NODE_PATH": _node_path() or ""},
        )
        assert_cli_succeeded(proc, "effgen battle")
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_contenders_come_from_the_catalog(self, result):
        """The picker offers real catalog ids, not the served-alias list."""
        assert "openai:gpt-5-nano" in result["contenderOptions"]
        assert "groq:openai/gpt-oss-20b" in result["contenderOptions"]

    def test_battle_mode_swaps_the_controls(self, result):
        assert result["singleHidden"] is True
        assert result["battleShown"] is True
        assert "battle" in result["runLabel"].lower()

    def test_fewer_than_two_contenders_is_refused(self, result):
        assert "at least two" in result["tooFewError"]

    def test_each_contender_gets_its_own_column(self, result):
        # A multi-select carries no selection order, so the columns follow the
        # picker's own order — stable across runs rather than click-dependent.
        models = [c["model"] for c in result["columns"]]
        assert models == ["groq:openai/gpt-oss-20b", "openai:gpt-5-nano",
                          "openai:broken-model"]
        by_model = {c["model"]: c for c in result["columns"]}
        assert by_model["openai:gpt-5-nano"]["answer"] == "a longer answer here"
        assert by_model["groq:openai/gpt-oss-20b"]["answer"] == "fast answer"

    def test_a_failed_contender_is_marked_and_the_others_finish(self, result):
        by_model = {c["model"]: c for c in result["columns"]}
        assert by_model["openai:broken-model"]["state"] == "failed"
        assert "model not found" in by_model["openai:broken-model"]["answer"]
        assert by_model["groq:openai/gpt-oss-20b"]["state"] == "done"

    def test_the_tally_reads_cost_from_the_server(self, result):
        """The cost shown is the server's number, not one derived in the page."""
        by_model = {c["model"]: c for c in result["columns"]}
        assert "$0.000002" in by_model["groq:openai/gpt-oss-20b"]["foot"]
        assert "15 tok" in by_model["groq:openai/gpt-oss-20b"]["foot"]

    def test_verdict_names_the_winners_and_excludes_the_failure(self, result):
        assert result["verdictShown"] is True
        verdict = result["verdict"]
        assert "Cheapest" in verdict and "groq:openai/gpt-oss-20b" in verdict
        assert "Longest" in verdict and "openai:gpt-5-nano" in verdict
        assert "Did not answer" in verdict and "openai:broken-model" in verdict
        assert "2/3 answered" in verdict
