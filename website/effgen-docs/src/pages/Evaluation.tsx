import { ClipboardCheck } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { siteData } from '../siteData';

const evalOptions = siteData.cli.command_options['eval'] ?? [];

export default function Evaluation() {
  return (
    <DocPage
      subtitle="Scoring an agent against a test suite, and failing a build when the score drops."
      icon={<ClipboardCheck size={48} />}
    >
      <p>
        <code>effgen eval</code> runs an agent over a suite of cases, scores each answer, and exits
        non-zero when the suite's accuracy falls below a threshold — which is what makes it a CI
        gate rather than a report. Five suites ship, your own <code>.jsonl</code> works the same way,
        and a saved baseline turns "is this good enough" into "is this worse than last time".
      </p>

      <h2>Run one</h2>

      <CodeBlock language="bash" filename="terminal" code={`effgen eval --suite math -m openai:gpt-5-nano --max-cases 5`} />

      <Terminal
        command="effgen eval --suite math -m openai:gpt-5-nano --max-cases 5"
        output={`Limited to first 5 cases
Loading model openai:gpt-5-nano...
Running math suite (5 cases, scoring=contains)...

Evaluation Results: math
Metric         Value
-------------  ------------
Accuracy       100.0% (5/5)
Avg Latency    2.9676s
Total Tokens   1158
Tool Accuracy  0.0%

  By Difficulty:
    easy    : 100.0% (5/5)

  Exit gate: PASS — accuracy 100.0% >= --fail-under 50%`}
        caption={
          <>
            <code>--max-cases</code> is how you check the wiring before spending a full suite's
            worth of calls.
          </>
        }
      />

      <h2>The built-in suites</h2>

      <CodeBlock filename="suites.py" code={`from effgen.eval import list_suites

for name, description in list_suites().items():
    print(f"{name:14} {description}")`} />

      <Terminal
        command="python suites.py"
        output={`math           77 cases — Math problems — basic arithmetic to calculus
tool_use       93 cases — Tool-use scenarios across built-in tools
reasoning      40 cases — Multi-step reasoning problems
safety         40 cases — Prompt injection and safety tests
conversation   20 cases — Multi-turn conversation evaluations`}
        caption={
          <>
            Read from <code>effgen.eval.list_suites()</code>, so the case counts are the installed
            package's. Cases carry a difficulty, which <code>--difficulty</code> filters on and the
            results break down by.
          </>
        }
      />

      <h3>Your own cases</h3>

      <CodeBlock
        language="json"
        filename="cases.jsonl"
        code={`{"query": "What is the capital of France?", "expected_output": "Paris", "difficulty": "easy"}
{"query": "Summarise this refund policy in one sentence.", "expected_output": "refund", "difficulty": "medium"}
{"query": "Which tool would you use to fetch a web page?", "expected_output": "url_fetch", "difficulty": "hard"}`}
        caption={
          <>
            Pass the path to <code>--suite</code> instead of a name. One JSON object per line, or a{' '}
            <code>.json</code> array. <code>difficulty</code> is optional.
          </>
        }
      />

      <h2>Scoring</h2>

      <ApiTable
        headers={['Mode', 'Passes when', 'Continuous']}
        rows={[
          [<code>exact_match</code>, 'The output equals the expected value exactly.', 'No — 0 or 1'],
          [<code>contains</code>, 'The expected value appears in the output. The default.', 'No — 0 or 1'],
          [<code>regex</code>, 'The output matches the expected value as a pattern.', 'No — 0 or 1'],
          [
            <code>semantic_similarity</code>,
            <>
              Cosine similarity clears <code>--threshold</code>. Needs sentence-transformers.
            </>,
            'Yes',
          ],
          [
            <code>llm_judge</code>,
            <>
              A model grades the answer and the score clears <code>--threshold</code>.
            </>,
            'Yes',
          ],
        ]}
        caption={
          <>
            <code>--threshold</code> only affects the two continuous modes; the other three are
            already binary. <strong>The suite-level gate is <code>--fail-under</code></strong>, and
            it is the one that decides the exit code.
          </>
        }
      />

      <Callout type="warning" title="exact_match is stricter than it sounds">
        <p>
          A model that answers "$0.05" where the case expects "0.05", or "5 minutes" where it expects
          "5", scores zero under <code>exact_match</code> while being entirely correct. That is the
          failing run below. Use <code>contains</code> for free-form answers, and{' '}
          <code>exact_match</code> only where the output format is pinned down.
        </p>
      </Callout>

      <h2>The CI gate</h2>

      <p>
        <code>--fail-under</code> is the minimum suite accuracy for a zero exit code, and it defaults
        to <code>0.5</code>. Nothing else about the command changes; the exit code is the whole
        mechanism.
      </p>

      <CodeBlock language="bash" filename="terminal" code={`effgen eval --suite math -m openai:gpt-5-nano --max-cases 5 --fail-under 0.9
echo "exit $?"`} />

      <Terminal command="effgen eval --suite math … --fail-under 0.9" output={`Limited to first 5 cases
Loading model openai:gpt-5-nano...
Running math suite (5 cases, scoring=contains)...

Evaluation Results: math
Metric         Value
-------------  ------------
Accuracy       100.0% (5/5)
Avg Latency    3.7163s
Total Tokens   1542
Tool Accuracy  0.0%

  By Difficulty:
    easy    : 100.0% (5/5)

  Exit gate: PASS — accuracy 100.0% >= --fail-under 90%
exit 0`} />

      <CodeBlock language="bash" filename="terminal" code={`effgen eval --suite reasoning -m openai:gpt-5-nano --max-cases 4 \\
  --scoring exact_match --fail-under 0.9
echo "exit $?"`} />

      <Terminal
        command="effgen eval --suite reasoning … --scoring exact_match --fail-under 0.9"
        output={`Limited to first 4 cases
Loading model openai:gpt-5-nano...
Running reasoning suite (4 cases, scoring=exact_match)...

Evaluation Results: reasoning
Metric         Value
-------------  ----------
Accuracy       0.0% (0/4)
Avg Latency    4.4750s
Total Tokens   2103
Tool Accuracy  75.0%

  By Difficulty:
    medium  : 0.0% (0/4)

  Failed cases (4):
    - If all roses are flowers and some flowers fade quickly, can ...
      Expected: no
      Got:      No. All roses are flowers, but “some flo
    - A bat and a ball cost $1.10. The bat costs $1.00 more than t...
      Expected: 0.05
      Got:      $0.05
    - If it takes 5 machines 5 minutes to make 5 widgets, how long...
      Expected: 5
      Got:      5 minutes
    - In a lake, there is a patch of lily pads. Every day, the pat...
      Expected: 47
      Got:      47 days

  Exit gate: FAIL — accuracy 0.0% < --fail-under 90%
exit 1`}
        maxLines={26}
        caption={
          <>
            Exit 1, and the failing cases are printed with what was expected and what came back — so
            the reason here is visible without opening a report. All four answers are correct; the
            scoring mode is wrong for them.
          </>
        }
      />

      <CodeBlock
        language="yaml"
        filename=".github/workflows/eval.yml"
        code={`- name: Evaluate the agent
  env:
    OPENAI_API_KEY: \${{ secrets.OPENAI_API_KEY }}
  run: |
    effgen eval --suite ./tests/cases.jsonl \\
      -m openai:gpt-5-nano \\
      --temperature 0 \\
      --fail-under 0.85 \\
      --report eval.html

- uses: actions/upload-artifact@v4
  if: always()
  with:
    name: eval-report
    path: eval.html`}
        caption={
          <>
            <code>--temperature 0</code> is what makes two runs comparable where the provider
            supports it. Upload the report on failure as well as success — that is when someone needs
            to read it.
          </>
        }
      />

      <h2>Baselines and regressions</h2>

      <p>
        <code>--save-baseline</code> stores the run; <code>--compare-baseline</code> scores the next
        one against it and prints the deltas. <strong>A regression fails the build whatever{' '}
        <code>--fail-under</code> says</strong> — an agent that is still above the absolute
        threshold but noticeably worse than last week is the case this exists for.
      </p>

      <CodeBlock language="bash" filename="terminal" code={`effgen eval --suite math -m openai:gpt-5-nano --max-cases 5 --save-baseline --baseline-dir baselines
effgen eval --suite math -m openai:gpt-5-nano --max-cases 5 --compare-baseline --baseline-dir baselines`} />

      <Terminal
        command="effgen eval … --save-baseline && effgen eval … --compare-baseline"
        output={`Limited to first 5 cases
Loading model openai:gpt-5-nano...
Running math suite (5 cases, scoring=contains)...

Evaluation Results: math
Metric         Value
-------------  ------------
Accuracy       100.0% (5/5)
Avg Latency    3.2076s
Total Tokens   1094
Tool Accuracy  0.0%

  By Difficulty:
    easy    : 100.0% (5/5)

  Baseline saved to baselines/eval_baseline_math.json

  Exit gate: PASS — accuracy 100.0% >= --fail-under 50%
Limited to first 5 cases
Loading model openai:gpt-5-nano...
Running math suite (5 cases, scoring=contains)...

Evaluation Results: math
Metric         Value
-------------  ------------
Accuracy       100.0% (5/5)
Avg Latency    3.1525s
Total Tokens   1222
Tool Accuracy  0.0%

  By Difficulty:
    easy    : 100.0% (5/5)

# Regression Report: math

**Baseline:** 1.0.0  
**Current:** 1.0.0  
**Status:** PASS

## Metrics

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| accuracy | 1.0000 | 1.0000 | +0.0% |
| avg_latency | 3.2076 | 3.1525 | -1.7% |
| total_tokens | 1094.0000 | 1222.0000 | +11.7% |
| avg_tool_accuracy | 0.0000 | 0.0000 | N/A |

  Exit gate: PASS — accuracy 100.0% >= --fail-under 50%`}
        maxLines={30}
        caption={
          <>
            Baselines live in <code>./.effgen/baselines</code> under the current directory by
            default, created if missing; <code>--baseline-dir</code> moves them, which is what the
            capture above did. Check them in if the comparison should mean anything across machines.
          </>
        }
      />

      <h2>Options</h2>

      <ParamTable
        nameLabel="Flag"
        params={evalOptions.map((option) => ({
          name: option.name,
          description: option.description,
        }))}
        caption={
          <>
            Every flag <code>effgen eval --help</code> declares, read from the binary.
          </>
        }
      />

      <h2>From Python</h2>

      <CodeBlock filename="evaluate.py" code={`from effgen import Agent, AgentConfig
from effgen.eval import AgentEvaluator, MathSuite
from effgen.eval.evaluator import ScoringMode

agent = Agent(AgentConfig(model="gpt-5-nano", provider="openai"))
evaluator = AgentEvaluator(agent, scoring=ScoringMode.CONTAINS)

results = evaluator.run_suite(MathSuite(), max_cases=5)
print(results.summary())`} />

      <Terminal command="python evaluate.py" output={`{'suite': 'math', 'total': 5, 'passed': 5, 'accuracy': 1.0, 'avg_latency': 2.7936, 'total_tokens': 1031, 'total_cost_usd': 0.000271, 'avg_tool_accuracy': 0.0, 'by_difficulty': {'easy': {'total': 5, 'passed': 5, 'accuracy': 1.0}}, 'error_count': 0, 'error': None, 'metadata': {'scoring': 'contains', 'pass_threshold': 0.5, 'num_cases': 5}, 'results': [{'query': 'What is 2 + 3?', 'expected_output': '5', 'agent_output': '5', 'score': 1.0, 'passed': True, 'latency': 3.1914, 'tokens_used': 163, 'cost_usd': 5.645e-05, 'tool_accuracy': 0.0, 'tools_called': [], 'difficulty': 'easy', 'details': {}, 'error': None}, {'query': 'What is 15 * 7?', 'expected_output': '105', 'agent_output': '105', 'score': 1.0, 'passed': True, 'latency': 2.8825, 'tokens_used': 204, 'cost_usd': 5.885e-05, 'tool_accuracy': 0.0, 'tools_called': [], 'difficulty': 'easy', 'details': {}, 'error': None}, {'query': 'What is 144 / 12?', 'expected_output': '12', 'agent_output': '12', 'score': 1.0, 'passed': True, 'latency': 2.7797, 'tokens_used': 223, 'cost_usd': 5.945e-05, 'tool_accuracy': 0.0, 'tools_called': [], 'difficulty': 'easy', 'details': {}, 'error': None}, {'query': 'What is 256 - 189?', 'expected_output': '67', 'agent_output': '67', 'score': 1.0, 'passed': True, 'latency': 2.5313, 'tokens_used': 179, 'cost_usd': 3.485e-05, 'tool_accuracy': 0.0, 'tools_called': [], 'difficulty': 'easy', 'details': {}, 'error': None}, {'query': 'What is 3^5?', 'expected_output': '243', 'agent_output': '243', 'score': 1.0, 'passed': True, 'latency': 2.5829, 'tokens_used': 262, 'cost_usd': 6.14e-05, 'tool_accuracy': 0.0, 'tools_called': [], 'difficulty': 'easy', 'details': {}, 'error': None}]}`} maxLines={12} />

      <ApiTable
        headers={['Object', 'What it is']}
        rows={[
          [
            <code>AgentEvaluator(agent, scoring=…, pass_threshold=0.5, judge_agent=None)</code>,
            <>
              Wraps an <Link to="/agents">agent</Link>. <code>judge_agent</code> is the grader under{' '}
              <code>LLM_JUDGE</code>; without one, the agent grades itself.
            </>,
          ],
          [
            <code>evaluator.run_suite(suite, max_cases=None, progress_callback=None)</code>,
            <>
              Returns a <code>SuiteResults</code>.
            </>,
          ],
          [
            <code>SuiteResults</code>,
            <>
              <code>accuracy</code>, <code>avg_latency</code>, <code>avg_tool_accuracy</code>,{' '}
              <code>total_tokens</code>, <code>total_cost_usd</code>, <code>error_count</code>,{' '}
              <code>summary()</code>, <code>to_json()</code>, <code>to_markdown()</code>.
            </>,
          ],
          [
            <>
              <code>MathSuite()</code>, <code>ToolUseSuite()</code>, <code>ReasoningSuite()</code>,{' '}
              <code>SafetySuite()</code>, <code>ConversationSuite()</code>
            </>,
            'The five built-in suites as objects, for use with run_suite().',
          ],
          [
            <>
              <code>get_suite(name)</code> / <code>list_suites()</code>
            </>,
            'Look one up by name, or list them with their case counts.',
          ],
          [
            <code>RegressionTracker()</code>,
            <>
              <code>save_baseline(name, results, version=…)</code> and{' '}
              <code>compare(name, results, version=…)</code>, whose report has{' '}
              <code>to_markdown()</code>.
            </>,
          ],
        ]}
        caption={
          <>
            All from <code>effgen.eval</code>. <code>ScoringMode</code> and <code>Difficulty</code>{' '}
            are the two enums.
          </>
        }
      />

      <h2>Reports</h2>

      <p>
        <code>--report out.html</code> writes a self-contained report — pass rate, the exit gate, the
        by-difficulty breakdown and every case — that opens with no network access.{' '}
        <code>-o</code> picks its format from the extension: <code>.html</code> renders the report,{' '}
        <code>.md</code> writes Markdown, anything else writes JSON. And a result saved earlier can
        be rendered without re-running anything:
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen eval --suite math -m openai:gpt-5-nano --json > eval.json
effgen report eval.json`}
        continues
        caption={
          <>
            <code>effgen report</code> infers the report kind from the JSON shape and refuses a
            document that carries none of the fields it renders, rather than writing an empty page.
          </>
        }
      />

      <Terminal command="effgen eval --suite math … --max-cases 3 --json" output={`{
  "suite": "math",
  "total": 3,
  "passed": 3,
  "accuracy": 1.0,
  "avg_latency": 3.574,
  "total_tokens": 909,
  "total_cost_usd": 0.00030235,
  "avg_tool_accuracy": 0.0,
  "by_difficulty": {
    "easy": {
      "total": 3,
      "passed": 3,
      "accuracy": 1.0
    }
  },
  "error_count": 0,
  "error": null,
  "metadata": {
    "scoring": "contains",
    "pass_threshold": 0.5,
    "num_cases": 3,
    "model": "openai:gpt-5-nano",
    "fail_under": 0.5
  },
  "results": [
    {
      "query": "What is 2 + 3?",
      "expected_output": "5",
      "agent_output": "5",
      "score": 1.0,`} maxLines={22} />

      <h2>Your own tasks: effgen bench</h2>

      <p>
        New in 1.2.0, <code>effgen bench</code> runs a suite of your own tasks against a model and
        reports what the agent spent to answer them: accuracy beside model calls, tool calls,
        tokens, time and cost. <code>effgen bench compare</code> pairs two saved runs task by task
        and prints a noise band beside every difference, so a change within run-to-run variation is
        not reported as a result. A suite is a strict YAML or JSON file — a name, the tools, a
        scorer, agent settings and a list of tasks with expected answers — and{' '}
        <code>effgen bench init</code> writes a starter one.
      </p>

      <CodeBlock
        language="bash"
        filename="terminal"
        code={`effgen bench init
effgen bench run bench-suite.yaml --model my-model --base-url http://127.0.0.1:8000/v1 --out runs/a
effgen bench run bench-suite.yaml --model my-model --base-url http://127.0.0.1:8000/v1 --out runs/b
effgen bench compare runs/a runs/b`}
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            'Accuracy near zero on answers that look right',
            <>
              <code>exact_match</code> against free-form output.
            </>,
            <>
              Switch to <code>contains</code>, or pin the format in the prompt so an exact match is
              a fair test.
            </>,
          ],
          [
            'Two runs of the same suite disagree',
            'Sampling. The model is not deterministic at its default temperature.',
            <>
              <code>--temperature 0</code>, where the provider supports it. Compare baselines rather
              than absolute numbers otherwise.
            </>,
          ],
          [
            <>
              The build passes but the agent is worse
            </>,
            <>
              <code>--fail-under</code> is an absolute floor; it says nothing about the trend.
            </>,
            <>
              Add <code>--compare-baseline</code>. A regression fails regardless of the floor.
            </>,
          ],
          [
            <>
              <code>--compare-baseline</code> finds nothing to compare
            </>,
            'No baseline was saved for that suite in that directory.',
            <>
              Run <code>--save-baseline</code> once, and point both runs at the same{' '}
              <code>--baseline-dir</code>. A relative default directory means CI starts empty every
              time.
            </>,
          ],
          [
            <>
              <code>--threshold</code> appears to do nothing
            </>,
            'The scoring mode is already binary.',
            <>
              It only applies to <code>semantic_similarity</code> and <code>llm_judge</code>. The
              suite gate is <code>--fail-under</code>.
            </>,
          ],
          [
            'Tool accuracy is 0% on a suite that is about tools',
            'The agent had no tools configured, so it answered from the model alone.',
            <>
              Pass <code>--preset</code>, or evaluate an agent you built with tools —{' '}
              <Link to="/tools">Tools</Link>.
            </>,
          ],
          [
            'The run costs more than expected',
            'A full suite is 77, 93, 40, 40 or 20 model calls, once per case.',
            <>
              <code>--max-cases</code> and <code>--difficulty</code> narrow it.{' '}
              <Link to="/cost">Set a budget</Link> before a long run.
            </>,
          ],
        ]}
      />

      <Callout type="note" title="New in 1.0.0">
        <p>
          <code>--provider</code> and <code>--temperature</code> reached <code>eval</code> and{' '}
          <code>compare</code> in this release, and <code>--json</code> on a pipe is now a single
          valid document with no spinner or table mixed into it — which is what makes{' '}
          <code>effgen eval --json | jq</code> reliable in a build. The shareable HTML report and{' '}
          <code>effgen report</code> are new too.
        </p>
      </Callout>

      <SeeAlso paths={['/compare', '/cost', '/observability']} />
    </DocPage>
  );
}
