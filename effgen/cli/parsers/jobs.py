"""Argument declarations for the commands that run a job over many calls.

Workflows, batches, evaluation, model comparison and battles, cost and
report rendering. ``effgen.cli._main.create_parser`` calls these in the
order the top-level ``--help`` lists them; each attaches one command to the
given subparsers action and returns nothing.
"""

from __future__ import annotations

import argparse


def add_workflow_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen workflow`` and its run/validate subcommands."""
    workflow_parser = subparsers.add_parser('workflow', help='Run a DAG-based workflow')
    workflow_parser.set_defaults(_group_parser=workflow_parser)
    workflow_subparsers = workflow_parser.add_subparsers(dest='workflow_command', help='Workflow command')

    workflow_run = workflow_subparsers.add_parser('run', help='Run a workflow from YAML file')
    workflow_run.add_argument('file', help='Path to workflow YAML file')
    workflow_run.add_argument('-m', '--model', help='Default model for all agents')
    workflow_run.add_argument('-i', '--input', action='append', nargs=2, metavar=('NODE', 'TASK'),
                              help='Input for a specific node (can be repeated)')
    workflow_run.add_argument('--task', help='A single task string routed to the '
                              'workflow entry node(s) (alternative to --input)')
    workflow_run.add_argument('--json', dest='output_json', action='store_true',
                              help='Emit the workflow result as JSON to stdout (for CI gating)')
    workflow_run.add_argument('--diagram', action='store_true',
                              help='Draw the workflow as a dependency graph (nodes by '
                                   'level, edges, per-node status/duration/cost)')
    workflow_run.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                              help='Quiet output (errors only); --json still emits to stdout')

    workflow_validate = workflow_subparsers.add_parser('validate', help='Validate a workflow YAML file')
    workflow_validate.add_argument('file', help='Path to workflow YAML file')
    workflow_validate.add_argument('--json', dest='output_json', action='store_true',
                                   help='Emit the validation result as JSON to stdout')
    workflow_validate.add_argument('--diagram', action='store_true',
                                   help='Draw the workflow dependency graph (nodes by level, edges)')
    workflow_validate.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                                   help='Quiet output (errors only); --json still emits to stdout')

def add_batch_parser(subparsers: argparse._SubParsersAction, *, preset_choices: list[str]) -> None:
    """Declare ``effgen batch`` — run many queries from a file."""
    batch_parser = subparsers.add_parser(
        'batch', help='Run batch queries from a file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen batch queries.jsonl -o answers.jsonl -m gpt-5-nano\n"
            "  effgen batch -i rows.csv --query-field question -o out.csv --excel\n"
            "  effgen batch prompts.txt --json -q | jq '.rows[].output'\n"
        ),
    )
    batch_parser.add_argument('input_file', nargs='?', default=None, metavar='INPUT',
                              help='Input file (JSONL, CSV, JSON, or plain text). '
                                   'Same as -i/--input.')
    batch_parser.add_argument('-i', '--input', help='Input file (JSONL, CSV, JSON, or plain text)')
    batch_parser.add_argument('-o', '--output',
                              help='Output file (JSONL, CSV, or JSON). .jsonl rows are '
                                   'written as each query finishes, so their file order '
                                   'is completion order, not input order, at any '
                                   '--concurrency above 1; .csv/.json rows are written '
                                   'once at the end in input order. Every row carries an '
                                   '"index" field back to its input position — sort on '
                                   'it if your consumer assumes line N corresponds to '
                                   'input row N.')
    batch_parser.add_argument('-c', '--concurrency', type=int, default=5, help='Max concurrent queries (default: 5)')
    batch_parser.add_argument('--batch-size', type=int, default=0, help='Batch size (0 = all at once)')
    batch_parser.add_argument('--timeout', type=float, default=120.0, help='Timeout per query in seconds')
    batch_parser.add_argument('--retries', type=int, default=1, help='Retries for failed queries')
    batch_parser.add_argument('-m', '--model', help='Model to use')
    batch_parser.add_argument('--preset', choices=preset_choices,
                              help='Use a preset agent configuration')
    batch_parser.add_argument(
        '--guardrails', metavar='NAME',
        help='Apply a guardrail preset to redact/block PII and screen for '
             'prompt injection on every row: "strict", "standard" (alias '
             '"default"/"balanced"), "phi" (alias "hipaa"/"deidentify"), '
             '"minimal", or "none".',
    )
    batch_parser.add_argument(
        '--system-prompt', '--persona', dest='system_prompt', metavar='TEXT',
        help='System prompt applied to every row, e.g. a target language, '
             'glossary, and tone instruction for a localization batch '
             '("Translate into formal European French (vous); keep {placeholders} '
             'and HTML tags verbatim."). Overrides the preset\'s default prompt.',
    )
    batch_parser.add_argument('--query-field', default='query', help='Field name for queries in JSONL/CSV (default: query)')
    batch_parser.add_argument('--max-tokens', type=int, default=None,
                              help='Max output tokens per query (raise for token-heavy or reasoning models)')
    batch_parser.add_argument('--temperature', type=float, default=None,
                              help='Sampling temperature per query (0 for deterministic reruns where the provider supports it)')
    batch_parser.add_argument('--schema', dest='schema_path', default=None,
                              help='JSON Schema file; each row is validated against it and its parsed object is written')
    batch_parser.add_argument('--output-model', dest='output_model', default=None,
                              help='Pydantic model as module:ClassName to validate each row against')
    batch_parser.add_argument('--strict', action='store_true',
                              help='Abort on the first malformed input line instead of skipping it')
    batch_parser.add_argument('--resume', action='store_true',
                              help='Skip input rows already present in the JSONL --output file and append the rest')
    batch_parser.add_argument(
        '--excel', '--bom', dest='excel_bom', action='store_true',
        help='Prepend a UTF-8 BOM to CSV output so Excel on Windows opens '
             'non-Latin scripts (Arabic, CJK, Devanagari, ...) correctly on '
             'double-click. Only affects --output ending in .csv.',
    )
    batch_parser.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                              help='Quiet output (suppress the progress bar)')
    batch_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                              help='Disable the live progress bar (plain output)')
    batch_parser.add_argument('--json', dest='output_json', action='store_true',
                              help='Emit the job summary and every row as a JSON '
                                   'document to stdout (for piping to jq), in addition '
                                   'to any -o file. Human output goes to stderr; '
                                   'combine with -q for clean stdout.')

def add_eval_parser(subparsers: argparse._SubParsersAction, *, preset_choices: list[str]) -> None:
    """Declare ``effgen eval`` — score an agent against a test suite."""
    eval_parser = subparsers.add_parser('eval', help='Evaluate an agent against a test suite')
    eval_parser.add_argument('--suite', required=True,
                              help='Built-in suite name (math, tool_use, reasoning, safety, '
                                   'conversation) OR a path to your own .jsonl/.json test cases')
    eval_parser.add_argument('-m', '--model', help='Model to use')
    eval_parser.add_argument(
        '--provider',
        help='Provider for a bare model id (e.g. openai, groq, cerebras, gemini, '
             'together, fireworks, replicate, anthropic, hf). '
             'Equivalent to the "provider:model" prefix.',
    )
    eval_parser.add_argument('--preset', choices=preset_choices,
                              help='Use a preset agent configuration')
    eval_parser.add_argument('--scoring', choices=['exact_match', 'contains', 'regex', 'semantic_similarity', 'llm_judge'],
                              default='contains', help='Scoring mode (default: contains)')
    eval_parser.add_argument('--threshold', type=float, default=0.5,
                              help='Per-case pass score for continuous scoring modes '
                                   '(semantic_similarity, llm_judge); has no effect on '
                                   'exact_match/contains/regex, whose scores are already binary '
                                   '(0 or 1) (default: 0.5). Use --fail-under to gate the exit '
                                   'code on suite accuracy.')
    eval_parser.add_argument('--fail-under', type=float, default=0.5, metavar='ACCURACY',
                              help='Minimum suite accuracy required for a zero exit code '
                                   '(default: 0.5). This is the CI gate; a --compare-baseline '
                                   'regression always fails regardless of this value.')
    eval_parser.add_argument('--temperature', type=float, default=None,
                              help='Sampling temperature for the evaluated agent (0 for '
                                   'deterministic, reproducible scoring where the provider '
                                   'supports it; default: the model/preset default)')
    eval_parser.add_argument('--save-baseline', action='store_true',
                              help='Save results as regression baseline')
    eval_parser.add_argument('--compare-baseline', action='store_true',
                              help='Compare results against stored baseline')
    eval_parser.add_argument('--baseline-dir', dest='baseline_dir', default=None, metavar='DIR',
                              help='Directory for --save-baseline/--compare-baseline files '
                                   '(default: ./.effgen/baselines under the current directory, '
                                   'created if missing). A baseline saved under the installed '
                                   'package tree by an older effGen version is still read.')
    eval_parser.add_argument('-o', '--output',
                              help='Output file for results. The extension chooses the '
                                   'format: .html renders the shareable report, .md writes '
                                   'Markdown, anything else writes JSON.')
    eval_parser.add_argument('--report', metavar='PATH.html',
                              help='Write a self-contained HTML report to PATH — pass rate, '
                                   'exit gate, by-difficulty breakdown, and every case. The '
                                   'file opens offline with no external references.')
    eval_parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                              help='Filter test cases by difficulty')
    eval_parser.add_argument('--max-cases', type=int, default=None,
                              help='Only run the first N cases (quick subsample)')
    eval_parser.add_argument('--json', dest='output_json', action='store_true',
                              help='Emit the results object as JSON to stdout (for CI gating)')
    eval_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                              help='Disable the live progress bar (plain output)')

def add_compare_parser(subparsers: argparse._SubParsersAction, *, preset_choices: list[str]) -> None:
    """Declare ``effgen compare`` — bake several models off on one suite."""
    compare_parser = subparsers.add_parser(
        'compare', help='Compare multiple models on a test suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen compare --models gpt-5-nano,groq:openai/gpt-oss-20b --suite math\n"
            "  effgen compare --models gpt-5-nano,gpt-5-mini --suite reasoning --optimize cost\n"
            "  effgen compare --models a,b --suite ./cases.jsonl --json | jq .recommendations\n"
            "\n"
            "compare reports a bake-off and always exits 0; use `eval --fail-under` "
            "to gate a build.\n"
        ),
    )
    compare_parser.add_argument('-m', '--models', required=True,
                                 help='Comma-separated model ids. Use a '
                                      'provider:model prefix to pin a provider '
                                      'for a bare id (e.g. '
                                      'groq:openai/gpt-oss-20b,gpt-5-nano).')
    compare_parser.add_argument('--suite', required=True,
                                 help='Built-in suite name (math, tool_use, '
                                      'reasoning, safety, conversation) OR a path '
                                      'to your own .jsonl/.json test cases')
    compare_parser.add_argument('--scoring', choices=['exact_match', 'contains', 'regex', 'semantic_similarity', 'llm_judge'],
                                 default='contains', help='Scoring mode (default: contains)')
    compare_parser.add_argument('--threshold', type=float, default=0.5,
                                 help='Per-case pass score for continuous scoring modes '
                                      '(semantic_similarity, llm_judge); has no effect on '
                                      'exact_match/contains/regex, whose scores are already '
                                      'binary (0 or 1) (default: 0.5). compare always exits 0 — '
                                      'it reports a bake-off rather than gating a build; use '
                                      '`eval --fail-under` for CI gating.')
    compare_parser.add_argument('--temperature', type=float, default=None,
                                 help='Sampling temperature for every compared model (0 for '
                                      'deterministic, reproducible scoring where the provider '
                                      'supports it; default: the model/preset default)')
    compare_parser.add_argument(
        '--provider',
        help='Provider applied to any bare model id in --models that has no '
             '"provider:" prefix of its own (e.g. openai, groq, cerebras, gemini, '
             'together, fireworks, replicate, anthropic, hf).',
    )
    compare_parser.add_argument('--max-cases', type=int, default=None,
                                 help='Only run the first N cases (quick bake-off '
                                      'on a big suite)')
    compare_parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                                 help='Filter test cases by difficulty')
    compare_parser.add_argument('-o', '--output',
                                 help='Output file for results. The extension chooses the '
                                      'format: .html renders the shareable report, .md writes '
                                      'Markdown, anything else writes JSON.')
    compare_parser.add_argument('--report', metavar='PATH.html',
                                 help='Write a self-contained HTML report to PATH — the '
                                      'recommended model and why, a per-model table, and '
                                      'accuracy/cost/latency charts. The file opens offline '
                                      'with no external references.')
    compare_parser.add_argument('--json', dest='output_json', action='store_true',
                                 help='Emit the comparison matrix as JSON to stdout (for CI gating)')
    compare_parser.add_argument('--preset', choices=preset_choices,
                                 help='Use a preset agent configuration')
    compare_parser.add_argument('--optimize', choices=['accuracy', 'cost', 'latency'],
                                 default='accuracy',
                                 help="What the recommendation optimizes for (default: accuracy — "
                                      "highest accuracy, tie-broken on lower latency then fewer "
                                      "tokens). 'cost'/'latency' recommend the cheapest/fastest "
                                      "model among those meeting --threshold accuracy (falling "
                                      "back to the full field if none qualify), tie-broken on "
                                      "higher accuracy.")
    compare_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                                 help='Disable the live progress bar (plain output)')
    compare_parser.add_argument('--judge', metavar='MODEL',
                                 help='Model that grades answers under --scoring llm_judge. '
                                      'Without it each model grades its own answers; naming a '
                                      'judge has one model grade the whole field. The judge is '
                                      'named in the output.')

def add_battle_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen battle`` — race several models on one prompt."""
    battle_parser = subparsers.add_parser(
        'battle', help='Race several models on one prompt, side by side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen battle \"Explain a B-tree in two sentences.\" \\\n"
            "      -m openai:gpt-5-nano,groq:openai/gpt-oss-20b\n"
            "  effgen battle \"Write a haiku about caching.\" -m a,b,c --judge openai:gpt-5-mini\n"
            "  effgen battle \"...\" -m a,b --json | jq '.contenders[].cost_usd'\n"
            "  effgen battle \"...\" -m a,b --report battle.html\n"
            "\n"
            "Every model answers the same prompt at once. On a terminal the answers\n"
            "stream side by side, each column showing its own time to first token,\n"
            "elapsed time, tokens and cost, and a verdict panel closes the race.\n"
            "\n"
            "The verdict reports what was measured — fastest, cheapest, longest —\n"
            "and needs no judge. --judge names a separate model to pick a winner on\n"
            "quality; that pick is reported apart from the measurements and names\n"
            "the judge. A model that fails is reported as failed and cannot win.\n"
            "\n"
            "Piped output, --json, --no-animation and NO_COLOR skip the live view\n"
            "and print one structured result carrying every model's full answer.\n"
        ),
    )
    battle_parser.add_argument('prompt', help='The prompt every model answers')
    battle_parser.add_argument('-m', '--models', required=True, metavar='A,B[,C]',
                                help='Comma-separated model ids to race (at least two), '
                                     'e.g. openai:gpt-5-nano,groq:openai/gpt-oss-20b')
    battle_parser.add_argument('--judge', metavar='MODEL',
                                help='Model asked to pick the best answer. Optional — the '
                                     'measured outcomes need no judge.')
    battle_parser.add_argument('--temperature', type=float, default=None,
                                help='Sampling temperature applied to every model')
    battle_parser.add_argument('--max-tokens', type=int, default=None, dest='max_tokens',
                                help='Output token cap applied to every model')
    battle_parser.add_argument('--system-prompt', dest='system_prompt', metavar='TEXT',
                                help='System prompt applied to every model')
    battle_parser.add_argument('-o', '--output', metavar='PATH',
                                help='Save the battle. The extension chooses the format: '
                                     '.md writes Markdown, anything else writes JSON.')
    battle_parser.add_argument('--report', metavar='PATH.html',
                                help='Write a self-contained HTML report of the battle')
    battle_parser.add_argument('--json', dest='output_json', action='store_true',
                                help='Print the battle as JSON to stdout (no live view)')
    battle_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                                help='Skip the live side-by-side view and print the result')

def add_cost_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen cost`` — the spend dashboard and budget subcommands."""
    _cost_output_help = ('Output file for the spend summary. The extension chooses the '
                         'format: .html renders the shareable report, anything else '
                         'writes JSON.')
    _cost_report_help = ('Write a self-contained HTML spend report to PATH — total against '
                         'the daily budget, a per-provider/model table, and cost-share '
                         'charts. The file opens offline with no external references.')
    cost_parser = subparsers.add_parser('cost', help='View cost spend and manage budgets')
    cost_parser.add_argument('--json', dest='output_json', action='store_true', help='Output as JSON')
    cost_parser.add_argument('-o', '--output', help=_cost_output_help)
    cost_parser.add_argument('--report', metavar='PATH.html', help=_cost_report_help)
    cost_subparsers = cost_parser.add_subparsers(dest='cost_command', help='Cost command')
    for _cost_sub, _cost_help in (
        ('today', 'Show per-provider/model spend for the last 24 hours'),
        ('week', 'Show rolling 7-day spend summary'),
        ('by-provider', 'Show lifetime totals grouped by provider'),
    ):
        _cost_period = cost_subparsers.add_parser(_cost_sub, help=_cost_help)
        _cost_period.add_argument('--json', dest='output_json', action='store_true',
                                  default=argparse.SUPPRESS, help='Output as JSON')
        _cost_period.add_argument('-o', '--output', default=argparse.SUPPRESS,
                                  help=_cost_output_help)
        _cost_period.add_argument('--report', metavar='PATH.html',
                                  default=argparse.SUPPRESS, help=_cost_report_help)
    cost_prune = cost_subparsers.add_parser(
        'prune', help='Delete old events from the local spend ledger')
    cost_prune.add_argument('--older-than-days', type=float, metavar='DAYS',
                            help='Delete events older than DAYS (default: 90).')
    cost_prune.add_argument('--keep-rows', type=int, metavar='N',
                            help='Keep the newest N events and delete the rest.')
    cost_prune.add_argument('--dry-run', action='store_true',
                            help='Report what would be deleted without deleting it.')
    cost_prune.add_argument('--json', dest='output_json', action='store_true',
                            default=argparse.SUPPRESS,
                            help='Output as JSON')

    cost_set_budget = cost_subparsers.add_parser('set-budget', help='Set a daily spend budget')
    cost_set_budget.add_argument('amount', type=float, help='Daily budget in USD (e.g. 1.0)')
    cost_subparsers.add_parser('clear-budget', help='Remove configured budget limits')

def add_report_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen report`` — render a saved result JSON as HTML."""
    from effgen.ui.report_html import REPORT_KINDS as _REPORT_KINDS
    report_parser = subparsers.add_parser(
        'report',
        help='Render a saved run/compare/eval/cost/loadtest JSON result as an HTML report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen eval --suite math --json > eval.json && effgen report eval.json\n"
            "  effgen run 'summarize this' -o run.json && effgen report run.json\n"
            "  effgen report bakeoff.json -o bakeoff.html\n"
            "  effgen report spend.json --kind cost\n"
            "\n"
            "The report kind is inferred from the JSON shape; --kind overrides it.\n"
            "A document that carries none of the fields the kind renders is\n"
            "refused, and no file is written.\n"
            "The written file is self-contained and opens with no network access.\n"
        ),
    )
    report_parser.add_argument('result',
                               help='Path to a JSON result saved from run/compare/eval/cost/loadtest')
    report_parser.add_argument('-o', '--output', metavar='PATH.html',
                               help='Where to write the HTML report '
                                    '(default: the result path with an .html extension)')
    report_parser.add_argument('--kind', choices=list(_REPORT_KINDS),
                               help='Report shape to render, when the JSON cannot be identified')
