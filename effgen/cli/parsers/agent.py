"""Argument declarations for the commands that run an agent.

``effgen.cli._main.create_parser`` calls these in the order the top-level
``--help`` lists them; each attaches one command (or one alias pair) to the
given subparsers action and returns nothing.
"""

from __future__ import annotations

import argparse


def add_run_parser(subparsers: argparse._SubParsersAction, *, preset_choices: list[str]) -> None:
    """Declare ``effgen run`` — run an agent on one task."""
    run_parser = subparsers.add_parser(
        'run', help='Run an agent with a task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen run \"What is 25 * 17?\" -t calculator\n"
            "  effgen run \"Summarize this\" --file report.pdf -m gpt-5-nano\n"
            "  effgen run \"Draft a reply\" --persona \"terse, formal\" --json | jq .output\n"
            "\n"
            "Environment:\n"
            "  EFFGEN_WORKSPACE   directory where the file and shell tools read\n"
            "                     and write by default, and the only directory\n"
            "                     sandboxed code may write to. Set it to keep\n"
            "                     files an agent generates out of the current\n"
            "                     directory (created if missing). Unset: the\n"
            "                     current dir.\n"
        ),
    )
    run_parser.add_argument('task', nargs='?', default=None, help='Task description (launches interactive wizard if not provided)')
    run_parser.add_argument('-m', '--model', help='Model to use')
    run_parser.add_argument(
        '--provider',
        help='Provider for a bare model id (e.g. openai, groq, cerebras, gemini, '
             'together, fireworks, replicate, anthropic, hf). '
             'Equivalent to the "provider:model" prefix.',
    )
    run_parser.add_argument('-v', '--verbose', action='store_true', default=argparse.SUPPRESS,
                            help='Verbose output (show DEBUG/INFO logs)')
    run_parser.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                            help='Quiet output (errors only)')
    run_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                            help='Disable live spinners/progress animation')
    run_parser.add_argument('-n', '--name', help='Agent name')
    run_parser.add_argument('-t', '--tools', nargs='+', help='Tools to enable')
    run_parser.add_argument('-c', '--config', help='Configuration file')
    run_parser.add_argument(
        '--system-prompt', '--persona', dest='system_prompt', metavar='TEXT',
        help='Custom persona / system prompt for this run, e.g. '
             '"You are a patient Socratic tutor who never gives the answer."',
    )
    run_parser.add_argument('--temperature', type=float, help='Temperature')
    run_parser.add_argument('--max-tokens', type=int,
                            help='Max output tokens (raise for token-heavy or '
                                 'reasoning models, e.g. gpt-5/o-series, which '
                                 'spend part of the budget on hidden reasoning '
                                 'before any visible text)')
    run_parser.add_argument('--max-iterations', type=int, help='Max iterations')
    run_parser.add_argument('--mode', choices=['auto', 'single', 'sub_agents'], help='Execution mode')
    run_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')
    run_parser.add_argument('--stream', action='store_true', help='Stream output')
    run_parser.add_argument('-o', '--output',
                            help='Write the full result as a JSON document to this '
                                 'file (output, success, tool_calls, tokens, cost, '
                                 'trace, citations, metadata)')
    run_parser.add_argument('--card', metavar='PATH.html',
                            help='Write a shareable HTML card for this run to PATH — '
                                 'the task, the answer, the tool trace with per-step '
                                 'durations, sources and citations, and tokens/cost/'
                                 'latency. The file is self-contained and opens with '
                                 'no network access. Terminal and --json output are '
                                 'unchanged.')
    run_parser.add_argument('--json', dest='output_json', action='store_true',
                            help='Emit that same JSON result object to stdout (for '
                                 'piping to jq). Human output goes to stderr; '
                                 'combine with -q for clean stdout.')
    run_parser.add_argument('--preset', choices=preset_choices,
                            help='Use a preset agent configuration')
    run_parser.add_argument(
        '--guardrails', metavar='NAME',
        help='Apply a guardrail preset to redact/block PII and screen for '
             'prompt injection before the task reaches the model: '
             '"strict", "standard" (alias "default"/"balanced"), "phi" '
             '(alias "hipaa"/"deidentify"), "minimal", or "none". Also '
             'honored from a `-c/--config` file\'s "guardrails" key.',
    )
    run_parser.add_argument('--show-thread', dest='show_thread', action='store_true',
                            help='Print the run\'s conversation after the answer: '
                                 'the persona and the question the model was framed '
                                 'by, every thought, every tool call with its input, '
                                 'every result and how the run ended. Secrets are '
                                 'redacted, and nothing that differs between two runs '
                                 'of the same path is printed, so two of these diff '
                                 'cleanly. The same conversation is in --json under '
                                 'metadata.thread.')
    run_parser.add_argument('--explain', action='store_true',
                            help='Show why the agent chose each tool')
    run_parser.add_argument('--trace', action='store_true',
                            help='Show a step-by-step timeline with per-step durations')
    run_parser.add_argument('--checkpoint-dir', help='Directory to write agent checkpoints')
    run_parser.add_argument('--checkpoint-interval', type=int, default=0,
                            help='Checkpoint every N iterations (requires --checkpoint-dir)')
    run_parser.add_argument(
        '--session-id', metavar='ID',
        help='Persistent conversation session id (shared with `effgen chat '
             '--session-id` and `effgen sessions`). Recalls prior turns and '
             'saves new ones. (Distinct from `effgen resume --checkpoint`, which '
             'restores a mid-run checkpoint snapshot.)',
    )
    run_parser.add_argument(
        '-f', '--file', '--input', dest='input_files', action='append', metavar='PATH',
        help='Attach a file to the task. An image (.png/.jpg/.gif/.webp/...) is '
             'passed as multimodal input; a document (.pdf/.docx/.xlsx/.txt/'
             '.md/.csv/...) or a source file (.py/.js/.ts/.go/.rs/.java/.sql/'
             '...) is read and prepended to the task as context. Any other file '
             'that decodes as UTF-8 text is read as plain text. Repeatable.',
    )

def add_resume_parser(subparsers: argparse._SubParsersAction, *, preset_choices: list[str]) -> None:
    """Declare ``effgen resume`` — continue a run from a checkpoint."""
    resume_parser = subparsers.add_parser(
        'resume',
        help='Resume an interrupted agent run from a saved checkpoint snapshot '
             '(distinct from a conversation session — see `--session-id`)')
    resume_parser.add_argument('--checkpoint', required=True,
                               help='Checkpoint id, JSON path, or directory (uses latest)')
    resume_parser.add_argument('-m', '--model', help='Model to use')
    resume_parser.add_argument('--preset', choices=preset_choices,
                               help='Preset to resume the run under')

def add_chat_parser(subparsers: argparse._SubParsersAction, *, preset_choices: list[str]) -> None:
    """Declare ``effgen chat`` — an interactive session with one agent."""
    chat_parser = subparsers.add_parser(
        'chat', help='Interactive chat mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  effgen chat -m gpt-5-nano --provider openai\n"
            "  effgen chat --preset research -t calculator wikipedia\n"
            "  effgen chat --session-id support-42   # resume a saved session\n"
            "\n"
            "In-session slash commands: /help /model /tools /cost /trace /reset "
            "/save /load /doctor /exit\n"
        ),
    )
    chat_parser.add_argument('-m', '--model', help='Model to use')
    chat_parser.add_argument(
        '--provider',
        help='Provider for a bare model id (e.g. openai, groq, cerebras, gemini). '
             'Equivalent to the "provider:model" prefix.',
    )
    chat_parser.add_argument(
        '--preset', choices=preset_choices,
        help='Agent preset for the session (e.g. math, research) — attaches the '
             "preset's tools and system prompt, same as `effgen run --preset`",
    )
    chat_parser.add_argument(
        '-t', '--tools', nargs='+', metavar='TOOL',
        help='Tools to enable for the session, same as `effgen run --tools` '
             '(e.g. calculator wikipedia). Also addable mid-session with /tools.',
    )
    chat_parser.add_argument(
        '--system-prompt', '--persona', dest='system_prompt', metavar='TEXT',
        help='Custom persona / system prompt for the session, e.g. '
             '"You are a patient Socratic tutor who never gives the answer." '
             'Steers every reply (unlike --preset, which only labels the session).',
    )
    chat_parser.add_argument(
        '--guardrails', metavar='NAME',
        help='Apply a guardrail preset to redact/block PII and screen for '
             'prompt injection on every turn: "strict", "standard" (alias '
             '"default"/"balanced"), "phi" (alias "hipaa"/"deidentify"), '
             '"minimal", or "none". Carries across a /model or /tools rebuild.',
    )
    chat_parser.add_argument('--temperature', type=float, help='Temperature')
    chat_parser.add_argument('--max-tokens', type=int,
                             help='Max output tokens per reply (raise for token-heavy '
                                  'or reasoning models, e.g. gpt-5/o-series, which '
                                  'spend part of the budget on hidden reasoning '
                                  'before any visible text)')
    chat_parser.add_argument('--no-sub-agents', action='store_true', help='Disable sub-agents')
    chat_parser.add_argument('-v', '--verbose', action='store_true', default=argparse.SUPPRESS,
                             help='Verbose output (show DEBUG/INFO logs)')
    chat_parser.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                             help='Quiet output (errors only)')
    chat_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                             help='Disable live spinners/progress animation')
    chat_parser.add_argument(
        '--session-id', '--resume', dest='session_id', metavar='ID',
        help='Continue a persistent session by id (same store as '
             '`effgen run --session-id` and `effgen sessions list`). Prior turns '
             'are recalled and new turns are saved; a new id starts a fresh session.',
    )

def add_code_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen code`` — the coding agent and its permission modes."""
    code_parser = subparsers.add_parser(
        'code', help='Coding agent: writes code, runs it, and fixes what fails',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Run a coding agent over a workspace. It proposes an approach, '
            'writes files, executes code in the configured sandbox, reads the '
            'real output, and iterates until the task is done or the iteration '
            'cap is reached. Each file change is shown as a unified diff before '
            'it is written, and --undo reverses the last applied edit. '
            'On a terminal with no task it opens an interactive session with '
            'slash commands (/plan, /review, /diff, /apply, /undo, /run, '
            '/context, /git, /model, /help); a task, -p, piped stdin or --json '
            'runs once and exits. --review makes a run read-only, and '
            '--session-id continues a saved session.'
        ),
        epilog=(
            "Examples:\n"
            "  effgen code                         # interactive session\n"
            "  effgen code \"write fib.py with fib(n) and print fib(10)\" --auto-edit\n"
            "  effgen code -p \"add a --retries flag to cli.py\" --json | jq .files_written\n"
            "  cat pytest.log | effgen code -p \"why did this fail?\"\n"
            "  effgen code --undo                  # revert the last applied edit\n"
            "  effgen code \"fix the failing test\" --auto-edit --commit\n"
            "  effgen code --review                # review everything uncommitted\n"
            "  effgen code --review staged -p \"is this safe to merge?\"\n"
            "  effgen code --review main...HEAD --json | jq -r .answer\n"
            "  effgen code -f src/app.py -f src/db.py -p \"where can this raise?\"\n"
            "  effgen code --session-id refactor-42 # continue a saved session\n"
            "\n"
            "Review mode is read-only in the tools it holds, not only in the\n"
            "mode it runs in: the file tool is narrowed to reading and the git\n"
            "tool is the read-only surface, so there is nothing attached that\n"
            "writes a file, runs code or runs a shell command. The change under\n"
            "review is handed to the model as context. --review cannot be\n"
            "combined with a permission flag, --commit or --undo.\n"
            "\n"
            "--session-id restores the conversation, the files in context, the\n"
            "files the session wrote, and (on a terminal, when no permission\n"
            "flag was given) the permission mode. The workspace always comes\n"
            "from -w/--workspace or the current directory; a stored one that\n"
            "differs is reported and not adopted. Edits proposed by /plan are\n"
            "not stored -- re-run /plan against the current files.\n"
            "\n"
            "In a git repository the branch, the short status and a bounded file\n"
            "layout (ignored files excluded) become part of the agent's context,\n"
            "and an AGENTS.md in the workspace is read as project instructions.\n"
            "The confirmed commit behind --commit is the only repository change a\n"
            "session makes: push, reset, checkout, rebase, tag and stash are refused\n"
            "in every mode, from the shell too.\n"
            "\n"
            "Permission modes (pick at most one):\n"
            "  --plan        propose only; show the diffs, write nothing, run nothing\n"
            "  (default)     with a terminal, show each diff and confirm every write\n"
            "                and command; without one, behave as --plan\n"
            "  --auto-edit   apply writes and sandboxed runs; confirm shell commands\n"
            "  --yes         apply writes, runs and shell commands without asking\n"
            "\n"
            "Exit codes: 0 completed, 1 failed, 2 completed but changes were\n"
            "withheld because there was no terminal to confirm on (including a\n"
            "--commit that could not be confirmed).\n"
            "\n"
            "Stdin: when it is not a terminal it is read to EOF before the run\n"
            "starts -- with a task it is folded in as context, without one it is\n"
            "the task. A pipe that stays open therefore holds the run; after a\n"
            "couple of seconds the wait is reported on stderr. Pass < /dev/null\n"
            "to skip the read.\n"
            "\n"
            "Environment:\n"
            "  EFFGEN_WORKSPACE          the directory the agent reads and writes,\n"
            "                            and the only one sandboxed code may write\n"
            "                            to (created if missing). Unset: the\n"
            "                            current directory. -w/--workspace\n"
            "                            overrides it.\n"
            "  EFFGEN_SANDBOX_BACKEND    docker|subprocess. Docker confines the\n"
            "                            filesystem and network for executed code;\n"
            "                            the subprocess fallback isolates the\n"
            "                            network and confines writes to the\n"
            "                            workspace, leaving the rest of the\n"
            "                            filesystem readable but read-only.\n"
        ),
    )
    code_parser.add_argument('task', nargs='?', default=None,
                             help='What to build, change or debug (omit on a '
                                  'terminal to open an interactive session)')
    code_parser.add_argument('-p', '--print', dest='print_task', nargs='?', const='',
                             metavar='TASK',
                             help='Run one task and print the result. Takes the task '
                                  'directly, or reads it from stdin when given alone.')
    code_parser.add_argument('-m', '--model', help='Model to use')
    code_parser.add_argument(
        '--provider',
        help='Provider for a bare model id (e.g. openai, groq, cerebras, gemini, '
             'together, fireworks, replicate, anthropic, hf). '
             'Equivalent to the "provider:model" prefix.',
    )
    code_parser.add_argument('-w', '--workspace', metavar='DIR',
                             help='Directory the agent reads and writes (created if '
                                  'missing). Sets EFFGEN_WORKSPACE for the run; '
                                  'nothing outside it is written.')
    code_parser.add_argument('--plan', dest='plan_only', action='store_true',
                             help='Propose the change without writing a file or '
                                  'running a command')
    code_parser.add_argument('--auto-edit', dest='auto_edit', action='store_true',
                             help='Apply file writes and sandboxed runs without '
                                  'asking; shell commands still need confirmation')
    code_parser.add_argument('-y', '--yes', dest='assume_yes', action='store_true',
                             help='Apply writes, sandboxed runs and shell commands '
                                  'without asking (still confined to the workspace)')
    code_parser.add_argument(
        '--review', nargs='?', const='uncommitted', metavar='TARGET',
        help='Review instead of change: read-only, with no tool that writes a '
             'file or runs a command. TARGET is uncommitted (the default), '
             'staged, or any revision or range git accepts (HEAD~3, '
             'main...HEAD). Combine with -f/--file, or use -f/--file alone to '
             'review files outside a repository.',
    )
    code_parser.add_argument(
        '-f', '--file', dest='review_files', action='append', metavar='PATH',
        help='Include a workspace file in the review, in full. Repeatable. '
             'Used with --review, or on its own to review files without a diff.',
    )
    code_parser.add_argument(
        '--session-id', '--resume', dest='session_id', metavar='ID',
        help='Continue a persistent session by id (the same store as `effgen '
             'chat --session-id` and `effgen sessions list`). Prior turns are '
             'recalled and new ones saved, along with the workspace, files in '
             'context and files written; a new id starts a fresh session.',
    )
    code_parser.add_argument('--commit', action='store_true',
                             help='After the run, offer to commit the files it '
                                  'wrote (y/N; needs --yes without a terminal). '
                                  'Only those files are committed, and it never '
                                  'pushes, amends, resets or discards your work.')
    code_parser.add_argument('--commit-message', metavar='MSG',
                             help='Commit message for --commit (default: a message '
                                  'naming the changed files and the task)')
    code_parser.add_argument('--undo', action='store_true',
                             help='Reverse the last applied edit(s) in the '
                                  'workspace instead of running a task, restoring '
                                  'the previous file content')
    code_parser.add_argument('--undo-count', type=int, default=1, metavar='N',
                             help='With --undo, how many recent edits to reverse '
                                  '(default 1)')
    code_parser.add_argument('--max-iterations', type=int,
                             help='Iteration cap for the plan/run/fix loop '
                                  '(default: the coding preset\'s)')
    code_parser.add_argument('--temperature', type=float, help='Temperature')
    code_parser.add_argument('--max-tokens', type=int,
                             help='Max output tokens per call (raise for reasoning '
                                  'models that spend part of the budget before any '
                                  'visible text)')
    code_parser.add_argument('--json', dest='output_json', action='store_true',
                             help='Emit the result as one JSON document on stdout '
                                  '(answer, files_written, diffs, actions, tokens, '
                                  'cost). Human output goes to stderr.')
    code_parser.add_argument('-v', '--verbose', action='store_true', default=argparse.SUPPRESS,
                             help='Verbose output (show DEBUG/INFO logs)')
    code_parser.add_argument('-q', '--quiet', action='store_true', default=argparse.SUPPRESS,
                             help='Quiet output (answer only)')
    code_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                             help='Disable live spinners/progress animation')

def add_quickstart_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen quickstart`` and its ``tutorial`` alias from one block."""
    # documented alias so both names a newcomer might try lead to the same
    # guided run rather than one dead-ending.
    _qs_help = {
        'quickstart': 'Guided first run: pick a model, run an agent, then write and run code',
        'tutorial': 'Alias of quickstart — the same guided first run',
    }
    for _qs_name in ('quickstart', 'tutorial'):
        qs_parser = subparsers.add_parser(
            _qs_name,
            help=_qs_help[_qs_name],
            description=_qs_help['quickstart']
            + ('.  (`effgen tutorial` is an alias of `effgen quickstart`.)'
               if _qs_name == 'tutorial' else '.')
            + '  The coding step writes and runs a small program inside '
              '~/.effgen/quickstart-code; the files it changes stay there.'
              '  `--init DIR` instead scaffolds a project in DIR — a config, an '
              '.env template, a runnable example and the next three commands — '
              'and makes no model call.',
        )
        qs_parser.add_argument('--init', nargs='?', const='.', metavar='DIR',
                               help='Scaffold a project in DIR (default: the current '
                                    'directory): effgen.yaml, .env.example, example.py '
                                    'and .gitignore, plus a daily spend cap when none '
                                    'is set. Prompts for nothing and calls no model; '
                                    'every other quickstart step is skipped.')
        qs_parser.add_argument('--force', action='store_true',
                               help='With --init, overwrite a scaffolded file that is '
                                    'already there (default: keep it)')
        qs_parser.add_argument('--budget', type=float, metavar='USD',
                               help='With --init, the daily spend cap to set when none '
                                    'is configured (default: 1.00; 0 sets none). An '
                                    'existing cap is never changed.')
        qs_parser.add_argument('-m', '--model', help='Model to use (skips the model prompt)')
        qs_parser.add_argument('--provider', help='Provider for a bare model id')
        qs_parser.add_argument('--task', help='Task to run (defaults to a sample task)')
        qs_parser.add_argument('-y', '--yes', action='store_true',
                               help='Run non-interactively with sensible defaults')
        qs_parser.add_argument('--code', action='store_true',
                               help='Include the coding step without asking (needed '
                                    'with --yes, which otherwise skips it)')
        qs_parser.add_argument('--no-code', dest='no_code', action='store_true',
                               help='Skip the coding step')

def add_debug_parser(subparsers: argparse._SubParsersAction, *, preset_choices: list[str]) -> None:
    """Declare ``effgen debug`` — run an agent in interactive debug mode."""
    debug_parser = subparsers.add_parser('debug', help='Run an agent in interactive debug mode')
    debug_parser.add_argument('task', help='Task to execute')
    debug_parser.add_argument('-m', '--model', help='Model to use')
    debug_parser.add_argument('--provider',
                              help='Provider for a bare model id (e.g. groq). '
                                   'Equivalent to the provider:model prefix.')
    debug_parser.add_argument('--preset', choices=preset_choices,
                              help='Use a preset agent configuration')
    debug_parser.add_argument('--step', action='store_true', help='Step through each iteration')
