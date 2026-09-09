"""Interactive REPL and non-interactive CLI for exploring prompt templates.
REPL commands:
  select <name>           — pick a prompt from the registry
  set <key> <value>       — bind a variable (JSON-decoded if possible)
  unset <key>             — remove a variable binding
  render                  — print the rendered prompt using current vars
  run [--model <id>] [--max-tokens N] [--temperature T]
                          — run rendered prompt through a model
  save [<path>]           — save session to JSON
  load <path>             — restore a saved session
  reload                  — re-import the template module (hot-reload)
  list [--domain D]       — list registered prompts
  show <name>             — show prompt details
  help                    — show this help
  exit / quit / Ctrl-D    — exit the REPL

Non-interactive (CLI):
  effgen prompts render <name> --input <file.json>
  effgen prompts run <name> --input <file.json> --model <id>
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from effgen.prompts.library.base import LibraryPrompt

if TYPE_CHECKING:
    from effgen.prompts.library.eval import RunOutput
from effgen.prompts.library.registry import registry
from effgen.prompts.library.session import PlaygroundSession

try:
    from rich.panel import Panel

    from effgen.ui.theme import get_console as _get_console
    _RICH = True
    _console = _get_console()
except ImportError:
    _RICH = False
    _console = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print(msg: str) -> None:
    if _RICH and _console:
        _console.print(msg, highlight=False)
    else:
        print(msg)


def _plain(text: str) -> str:
    """Neutralize console-markup syntax in text that came from a model or a file.

    Rendered prompts, model output and error details routinely contain square
    brackets. Left unescaped, ``[empty result]`` is consumed as a style tag and
    disappears, and a ``[/…]`` sequence aborts the command with a markup error
    instead of showing the answer.
    """
    if not (_RICH and _console):
        return text
    from rich.markup import escape

    return escape(text)


def _print_prompt(text: str) -> None:
    if _RICH and _console:
        _console.print(Panel(_plain(text), title="Rendered Prompt", border_style="green"))
    else:
        print("\n--- Rendered Prompt ---")
        print(text)
        print("--- End ---")


def _print_output(text: str, model: str) -> None:
    if _RICH and _console:
        _console.print(
            Panel(_plain(text), title=f"Model Output ({model})", border_style="blue")
        )
    else:
        print(f"\n--- Model Output ({model}) ---")
        print(text)
        print("--- End ---")


def _print_err(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[red]Error:[/red] {_plain(msg)}", highlight=False)
    else:
        print(f"Error: {msg}", file=sys.stderr)


def _print_warn(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[yellow]Warning:[/yellow] {_plain(msg)}", highlight=False)
    else:
        print(f"Warning: {msg}")


def _print_ok(msg: str) -> None:
    if _RICH and _console:
        _console.print(f"[green]{_plain(msg)}[/green]", highlight=False)
    else:
        print(msg)


def _resolve_prompt(name: str) -> tuple[LibraryPrompt, str]:
    """Resolve exact prompt names and base names with a clear default variant."""
    try:
        return registry.get(name), name
    except KeyError as exc:
        candidates = [p for p in registry.all() if p.name.startswith(f"{name}.")]
        if not candidates:
            raise

        zero_shot = [p for p in candidates if p.variant == "zero_shot"]
        if len(zero_shot) == 1:
            prompt = zero_shot[0]
            return prompt, prompt.name
        if len(candidates) == 1:
            prompt = candidates[0]
            return prompt, prompt.name

        options = ", ".join(p.name for p in sorted(candidates, key=lambda p: p.name))
        raise KeyError(f"Prompt '{name}' is ambiguous; use one of: {options}") from exc


def _suggest_prompt_names(name: str, n: int = 3) -> list[str]:
    """Closest registered prompt names to *name*, for a not-found error."""
    import difflib

    try:
        candidates = [p.name for p in registry.all()]
    except Exception:  # noqa: BLE001
        return []
    return difflib.get_close_matches(name, candidates, n=n, cutoff=0.5)


def _key_error_message(exc: KeyError, name: str | None = None) -> str:
    msg = str(exc.args[0]) if exc.args else str(exc)
    if name and "not found in registry" in msg:
        suggestions = _suggest_prompt_names(name)
        if suggestions:
            msg += f". Did you mean: {', '.join(suggestions)}?"
    return msg


# ---------------------------------------------------------------------------
# Variable parsing
# ---------------------------------------------------------------------------

def _parse_value(raw: str) -> Any:
    """Try JSON decoding, fall back to plain string."""
    stripped = raw.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _split_set_args(line: str) -> tuple[str, str] | None:
    """Parse 'key value' from a `set` command line.

    The value part may be a JSON literal or a bare string (possibly
    space-separated after the key).  We peel off the first token as the
    key and treat everything after it as the raw value.
    """
    parts = line.split(None, 1)
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Hot-reload helper
# ---------------------------------------------------------------------------

def _reload_prompt(name: str) -> bool:
    """Re-import the module that registered *name*.

    Walk sys.modules for any module under effgen.prompts.library.domains
    whose file contains the prompt's domain, and reload it.  After reload
    the registry should contain the updated template.
    """
    try:
        p = registry.get(name)
    except KeyError:
        _print_err(f"Prompt '{name}' not in registry")
        return False

    domain = p.domain
    domain_pkg = f"effgen.prompts.library.domains.{domain}"
    reloaded: list[str] = []
    for modname, mod in list(sys.modules.items()):
        if not (modname == domain_pkg or modname.startswith(domain_pkg + ".")):
            continue
        try:
            importlib.reload(mod)
            reloaded.append(modname)
        except Exception as exc:
            _print_warn(f"Reload of {modname} raised: {exc}")

    if reloaded:
        _print_ok(f"Reloaded: {', '.join(reloaded)}")
        # Force registry re-discovery next access
        registry._discovered = False
        return True

    _print_warn("No matching modules found to reload")
    return False


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------

def _run_prompt(
    rendered: str,
    model: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> "RunOutput":
    """Call a model via effgen's model loader and return text + metadata."""
    from effgen.prompts.library.eval import PromptEval
    evaluator = PromptEval()
    return evaluator.run_model(
        rendered, model, max_tokens=max_tokens, temperature=temperature
    )


def _print_run_footer(result: "RunOutput") -> None:
    """Print a one-line token/cost/latency footer under a model output."""
    parts: list[str] = []
    if result.prompt_tokens is not None and result.completion_tokens is not None:
        total = result.prompt_tokens + result.completion_tokens
        parts.append(
            f"tokens: {total} ({result.prompt_tokens} in / {result.completion_tokens} out)"
        )
    elif result.tokens_used is not None:
        parts.append(f"tokens: {result.tokens_used}")
    if result.cost_usd is not None:
        parts.append(f"cost: ${result.cost_usd:.6f}")
    if result.latency_ms is not None:
        parts.append(f"latency: {result.latency_ms:.0f} ms")
    if not parts:
        return
    line = "  ·  ".join(parts)
    if _RICH and _console:
        _console.print(f"[dim]{line}[/dim]", highlight=False)
    else:
        print(line)


def _print_shape_verdict(prompt: LibraryPrompt, output: str) -> None:
    """Print whether *output* parses / matches the template's expected_shape.

    Runs for templates that declare an ``expected_shape`` or a ``structured``
    variant, so a structured-output author sees the same pass/fail signal the
    eval harness uses without a second command.
    """
    from effgen.prompts.library.eval import PromptEval

    spec = prompt.expected_shape
    if spec is None and prompt.variant != "structured":
        return
    if spec is None:
        # A structured template with no explicit shape: check it is JSON.
        parsed = PromptEval._extract_json_object(output)
        if parsed is not None:
            _print_ok("✓ valid JSON")
        else:
            _print_warn("output did not parse as a JSON object")
        return
    ok, msg = PromptEval._check_shape(output, spec)
    if ok:
        _print_ok("✓ output matches expected_shape")
    else:
        _print_warn(f"output does not match expected_shape: {msg}")


def _report_empty_result(result: "RunOutput", model: str) -> None:
    """Show a marker in place of a blank panel for an empty/truncated result."""
    if result.truncated:
        cap = result.max_tokens
        cap_note = f" (currently {cap})" if cap is not None else ""
        detail = (
            "the output-token budget was spent before any visible text "
            f"(finish_reason={result.finish_reason!r}). "
            f"Raise the cap with --max-tokens{cap_note}."
        )
    else:
        detail = "the model returned no text."
    _print_output(f"[empty result] {detail}", model)
    _print_err(f"Model returned no usable output — {detail}")


def _report_truncated_result(result: "RunOutput") -> None:
    """Flag an answer the model was still writing when the budget ran out."""
    cap = result.max_tokens
    cap_note = f" (currently {cap})" if cap is not None else ""
    _print_err(
        "Output is incomplete — the model stopped at the token budget "
        f"(finish_reason={result.finish_reason!r}). Raise the cap with "
        f"--max-tokens{cap_note} to get the full answer."
    )


# ---------------------------------------------------------------------------
# Non-interactive entry-points
# ---------------------------------------------------------------------------

def _unknown_input_keys(p: LibraryPrompt, inputs: dict[str, Any]) -> list[str]:
    """Supplied keys the template can't accept — not in the schema's declared
    ``properties`` and not absorbed by a ``**kwargs`` render signature.

    Templates render with a fixed keyword signature, so an undeclared key would
    otherwise reach ``template(**inputs)`` and raise a raw ``TypeError`` naming
    the private render function. Returns the offending keys so the caller can
    report a schema-anchored message instead.
    """
    import inspect

    props = (p.input_schema or {}).get("properties")
    if not isinstance(props, dict) or not props:
        return []
    try:
        sig = inspect.signature(p.template)
        if any(
            par.kind == inspect.Parameter.VAR_KEYWORD
            for par in sig.parameters.values()
        ):
            return []
    except (TypeError, ValueError):
        pass
    return [k for k in inputs if k not in props]


def _report_unknown_keys(p: LibraryPrompt, unknown: list[str]) -> None:
    valid = ", ".join(sorted((p.input_schema or {}).get("properties", {}))) or "—"
    for key in unknown:
        _print_err(f"unknown input key '{key}'; valid keys: {valid}")
    _print(f"Run 'effgen prompts show {p.name}' to see the schema and a valid fixture example.")


def _validate_or_report(p: LibraryPrompt, inputs: dict[str, Any]) -> bool:
    """Validate a non-empty --input object against the prompt's input_schema.

    Returns True when it's safe to render. An empty *inputs* dict means no
    --input was supplied (the fixture alone drives the render), so it skips
    validation. Reports one message per violation and leaves rendering to
    the caller to skip.
    """
    if not inputs:
        return True
    unknown = _unknown_input_keys(p, inputs)
    if unknown:
        _report_unknown_keys(p, unknown)
        return False
    errors = p.validate_input(inputs)
    if not errors:
        return True
    _print_err(f"Input for '{p.name}' does not match its input_schema:")
    for msg in errors:
        _print_err(f"  {msg}")
    _print(f"Run 'effgen prompts show {p.name}' to see the schema and a valid fixture example.")
    return False


def cmd_render(name: str, inputs: dict[str, Any]) -> int:
    """Non-interactive render: print rendered prompt to stdout."""
    try:
        p, _resolved_name = _resolve_prompt(name)
    except KeyError as exc:
        _print_err(_key_error_message(exc, name))
        return 1
    if not _validate_or_report(p, inputs):
        return 1
    try:
        merged = {**p.fixture, **inputs}
        rendered = p.template(**merged)
    except Exception as exc:
        _print_err(f"Render failed: {exc}")
        return 1
    _print_prompt(rendered)
    return 0


def cmd_run(
    name: str,
    inputs: dict[str, Any],
    model: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> int:
    """Non-interactive run: render + call model + print output.

    Returns non-zero when the model produces no usable text (an empty answer or
    a budget-exhausted/truncated reasoning result), so a scripted caller can
    tell an empty billed result from a good one.

    Args:
        name: The library prompt to render.
        inputs: Values for the prompt's declared inputs.
        model: The model id to call.
        max_tokens: Output cap for the call.
        temperature: Sampling temperature for the call.
    """
    try:
        p, _resolved_name = _resolve_prompt(name)
    except KeyError as exc:
        _print_err(_key_error_message(exc, name))
        return 1
    if not _validate_or_report(p, inputs):
        return 1
    try:
        merged = {**p.fixture, **inputs}
        rendered = p.template(**merged)
    except Exception as exc:
        _print_err(f"Render failed: {exc}")
        return 1
    _print_prompt(rendered)
    try:
        result = _run_prompt(
            rendered, model, max_tokens=max_tokens, temperature=temperature
        )
    except Exception as exc:
        _print_err(f"Model call failed: {exc}")
        return 1
    if result.is_empty:
        _report_empty_result(result, model)
        return 1
    _print_output(result.text, model)
    _print_run_footer(result)
    _print_shape_verdict(p, result.text)
    if result.truncated:
        _report_truncated_result(result)
        return 1
    return 0


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

_HELP = """\
REPL commands:
  select <name>           pick a prompt from the registry
  set <key> <value>       bind a variable (JSON or plain string)
  unset <key>             remove a variable
  render                  print the rendered prompt
  run [--model <id>] [--max-tokens N] [--temperature T]
                          run rendered prompt through a model
  save [<path>]           save session to JSON
  load <path>             restore a saved session
  reload                  hot-reload the template module
  list [--domain <d>]     list registered prompts
  show <name>             show prompt details
  help                    show this help
  exit / quit / Ctrl-D    exit
"""


class PlaygroundREPL:
    """Interactive REPL for the prompt playground."""

    def __init__(self, default_model: str | None = None) -> None:
        self.session = PlaygroundSession()
        # A run with no --model resolves a broadly-available model (a keyed
        # cloud model if one is configured, else a small local one) at call
        # time, rather than defaulting to a provider a newcomer may not have a
        # key for.
        self.default_model = default_model
        self._running = False

    def _resolve_run_model(self) -> tuple[str | None, str | None]:
        """Return ``(model_id, note)`` for a run with no explicit --model."""
        if self.default_model:
            return self.default_model, None
        try:
            from effgen.cli._main import _quickstart_suggest_model

            model_id, _provider, reason = _quickstart_suggest_model()
            return model_id, reason
        except Exception:  # noqa: BLE001
            return None, None

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Start the interactive REPL loop.  Returns exit code."""
        self._running = True
        _print("[bold cyan]effGen Prompt Playground[/bold cyan]" if _RICH else "effGen Prompt Playground")
        _print("Type [bold]help[/bold] for commands, [bold]exit[/bold] to quit." if _RICH else "Type 'help' for commands, 'exit' to quit.")
        _print("")
        try:
            while self._running:
                try:
                    line = self._readline()
                except EOFError:
                    _print("\nBye!")
                    break
                line = line.strip()
                if not line:
                    continue
                self._dispatch(line)
        except KeyboardInterrupt:
            _print("\nInterrupted. Bye!")
        return 0

    def _readline(self) -> str:
        prompt_label = f"[{self.session.prompt_name or '(none)'}]> "
        try:
            return input(prompt_label)
        except EOFError:
            raise

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, line: str) -> None:
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        dispatch = {
            "select": self._cmd_select,
            "set": self._cmd_set,
            "unset": self._cmd_unset,
            "render": lambda _: self._cmd_render(),
            "run": self._cmd_run,
            "save": self._cmd_save,
            "load": self._cmd_load,
            "reload": lambda _: self._cmd_reload(),
            "list": self._cmd_list,
            "show": self._cmd_show,
            "help": lambda _: _print(_HELP),
            "exit": lambda _: self._exit(),
            "quit": lambda _: self._exit(),
        }

        handler = dispatch.get(cmd)
        if handler is None:
            _print_err(f"Unknown command '{cmd}'. Type 'help' for a list.")
            return
        handler(rest)

    def _exit(self) -> None:
        self._running = False
        _print("Bye!")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_select(self, rest: str) -> None:
        name = rest.strip()
        if not name:
            _print_err("Usage: select <prompt-name>")
            return
        try:
            p, resolved_name = _resolve_prompt(name)
        except KeyError as exc:
            _print_err(f"{_key_error_message(exc, name)}. Try 'list' to see available prompts.")
            return
        self.session.prompt_name = resolved_name
        self.session.variables = dict(p.fixture)  # seed with fixture defaults
        _print_ok(f"Selected: {resolved_name}")
        _print(f"  Domain: {p.domain}  Variant: {p.variant}")
        _print(f"  Pre-loaded fixture inputs: {list(p.fixture.keys())}")

    def _cmd_set(self, rest: str) -> None:
        parsed = _split_set_args(rest)
        if parsed is None:
            _print_err("Usage: set <key> <value>")
            return
        key, raw_value = parsed
        value = _parse_value(raw_value)
        self.session.set_var(key, value)
        _print_ok(f"Set {key} = {json.dumps(value)}")

    def _cmd_unset(self, rest: str) -> None:
        key = rest.strip()
        if not key:
            _print_err("Usage: unset <key>")
            return
        if self.session.unset_var(key):
            _print_ok(f"Unset {key}")
        else:
            _print_warn(f"Variable '{key}' was not set")

    def _cmd_render(self) -> None:
        if not self.session.prompt_name:
            _print_err("No prompt selected. Use 'select <name>' first.")
            return
        try:
            p = registry.get(self.session.prompt_name)
        except KeyError:
            _print_err(f"Prompt '{self.session.prompt_name}' not in registry")
            return
        unknown = _unknown_input_keys(p, self.session.variables)
        if unknown:
            _report_unknown_keys(p, unknown)
            return
        try:
            rendered = p.template(**self.session.variables)
        except Exception as exc:
            _print_err(f"Render failed: {exc}")
            return
        self.session.add_render(rendered)
        _print_prompt(rendered)

    @staticmethod
    def _parse_run_args(rest: str) -> tuple[str | None, int | None, float | None]:
        """Parse ``run`` flags: ``--model/-m``, ``--max-tokens``, ``--temperature``."""
        model: str | None = None
        max_tokens: int | None = None
        temperature: float | None = None
        args = rest.split()
        i = 0
        while i < len(args):
            token = args[i]
            has_next = i + 1 < len(args)
            if token in ("--model", "-m") and has_next:
                model = args[i + 1]
                i += 2
            elif token == "--max-tokens" and has_next:
                try:
                    max_tokens = int(args[i + 1])
                except ValueError:
                    _print_warn(f"Ignoring non-integer --max-tokens '{args[i + 1]}'")
                i += 2
            elif token == "--temperature" and has_next:
                try:
                    temperature = float(args[i + 1])
                except ValueError:
                    _print_warn(f"Ignoring non-numeric --temperature '{args[i + 1]}'")
                i += 2
            else:
                i += 1
        return model, max_tokens, temperature

    def _cmd_run(self, rest: str) -> None:
        if not self.session.prompt_name:
            _print_err("No prompt selected. Use 'select <name>' first.")
            return

        model, max_tokens, temperature = self._parse_run_args(rest)
        note = None
        if model is None:
            model, note = self._resolve_run_model()
        if model is None:
            _print_err(
                "No model specified and none configured. "
                "Use 'run --model <id>' (e.g. run --model groq:openai/gpt-oss-20b)."
            )
            return

        try:
            p = registry.get(self.session.prompt_name)
        except KeyError:
            _print_err(f"Prompt '{self.session.prompt_name}' not in registry")
            return
        unknown = _unknown_input_keys(p, self.session.variables)
        if unknown:
            _report_unknown_keys(p, unknown)
            return
        try:
            rendered = p.template(**self.session.variables)
        except Exception as exc:
            _print_err(f"Render failed: {exc}")
            return

        note_suffix = f" ({note})" if note else ""
        _print(f"Running with model: {model}{note_suffix} ...")
        try:
            result = _run_prompt(
                rendered, model, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as exc:
            _print_err(f"Model call failed: {exc}")
            return
        if result.is_empty:
            _report_empty_result(result, model)
            return
        self.session.add_run(model=model, rendered=rendered, output=result.text)
        _print_output(result.text, model)
        _print_run_footer(result)
        _print_shape_verdict(p, result.text)
        if result.truncated:
            _report_truncated_result(result)

    def _cmd_save(self, rest: str) -> None:
        path_str = rest.strip()
        path = Path(path_str) if path_str else None
        try:
            saved_path = self.session.save(path)
            _print_ok(f"Session saved to: {saved_path}")
        except Exception as exc:
            _print_err(f"Save failed: {exc}")

    def _cmd_load(self, rest: str) -> None:
        path_str = rest.strip()
        if not path_str:
            _print_err("Usage: load <path>")
            return
        path = Path(path_str)
        if not path.exists():
            _print_err(f"File not found: {path}")
            return
        try:
            self.session = PlaygroundSession.load(path)
            _print_ok(f"Session loaded: {self.session.summary()}")
        except Exception as exc:
            _print_err(f"Load failed: {exc}")

    def _cmd_reload(self) -> None:
        if not self.session.prompt_name:
            _print_err("No prompt selected. Use 'select <name>' first.")
            return
        _reload_prompt(self.session.prompt_name)

    def _cmd_list(self, rest: str) -> None:
        domain = None
        args = rest.split()
        i = 0
        while i < len(args):
            if args[i] == "--domain" and i + 1 < len(args):
                domain = args[i + 1]
                i += 2
            else:
                i += 1
        prompts = registry.search(domain=domain)
        if not prompts:
            _print("No prompts registered.")
            return
        _print(f"{'Name':<45} {'Domain':<12} {'Variant':<12} Description")
        _print("-" * 100)
        for p in prompts:
            _print(f"{p.name:<45} {p.domain:<12} {p.variant:<12} {p.description}")
        _print(f"\nTotal: {len(prompts)} prompt(s)")

    def _cmd_show(self, rest: str) -> None:
        name = rest.strip()
        if not name:
            _print_err("Usage: show <prompt-name>")
            return
        try:
            p, _resolved_name = _resolve_prompt(name)
        except KeyError as exc:
            _print_err(_key_error_message(exc, name))
            return
        _print(f"\nName:        {p.name}")
        _print(f"Domain:      {p.domain}")
        _print(f"Variant:     {p.variant}")
        _print(f"Description: {p.description}")
        _print(f"Tags:        {', '.join(p.tags) or '—'}")
        _print("\nInput Schema:")
        _print("  " + json.dumps(p.input_schema, indent=2).replace("\n", "\n  "))
        _print("\nFixture:")
        _print("  " + json.dumps(p.fixture, indent=2).replace("\n", "\n  "))
