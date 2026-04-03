"""
Visual Agent Demo — MLX on Apple Silicon

A Gradio app with two tabs:
  1. Agent Visualizer — watch the agent reason step-by-step
  2. Code Editor — view, edit, and run the agent code live

Uses LFM2.5's native tool call format on Apple Silicon via MLX.

Requirements:
    pip install gradio mlx-lm

Usage:
    python examples/basic/agent_viz_mlx.py --autoload
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import re
import sys
import textwrap
import traceback
from typing import Iterator

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Tools — OpenAI-format schemas + Python callables
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, log, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '98765 * 43210'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_today",
            "description": "Get today's date in ISO format.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "string_length",
            "description": "Count the number of characters in a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The string to measure"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reverse_string",
            "description": "Reverse a given string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "String to reverse"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_convert",
            "description": "Convert between units. Supports: celsius/fahrenheit, miles/km, kg/lbs, meters/feet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]


def _safe_eval(expr: str) -> str:
    allowed = {"__builtins__": {}}
    safe_ns = {
        "math": math, "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt, "log": math.log,
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
    }
    return str(eval(expr, allowed, safe_ns))


TOOL_HANDLERS = {
    "calculator": lambda **kw: _safe_eval(kw.get("expression", "0")),
    "date_today": lambda **kw: datetime.date.today().isoformat(),
    "string_length": lambda **kw: str(len(kw.get("text", ""))),
    "reverse_string": lambda **kw: kw.get("text", "")[::-1],
    "unit_convert": lambda **kw: _unit_convert(
        float(kw.get("value", 0)), kw.get("from_unit", ""), kw.get("to_unit", "")
    ),
}


def _unit_convert(val: float, src: str, dst: str) -> str:
    src, dst = src.lower(), dst.lower()
    conversions = {
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("miles", "km"): lambda v: v * 1.60934,
        ("km", "miles"): lambda v: v / 1.60934,
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v / 2.20462,
        ("meters", "feet"): lambda v: v * 3.28084,
        ("feet", "meters"): lambda v: v / 3.28084,
    }
    fn = conversions.get((src, dst))
    if fn is None:
        return f"Unknown conversion: {src} -> {dst}"
    return f"{fn(val):.4f} {dst}"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None


def load_model(model_id: str) -> str:
    global _model, _tokenizer
    if _model is not None:
        del _model, _tokenizer
        _model = _tokenizer = None
        import gc; gc.collect()
    try:
        from mlx_lm import load
        _model, _tokenizer = load(model_id)
        return f"Loaded: {model_id}"
    except ImportError:
        return "Error: pip install mlx-lm"
    except Exception as e:
        return f"Error: {e}"


def generate_text(messages: list[dict], tools=None, max_tokens: int = 512) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        template_kwargs["tools"] = tools

    formatted = _tokenizer.apply_chat_template(messages, **template_kwargs)
    sampler = make_sampler(temp=0.1, top_p=0.1)
    return generate(
        _model, _tokenizer, prompt=formatted,
        max_tokens=max_tokens, sampler=sampler, verbose=False,
    )


# ---------------------------------------------------------------------------
# Tool Call Parsing (LFM2.5 native format)
# ---------------------------------------------------------------------------

def parse_tool_calls(text: str) -> list[dict]:
    """Parse <|tool_call_start|>[fn(arg=val)]<|tool_call_end|>"""
    calls = []
    pattern = r"<\|tool_call_start\|>\s*\[(.+?)\]\s*<\|tool_call_end\|>"
    for match in re.finditer(pattern, text, re.DOTALL):
        inner = match.group(1).strip()
        fn_match = re.match(r"(\w+)\((.+)\)", inner, re.DOTALL)
        if fn_match:
            name = fn_match.group(1)
            args_str = fn_match.group(2)
            arguments = {}
            for arg_match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
                arguments[arg_match.group(1)] = arg_match.group(2)
            for arg_match in re.finditer(r'(\w+)\s*=\s*([0-9.]+)(?=[,\s\)]|$)', args_str):
                if arg_match.group(1) not in arguments:
                    arguments[arg_match.group(1)] = arg_match.group(2)
            calls.append({"name": name, "arguments": arguments})
    return calls


def extract_text_before_tool_call(text: str) -> str:
    idx = text.find("<|tool_call_start|>")
    return text[:idx].strip() if idx > 0 else ""


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

def run_agent(task: str, max_steps: int = 6) -> Iterator[list[dict]]:
    """Run the agent, yielding step list after each step."""
    if _model is None:
        yield [{"type": "error", "step": 0, "content": "Model not loaded."}]
        return

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools to answer questions accurately. Always use the calculator for math."},
        {"role": "user", "content": task},
    ]
    steps: list[dict] = []

    for step_num in range(1, max_steps + 1):
        response = generate_text(messages, tools=TOOL_SCHEMAS)
        tool_calls = parse_tool_calls(response)
        reasoning = extract_text_before_tool_call(response)

        if reasoning:
            steps.append({"type": "thought", "step": step_num, "content": reasoning})
            yield list(steps)

        if tool_calls:
            for tc in tool_calls:
                args_display = ", ".join(f'{k}="{v}"' for k, v in tc["arguments"].items())
                steps.append({"type": "action", "step": step_num, "content": f'{tc["name"]}({args_display})'})
                yield list(steps)

                handler = TOOL_HANDLERS.get(tc["name"])
                try:
                    result = handler(**tc["arguments"]) if handler else f"Unknown tool: {tc['name']}"
                except Exception as e:
                    result = f"Error: {e}"

                steps.append({"type": "observation", "step": step_num, "content": str(result)})
                yield list(steps)

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "tool", "content": str(result)})
        else:
            # Check plain-text tool call: [fn(...)]
            plain_call = re.search(r"\[(\w+)\((.*?)\)\]", response)
            if plain_call:
                name, args_str = plain_call.group(1), plain_call.group(2).strip()
                arguments = {}
                for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
                    arguments[m.group(1)] = m.group(2)

                steps.append({"type": "action", "step": step_num, "content": f"{name}({args_str})"})
                yield list(steps)

                handler = TOOL_HANDLERS.get(name)
                result = handler(**arguments) if handler else f"Unknown tool: {name}"
                steps.append({"type": "observation", "step": step_num, "content": str(result)})
                yield list(steps)

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "tool", "content": str(result)})
                continue

            # Final answer
            clean = response.strip()
            for tok in ["<|im_end|>", "<|endoftext|>", "<|tool_call_start|>", "<|tool_call_end|>"]:
                clean = clean.replace(tok, "")
            steps.append({"type": "answer", "step": step_num, "content": clean.strip() or "(empty)"})
            yield list(steps)
            return

    steps.append({"type": "error", "step": max_steps, "content": f"Max steps ({max_steps}) reached."})
    yield list(steps)


# ---------------------------------------------------------------------------
# Step Renderer
# ---------------------------------------------------------------------------

def render_steps(steps: list[dict]) -> str:
    STYLES = {
        "thought":     ("Thought",      "#e8f4fd", "#1976d2"),
        "action":      ("Tool Call",    "#fff3e0", "#e65100"),
        "observation": ("Result",       "#e8f5e9", "#2e7d32"),
        "answer":      ("Final Answer", "#f3e5f5", "#7b1fa2"),
        "error":       ("Error",        "#ffebee", "#c62828"),
    }
    if not steps:
        return ""
    parts = []
    for step in steps:
        label, bg, color = STYLES.get(step["type"], ("Step", "#f5f5f5", "#333"))
        content = step["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        is_answer = step["type"] == "answer"
        parts.append(f"""
        <div style="margin:8px 0;padding:12px 16px;border-radius:8px;background:{bg};
                    border-left:4px solid {color};
                    {'box-shadow:0 2px 8px rgba(0,0,0,0.1);' if is_answer else ''}">
            <div style="font-weight:600;color:{color};margin-bottom:4px;font-size:12px;
                        text-transform:uppercase;letter-spacing:0.5px;">
                Step {step['step']} &mdash; {label}
            </div>
            <div style="color:#333;font-size:{'16px' if is_answer else '14px'};line-height:1.5;
                        font-weight:{'600' if is_answer else '400'};">
                {content}
            </div>
        </div>""")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Default code snippet for the editor
# ---------------------------------------------------------------------------

DEFAULT_CODE = textwrap.dedent('''\
    # === effGen MLX Agent — Editable Code ===
    # Modify this code and click "Run Code" to execute it.
    # The model is pre-loaded as `model` and `tokenizer`.
    # Available: generate_text(), parse_tool_calls(), TOOL_HANDLERS, TOOL_SCHEMAS

    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    # --- 1. Define your prompt ---
    task = "What is 2**16 + 42?"

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": task},
    ]

    # --- 2. Apply chat template with tools ---
    formatted = tokenizer.apply_chat_template(
        messages, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True
    )
    print("=== Formatted Prompt (last 300 chars) ===")
    print(formatted[-300:])
    print()

    # --- 3. Generate ---
    sampler = make_sampler(temp=0.1, top_p=0.1)
    response = generate(model, tokenizer, prompt=formatted, max_tokens=200, sampler=sampler, verbose=False)
    print("=== Raw Model Response ===")
    print(response)
    print()

    # --- 4. Parse tool calls ---
    tool_calls = parse_tool_calls(response)
    if tool_calls:
        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            print(f"=== Tool Call: {name}({args}) ===")
            handler = TOOL_HANDLERS.get(name)
            if handler:
                result = handler(**args)
                print(f"Result: {result}")
            else:
                print(f"Unknown tool: {name}")
            print()

        # --- 5. Feed result back and get final answer ---
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "tool", "content": str(result)})
        formatted2 = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True
        )
        final = generate(model, tokenizer, prompt=formatted2, max_tokens=200, sampler=sampler, verbose=False)
        print("=== Final Answer ===")
        print(final)
    else:
        print("=== Direct Answer (no tool call) ===")
        print(response)
''')


def run_user_code(code: str) -> str:
    """Execute user code with model/tokenizer in scope. Returns captured output."""
    if _model is None:
        return "Error: Model not loaded. Click 'Load Model' first."

    import io
    import contextlib

    # Build execution namespace with useful references
    exec_globals = {
        "__builtins__": __builtins__,
        "model": _model,
        "tokenizer": _tokenizer,
        "generate_text": generate_text,
        "parse_tool_calls": parse_tool_calls,
        "extract_text_before_tool_call": extract_text_before_tool_call,
        "TOOL_SCHEMAS": TOOL_SCHEMAS,
        "TOOL_HANDLERS": TOOL_HANDLERS,
        "run_agent": run_agent,
        "render_steps": render_steps,
        "math": math,
        "re": re,
        "datetime": datetime,
    }

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, exec_globals)
    except Exception:
        stderr_capture.write(traceback.format_exc())

    output = stdout_capture.getvalue()
    errors = stderr_capture.getvalue()

    result = ""
    if output:
        result += output
    if errors:
        if result:
            result += "\n"
        result += f"--- ERRORS ---\n{errors}"

    return result or "(No output)"


# ---------------------------------------------------------------------------
# Code snippet templates
# ---------------------------------------------------------------------------

CODE_SNIPPETS = {
    "Basic generation": textwrap.dedent('''\
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = "Explain what unified memory means in 2 sentences."
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampler = make_sampler(temp=0.1, top_p=0.1)
        result = generate(model, tokenizer, prompt=formatted, max_tokens=200, sampler=sampler, verbose=False)
        print(result)
    '''),
    "Tool call + execution": DEFAULT_CODE,
    "Multi-step agent loop": textwrap.dedent('''\
        # Run the full agent loop and print each step
        task = "What is sqrt(625) * 3? Then convert the result from miles to km."

        for steps in run_agent(task, max_steps=6):
            pass  # let it complete

        for s in steps:
            step_type = s["type"].upper()
            print(f"[{step_type:12}] {s['content']}")
    '''),
    "Custom tool": textwrap.dedent('''\
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # Define a custom tool
        my_tools = [
            {
                "type": "function",
                "function": {
                    "name": "fibonacci",
                    "description": "Get the nth Fibonacci number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {"type": "integer", "description": "Which Fibonacci number (0-indexed)"}
                        },
                        "required": ["n"],
                    },
                },
            }
        ]

        def fib(n):
            a, b = 0, 1
            for _ in range(int(n)):
                a, b = b, a + b
            return a

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the 20th Fibonacci number?"},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tools=my_tools, tokenize=False, add_generation_prompt=True
        )

        sampler = make_sampler(temp=0.1)
        response = generate(model, tokenizer, prompt=formatted, max_tokens=100, sampler=sampler, verbose=False)
        print("Model response:", response)

        # Parse and execute
        calls = parse_tool_calls(response)
        if calls:
            n = calls[0]["arguments"].get("n", "10")
            result = fib(n)
            print(f"fibonacci({n}) = {result}")
        else:
            print("(Model answered directly)")
    '''),
    "Inspect chat template": textwrap.dedent('''\
        # See exactly what the model receives as input
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 42 * 99?"},
        ]

        # Without tools
        plain = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("=== Without tools ===")
        print(plain)
        print()

        # With tools
        with_tools = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, tokenize=False, add_generation_prompt=True
        )
        print("=== With tools ===")
        print(with_tools)
    '''),
}


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app(default_model: str):
    import gradio as gr

    with gr.Blocks(title="effGen Agent Visualizer") as app:
        gr.Markdown("# effGen — Agent Visualizer + Code Editor")

        # --- Model bar (shared across tabs) ---
        with gr.Row():
            model_id = gr.Textbox(value=default_model, label="Model ID", scale=4)
            load_btn = gr.Button("Load Model", variant="primary", scale=1)
            status = gr.Textbox(label="Status", interactive=False, scale=2)
        load_btn.click(fn=load_model, inputs=[model_id], outputs=[status])

        # --- Tabs ---
        with gr.Tabs():

            # ==================== TAB 1: Agent Visualizer ====================
            with gr.Tab("Agent Visualizer"):
                gr.Markdown(
                    "Watch the agent reason step-by-step. "
                    "**Tools:** calculator, date_today, string_length, reverse_string, unit_convert"
                )

                with gr.Row():
                    task_input = gr.Textbox(
                        placeholder="Ask something that needs tools...",
                        label="Task", scale=4, lines=2,
                    )
                    run_btn = gr.Button("Run Agent", variant="primary", scale=1, size="lg")

                gr.Examples(
                    examples=[
                        ["What is 98765 * 43210?"],
                        ["What is today's date?"],
                        ["What is sqrt(144) + 2^10?"],
                        ["Convert 100 celsius to fahrenheit."],
                        ["How many characters are in 'Hello World'? Then reverse it."],
                        ["What is 2^16? Then convert 65536 meters to feet."],
                    ],
                    inputs=[task_input],
                )

                steps_html = gr.HTML(
                    value='<div style="padding:40px;text-align:center;color:#aaa;font-size:15px;">'
                          'Click <b>Run Agent</b> to see reasoning steps.</div>',
                )

                def run_and_render(task):
                    for steps in run_agent(task):
                        yield render_steps(steps)

                run_btn.click(fn=run_and_render, inputs=[task_input], outputs=[steps_html])
                task_input.submit(fn=run_and_render, inputs=[task_input], outputs=[steps_html])

            # ==================== TAB 2: Code Editor ====================
            with gr.Tab("Code Editor"):
                gr.Markdown(
                    "Edit and run Python code with the loaded MLX model. "
                    "Available in scope: `model`, `tokenizer`, `generate_text()`, "
                    "`parse_tool_calls()`, `TOOL_SCHEMAS`, `TOOL_HANDLERS`, `run_agent()`."
                )

                # Snippet selector
                snippet_names = list(CODE_SNIPPETS.keys())
                snippet_dropdown = gr.Dropdown(
                    choices=snippet_names,
                    value=snippet_names[1],
                    label="Load snippet",
                    interactive=True,
                )

                code_editor = gr.Code(
                    value=DEFAULT_CODE,
                    language="python",
                    label="Python Code",
                    lines=30,
                )

                with gr.Row():
                    run_code_btn = gr.Button("Run Code", variant="primary", scale=1, size="lg")
                    clear_output_btn = gr.Button("Clear Output", scale=1)

                code_output = gr.Textbox(
                    value="",
                    label="Output",
                    lines=20,
                    interactive=False,
                )

                # Load snippet into editor
                def load_snippet(name):
                    return CODE_SNIPPETS.get(name, DEFAULT_CODE)

                snippet_dropdown.change(fn=load_snippet, inputs=[snippet_dropdown], outputs=[code_editor])

                # Run code
                run_code_btn.click(fn=run_user_code, inputs=[code_editor], outputs=[code_output])

                # Clear
                clear_output_btn.click(fn=lambda: "", outputs=[code_output])

    return app


def main():
    parser = argparse.ArgumentParser(description="Agent Visualizer + Code Editor")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--autoload", action="store_true")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("pip install gradio"); sys.exit(1)

    if args.autoload:
        print(f"Loading: {args.model}")
        print(load_model(args.model))

    build_app(args.model).launch(
        server_port=args.port, share=args.share,
        theme=gr.themes.Soft(),
        css="footer { display: none !important; }",
    )


if __name__ == "__main__":
    main()
