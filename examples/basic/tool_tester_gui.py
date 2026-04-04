"""
effGen Tool Tester — Interactive playground for testing all tools.

A Gradio web app that lets users:
  - Load a model (MLX, vLLM, Transformers, or API)
  - Browse all 14 built-in tools with descriptions and schemas
  - Fill in parameters visually and execute tools
  - Test tools through an agent with a loaded model
  - View results, errors, and execution timing
  - Inspect OpenAI-format tool schemas
  - Batch-test tools with multiple inputs

Requirements:
    pip install gradio mlx-lm

Usage:
    python examples/basic/tool_tester_gui.py
    python examples/basic/tool_tester_gui.py --port 7864 --share
    python examples/basic/tool_tester_gui.py --autoload
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import sys
import time
import traceback
from html import escape as _html_escape
from typing import Any, Iterator

from effgen.tools.base_tool import ToolResult

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalog & state
# ---------------------------------------------------------------------------

MODEL_CATALOG = {
    "MLX (Apple Silicon)": {
        "LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit": {"engine": "mlx", "desc": "LFM2.5 1.2B (8-bit) — fast, agentic"},
        "mlx-community/Qwen2.5-3B-Instruct-4bit": {"engine": "mlx", "desc": "Qwen2.5 3B (4-bit)"},
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit": {"engine": "mlx", "desc": "Mistral 7B (4-bit)"},
        "mlx-community/Qwen2-VL-2B-Instruct-4bit": {"engine": "mlx_vlm", "desc": "Qwen2-VL 2B Vision (4-bit)"},
    },
    "Local (vLLM / Transformers)": {
        "microsoft/Phi-3-mini-4k-instruct": {"engine": "auto", "desc": "Phi-3 Mini 3.8B"},
        "mistralai/Mistral-7B-Instruct-v0.2": {"engine": "auto", "desc": "Mistral 7B"},
        "meta-llama/Meta-Llama-3.1-8B-Instruct": {"engine": "auto", "desc": "Llama 3.1 8B"},
        "Qwen/Qwen2-7B-Instruct": {"engine": "auto", "desc": "Qwen2 7B"},
        "google/gemma-2b-it": {"engine": "auto", "desc": "Gemma 2B"},
    },
    "API Models": {
        "gpt-4-turbo-preview": {"engine": "api", "desc": "GPT-4 Turbo (OpenAI)"},
        "gpt-3.5-turbo": {"engine": "api", "desc": "GPT-3.5 Turbo (OpenAI)"},
        "claude-sonnet-4.5": {"engine": "api", "desc": "Claude Sonnet 4.5 (Anthropic)"},
        "claude-haiku-4": {"engine": "api", "desc": "Claude Haiku 4 (Anthropic)"},
        "gemini-pro": {"engine": "api", "desc": "Gemini Pro (Google)"},
    },
}

_model = None
_model_name = None
_agent = None


def _get_all_model_choices() -> list[str]:
    """Flatten model catalog into dropdown choices."""
    choices = []
    for group, models in MODEL_CATALOG.items():
        for model_id in models:
            choices.append(model_id)
    return choices


def _get_engine_for_model(model_id: str) -> str | None:
    """Look up the engine type for a model ID."""
    for group, models in MODEL_CATALOG.items():
        if model_id in models:
            eng = models[model_id]["engine"]
            return None if eng in ("auto", "api") else eng
    return None


def load_model_op(model_id: str, custom_model: str) -> str:
    """Load a model and return status string."""
    global _model, _model_name, _agent

    chosen = custom_model.strip() if custom_model.strip() else model_id
    if not chosen or not chosen.strip():
        return "Select a model first."

    # Unload previous
    if _model is not None:
        try:
            _model.unload()
        except Exception:
            pass
        _model = None
        _agent = None
        gc.collect()

    try:
        from effgen.models import load_model
        engine = _get_engine_for_model(chosen.strip())
        kwargs = {}
        if engine:
            kwargs["engine"] = engine

        _model = load_model(chosen.strip(), **kwargs)
        _model_name = chosen.strip()
        return f"Loaded: {_model_name} ({type(_model).__name__})"
    except Exception as e:
        return f"Error: {e}"


def unload_model_op() -> str:
    """Unload the current model."""
    global _model, _model_name, _agent
    if _model is not None:
        try:
            _model.unload()
        except Exception:
            pass
        _model = None
        _model_name = None
        _agent = None
        gc.collect()
        return "Model unloaded."
    return "No model loaded."


# ---------------------------------------------------------------------------
# Agent-based tool testing
# ---------------------------------------------------------------------------

def build_agent_with_tools(selected_tools: list[str]) -> str:
    """Build an agent with the selected tools and loaded model."""
    global _agent

    if _model is None:
        return "Load a model first."

    if not selected_tools:
        return "Select at least one tool."

    try:
        from effgen.core.agent import Agent, AgentConfig

        tools = []
        for tool_key in selected_tools:
            tool = _get_tool_instance(tool_key)
            if tool:
                tools.append(tool)

        config = AgentConfig(
            name="tool_tester_agent",
            model=_model,
            tools=tools,
            system_prompt="You are a helpful assistant. Use the provided tools to answer questions accurately. Always use the most appropriate tool for the task.",
            max_iterations=6,
            temperature=0.3,
            tool_calling_mode="auto",
        )

        _agent = Agent(config=config)

        tool_names = [t.name for t in tools]
        return (
            f"Agent ready with {len(tools)} tools: {', '.join(tool_names)}\n"
            f"Model: {_model_name}"
        )
    except Exception as e:
        return f"Error building agent: {e}"


def run_agent_with_prompt(prompt: str) -> Iterator[str]:
    """Run the agent on a prompt and stream the result."""
    if _agent is None:
        yield "Build an agent first (select tools and click 'Build Agent')."
        return
    if not prompt.strip():
        yield "Enter a prompt."
        return

    try:
        from effgen.core.agent import AgentResponse

        start = time.perf_counter()
        result = _agent.run(prompt)
        elapsed = (time.perf_counter() - start) * 1000

        output = f"--- Agent Response ({elapsed:.0f} ms) ---\n"

        if isinstance(result, AgentResponse):
            output += f"--- Success: {result.success} ---\n"
            output += f"--- Iterations: {result.iterations} | Tool Calls: {result.tool_calls} ---\n\n"

            if result.success:
                output += result.output
            else:
                output += f"AGENT ERROR: {result.output}\n"

            # Show execution trace for debugging
            if result.execution_trace:
                output += "\n\n--- Execution Trace ---\n"
                for entry in result.execution_trace:
                    role = entry.get("role", "unknown")
                    content = entry.get("content", "")
                    if isinstance(content, str) and len(content) > 300:
                        content = content[:300] + "..."
                    output += f"\n[{role}] {content}\n"

            if result.metadata:
                output += f"\n--- Metadata ---\n{json.dumps(result.metadata, indent=2, default=str)}"
        elif hasattr(result, "output"):
            output += f"\n{result.output}"
        else:
            output += f"\n{result}"

        yield output
    except Exception as e:
        yield f"Agent error: {e}\n\n{traceback.format_exc()}"

# ---------------------------------------------------------------------------
# Tool catalog — metadata for all built-in tools
# ---------------------------------------------------------------------------

TOOL_INFO = {
    "calculator": {
        "class": "Calculator",
        "desc": "Math calculations, expression evaluation, unit conversions, and statistics",
        "example_input": '{"expression": "sqrt(16) + 2**3", "operation": "calculate"}',
        "params": ["expression (str, required)", "operation (str: calculate|convert_units|statistics)", "from_unit (str)", "to_unit (str)", "precision (int, 0-15)"],
    },
    "python_repl": {
        "class": "PythonREPL",
        "desc": "Execute Python code in a persistent REPL with state across calls",
        "example_input": '{"code": "x = 42\\nprint(f\'The answer is {x}\')"}',
        "params": ["code (str, required)"],
    },
    "web_search": {
        "class": "WebSearch",
        "desc": "Search the web using DuckDuckGo (no API key needed)",
        "example_input": '{"query": "Python asyncio tutorial", "max_results": 3}',
        "params": ["query (str, required)", "max_results (int, default 5)"],
    },
    "code_executor": {
        "class": "CodeExecutor",
        "desc": "Run code in a sandboxed environment with timeout protection",
        "example_input": '{"code": "print(sum(range(10)))", "language": "python"}',
        "params": ["code (str, required)", "language (str, default python)", "timeout (int)"],
    },
    "file_operations": {
        "class": "FileOperations",
        "desc": "Read, write, list, and search files on the filesystem",
        "example_input": '{"operation": "list", "path": "."}',
        "params": ["operation (str: read|write|list|search|info)", "path (str, required)", "content (str)", "pattern (str)"],
    },
    "bash": {
        "class": "BashTool",
        "desc": "Execute shell commands safely with output capture",
        "example_input": '{"command": "echo Hello World && date"}',
        "params": ["command (str, required)", "timeout (int)", "working_dir (str)"],
    },
    "json_tool": {
        "class": "JSONTool",
        "desc": "Parse, query, validate, and transform JSON data",
        "example_input": '{"operation": "parse", "data": "{\\\"name\\\": \\\"effGen\\\", \\\"version\\\": \\\"0.1.3\\\"}"}',
        "params": ["operation (str: parse|query|validate|format)", "data (str, required)", "query (str)", "schema (str)"],
    },
    "datetime": {
        "class": "DateTimeTool",
        "desc": "Date/time operations, formatting, and timezone conversions",
        "example_input": '{"operation": "now"}',
        "params": ["operation (str: now|format|parse|diff|convert_tz)", "date_string (str)", "format (str)", "timezone (str)"],
    },
    "text_processing": {
        "class": "TextProcessingTool",
        "desc": "Text analysis: word count, regex matching, comparison, and transformation",
        "example_input": '{"operation": "analyze", "text": "Hello World! This is effGen."}',
        "params": ["operation (str: analyze|regex|compare|transform)", "text (str, required)", "pattern (str)", "replacement (str)"],
    },
    "url_fetch": {
        "class": "URLFetchTool",
        "desc": "Fetch and extract text content from URLs",
        "example_input": '{"url": "https://httpbin.org/json"}',
        "params": ["url (str, required)", "method (str, default GET)", "headers (dict)", "extract_text (bool)"],
    },
    "wikipedia": {
        "class": "WikipediaTool",
        "desc": "Search and retrieve Wikipedia articles",
        "example_input": '{"query": "Python programming language", "sentences": 3}',
        "params": ["query (str, required)", "sentences (int, default 5)", "language (str, default en)"],
    },
    "weather": {
        "class": "WeatherTool",
        "desc": "Get weather information using Open-Meteo (free, no API key)",
        "example_input": '{"location": "San Francisco"}',
        "params": ["location (str, required)", "units (str: metric|imperial)"],
    },
    "retrieval": {
        "class": "Retrieval",
        "desc": "Semantic search over a knowledge base using RAG + BM25",
        "example_input": '{"query": "How does effGen handle tool calling?"}',
        "params": ["query (str, required)", "top_k (int, default 5)", "collection (str)"],
    },
    "agentic_search": {
        "class": "AgenticSearch",
        "desc": "Exact string search over files using ripgrep",
        "example_input": '{"query": "BaseTool", "path": "effgen/tools/"}',
        "params": ["query (str, required)", "path (str)", "file_pattern (str)", "max_results (int)"],
    },
}

# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------

_tool_instances: dict[str, Any] = {}


def _get_tool_instance(tool_key: str):
    """Lazy-load and cache a tool instance."""
    if tool_key in _tool_instances:
        return _tool_instances[tool_key]

    try:
        from effgen.tools.builtin import (
            Calculator, PythonREPL, WebSearch, CodeExecutor, FileOperations,
            Retrieval, AgenticSearch, BashTool, WeatherTool, JSONTool,
            DateTimeTool, TextProcessingTool, URLFetchTool, WikipediaTool,
        )
        mapping = {
            "calculator": Calculator,
            "python_repl": PythonREPL,
            "web_search": WebSearch,
            "code_executor": CodeExecutor,
            "file_operations": FileOperations,
            "retrieval": Retrieval,
            "agentic_search": AgenticSearch,
            "bash": BashTool,
            "weather": WeatherTool,
            "json_tool": JSONTool,
            "datetime": DateTimeTool,
            "text_processing": TextProcessingTool,
            "url_fetch": URLFetchTool,
            "wikipedia": WikipediaTool,
        }
        cls = mapping.get(tool_key)
        if cls:
            _tool_instances[tool_key] = cls()
            return _tool_instances[tool_key]
    except ImportError as e:
        logger.warning(f"Could not import tool {tool_key}: {e}")
    return None


def _get_tool_schema(tool_key: str) -> dict | None:
    """Get the OpenAI-format schema for a tool."""
    tool = _get_tool_instance(tool_key)
    if tool is None:
        return None

    try:
        meta = tool.metadata
        properties = {}
        required = []

        for param in meta.parameters:
            prop: dict[str, Any] = {
                "type": param.type.value,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": meta.name,
                "description": meta.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Execute tool
# ---------------------------------------------------------------------------

def execute_tool(tool_key: str, input_json: str) -> str:
    """Execute a tool with the given JSON input and return results."""
    if not tool_key:
        return "Select a tool first."

    tool = _get_tool_instance(tool_key)
    if tool is None:
        return f"Could not load tool: {tool_key}"

    try:
        params = json.loads(input_json) if input_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"Invalid JSON input: {e}"

    start = time.perf_counter()
    try:
        # Run async execute — use asyncio.run() for clean event loop handling
        result = asyncio.run(tool.execute(**params))

        elapsed = (time.perf_counter() - start) * 1000

        output = f"--- Tool: {tool_key} ---\n"
        output += f"--- Time: {elapsed:.2f} ms ---\n"

        if isinstance(result, ToolResult):
            output += f"--- Success: {result.success} ---\n\n"

            if result.success:
                if isinstance(result.output, dict):
                    output += json.dumps(result.output, indent=2, default=str)
                elif result.output is not None:
                    output += str(result.output)
                else:
                    output += "(empty result)"
            else:
                output += f"ERROR: {result.error}\n"
                if result.output is not None:
                    output += f"\nPartial output: {result.output}"

            if result.metadata:
                output += f"\n\nMetadata: {json.dumps(result.metadata, indent=2, default=str)}"
        else:
            output += "\n"
            output += json.dumps(result, indent=2, default=str)

        return output
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        return (
            f"--- Tool: {tool_key} ---\n"
            f"--- Time: {elapsed:.2f} ms ---\n"
            f"--- ERROR ---\n{traceback.format_exc()}"
        )


# ---------------------------------------------------------------------------
# Batch test
# ---------------------------------------------------------------------------

def batch_test(tool_key: str, batch_json: str) -> str:
    """Run multiple test cases against a tool."""
    if not tool_key:
        return "Select a tool first."

    try:
        test_cases = json.loads(batch_json)
        if not isinstance(test_cases, list):
            return "Batch input must be a JSON array of test case objects."
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"

    results = []
    total_time = 0.0
    pass_count = 0
    fail_count = 0

    for i, case in enumerate(test_cases, 1):
        case_json = json.dumps(case)
        result = execute_tool(tool_key, case_json)

        # Extract timing and pass/fail
        for line in result.splitlines():
            if "Time:" in line:
                try:
                    ms = float(line.split("Time:")[1].strip().replace("ms", "").strip())
                    total_time += ms
                except (ValueError, IndexError):
                    pass
        if "--- Success: True ---" in result:
            pass_count += 1
        elif "--- Success: False ---" in result:
            fail_count += 1

        results.append(f"=== Test Case {i} ===\nInput: {case_json}\n{result}\n")

    avg_time = total_time / len(test_cases) if test_cases else 0.0
    summary = (
        f"=== BATCH SUMMARY ===\n"
        f"Tool: {tool_key}\n"
        f"Cases: {len(test_cases)} (Passed: {pass_count}, Failed: {fail_count})\n"
        f"Total time: {total_time:.2f} ms\n"
        f"Avg time: {avg_time:.2f} ms\n\n"
    )

    return summary + "\n".join(results)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def on_tool_select(tool_key: str) -> tuple[str, str, str]:
    """When a tool is selected, update info, example, and schema."""
    info = TOOL_INFO.get(tool_key, {})

    # Info panel
    params_list = "\n".join(f"  - {p}" for p in info.get("params", []))
    info_text = (
        f"**{info.get('class', tool_key)}** — {info.get('desc', 'No description')}\n\n"
        f"**Parameters:**\n{params_list}"
    )

    # Example input
    example = info.get("example_input", "{}")

    # Schema
    schema = _get_tool_schema(tool_key)
    schema_text = json.dumps(schema, indent=2) if schema else "Schema not available."

    return info_text, example, schema_text


def render_tool_catalog() -> str:
    """Render the full tool catalog as HTML."""
    rows = []
    for key, info in TOOL_INFO.items():
        rows.append(
            f"<tr>"
            f'<td style="padding:10px;font-weight:600;font-family:monospace;">{_html_escape(key)}</td>'
            f'<td style="padding:10px;">{_html_escape(info["class"])}</td>'
            f'<td style="padding:10px;">{_html_escape(info["desc"])}</td>'
            f'<td style="padding:10px;font-size:12px;">{len(info["params"])} params</td>'
            f"</tr>"
        )

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
            <tr style="background:#f0f0f0;border-bottom:2px solid #ddd;">
                <th style="padding:10px;text-align:left;">Tool ID</th>
                <th style="padding:10px;text-align:left;">Class</th>
                <th style="padding:10px;text-align:left;">Description</th>
                <th style="padding:10px;text-align:left;">Params</th>
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app(default_model: str):
    import gradio as gr

    tool_keys = list(TOOL_INFO.keys())
    all_model_choices = _get_all_model_choices()

    with gr.Blocks(title="effGen Tool Tester") as app:

        gr.Markdown(
            "# effGen Tool Tester\n"
            "Load a model, browse tools, test them directly or through an agent."
        )

        # --- Model Loading Bar (shared across all tabs) ---
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=all_model_choices,
                value=default_model,
                label="Model",
                allow_custom_value=True,
                scale=3,
            )
            custom_model = gr.Textbox(
                label="Custom Model ID",
                placeholder="org/model-name (overrides dropdown)",
                scale=2,
            )
            load_btn = gr.Button("Load Model", variant="primary", scale=1)
            unload_btn = gr.Button("Unload", variant="stop", scale=1)
            model_status = gr.Textbox(
                label="Model Status",
                interactive=False,
                scale=2,
                value="No model loaded. Direct tool testing works without a model.",
            )

        load_btn.click(
            fn=load_model_op,
            inputs=[model_dropdown, custom_model],
            outputs=[model_status],
        )
        unload_btn.click(
            fn=unload_model_op,
            outputs=[model_status],
        )

        with gr.Tabs():

            # ==================== TAB 1: Browse Tools ====================
            with gr.Tab("Browse Tools"):
                gr.Markdown("### All Built-in Tools")
                gr.HTML(value=render_tool_catalog())

                gr.Markdown("""
### Tool Categories
| Category | Tools |
|----------|-------|
| **Computation** | calculator |
| **Code Execution** | python_repl, code_executor, bash |
| **Information Retrieval** | web_search, url_fetch, wikipedia, weather, retrieval, agentic_search |
| **Data Processing** | json_tool, text_processing, datetime |
| **File Operations** | file_operations |
""")

            # ==================== TAB 2: Test Tool ====================
            with gr.Tab("Test Tool"):
                gr.Markdown("### Interactive Tool Testing")

                with gr.Row():
                    tool_select = gr.Dropdown(
                        choices=tool_keys,
                        value="calculator",
                        label="Select Tool",
                        scale=1,
                    )

                tool_info = gr.Markdown(value="Select a tool to see its info.")

                gr.Markdown("### Input")
                tool_input = gr.Code(
                    value='{"expression": "sqrt(16) + 2**3", "operation": "calculate"}',
                    language="json",
                    label="Parameters (JSON)",
                    lines=8,
                )

                with gr.Row():
                    run_btn = gr.Button("Execute Tool", variant="primary", size="lg")
                    load_example_btn = gr.Button("Load Example Input")

                tool_output = gr.Textbox(
                    label="Output",
                    lines=15,
                    interactive=False,
                )

                # Wire tool selection
                _example_store = gr.State(value='{"expression": "sqrt(16) + 2**3", "operation": "calculate"}')

                def update_on_select(tool_key):
                    info_text, example, _ = on_tool_select(tool_key)
                    return info_text, example

                tool_select.change(
                    fn=update_on_select,
                    inputs=[tool_select],
                    outputs=[tool_info, _example_store],
                )

                load_example_btn.click(
                    fn=lambda ex: ex,
                    inputs=[_example_store],
                    outputs=[tool_input],
                )

                run_btn.click(
                    fn=execute_tool,
                    inputs=[tool_select, tool_input],
                    outputs=[tool_output],
                )

            # ==================== TAB 3: Schema Inspector ====================
            with gr.Tab("Schema Inspector"):
                gr.Markdown(
                    "### OpenAI-Format Tool Schemas\n"
                    "View the schemas that the agent framework sends to LLMs for tool calling."
                )

                schema_tool_select = gr.Dropdown(
                    choices=tool_keys,
                    value="calculator",
                    label="Select Tool",
                )

                schema_output = gr.Code(
                    value="Select a tool to view its schema.",
                    language="json",
                    label="Tool Schema (OpenAI Format)",
                    lines=30,
                )

                def show_schema(tool_key):
                    schema = _get_tool_schema(tool_key)
                    return json.dumps(schema, indent=2) if schema else "Schema not available."

                schema_tool_select.change(
                    fn=show_schema,
                    inputs=[schema_tool_select],
                    outputs=[schema_output],
                )

                all_schemas_btn = gr.Button("Show All Schemas")

                def show_all_schemas():
                    schemas = []
                    for key in tool_keys:
                        schema = _get_tool_schema(key)
                        if schema:
                            schemas.append(schema)
                    return json.dumps(schemas, indent=2)

                all_schemas_btn.click(
                    fn=show_all_schemas,
                    outputs=[schema_output],
                )

                gr.Markdown("""
### Schema Usage
These schemas are passed to the LLM via `tools` parameter in the chat template.
The model uses them to decide which tool to call and with what arguments.

```python
# Example: passing schemas to an MLX model
formatted = tokenizer.apply_chat_template(
    messages,
    tools=schemas,  # <-- these schemas
    tokenize=False,
    add_generation_prompt=True,
)
```
""")

            # ==================== TAB 4: Batch Test ====================
            with gr.Tab("Batch Test"):
                gr.Markdown(
                    "### Batch Testing\n"
                    "Run multiple test cases against a tool and compare results."
                )

                batch_tool_select = gr.Dropdown(
                    choices=tool_keys,
                    value="calculator",
                    label="Select Tool",
                )

                batch_input = gr.Code(
                    value=json.dumps([
                        {"expression": "2 + 2"},
                        {"expression": "sqrt(144)"},
                        {"expression": "3.14 * 10**2"},
                        {"expression": "log(1000, 10)"},
                    ], indent=2),
                    language="json",
                    label="Test Cases (JSON Array)",
                    lines=15,
                )

                batch_btn = gr.Button("Run Batch Test", variant="primary", size="lg")

                batch_output = gr.Textbox(
                    label="Batch Results",
                    lines=25,
                    interactive=False,
                )

                batch_btn.click(
                    fn=batch_test,
                    inputs=[batch_tool_select, batch_input],
                    outputs=[batch_output],
                )

                gr.Markdown("""
### Batch Test Tips
- Input must be a JSON **array** of objects
- Each object is a separate test case
- Results include timing per case and summary statistics
- Use this to verify tool behavior across edge cases
""")

            # ==================== TAB 5: Agent Tool Test ====================
            with gr.Tab("Agent Tool Test"):
                gr.Markdown(
                    "### Test Tools Through an Agent\n"
                    "Load a model above, select tools, then send a prompt. "
                    "The agent will reason and call tools automatically."
                )

                with gr.Row():
                    agent_tools_select = gr.CheckboxGroup(
                        choices=tool_keys,
                        value=["calculator", "datetime", "text_processing"],
                        label="Select Tools for Agent",
                    )

                with gr.Row():
                    build_agent_btn = gr.Button("Build Agent", variant="primary")
                    agent_status = gr.Textbox(
                        label="Agent Status",
                        interactive=False,
                        scale=3,
                    )

                build_agent_btn.click(
                    fn=build_agent_with_tools,
                    inputs=[agent_tools_select],
                    outputs=[agent_status],
                )

                gr.Markdown("### Send a Prompt")
                agent_prompt = gr.Textbox(
                    label="Prompt",
                    lines=2,
                    placeholder="Ask something that requires tools...",
                )

                gr.Examples(
                    examples=[
                        ["What is sqrt(144) + 2^10?"],
                        ["What is today's date and day of the week?"],
                        ["Analyze the text: 'The quick brown fox jumps over the lazy dog'"],
                        ["Calculate 3.14159 * 100^2 and round to 2 decimal places"],
                        ["What is 72 fahrenheit in celsius?"],
                    ],
                    inputs=[agent_prompt],
                )

                run_agent_btn = gr.Button("Run Agent", variant="primary", size="lg")

                agent_output = gr.Textbox(
                    label="Agent Output",
                    lines=20,
                    interactive=False,
                )

                run_agent_btn.click(
                    fn=run_agent_with_prompt,
                    inputs=[agent_prompt],
                    outputs=[agent_output],
                )
                agent_prompt.submit(
                    fn=run_agent_with_prompt,
                    inputs=[agent_prompt],
                    outputs=[agent_output],
                )

                gr.Markdown("""
### How It Works
1. **Load a model** using the bar at the top
2. **Select tools** the agent should have access to
3. **Click Build Agent** to create the agent with those tools
4. **Send a prompt** — the agent will reason, call tools, and return the result

The agent uses the model to decide *which* tool to call and *what arguments* to pass.
This tests the full pipeline: model reasoning + tool schema + tool execution.
""")

            # ==================== TAB 6: Quick Reference ====================
            with gr.Tab("Quick Reference"):
                gr.Markdown("""
### Creating Custom Tools

**Step 1: Subclass BaseTool**
```python
from effgen.tools.base_tool import BaseTool, ToolMetadata, ToolCategory, ParameterSpec, ParameterType

class MyTool(BaseTool):
    def __init__(self):
        super().__init__(metadata=ToolMetadata(
            name="my_tool",
            description="Does something useful",
            category=ToolCategory.DATA_PROCESSING,
            parameters=[
                ParameterSpec(name="input", type=ParameterType.STRING, description="Input text", required=True),
            ],
        ))

    async def _execute(self, input: str, **kwargs) -> dict:
        return {"result": input.upper()}
```

**Step 2: Register with an Agent**
```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig

model = load_model("LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit", engine="mlx")
agent = Agent(config=AgentConfig(
    name="my_agent",
    model=model,
    tools=[MyTool()],
))
result = agent.run("Use my_tool on 'hello'")
```

**Step 3: Or use the Plugin System**
```python
from effgen.tools.plugin import ToolPlugin

class MyPlugin(ToolPlugin):
    name = "my_tools"
    version = "1.0.0"
    tools = [MyTool]
```
Save to `~/.effgen/plugins/my_tools.py` for auto-discovery.

### Tool Builder GUI
Use the **Tool Builder** (`tool_builder_gui.py`) to visually create tools
without writing boilerplate code.

### Parameter Types
| Type | Python | Use For |
|------|--------|---------|
| `STRING` | `str` | Text input, queries, expressions |
| `INTEGER` | `int` | Counts, indices, limits |
| `FLOAT` | `float` | Measurements, scores, thresholds |
| `BOOLEAN` | `bool` | Flags, toggles |
| `ARRAY` | `list` | Collections, batch inputs |
| `OBJECT` | `dict` | Structured data, configs |
| `ANY` | `Any` | Flexible input |
""")

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="effGen Tool Tester GUI")
    parser.add_argument(
        "--model",
        default="LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit",
        help="Default model ID to pre-fill",
    )
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--autoload", action="store_true", help="Load model on startup")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required. Install with:")
        print("  pip install gradio")
        sys.exit(1)

    if args.autoload:
        print(f"Loading model: {args.model}")
        result = load_model_op(args.model, "")
        print(result)

    app = build_app(args.model)
    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="footer { display: none !important; }",
    )


if __name__ == "__main__":
    main()
