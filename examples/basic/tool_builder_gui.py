"""
effGen Tool Builder — Visual GUI for creating custom tools.

A Gradio web app that lets users:
  - Define tool metadata (name, description, category, tags)
  - Add parameters with types, constraints, and defaults
  - Write the tool's execute function
  - Test the tool live with sample inputs
  - Export the full BaseTool subclass or plugin code

Requirements:
    pip install gradio

Usage:
    python examples/basic/tool_builder_gui.py
    python examples/basic/tool_builder_gui.py --port 7863 --share
"""

from __future__ import annotations

import argparse
import builtins as _builtins_mod
import json
import logging
import sys
import textwrap
import traceback
from html import escape as _html_escape
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = {
    "information_retrieval": "Information Retrieval — web search, data lookup",
    "code_execution": "Code Execution — run code, scripts, REPLs",
    "file_operations": "File Operations — read, write, search files",
    "computation": "Computation — math, calculations, conversions",
    "communication": "Communication — messaging, notifications",
    "data_processing": "Data Processing — parse, transform, validate",
    "system": "System — shell commands, OS operations",
    "external_api": "External API — third-party service integrations",
}

PARAM_TYPES = {
    "string": "String — text input",
    "integer": "Integer — whole numbers",
    "float": "Float — decimal numbers",
    "boolean": "Boolean — true/false",
    "array": "Array — list of values",
    "object": "Object — dictionary/map",
    "any": "Any — accepts any type",
}

COST_LEVELS = ["low", "medium", "high"]

# ---------------------------------------------------------------------------
# Parameter state
# ---------------------------------------------------------------------------

_parameters: list[dict[str, Any]] = []


def add_parameter(
    name: str, param_type: str, description: str,
    required: bool, default_val: str,
    enum_vals: str, min_val: str, max_val: str,
) -> tuple[str, str]:
    """Add a parameter to the current tool definition."""
    if not name.strip():
        return _render_params_table(), "Parameter name is required."

    name = name.strip().replace(" ", "_").lower()

    # Check for duplicates
    if any(p["name"] == name for p in _parameters):
        return _render_params_table(), f"Parameter '{name}' already exists."

    param = {
        "name": name,
        "type": param_type,
        "description": description.strip() or f"The {name} parameter",
        "required": required,
    }

    if default_val.strip():
        param["default"] = default_val.strip()
    if enum_vals.strip():
        param["enum"] = [v.strip() for v in enum_vals.split(",") if v.strip()]
    if min_val.strip():
        try:
            param["min_value"] = float(min_val) if "." in min_val else int(min_val)
        except ValueError:
            pass
    if max_val.strip():
        try:
            param["max_value"] = float(max_val) if "." in max_val else int(max_val)
        except ValueError:
            pass

    _parameters.append(param)
    return _render_params_table(), f"Added parameter: {name} ({param_type})"


def remove_parameter(name: str) -> tuple[str, str]:
    """Remove a parameter by name."""
    global _parameters
    before = len(_parameters)
    _parameters = [p for p in _parameters if p["name"] != name.strip()]
    if len(_parameters) < before:
        return _render_params_table(), f"Removed parameter: {name}"
    return _render_params_table(), f"Parameter '{name}' not found."


def clear_parameters() -> tuple[str, str]:
    """Clear all parameters."""
    _parameters.clear()
    return _render_params_table(), "All parameters cleared."


def _render_params_table() -> str:
    """Render parameters as an HTML table."""
    if not _parameters:
        return (
            '<div style="padding:20px;text-align:center;color:#999;">'
            "No parameters defined yet. Add one above.</div>"
        )

    rows = []
    for p in _parameters:
        req = "Yes" if p.get("required") else "No"
        default = p.get("default", "—")
        enum = ", ".join(p["enum"]) if p.get("enum") else "—"
        constraints = []
        if p.get("min_value") is not None:
            constraints.append(f"min={p['min_value']}")
        if p.get("max_value") is not None:
            constraints.append(f"max={p['max_value']}")
        constraint_str = ", ".join(constraints) if constraints else "—"

        rows.append(
            f"<tr>"
            f'<td style="padding:8px;font-weight:600;">{_html_escape(str(p["name"]))}</td>'
            f'<td style="padding:8px;"><code>{_html_escape(str(p["type"]))}</code></td>'
            f'<td style="padding:8px;">{_html_escape(str(p["description"]))}</td>'
            f'<td style="padding:8px;">{req}</td>'
            f'<td style="padding:8px;">{_html_escape(str(default))}</td>'
            f'<td style="padding:8px;">{_html_escape(str(enum))}</td>'
            f'<td style="padding:8px;">{_html_escape(str(constraint_str))}</td>'
            f"</tr>"
        )

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px;">
        <thead>
            <tr style="background:#f0f0f0;border-bottom:2px solid #ddd;">
                <th style="padding:8px;text-align:left;">Name</th>
                <th style="padding:8px;text-align:left;">Type</th>
                <th style="padding:8px;text-align:left;">Description</th>
                <th style="padding:8px;text-align:left;">Required</th>
                <th style="padding:8px;text-align:left;">Default</th>
                <th style="padding:8px;text-align:left;">Enum</th>
                <th style="padding:8px;text-align:left;">Constraints</th>
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# Default implementation template
# ---------------------------------------------------------------------------

DEFAULT_IMPL = textwrap.dedent('''\
    # Write your tool's execute logic here.
    # Available variables: all parameters defined in the Parameters tab.
    # Return a dict with your results.
    #
    # Example:
    #   result = some_computation(expression)
    #   return {"result": result, "formatted": str(result)}

    # --- Your implementation ---
    return {"result": "Hello from my custom tool!", "input": expression}
''')


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def test_tool(impl_code: str, test_input_json: str) -> str:
    """Execute the tool implementation with test inputs.

    Uses restricted builtins to prevent arbitrary system access.
    """
    try:
        inputs = json.loads(test_input_json) if test_input_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"Invalid JSON input: {e}"

    # Build a function from the implementation code
    func_code = f"def _test_execute(**kwargs):\n"
    for line in impl_code.splitlines():
        func_code += f"    {line}\n"

    # Execute with restricted builtins (no os, subprocess, etc.)
    import io
    import contextlib
    import math
    import datetime
    import re

    _SAFE_BUILTIN_NAMES = {
        "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
        "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "getattr", "hasattr", "hash", "hex", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min",
        "next", "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "type", "zip", "True", "False", "None",
        "ValueError", "TypeError", "KeyError", "IndexError",
        "RuntimeError", "StopIteration", "Exception",
    }
    restricted_builtins = {
        k: getattr(_builtins_mod, k)
        for k in _SAFE_BUILTIN_NAMES
        if hasattr(_builtins_mod, k)
    }
    restricted_builtins["__import__"] = None  # Block all imports

    exec_globals = {
        "__builtins__": restricted_builtins,
        "math": math,
        "datetime": datetime,
        "re": re,
        "json": json,
    }

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(func_code, exec_globals)
            result = exec_globals["_test_execute"](**inputs)

        output = ""
        printed = stdout.getvalue()
        if printed:
            output += f"--- stdout ---\n{printed}\n"
        output += f"--- return value ---\n{json.dumps(result, indent=2, default=str)}"
        return output
    except Exception:
        printed = stdout.getvalue()
        err = traceback.format_exc()
        output = ""
        if printed:
            output += f"--- stdout ---\n{printed}\n"
        output += f"--- error ---\n{err}"
        return output


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def generate_tool_code(
    tool_name: str, tool_desc: str, category: str,
    version: str, author: str, tags: str,
    timeout: int, cost: str, requires_auth: bool,
    impl_code: str,
) -> str:
    """Generate a complete BaseTool subclass."""
    # Sanitize class name
    class_name = "".join(
        word.capitalize() for word in tool_name.strip().replace("-", "_").split("_")
    ) if tool_name.strip() else "MyCustomTool"

    tool_id = tool_name.strip().lower().replace(" ", "_").replace("-", "_") or "my_custom_tool"
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags.strip() else []

    # Build parameter specs
    param_specs = []
    for p in _parameters:
        spec_parts = [
            f'                    ParameterSpec(\n'
            f'                        name="{p["name"]}",\n'
            f'                        type=ParameterType.{p["type"].upper()},\n'
            f'                        description="{p["description"]}",\n'
            f'                        required={p.get("required", False)},\n'
        ]
        if p.get("default") is not None:
            spec_parts.append(f'                        default="{p["default"]}",\n')
        if p.get("enum"):
            spec_parts.append(f'                        enum={p["enum"]},\n')
        if p.get("min_value") is not None:
            spec_parts.append(f'                        min_value={p["min_value"]},\n')
        if p.get("max_value") is not None:
            spec_parts.append(f'                        max_value={p["max_value"]},\n')
        spec_parts.append('                    ),')
        param_specs.append("".join(spec_parts))

    params_block = "\n".join(param_specs) if param_specs else ""

    # Build _execute signature
    exec_params = []
    for p in _parameters:
        type_map = {
            "string": "str", "integer": "int", "float": "float",
            "boolean": "bool", "array": "list", "object": "dict", "any": "Any",
        }
        py_type = type_map.get(p["type"], "Any")
        if p.get("required"):
            exec_params.append(f"{p['name']}: {py_type}")
        else:
            raw_default = p.get("default")
            if raw_default is None:
                default_repr = "None"
            elif py_type == "str":
                default_repr = repr(raw_default)
            else:
                default_repr = raw_default
            exec_params.append(f"{p['name']}: {py_type} | None = {default_repr}")

    exec_sig = ", ".join(exec_params)
    if exec_sig:
        exec_sig = f"\n        self,\n        {exec_sig},\n        **kwargs,\n    "
    else:
        exec_sig = "self, **kwargs"

    # Indent implementation
    impl_lines = []
    for line in impl_code.splitlines():
        impl_lines.append(f"        {line}" if line.strip() else "")
    impl_block = "\n".join(impl_lines)

    code = textwrap.dedent(f'''\
        """
        {tool_desc or 'Custom tool generated by effGen Tool Builder.'}
        """

        from __future__ import annotations

        import logging
        from typing import Any

        from effgen.tools.base_tool import (
            BaseTool,
            ParameterSpec,
            ParameterType,
            ToolCategory,
            ToolMetadata,
        )

        logger = logging.getLogger(__name__)


        class {class_name}(BaseTool):
            """
            {tool_desc or 'Custom tool.'}

            Generated by effGen Tool Builder GUI.
            """

            def __init__(self):
                super().__init__(
                    metadata=ToolMetadata(
                        name="{tool_id}",
                        description="{tool_desc or 'Custom tool'}",
                        category=ToolCategory.{category.upper()},
                        parameters=[
        {params_block}
                        ],
                        version="{version or '1.0.0'}",
                        author="{author or ''}",
                        requires_auth={requires_auth},
                        cost_estimate="{cost or 'low'}",
                        timeout_seconds={timeout or 30},
                        tags={tag_list},
                    )
                )

            async def _execute({exec_sig}) -> dict[str, Any]:
                """Execute the tool."""
        {impl_block}
    ''')

    return code


def generate_plugin_code(
    tool_name: str, tool_desc: str, category: str,
    version: str, author: str, tags: str,
    timeout: int, cost: str, requires_auth: bool,
    impl_code: str,
) -> str:
    """Generate a ToolPlugin wrapper around the tool."""
    tool_code = generate_tool_code(
        tool_name, tool_desc, category, version, author, tags,
        timeout, cost, requires_auth, impl_code,
    )

    class_name = "".join(
        word.capitalize() for word in tool_name.strip().replace("-", "_").split("_")
    ) if tool_name.strip() else "MyCustomTool"

    plugin_name = tool_name.strip().lower().replace(" ", "_").replace("-", "_") or "my_custom_tool"

    plugin_code = textwrap.dedent(f'''\
        """
        effGen Plugin: {tool_name or 'My Custom Tool'}

        Install:
            1. Save this file to ~/.effgen/plugins/{plugin_name}.py
            2. Or register via entry points in pyproject.toml

        Usage:
            from effgen.tools.plugin import PluginManager
            mgr = PluginManager()
            mgr.discover_all()  # auto-discovers from ~/.effgen/plugins/
        """

        from effgen.tools.plugin import ToolPlugin


    ''')

    # Append the tool class code
    plugin_code += tool_code

    # Append the plugin registration
    plugin_code += textwrap.dedent(f'''

        class {class_name}Plugin(ToolPlugin):
            """Plugin wrapper for {class_name}."""
            name = "{plugin_name}"
            version = "{version or '1.0.0'}"
            description = "{tool_desc or 'Custom tool plugin'}"
            tools = [{class_name}]
    ''')

    return plugin_code


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    import gradio as gr

    with gr.Blocks(title="effGen Tool Builder") as app:

        gr.Markdown(
            "# effGen Tool Builder\n"
            "Visually create, test, and export custom tools for the effGen framework."
        )

        with gr.Tabs():

            # ==================== TAB 1: Define Tool ====================
            with gr.Tab("1. Define"):
                gr.Markdown("### Tool Metadata")

                with gr.Row():
                    with gr.Column():
                        tool_name = gr.Textbox(
                            value="my_custom_tool",
                            label="Tool Name (snake_case)",
                            placeholder="e.g. sentiment_analyzer",
                        )
                        tool_desc = gr.Textbox(
                            value="A custom tool that does something useful.",
                            label="Description",
                            lines=2,
                            placeholder="What does this tool do?",
                        )
                        category = gr.Dropdown(
                            choices=list(CATEGORIES.keys()),
                            value="data_processing",
                            label="Category",
                        )

                    with gr.Column():
                        version = gr.Textbox(value="1.0.0", label="Version")
                        author = gr.Textbox(value="", label="Author", placeholder="Your name")
                        tags = gr.Textbox(
                            value="",
                            label="Tags (comma-separated)",
                            placeholder="e.g. text, nlp, analysis",
                        )

                with gr.Row():
                    timeout = gr.Slider(1, 300, value=30, step=1, label="Timeout (seconds)")
                    cost = gr.Dropdown(choices=COST_LEVELS, value="low", label="Cost Estimate")
                    requires_auth = gr.Checkbox(value=False, label="Requires Authentication")

                gr.Markdown("""
### Category Reference
| Category | Use For |
|----------|---------|
| `computation` | Math, calculations, conversions |
| `data_processing` | Parse, transform, validate data |
| `information_retrieval` | Web search, data lookup |
| `code_execution` | Run code, scripts, REPLs |
| `file_operations` | Read, write, search files |
| `system` | Shell commands, OS operations |
| `external_api` | Third-party service calls |
| `communication` | Messaging, notifications |
""")

            # ==================== TAB 2: Parameters ====================
            with gr.Tab("2. Parameters"):
                gr.Markdown("### Define Tool Parameters")

                with gr.Row():
                    with gr.Column(scale=2):
                        param_name = gr.Textbox(label="Name", placeholder="e.g. expression")
                        param_desc = gr.Textbox(label="Description", placeholder="What this parameter is for")
                    with gr.Column(scale=1):
                        param_type = gr.Dropdown(
                            choices=list(PARAM_TYPES.keys()),
                            value="string",
                            label="Type",
                        )
                        param_required = gr.Checkbox(value=True, label="Required")

                with gr.Accordion("Advanced Constraints", open=False):
                    with gr.Row():
                        param_default = gr.Textbox(label="Default Value", placeholder="Leave empty for none")
                        param_enum = gr.Textbox(label="Allowed Values (comma-separated)", placeholder="e.g. add, subtract, multiply")
                    with gr.Row():
                        param_min = gr.Textbox(label="Min Value", placeholder="For numeric types")
                        param_max = gr.Textbox(label="Max Value", placeholder="For numeric types")

                with gr.Row():
                    add_param_btn = gr.Button("Add Parameter", variant="primary")
                    remove_param_name = gr.Textbox(label="Remove by Name", placeholder="parameter name", scale=2)
                    remove_param_btn = gr.Button("Remove", variant="stop")
                    clear_params_btn = gr.Button("Clear All")

                param_status = gr.Textbox(label="Status", interactive=False)
                params_html = gr.HTML(value=_render_params_table())

                # Wire parameter operations
                add_param_btn.click(
                    fn=add_parameter,
                    inputs=[
                        param_name, param_type, param_desc, param_required,
                        param_default, param_enum, param_min, param_max,
                    ],
                    outputs=[params_html, param_status],
                )
                remove_param_btn.click(
                    fn=remove_parameter,
                    inputs=[remove_param_name],
                    outputs=[params_html, param_status],
                )
                clear_params_btn.click(
                    fn=clear_parameters,
                    outputs=[params_html, param_status],
                )

            # ==================== TAB 3: Implement ====================
            with gr.Tab("3. Implement"):
                gr.Markdown(
                    "### Write the Execute Function\n"
                    "Write the body of the `_execute` method. "
                    "Your defined parameters are available as local variables. "
                    "Return a dict with your results."
                )

                impl_code = gr.Code(
                    value=DEFAULT_IMPL,
                    language="python",
                    label="Implementation",
                    lines=25,
                )

                gr.Markdown("""
### Available Imports in Execute
Your generated tool will have access to standard Python libraries.
Add any additional imports at the top of the generated code.

### Tips
- Always return a `dict` with at least a `"result"` key
- Use `try/except` for error handling
- Raise `ValueError` for invalid inputs (the framework catches it)
- Keep implementations focused on a single responsibility
""")

            # ==================== TAB 4: Test ====================
            with gr.Tab("4. Test"):
                gr.Markdown(
                    "### Test Your Tool\n"
                    "Provide test inputs as JSON and run the implementation."
                )

                test_input = gr.Code(
                    value='{"expression": "hello world"}',
                    language="json",
                    label="Test Input (JSON)",
                    lines=8,
                )

                test_btn = gr.Button("Run Test", variant="primary", size="lg")

                test_output = gr.Textbox(
                    label="Test Output",
                    lines=15,
                    interactive=False,
                )

                test_btn.click(
                    fn=test_tool,
                    inputs=[impl_code, test_input],
                    outputs=[test_output],
                )

                gr.Markdown("""
### Test Examples
```json
{"expression": "2 + 2"}
{"text": "Hello World", "uppercase": true}
{"url": "https://example.com", "timeout": 5}
```
""")

            # ==================== TAB 5: Export ====================
            with gr.Tab("5. Export"):
                gr.Markdown(
                    "### Export Your Tool\n"
                    "Generate the full Python code for your tool as a **BaseTool subclass** "
                    "or as a **Plugin** ready to drop into `~/.effgen/plugins/`."
                )

                with gr.Row():
                    gen_tool_btn = gr.Button("Generate Tool Class", variant="primary")
                    gen_plugin_btn = gr.Button("Generate Plugin", variant="secondary")

                export_code = gr.Code(
                    label="Generated Code",
                    language="python",
                    lines=40,
                )

                gen_tool_btn.click(
                    fn=generate_tool_code,
                    inputs=[
                        tool_name, tool_desc, category, version, author, tags,
                        timeout, cost, requires_auth, impl_code,
                    ],
                    outputs=[export_code],
                )

                gen_plugin_btn.click(
                    fn=generate_plugin_code,
                    inputs=[
                        tool_name, tool_desc, category, version, author, tags,
                        timeout, cost, requires_auth, impl_code,
                    ],
                    outputs=[export_code],
                )

                gr.Markdown("""
### How to Use Your Tool

**Option A: As a standalone file**
1. Save the generated tool class to `effgen/tools/builtin/my_tool.py`
2. Import it in your agent code:
   ```python
   from effgen.tools.builtin.my_tool import MyCustomTool
   agent = Agent(config=AgentConfig(tools=[MyCustomTool()], ...))
   ```

**Option B: As a plugin**
1. Save the generated plugin to `~/.effgen/plugins/my_tool.py`
2. The plugin system auto-discovers it:
   ```python
   from effgen.tools.plugin import discover_plugins
   discover_plugins()  # auto-loads from ~/.effgen/plugins/
   ```

**Option C: Use with Agent Builder GUI**
1. Register the tool in the tool catalog
2. It will appear in the Agent Builder's tool selection
""")

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="effGen Tool Builder GUI")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    try:
        import gradio as gr
    except ImportError:
        print("Gradio is required. Install with:")
        print("  pip install gradio")
        sys.exit(1)

    app = build_app()
    app.launch(
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css="footer { display: none !important; }",
    )


if __name__ == "__main__":
    main()
