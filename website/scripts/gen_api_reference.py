#!/usr/bin/env python3
"""Derive the documentation site's API reference from the installed effGen
package and write it to ``effgen-docs/src/data/apiReference.json``.

The reference page used to be written by hand, which is how it came to describe
a model id the framework had dropped and to miss most of what the package
exports. This script replaces that: every name on the page is read out of
``effgen.__all__`` at the moment it is generated, with the signature Python
reports, the docstring the source carries, and the module the object is defined
in.

    python scripts/gen_api_reference.py            # write the JSON
    python scripts/gen_api_reference.py --check    # exit 1 if it is stale
    python scripts/gen_api_reference.py --stdout   # print, write nothing

Run it with the framework's environment on PATH::

    export PATH=/path/to/your/effgen/env/bin:$PATH   # or activate it however you do
    unset EFFGEN_BASE_URL OPENAI_BASE_URL OPENAI_API_BASE
    python scripts/gen_api_reference.py

``--check`` regenerates in memory and compares against the checked-in file,
ignoring ``derived_at``, so a reference that has fallen behind a release fails a
build instead of reaching a reader.

Two rules the output holds to, because the page states both:

* **Every name in the file is in ``effgen.__all__``**, so ``from effgen import
  <name>`` works for every row on the page.
* **Every name in ``effgen.__all__`` is in the file**, and every one of them is
  filed under exactly one area. An unknown module raises rather than landing in
  a catch-all, so a new subpackage in a future release is a failure here rather
  than a silent omission on the page.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import inspect
import json
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "effgen-docs" / "src" / "data" / "apiReference.json"

# The areas the page groups by, in the order it prints them, and which modules
# belong to each. A module is matched by its longest listed prefix, so
# ``effgen.models.routing.cost`` lands under routing rather than under models.
#
# The blurb is the one sentence the page prints under the area heading. It says
# what the group is for; it does not describe how good it is.
AREAS: list[dict] = [
    {
        "id": "agents",
        "title": "Agents",
        "blurb": "The agent itself, the configuration it runs under, the state it can be saved to, the record of the tools it called, and what a run that stopped short had got done.",
        "modules": [
            "effgen.core.agent",
            "effgen.core.agent_config",
            "effgen.core.agent_response",
            "effgen.core.state",
            "effgen.core.tool_call_record",
        ],
    },
    {
        "id": "presets-config",
        "title": "Presets and configuration",
        "blurb": "Building an agent from a named preset, and loading, validating and inspecting configuration.",
        "modules": [
            "effgen.presets.registry",
            "effgen.config.loader",
            "effgen.config.validator",
            "effgen._env",
        ],
    },
    {
        "id": "models",
        "title": "Models and providers",
        "blurb": "Loading a model, the adapter classes behind each provider, the local engines, and the trackers that record what a call cost and how long it took.",
        "modules": [
            "effgen.models.base",
            "effgen.models.model_loader",
            "effgen.models.registry",
            "effgen.models.auth",
            "effgen.models.openai_schema",
            "effgen.models.latency_tracker",
            "effgen.models._cost",
            "effgen.models._cost_store",
            "effgen.models._rate_limit",
            "effgen.models._rate_limit_store",
            "effgen.models.anthropic_adapter",
            "effgen.models.cerebras_adapter",
            "effgen.models.fireworks_adapter",
            "effgen.models.gemini_adapter",
            "effgen.models.groq_adapter",
            "effgen.models.hf_inference_adapter",
            "effgen.models.openai_adapter",
            "effgen.models.openai_compatible_adapter",
            "effgen.models.replicate_adapter",
            "effgen.models.together_adapter",
            "effgen.models.transformers_engine",
            "effgen.models.vllm_engine",
        ],
    },
    {
        "id": "catalogs",
        "title": "Provider catalogs",
        "blurb": "Per-provider lookups over the bundled catalog: what a provider serves, which of its models take tools, and what each one costs.",
        "modules": [
            "effgen.models.cerebras_models",
            "effgen.models.fireworks_models",
            "effgen.models.gemini_models",
            "effgen.models.groq_models",
            "effgen.models.hf_inference_models",
            "effgen.models.openai_models",
            "effgen.models.replicate_models",
            "effgen.models.together_models",
        ],
    },
    {
        "id": "routing",
        "title": "Routing and fallback",
        "blurb": "Choosing which model answers a call, what happens when it will not, and the breaker that stops a failing dependency being retried.",
        "modules": [
            "effgen.models.router",
            "effgen.models.routing",
            "effgen.core.router",
            "effgen.tools.fallback",
            "effgen.utils.circuit_breaker",
        ],
    },
    {
        "id": "errors",
        "title": "Errors",
        "blurb": "The typed exceptions a run raises, and what each one means about whether retrying is worth anything.",
        "modules": [
            "effgen.errors",
            "effgen.models.errors",
        ],
    },
    {
        "id": "tools",
        "title": "Tools and execution",
        "blurb": "Writing a tool, registering one, and running generated code inside the sandbox.",
        "modules": [
            "effgen.tools.base_tool",
            "effgen.tools.function_tool",
            "effgen.tools.registry",
            "effgen.execution.sandbox",
            "effgen.execution.validators",
        ],
    },
    {
        "id": "messages",
        "title": "Messages and multimodal",
        "blurb": "The message parts a conversation is made of, and the helpers that turn a file, a URL or bytes into one.",
        "modules": [
            "effgen.core.messages",
            "effgen.core.multimodal",
            "types",
        ],
    },
    {
        "id": "memory",
        "title": "Memory and context",
        "blurb": "What an agent keeps between turns, where it is stored, and the strategies that decide what to drop when the window fills.",
        "modules": [
            "effgen.memory.short_term",
            "effgen.memory.long_term",
            "effgen.memory.vector_store",
            "effgen.memory.compaction",
        ],
    },
    {
        "id": "prompts",
        "title": "Prompts",
        "blurb": "The template library, prompt chains, the optimizer, and the builders that assemble a system prompt from the tools an agent holds.",
        "modules": ["effgen.prompts"],
    },
    {
        "id": "orchestration",
        "title": "Orchestration and workflows",
        "blurb": "Splitting work across agents, running a DAG, checkpointing it, batching many tasks, and the middleware that wraps every step.",
        "modules": [
            "effgen.core.task",
            "effgen.core.orchestrator",
            "effgen.core.aggregation",
            "effgen.core.batch",
            "effgen.core.workflow",
            "effgen.core.workflow_checkpoint",
            "effgen.core.middleware",
        ],
    },
    {
        "id": "guardrails",
        "title": "Guardrails",
        "blurb": "Checks that run before a prompt reaches a model and after an answer comes back, and the presets that bundle them.",
        "modules": ["effgen.guardrails"],
    },
    {
        "id": "domains",
        "title": "Domains",
        "blurb": "Subject-matter packs that add vocabulary, tools and a system prompt for one field.",
        "modules": ["effgen.domains"],
    },
    {
        "id": "evaluation",
        "title": "Evaluation",
        "blurb": "Test cases, suites, model comparisons and the regression tracker a CI gate reads.",
        "modules": ["effgen.eval"],
    },
    {
        "id": "observability",
        "title": "Observability",
        "blurb": "Service level objectives, the tracker that measures them, and the alerts that fire when one is missed.",
        "modules": ["effgen.observability"],
    },
    {
        "id": "security",
        "title": "Security",
        "blurb": "Verifying that the installed package is the package that was published.",
        "modules": ["effgen.security"],
    },
    {
        "id": "hardware",
        "title": "Hardware",
        "blurb": "Reading what the GPUs on this machine are doing, and picking one to load weights onto.",
        "modules": ["effgen.gpu"],
    },
]

# A handful of names whose module does not put them where a reader would look.
# `RateLimitExceeded` is defined beside the coordinator that raises it, but it is
# an exception, and the page's errors table is where someone goes to find out
# what a run can raise.
NAME_AREA: dict[str, str] = {
    "RateLimitExceeded": "errors",
}

# Names a reader arrives looking for that are not in `effgen.__all__`, with the
# import that does work. Each one is checked by this script, so a path that
# stops working fails here rather than on the page.
NOT_EXPORTED: list[tuple[str, str, str]] = [
    (
        "AgentResponse",
        "effgen.core.agent",
        "What `Agent.run()` returns. It is not exported from the top level because you never construct one.",
    ),
    (
        "ToolResult",
        "effgen.tools.base_tool",
        "What `await tool.execute(**kwargs)` returns: `success`, `output`, `error`, `execution_time`, `metadata`, `timestamp`.",
    ),
    (
        "ToolCategory",
        "effgen.tools.base_tool",
        "The category a tool files itself under, which is what the tool gallery groups by.",
    ),
    (
        "ParameterSpec",
        "effgen.tools.base_tool",
        "One declared parameter of a tool, as it appears in `metadata.parameters`.",
    ),
    (
        "ToolMetadata",
        "effgen.tools.base_tool",
        "A tool's own description of itself: name, description, category, parameters, timeout.",
    ),
    (
        "AgentMode",
        "effgen.core.agent_config",
        "The value `AgentConfig.mode` takes.",
    ),
    (
        "get_registry",
        "effgen.tools",
        "The tool registry. `get_registry().get_tool_sync('calculator')` is how you get a built-in tool instance by name.",
    ),
]

# Sections a Google-style docstring uses for its parameter list. `Attributes:`
# is what the dataclasses use, and it documents the same names the generated
# `__init__` takes.
PARAM_SECTIONS = ("Args", "Arguments", "Attributes", "Parameters", "Keyword Args")
SECTION_RE = re.compile(
    r"^(Args|Arguments|Attributes|Parameters|Keyword Args|Returns|Yields|Raises|Example|Examples|"
    r"Note|Notes|Warning|Warnings|See Also|References|Usage|Todo)\s*:\s*$"
)


def clean(text: str) -> str:
    """Docstring markup as prose.

    The framework's docstrings are reStructuredText: ``literal`` for code,
    :class:`Name` for a cross-reference. The page renders text, not reST, so
    double backticks become single ones and a role becomes the name it points
    at.
    """
    text = re.sub(r":(?:class|func|meth|attr|mod|exc|data|obj|ref):`~?([^`]+)`", r"`\1`", text)
    text = text.replace("``", "`")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sections(doc: str) -> tuple[str, dict[str, list[str]]]:
    """The docstring body, and its Google-style sections keyed by name."""
    body: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in doc.splitlines():
        match = SECTION_RE.match(line.strip())
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            continue
        if current is None:
            body.append(line)
        else:
            sections[current].append(line)
    return "\n".join(body), sections


def first_paragraph(body: str) -> str:
    """The opening paragraph of a docstring — what the page shows as the summary."""
    para: list[str] = []
    for line in body.strip().splitlines():
        if not line.strip():
            if para:
                break
            continue
        para.append(line.strip())
    return clean(" ".join(para))


def parse_params(lines: list[str]) -> dict[str, str]:
    """`name: description` rows out of an Args or Attributes section.

    A description that wraps onto following indented lines is joined back
    together, which is how every multi-line default in `AgentConfig` is
    written.
    """
    out: dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    row = re.compile(r"^(\*{0,2}\w+)\s*(\([^)]*\))?\s*:\s*(.*)$")

    def flush() -> None:
        if name:
            out[name] = clean(" ".join(buf))

    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        match = row.match(line.strip())
        if match and (name is None or indent <= base_indent):
            flush()
            name = match.group(1).lstrip("*")
            buf = [match.group(3)]
            base_indent = indent
        elif name:
            buf.append(line.strip())
        else:
            base_indent = indent
    flush()
    return out


def parse_raises(lines: list[str]) -> list[dict[str, str]]:
    rows = parse_params(lines)
    return [{"name": k, "description": v} for k, v in rows.items()]


def annotation_of(param: inspect.Parameter) -> str | None:
    if param.annotation is inspect.Parameter.empty:
        return None
    return clean_annotation(param.annotation)


def clean_annotation(value) -> str:
    """An annotation as the page prints it.

    ``from __future__ import annotations`` is on across the framework, so most
    annotations arrive as strings and ``inspect`` renders them quoted. The
    quotes are an artefact of how they are stored, not part of the type, so they
    come off — and a fully qualified name is shortened to the one the import
    line uses.
    """
    text = value if isinstance(value, str) else getattr(value, "__name__", None) or str(value)
    text = re.sub(r"<class '([^']+)'>", r"\1", text).strip()
    while len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text.replace("typing.", "").replace("effgen.", "")


def default_of(param: inspect.Parameter) -> str | None:
    if param.default is inspect.Parameter.empty:
        return None
    value = param.default
    if isinstance(value, dataclasses._HAS_DEFAULT_FACTORY_CLASS):  # type: ignore[attr-defined]
        return "<factory>"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, str):
        return repr(value)
    return clean_annotation(repr(value))


def signature_of(obj) -> str | None:
    """The call signature, rebuilt so the annotations read as they are written.

    ``str(inspect.signature(...))`` would do, but it prints every annotation
    inside quotes on a module that uses postponed evaluation, which is every
    module here.
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None

    parts: list[str] = []
    seen_kw_only = False
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY and not seen_kw_only:
            parts.append("*")
            seen_kw_only = True
        prefix = {
            inspect.Parameter.VAR_POSITIONAL: "*",
            inspect.Parameter.VAR_KEYWORD: "**",
        }.get(param.kind, "")
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            seen_kw_only = True
        text = f"{prefix}{name}"
        annotation = annotation_of(param)
        if annotation:
            text += f": {annotation}"
        default = default_of(param)
        if default is not None:
            text += f" = {default}" if annotation else f"={default}"
        parts.append(text)

    rendered = f"({', '.join(parts)})"
    if sig.return_annotation is not inspect.Signature.empty:
        returned = clean_annotation(sig.return_annotation)
        if returned:
            rendered += f" -> {returned}"
    return rendered


def parameters_of(obj, docs: dict[str, str]) -> list[dict]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    out = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prefix = {
            inspect.Parameter.VAR_POSITIONAL: "*",
            inspect.Parameter.VAR_KEYWORD: "**",
        }.get(param.kind, "")
        out.append(
            {
                "name": f"{prefix}{name}",
                "type": annotation_of(param),
                "default": default_of(param),
                "required": param.default is inspect.Parameter.empty
                and param.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD),
                "keyword_only": param.kind is inspect.Parameter.KEYWORD_ONLY,
                "description": docs.get(name, ""),
            }
        )
    return out


def returns_of(obj, sections: dict[str, list[str]]) -> dict | None:
    annotation = None
    try:
        sig = inspect.signature(obj)
        if sig.return_annotation is not inspect.Signature.empty:
            annotation = clean_annotation(sig.return_annotation)
    except (TypeError, ValueError):
        pass
    described = clean(" ".join(sections.get("Returns", []) + sections.get("Yields", [])))
    if annotation in (None, "None") and not described:
        return None
    return {"type": annotation, "description": described}


def members_of(cls, exported: set[str]) -> list[dict]:
    """The public methods you can call on an instance of this class.

    Walking the MRO rather than just the class matters here: the framework
    assembles `Agent` out of eight mixins, so `run` — the method every reader
    arrives looking for — is not written on `Agent` itself. What is skipped is
    anything from outside effGen (`ToolCallList` inherits the whole of `list`,
    and that is Python's documentation, not this one) and anything from a base
    that is exported in its own right, which has its own entry on the page.
    """
    out = []
    seen: set[str] = set()
    for klass in cls.__mro__:
        if klass is cls:
            pass
        elif not (klass.__module__ or "").startswith("effgen"):
            continue
        elif klass.__name__ in exported:
            continue
        for name, member in sorted(vars(klass).items()):
            if name.startswith("_") or name in seen:
                continue
            func = member.__func__ if isinstance(member, (classmethod, staticmethod)) else member
            inherited = None if klass is cls else klass.__name__
            if isinstance(member, property):
                seen.add(name)
                out.append(
                    {
                        "name": name,
                        "kind": "property",
                        "signature": "",
                        "summary": first_paragraph(
                            split_sections(inspect.getdoc(member) or "")[0]
                        ),
                        "is_async": False,
                        "inherited_from": inherited,
                    }
                )
                continue
            if not (inspect.isfunction(func) or inspect.ismethod(func)):
                continue
            seen.add(name)
            kind = (
                "classmethod"
                if isinstance(member, classmethod)
                else "staticmethod"
                if isinstance(member, staticmethod)
                else "method"
            )
            out.append(
                {
                    "name": name,
                    "kind": kind,
                    "signature": signature_of(func) or "()",
                    "summary": first_paragraph(split_sections(inspect.getdoc(func) or "")[0]),
                    "is_async": inspect.iscoroutinefunction(func),
                    "inherited_from": inherited,
                }
            )
    return sorted(out, key=lambda m: m["name"])


def kind_of(name: str, obj) -> str:
    if isinstance(obj, types.ModuleType):
        return "module"
    if isinstance(obj, types.UnionType):
        return "alias"
    if inspect.isclass(obj):
        if issubclass(obj, BaseException):
            return "exception"
        if issubclass(obj, enum.Enum):
            return "enum"
        if dataclasses.is_dataclass(obj):
            return "dataclass"
        return "class"
    if inspect.isfunction(obj):
        return "function"
    return "value"


def module_of(obj) -> str:
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    return getattr(obj, "__module__", None) or ""


def area_of(module: str) -> str:
    best, best_len = None, -1
    for area in AREAS:
        for prefix in area["modules"]:
            if (module == prefix or module.startswith(prefix + ".")) and len(prefix) > best_len:
                best, best_len = area["id"], len(prefix)
    if best is None:
        raise SystemExit(
            f"no area covers module {module!r}; add it to AREAS in "
            f"{Path(__file__).name} rather than letting the name fall off the page"
        )
    return best


def describe(name: str, obj, exported: set[str]) -> dict:
    doc = inspect.getdoc(obj) or ""
    body, sections = split_sections(doc)
    param_docs: dict[str, str] = {}
    for section in PARAM_SECTIONS:
        param_docs.update(parse_params(sections.get(section, [])))

    kind = kind_of(name, obj)
    module = module_of(obj)
    entry: dict = {
        "name": name,
        "kind": kind,
        "module": module,
        "area": NAME_AREA.get(name) or area_of(module),
        "summary": first_paragraph(body),
        "signature": None,
        "params": [],
        "returns": None,
        "raises": parse_raises(sections.get("Raises", [])),
        "bases": [],
        "members": [],
        "values": [],
    }

    if kind in ("class", "dataclass", "exception", "enum"):
        entry["bases"] = [
            base.__name__ for base in obj.__bases__ if base is not object
        ]
        entry["members"] = members_of(obj, exported)
        if kind == "enum":
            entry["values"] = [
                {"name": member.name, "value": clean_annotation(repr(member.value))}
                for member in obj
            ]
        else:
            init = obj.__init__
            entry["signature"] = signature_of(obj)
            if entry["signature"]:
                entry["params"] = parameters_of(obj, param_docs)
            if not entry["summary"] and init.__doc__:
                entry["summary"] = first_paragraph(split_sections(inspect.getdoc(init) or "")[0])
    elif kind == "function":
        entry["signature"] = signature_of(obj)
        entry["params"] = parameters_of(obj, param_docs)
        entry["returns"] = returns_of(obj, sections)
        entry["is_async"] = inspect.iscoroutinefunction(obj)
    elif kind == "alias":
        # The docstring on a `X | Y` object is Python's own description of what
        # a union is, which says nothing about this one. The members are the
        # documentation.
        entry["summary"] = ""
        entry["values"] = [
            {"name": clean_annotation(arg), "value": ""} for arg in obj.__args__
        ]

    return entry


def collect_not_exported() -> list[dict]:
    import importlib

    out = []
    for name, module_path, what in NOT_EXPORTED:
        module = importlib.import_module(module_path)
        obj = getattr(module, name)  # raises here if the path ever stops working
        out.append(
            {
                "name": name,
                "module": module_path,
                "kind": kind_of(name, obj),
                "what": what,
            }
        )
    return out


def collect() -> dict:
    import effgen
    from importlib.metadata import version

    names = sorted(effgen.__all__)
    exported = set(names)
    entries = [describe(name, getattr(effgen, name), exported) for name in names]

    by_area: dict[str, list[str]] = {}
    for entry in entries:
        by_area.setdefault(entry["area"], []).append(entry["name"])

    kinds: dict[str, int] = {}
    for entry in entries:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1

    return {
        "derived_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "version": version("effgen"),
        "public_names": len(effgen.__all__),
        "kind_counts": {k: kinds[k] for k in sorted(kinds)},
        "areas": [
            {
                "id": area["id"],
                "title": area["title"],
                "blurb": area["blurb"],
                "count": len(by_area.get(area["id"], [])),
            }
            for area in AREAS
        ],
        "names": entries,
        "not_exported": collect_not_exported(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true", help="exit 1 if the checked-in file is stale"
    )
    ap.add_argument("--stdout", action="store_true", help="print it and write nothing")
    args = ap.parse_args()

    fresh = collect()
    text = json.dumps(fresh, indent=1, sort_keys=False) + "\n"

    if args.stdout:
        sys.stdout.write(text)
        return 0

    if args.check:
        if not OUT.exists():
            print(f"{OUT.relative_to(REPO)} does not exist", file=sys.stderr)
            return 1
        current = json.loads(OUT.read_text())
        a = {k: v for k, v in current.items() if k != "derived_at"}
        b = {k: v for k, v in fresh.items() if k != "derived_at"}
        if a != b:
            print(
                f"{OUT.relative_to(REPO)} no longer matches the installed package; "
                f"run: python scripts/gen_api_reference.py",
                file=sys.stderr,
            )
            return 1
        print(f"{OUT.relative_to(REPO)} matches the installed package")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)

    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  effGen            {fresh['version']}")
    print(f"  public names      {fresh['public_names']}")
    print(f"  areas             {len(fresh['areas'])}")
    for area in fresh["areas"]:
        print(f"    {area['count']:4d}  {area['title']}")
    print(f"  by kind           {fresh['kind_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
