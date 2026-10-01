#!/usr/bin/env python3
"""Run every code sample the effgen.org sites show.

The rule this harness exists to enforce: a sample printed on the landing site or
in the documentation has been run, verbatim, against the released framework. A
sample nobody can see is a sample nobody is testing, so extraction is mechanical
and reports what it found per file — a page you know has eight samples that
reports two is a broken extractor, not a clean page.

Where the samples live
----------------------
Nothing on these sites keeps its code in a fenced markdown block, so an
extractor that only reads markdown finds almost nothing:

  * the landing site passes template literals to ``<CodeSample code={`…`} />``
    and to ``<Terminal command="…" />``, sometimes through a ``const SAMPLE_X``;
  * the documentation does the same through ``<CodeBlock>``, ``<CodeTabs>`` and
    its own ``<Terminal>``;
  * the tool gallery and the capture pages read their samples out of JSON;
  * a couple of samples are bare template literals inside a ``<pre>``;
  * the repository's own markdown uses fences.

All five are read here.

How a sample is run
-------------------
    python            written to a file and executed by the framework's
                      interpreter
    bash/console/sh   executed by a real shell with the ``effgen`` binary on PATH
    ts/tsx/js         collected and typechecked in one ``tsc --noEmit`` pass
    json/yaml/toml    parsed
    text/output       not code; not run

Samples that cannot stand alone
-------------------------------
Two conventions, both machine-readable, both declared rather than inferred:

  * ``continues`` — a ``<CodeBlock continues>``, or a ``continues: true`` beside
    a ``code:`` in a data file, carries on from the block above it. The harness
    joins it to its predecessor and runs the pair as one file. This is the same
    prop the page uses to draw the "continues" chip, so the reader and the
    harness agree about what the sample is. It means the block *immediately*
    above in the same language: a sample that carries on from further back is
    not a ``continues`` block, and is written out so it stands on its own.
  * ``snippet_policy.json`` beside this file — an explicit entry, keyed by the
    sha-256 of the sample text, for a sample that cannot run on an ordinary
    machine (it needs a GPU, a paid tier, a second host, or it would send mail).
    Each entry carries a reason a reader would accept. Keying on the text and
    not on a line number means the entry survives the page being edited and
    stops applying the moment the sample changes.

Usage
-----
    python effgen-docs/test_all_snippets.py                 # extract and run
    python effgen-docs/test_all_snippets.py --list          # extract only
    python effgen-docs/test_all_snippets.py --per-file      # counts per file
    python effgen-docs/test_all_snippets.py --only PATTERN  # a subset
    python effgen-docs/test_all_snippets.py --out DIR       # where the log goes

Exit status is 0 only when every sample the harness ran passed.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POLICY_PATH = HERE / "snippet_policy.json"

# Directories whose .ts/.tsx files are scanned for samples.
SOURCE_DIRS = ["app", "components", "shared", "data", "effgen-docs/src"]
SKIP_DIR_NAMES = {"node_modules", "dist", ".next", "out", ".git"}

# The components that *render* a sample. Their own `code`/`command` templates are
# the component's markup — `${command}`, `${output}` — not anything a reader is
# shown, so scanning them would report the component's source as a broken sample.
RENDERERS = {
    "components/ui/CodeSample.tsx", "components/ui/Terminal.tsx",
    "components/ui/TerminalReplay.tsx", "components/syntaxHighlight.ts",
    "effgen-docs/src/components/CodeBlock.tsx",
    "effgen-docs/src/components/ui/Terminal.tsx",
    "shared/syntaxHighlight.ts",
}

# Markdown that ships with the repository.
MARKDOWN_FILES = ["README.md", "effgen-docs/README.md", "effgen-docs/installation.md"]

# JSON that a page renders as a code sample.
JSON_CODE_SOURCES = [
    ("effgen-docs/src/data/toolGallery.json", "items", "code", "python"),
]
CAPTURE_GLOBS = ["data/captures.*.json"]

# Props and object keys whose template literal is a code sample. `output`,
# `caption`, `title`, `className` and the style props are deliberately absent:
# they are what a sample printed or how it is drawn, not code.
CODE_KEYS = {"code": None, "command": "shell"}

# Interpolations that appear inside a sample, and what they stand for at render
# time. Anything not listed here makes the sample unresolvable and is reported
# rather than guessed at.
def _interpolation_table() -> dict[str, str]:
    table: dict[str, str] = {}
    effgen_json = ROOT / "data" / "effgen.json"
    version = "1.3.0"
    if effgen_json.exists():
        data = json.loads(effgen_json.read_text())
        version = data.get("version", version)
    table["version"] = version
    table["siteData.version"] = version
    # `<Terminal command={`${capture.produced_by}`} />` names the command that
    # produced a capture. The capture files hold it, so it is looked up rather
    # than guessed; a capture whose `produced_by` has gone is then a failure,
    # which is the point.
    for path in sorted(ROOT.glob("data/captures.*.json")):
        raw = json.loads(path.read_text())
        for group in ("captures", "documents"):
            for name, cap in (raw.get(group) or {}).items():
                produced = (cap or {}).get("produced_by")
                if produced:
                    # "<command> · <what was captured>" — the note after the
                    # separator describes the capture, it is not part of it.
                    produced = produced.split(" · ")[0].strip()
                    slug = (cap or {}).get("slug")
                    if slug:
                        table[f'webCapture("{slug}").produced_by'] = produced
                    table[f'webCapture("{name}").produced_by'] = produced
                    table.setdefault("capture.produced_by", produced)
    return table


LANG_ALIASES = {
    "py": "python",
    "python3": "python",
    "sh": "bash",
    "shell": "bash",
    "console": "bash",
    "zsh": "bash",
    "bash": "bash",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "yml": "yaml",
    "text": "text",
    "txt": "text",
    "output": "text",
    "diff": "text",
    "ini": "text",
    "dockerfile": "text",
    "docker": "text",
    "http": "text",
    "sql": "text",
    "mermaid": "text",
    "bibtex": "text",
    "env": "text",
    "csv": "text",
    "makefile": "text",
    "nginx": "text",
    "promql": "text",
    "bibtex": "text",
}

RUNNABLE = {"python", "bash", "json", "yaml", "toml", "typescript", "javascript"}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

@dataclass
class Sample:
    id: str
    site: str
    file: str
    line: int
    key: str
    language: str
    code: str
    filename: str | None = None
    continues: bool = False
    unresolved: list[str] = field(default_factory=list)
    digest: str = ""
    policy: dict | None = None
    result: dict | None = None

    def brief(self) -> str:
        return f"{self.file}:{self.line}"


def _iter_files() -> list[str]:
    out: list[str] = []
    for d in SOURCE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_NAMES]
            for fn in sorted(filenames):
                if fn.endswith((".ts", ".tsx")):
                    rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
                    if rel not in RENDERERS:
                        out.append(rel)
    return sorted(out)


# `<Figure command="…">` says what produced a screenshot — "GET /playground ·
# one run of the prompt shown". It is provenance, in the same slot a Terminal
# uses for a command, and it is not a command line.
PROVENANCE_TAGS = {"Figure"}


def _element_tag(src: str, start: int) -> str | None:
    """The name of the JSX element the position `start` sits inside."""
    j = src.rfind("<", max(0, start - 2000), start)
    while j > 0:
        m = re.match(r"<([A-Za-z][\w.]*)", src[j:])
        if m:
            return m.group(1)
        j = src.rfind("<", 0, j)
    return None


def _is_comment(src: str, i: int) -> bool:
    """True when `/` at `i` opens a real comment rather than sitting in prose.

    `<code>/rbac/*</code>` on a page about RBAC paths is text, but read as
    JavaScript its `/*` opens a block comment that never closes — and the
    scanner then skips the rest of the file, code blocks and all. `https://…`
    has the same shape for `//`.

    A comment opens at the start of a line, after whitespace, or directly after
    `{` (the JSX `{/* … */}` form). It never opens directly after a letter, a
    `>` or a `:`.
    """
    if src[i] != "/" or src[i + 1:i + 2] not in ("/", "*"):
        return False
    if i == 0:
        return True
    return src[i - 1] in " \t\n({[,;=&|?+"


def _is_apostrophe(src: str, i: int) -> bool:
    """True when the quote at `i` is prose, not the start of a string.

    A documentation page is mostly JSX text, and JSX text contains apostrophes:
    `the server's own{' '}`. Read as JavaScript, that apostrophe opens a string
    that runs to the quote in `{' '}` — and from there every quote in the file is
    paired one out of step, so the scanner walks straight past the code blocks
    below it. This is how an extractor goes quietly blind: it reports a smaller
    number and nothing looks wrong.

    A quote that directly follows a letter or a digit is an apostrophe. A string
    literal never starts there — it follows `(`, `[`, `{`, `,`, `=`, `:` or
    whitespace.
    """
    return src[i] == "'" and i > 0 and (src[i - 1].isalnum() or src[i - 1] == ".")


def _templates(src: str):
    """Yield (start, end) for every top-level template literal in `src`.

    Quoted strings, `//` and `/* */` are skipped so a backtick inside one of them
    does not open a literal, and `${…}` nesting is tracked so an interpolation
    that itself contains a template literal does not close the outer one.
    """
    n, i = len(src), 0
    while i < n:
        c = src[i]
        if c == "/" and _is_comment(src, i) and src[i + 1] == "/":
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and _is_comment(src, i) and src[i + 1] == "*":
            end = src.find("*/", i + 2)
            # An unterminated "comment" is prose that only looked like one.
            i = (i + 2) if end < 0 else end + 2
            continue
        if c in "'\"" and not _is_apostrophe(src, i):
            q, j = c, i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == q:
                    j += 1
                    break
                if src[j] == "\n":
                    break
                j += 1
            i = j
            continue
        if c == "`":
            start, j, depth = i, i + 1, 0
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "$" and j + 1 < n and src[j + 1] == "{":
                    depth += 1
                    j += 2
                    continue
                if src[j] == "}" and depth:
                    depth -= 1
                    j += 1
                    continue
                if src[j] == "`" and depth == 0:
                    break
                if src[j] == "`" and depth:
                    k, d2 = j + 1, 0
                    while k < n:
                        if src[k] == "\\":
                            k += 2
                            continue
                        if src[k] == "$" and k + 1 < n and src[k + 1] == "{":
                            d2 += 1
                            k += 2
                            continue
                        if src[k] == "}" and d2:
                            d2 -= 1
                            k += 1
                            continue
                        if src[k] == "`" and d2 == 0:
                            break
                        k += 1
                    j = k + 1
                    continue
                j += 1
            yield start, j
            i = j + 1
            continue
        i += 1


_KEY_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*(?:=\{|:|=)\s*$")

_QUOTED = r'"(?:[^"\\\n]|\\.)*"' r"|'(?:[^'\\\n]|\\.)*'"

# `command="…"`, `command={'…'}`, `code="…"` — the string-literal form.
#
# Written either as a JSX prop (`code="…"`) or as an object key (`code: "…"`).
# Both reach a reader the same way: the landing page's data files hold their
# samples in arrays of objects and hand them to `<CodeSample code={step.code}>`.
# Reading only the `=` form left 39 samples on the landing site unrun, three of
# them broken, so the `:` form is matched here too.
#
# A long command is usually wrapped in source as several literals joined with
# `+`. The continuation is part of the match and the pieces are joined before
# the sample runs, or the harness reports a truncated command as a failure.
_STRING_PROP_RE = re.compile(
    r"\b(command|code)\s*(?:=\s*\{?|:)\s*"
    rf"((?:{_QUOTED})(?:\s*\+\s*(?:{_QUOTED}))*)"
    r"\s*\}?"
)

_QUOTED_RE = re.compile(_QUOTED)

# A value with no whitespace anywhere is a bare token — an exit status, an error
# code, a route in a lookup table — that a page is quoting, not something a
# reader can run. Every real sample has a space or a newline in it, so this
# excludes the identifiers without excluding any command. A one-word value that
# names an executable is still a sample and stays.
_ONE_WORD_COMMAND_RE = re.compile(
    r"^(effgen|pip|pipx|python|python3|npm|npx|node|docker|curl|uv|bash|sh|make)\b"
)


def _join_quoted(text: str) -> str:
    """The value of one or more string literals joined with `+`."""
    return "".join(_unquote(m.group(0)) for m in _QUOTED_RE.finditer(text))


def _is_bare_token(value: str) -> bool:
    value = value.strip()
    return bool(value) and not re.search(r"\s", value) and not _ONE_WORD_COMMAND_RE.match(value)


def _object_head(src: str, pos: int) -> str:
    """The enclosing object literal, from its `{` up to `pos`.

    Used to read a sibling key — `continues: true` — off the same object the
    sample was written in, the way `_element_head` reads a JSX prop.
    """
    depth, i = 0, pos - 1
    while i >= 0 and pos - i < 4000:
        c = src[i]
        if c == "}":
            depth += 1
        elif c == "{":
            if depth == 0:
                return src[i:pos]
            depth -= 1
        i -= 1
    return ""


def _unquote(text: str) -> str:
    body = text[1:-1]
    out, i = [], 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(body[i + 1], body[i + 1]))
            i += 2
            continue
        out.append(body[i])
        i += 1
    return "".join(out)


def _in_masked_region(src: str, pos: int) -> bool:
    """True when `pos` sits inside a comment (a prop named in prose, not code)."""
    line_start = src.rfind("\n", 0, pos) + 1
    line = src[line_start:pos]
    if line.strip()[:1] == "*":
        return True
    return any(_is_comment(src, line_start + m.start()) for m in re.finditer(r"//", line))


def _key_before(src: str, start: int) -> str | None:
    head = src[max(0, start - 200):start]
    head = re.sub(r"\s+$", "", head) + " "
    m = _KEY_RE.search(head[:-1] + "")
    return m.group(1) if m else None


def _unescape(body: str, table: dict[str, str]) -> tuple[str, list[str]]:
    """Turn template-literal source into the string the browser renders."""
    out, unresolved = [], []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == "\\" and i + 1 < n:
            nxt = body[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(nxt, nxt))
            i += 2
            continue
        if c == "$" and i + 1 < n and body[i + 1] == "{":
            depth, j = 1, i + 2
            while j < n and depth:
                if body[j] == "{":
                    depth += 1
                elif body[j] == "}":
                    depth -= 1
                j += 1
            expr = body[i + 2:j - 1].strip()
            if expr in table:
                out.append(table[expr])
            else:
                unresolved.append(expr)
                out.append("{" + expr + "}")
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out), unresolved


def _element_head(src: str, start: int) -> str:
    """The props of the element the sample sits in — and of no other.

    A fixed-size window behind the backtick reaches back into whatever element
    came before, so a block with no `language` of its own inherited the language
    of the block above it. The window stops at the nearest element or object
    boundary instead.
    """
    lo = max(0, start - 600)
    head = src[lo:start]
    cut = max(head.rfind("<"), head.rfind("{\n"), head.rfind("},"))
    if cut > 0:
        head = head[cut:]
    return head


def _prop(head: str, name: str) -> str | None:
    m = re.search(name + r'\s*=\s*["\']([^"\']+)["\']', head)
    if m:
        return m.group(1)
    m = re.search(name + r'\s*=\s*\{\s*["\']([^"\']+)["\']\s*\}', head)
    if m:
        return m.group(1)
    m = re.search(name + r'\s*:\s*["\']([^"\']+)["\']', head)
    if m:
        return m.group(1)
    return None


def _guess_language(code: str) -> str:
    text = code.strip()
    if not text:
        return "text"
    first = text.splitlines()[0].strip()
    if first.startswith("@") and re.match(r"^@\w+\{", first):
        return "text"          # a bibtex entry, not a decorator
    if re.match(r"^(import |from |def |class |async def |@|print\(|#!/usr/bin/env python)", first):
        return "python"
    if re.match(r"^(effgen|pip|python|uv|npm|curl|export|cd |git |docker|kubectl|sudo|bash|\$ )", first):
        return "bash"
    if first.startswith("{") or first.startswith("["):
        return "json"
    return "text"


def _classify(lang_prop: str | None, filename: str | None, code: str) -> str:
    if lang_prop:
        return LANG_ALIASES.get(lang_prop.lower(), lang_prop.lower())
    if filename:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext:
            mapped = LANG_ALIASES.get(ext, ext)
            if mapped in RUNNABLE or mapped == "text":
                return mapped
        if filename.lower() in ("dockerfile", "makefile"):
            return "text"
    return _guess_language(code)


def extract() -> list[Sample]:
    table = _interpolation_table()
    samples: list[Sample] = []
    counter = collections.Counter()

    def add(site, file, line, key, language, code, filename=None,
            continues=False, unresolved=()):
        counter[site] += 1
        sid = f"{'L' if site == 'landing' else 'D' if site == 'docs' else 'R'}{counter[site]:04d}"
        digest = hashlib.sha256(code.strip().encode()).hexdigest()[:16]
        samples.append(Sample(id=sid, site=site, file=file, line=line, key=key,
                              language=language, code=code, filename=filename,
                              continues=continues, unresolved=list(unresolved),
                              digest=digest))

    # --- .ts / .tsx -------------------------------------------------------
    for rel in _iter_files():
        site = "docs" if rel.startswith("effgen-docs/") else "landing"
        src = (ROOT / rel).read_text(encoding="utf-8")
        # const NAME = `…` that is later handed to a code prop
        const_refs = set(re.findall(r"code=\{\s*([A-Za-z_$][\w$]*)\s*\}", src))
        const_refs |= set(re.findall(r"code:\s*([A-Za-z_$][\w$]*)\s*[,}]", src))
        for start, end in _templates(src):
            key = _key_before(src, start)
            body = src[start + 1:end]
            line = src.count("\n", 0, start) + 1
            head = _element_head(src, start)

            is_pre = key is None and re.search(r"<pre\b[^>]*>\s*\{\s*$", src[max(0, start - 400):start], re.S)
            if key in CODE_KEYS:
                pass
            elif key in const_refs:
                pass
            elif is_pre:
                key = "pre"
            else:
                continue

            code, unresolved = _unescape(body, table)
            code = textwrap.dedent(code) if code.startswith("\n") else code
            if not code.strip():
                continue
            lang_prop = _prop(head, "language")
            filename = _prop(head, "filename")
            if key == "command":
                if _element_tag(src, start) in PROVENANCE_TAGS:
                    continue
                language = "bash"
            elif key == "pre":
                language = _guess_language(code)
            elif site == "docs" and not lang_prop and not filename:
                # `CodeBlock`'s own default. Guessing here would disagree with
                # what the page actually renders.
                language = "python"
            else:
                language = _classify(lang_prop, filename, code)
            continues = bool(re.search(r"\bcontinues\b(?!\s*=\s*\{?\s*false)", head))
            add(site, rel, line, key, language, code, filename, continues, unresolved)

        # `command="python demo.py"` and `code="…"` — a one-line sample written
        # as an ordinary string rather than a template literal. These are the
        # commands under a `<Terminal>`, and there are several hundred of them;
        # an extractor that only reads template literals cannot see any of them.
        spans = list(_templates(src))
        for m in _STRING_PROP_RE.finditer(src):
            prop, quoted = m.group(1), m.group(2)
            if _in_masked_region(src, m.start()):
                continue
            # `render(code="def add(a, b): …")` inside a sample is an argument
            # of that sample, not a sample of its own.
            if any(a < m.start() < b for a, b in spans):
                continue
            value = _join_quoted(quoted)
            if not value.strip():
                continue
            if _is_bare_token(value):
                continue
            line = src.count("\n", 0, m.start()) + 1
            head = _element_head(src, m.start())
            if prop == "command" and _element_tag(src, m.start()) in PROVENANCE_TAGS:
                continue
            language = "bash" if prop == "command" else _classify(
                _prop(head, "language"), _prop(head, "filename"), value)
            # `continues` is a JSX prop on the element form and a sibling key on
            # the object form; both say the sample carries on from the one above.
            continues = bool(
                re.search(r"\bcontinues\b(?!\s*[=:]\s*\{?\s*false)", head)
                or re.search(r"\bcontinues\s*:\s*true\b", _object_head(src, m.start()))
            )
            add(site, rel, line, prop, language, value, _prop(head, "filename"),
                continues)

    # --- JSON that a page renders as code ---------------------------------
    for rel, list_key, field_name, language in JSON_CODE_SOURCES:
        path = ROOT / rel
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        items = raw.get(list_key, raw) if isinstance(raw, dict) else raw
        site = "docs" if rel.startswith("effgen-docs/") else "landing"
        for i, item in enumerate(items):
            code = (item or {}).get(field_name)
            if not code or not code.strip():
                continue
            add(site, rel, i + 1, field_name, language, code,
                filename=(item.get("name") or None))

    # --- captured terminal commands the pages show ------------------------
    for pattern in CAPTURE_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            raw = json.loads(path.read_text())
            rel = str(path.relative_to(ROOT))
            for group in ("captures", "documents"):
                for name, cap in sorted((raw.get(group) or {}).items()):
                    cmd = (cap or {}).get("command")
                    if not cmd or not cmd.strip():
                        continue
                    add("landing", rel, 0, f"{group}.{name}.command", "bash", cmd,
                        filename=name)

    # --- markdown ---------------------------------------------------------
    fence = re.compile(r"^```([A-Za-z0-9_+-]*)[^\n]*\n(.*?)^```", re.M | re.S)
    for rel in MARKDOWN_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        site = "docs" if rel.startswith("effgen-docs/") else "landing"
        for m in fence.finditer(src):
            lang = LANG_ALIASES.get(m.group(1).lower(), m.group(1).lower() or "")
            code = m.group(2)
            if not code.strip():
                continue
            add(site, rel, src.count("\n", 0, m.start()) + 1, "fence",
                lang or _guess_language(code), code)

    for s in samples:
        if s.filename:
            DECLARED_FILES.setdefault(s.file, set()).add(os.path.basename(s.filename))
    return samples


# Every filename a page declares on one of its samples, so a `python demo.py`
# under a block can be tied back to the block that writes demo.py.
DECLARED_FILES: dict[str, set] = {}


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

def load_policy() -> dict:
    if POLICY_PATH.exists():
        return json.loads(POLICY_PATH.read_text())
    return {"entries": {}}


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------

ENV = dict(os.environ)
for var in ("EFFGEN_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
    ENV.pop(var, None)
ENV["EFFGEN_NO_ANIM"] = "1"
ENV["PYTHONWARNINGS"] = "ignore"

# A sample that calls a provider needs a key. The keys are read from a file the
# runner points at and are never written to the log; set EFFGEN_ENV_FILE to the
# .env you want used, or leave it and the framework checkout beside this one is
# tried. Without keys the provider samples report `no-key` rather than failing,
# so the harness still says something true on a machine that has none.
def _load_keys() -> int:
    candidates = [os.environ.get("EFFGEN_ENV_FILE"),
                  str(ROOT.parent / "effGen" / ".env")]
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            continue
        loaded = 0
        for raw in open(cand, encoding="utf-8", errors="replace"):
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and value and name not in ENV:
                ENV[name] = value
                loaded += 1
        return loaded
    return 0


KEYS_LOADED = _load_keys()


def _run(cmd, cwd, timeout, stdin=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True,
                           timeout=timeout, input=stdin)
        return p.returncode, p.stdout[-4000:], p.stderr[-4000:]
    except subprocess.TimeoutExpired:
        return 124, "", f"no exit within {timeout}s"


MAGIC = re.compile(r"^\s*%{1,2}\w+", re.M)
RATE_LIMITED = re.compile(r"rate_limited|RESOURCE_EXHAUSTED|\b429\b|Too Many Requests")


def _as_python(code: str) -> str:
    """A notebook cell, as the Python that IPython would actually run.

    `%effgen_chat …` and `%%effgen_agent …` are cell magics the framework ships.
    They are what a reader types into a notebook, so the page shows them as they
    are typed — but `python` cannot parse them, and checking them by eye is how a
    magic that no longer exists survives. IPython's own transformer turns the
    cell into the Python it would execute, which is the same check every other
    sample gets.
    """
    if not MAGIC.search(code):
        return code
    from IPython.core.inputtransformer2 import TransformerManager
    return TransformerManager().transform_cell(code)


def run_python(sample: Sample, workdir: Path, timeout: int) -> dict:
    """Compile the sample, then execute it."""
    path = workdir / f"{sample.id}.py"
    try:
        source = _as_python(sample.code)
    except Exception as exc:
        return {"status": "fail", "stage": "notebook", "stderr": repr(exc)}
    path.write_text(source)
    rc, out, err = _run([sys.executable, "-m", "py_compile", str(path)], workdir, 60)
    if rc != 0:
        return {"status": "fail", "stage": "compile", "exit": rc, "stderr": err}
    if MAGIC.search(sample.code):  # noqa: E501 - see below
        # The transformed cell calls `get_ipython()`, which exists only inside a
        # running kernel. Parsing it is the whole of what can be checked here,
        # and it is the part that catches a magic that has been renamed.
        return {"status": "pass", "stage": "notebook",
                "note": "magic cell: transformed by IPython and parsed"}
    # A provider rate limit is the state of the environment, not a fault in the
    # sample, and it is exactly what a reader would wait out. Two retries, then
    # it is reported as the failure it is.
    for attempt in range(3):
        rc, out, err = _run([sys.executable, str(path)], workdir, timeout)
        if rc == 0 or not RATE_LIMITED.search(err):
            break
        if attempt < 2:
            time.sleep(20 * (attempt + 1))
    status = "pass" if rc == 0 else "fail"
    return {"status": status, "stage": "run", "exit": rc, "stdout": out,
            "stderr": err, "attempts": attempt + 1}


SHELL_PLACEHOLDER = re.compile(r"[<>]|\.\.\.|…|YOUR_|/path/to|example\.com")

# `effgen` subcommands that only read. These are executed for real, because a
# sample of `effgen tools list` is only true if that command still exists and
# still exits 0. Everything else is validated against the CLI's own argument
# parser instead of being run, so a sample that would start a server, spend
# money, send mail or write to a repository is checked without doing any of it.
CLI_READ_ONLY = {
    ("tools", "list"), ("tools", "info"), ("tools", "search"), ("tools", "categories"),
    ("models", "list"), ("models", "info"), ("models", "browse"), ("models", "search"),
    ("presets",), ("examples", "list"), ("config", "show"), ("config", "list"),
    ("prompts", "list"), ("prompts", "show"), ("prompts", "search"),
    ("runs", "list"), ("sessions", "list"), ("cost", "today"),
}
CLI_MUTATING_FLAG = re.compile(r"--(live|force|yes|write|apply|delete|rm|set-)")

# A command of the form `python demo.py` runs a file the page shows above it.
# The file's own sample is run on its own; what is checked here is that the page
# actually shows the file the command names.
RUNS_A_FILE = re.compile(
    r"^(?:python3?|uv run(?: python)?)\s+(?:-m\s+pytest\s+)?([\w./-]+\.py)\b")


_REDIRECT = re.compile(r"\s+\d?[<>]{1,2}\s*\S+")
_TRAILING_COMMENT = re.compile(r"""\s+\#(?=(?:[^"']*["'][^"']*["'])*[^"']*$).*$""")


def _effgen_invocations(code: str) -> list[str]:
    """Every `effgen …` command in a shell block, as the shell would see it.

    A sample is written for a reader, not for a parser: it wraps a long command
    over several lines with a trailing backslash, pipes the result into `jq`,
    chains with `&&`, and redirects into a file. All of that is shell plumbing
    around the command, and none of it belongs to the command the CLI has to
    accept — so it is unwrapped here before the parser is asked.
    """
    joined = re.sub(r"\\\n\s*", " ", code)
    # `<name>` is a command line saying "put a name here". Turn it into a word
    # before the redirect stripping below, which would otherwise read it as
    # `> name` and silently drop the argument.
    joined = re.sub(r"<([A-Za-z][\w .|-]*)>", r"\1", joined)
    out: list[str] = []
    for raw in joined.splitlines():
        line = re.sub(r"^\$\s+", "", raw.strip())
        if not line or line.startswith("#"):
            continue
        line = _TRAILING_COMMENT.sub("", line)
        for part in re.split(r"\|\||&&|[|;]", line):
            part = _REDIRECT.sub("", part).strip()
            if part.startswith("effgen"):
                out.append(part)
    return out


def _unknown_flags(line: str) -> list[str]:
    """The long options in `line` that the command it names does not accept."""
    import argparse
    from effgen.cli._main import create_parser

    words = [w for w in line.replace("…", " ").split() if w]
    sub = next((w for w in words[1:] if not w.startswith("-")), None)
    parser = create_parser()
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    target = parser
    if sub and actions and sub in actions[0].choices:
        target = actions[0].choices[sub]
        nested = [a for a in target._actions if isinstance(a, argparse._SubParsersAction)]
        rest = [w for w in words[words.index(sub) + 1:] if not w.startswith("-")]
        if nested and rest and rest[0] in nested[0].choices:
            target = nested[0].choices[rest[0]]
    known = {opt for a in target._actions for opt in a.option_strings}
    return [w.split("=")[0] for w in words
            if w.startswith("--") and w.split("=")[0] not in known]


def _effgen_path(line: str) -> tuple[str, ...]:
    parts = [p for p in line.split() if not p.startswith("-")]
    return tuple(parts[1:3])


def run_bash(sample: Sample, workdir: Path, timeout: int) -> dict:
    """Parse-check the block, then execute or parser-validate each effgen line."""
    code = sample.code
    # `effgen examples run <name>` is how a command line writes "put a name
    # here". To bash that is a redirect, so the placeholder is replaced with a
    # plain word before the block is parsed as shell.
    parsable = re.sub(r"<[A-Za-z][\w .|-]*>", "NAME", code)
    rc, out, err = _run(["bash", "-n"], workdir, 60, stdin=parsable)
    if rc != 0:
        return {"status": "fail", "stage": "parse", "exit": rc, "stderr": err}

    # `python demo.py` — the page must show demo.py.
    stripped = code.strip()
    if "\n" not in stripped:
        m = RUNS_A_FILE.match(re.sub(r"^\$\s+", "", stripped))
        if m:
            named = m.group(1)
            if "/" in named:
                # A path, not a bare filename: the sample is running a script
                # that ships in the framework repository rather than one the
                # page writes out. What can be checked is that it is still there.
                target = ROOT.parent / "effGen" / named
                if target.exists():
                    return {"status": "linked", "stage": "repo-file", "file": named}
                return {"status": "fail", "stage": "repo-file",
                        "stderr": f"runs {named}, which is not in the framework repository"}
            wanted = os.path.basename(named)
            if wanted in DECLARED_FILES.get(sample.file, set()):
                return {"status": "linked", "stage": "linked", "file": wanted}
            return {"status": "fail", "stage": "linked",
                    "stderr": f"runs {wanted}, which this page never shows"}

    ran, validated = 0, 0
    for line in _effgen_invocations(code):
        if "…" in line:
            # `effgen compare … --optimize cost` is the page saying "the command
            # above, plus this flag". It is not a command and never was, so what
            # is checked is that the flags it names still exist on that
            # sub-command — which is the thing the page is actually claiming.
            bad = _unknown_flags(line)
            validated += 1
            if bad:
                return {"status": "fail", "stage": "cli-flags", "line": line,
                        "stderr": "no such option on this command: " + ", ".join(bad)}
            continue
        if SHELL_PLACEHOLDER.search(line):
            # A placeholder is not a command; the flags around it are still
            # checked, with the placeholder standing in as a plain word.
            probe = SHELL_PLACEHOLDER.sub("PLACEHOLDER", line)
        else:
            probe = line
        path = _effgen_path(probe)
        executable = (path in CLI_READ_ONLY or path[:1] in CLI_READ_ONLY) \
            and not CLI_MUTATING_FLAG.search(probe) and probe == line
        if executable:
            rc, out, err = _run(["bash", "-lc", line], workdir, timeout)
            ran += 1
            if rc != 0:
                return {"status": "fail", "stage": "cli-run", "exit": rc,
                        "line": line, "stderr": (err or out)[-1500:]}
            continue
        rc, out, err = _run([sys.executable, "-c", CLI_CHECK, probe], workdir, 120)
        validated += 1
        if rc != 0:
            return {"status": "fail", "stage": "cli-parse", "exit": rc,
                    "line": line, "stderr": err[-1500:]}
    return {"status": "pass", "stage": "shell", "cli_run": ran, "cli_parsed": validated}


# Ask the CLI's own parser whether a command line is valid, without running it.
CLI_CHECK = r"""
import shlex, sys, io, contextlib
line = sys.argv[1]
argv = shlex.split(line)[1:]
from effgen.cli._main import create_parser
parser = create_parser()
buf = io.StringIO()
try:
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        parser.parse_args(argv)
except SystemExit as exc:
    if exc.code not in (0, None):
        sys.stderr.write(buf.getvalue())
        raise SystemExit(1)
"""


def run_json(sample: Sample, workdir: Path, timeout: int) -> dict:
    lines = [l for l in sample.code.splitlines() if l.strip()]
    jsonl = (sample.filename or "").endswith(".jsonl")
    try:
        if jsonl:
            for i, line in enumerate(lines, 1):
                try:
                    json.loads(line)
                except Exception as exc:
                    raise ValueError(f"line {i}: {exc}") from None
        else:
            json.loads(sample.code)
        return {"status": "pass", "stage": "parse", "lines": len(lines) if jsonl else 1}
    except Exception as exc:
        return {"status": "fail", "stage": "parse", "stderr": str(exc)}


def run_yaml(sample: Sample, workdir: Path, timeout: int) -> dict:
    try:
        import yaml
    except ImportError:
        return {"status": "skip", "reason": "pyyaml is not installed"}
    try:
        list(yaml.safe_load_all(sample.code))
        return {"status": "pass", "stage": "parse"}
    except Exception as exc:
        return {"status": "fail", "stage": "parse", "stderr": str(exc)}


def run_toml(sample: Sample, workdir: Path, timeout: int) -> dict:
    try:
        import tomllib
    except ImportError:
        return {"status": "skip", "reason": "tomllib is not available"}
    try:
        tomllib.loads(sample.code)
        return {"status": "pass", "stage": "parse"}
    except Exception as exc:
        return {"status": "fail", "stage": "parse", "stderr": str(exc)}


def typecheck(samples: list[Sample], workdir: Path) -> None:
    """Typecheck the TypeScript samples against the real client.

    The samples on `/docs/clients` import `effgen-client`, which ships in the
    framework repository at `clients/typescript`. Pointing the compiler at that
    source is what makes the check worth running: it fails if the page names a
    method or an error class the client does not export.

    A page's samples are compiled as one module, in page order, because that is
    what they are — the `chat` tab builds the `client` that the `stream` tab
    then iterates. Compiling each tab alone would only prove that a tab is not a
    whole program, which nobody claimed.
    """
    ts = [s for s in samples if s.language in ("typescript", "javascript")]
    if not ts:
        return
    tsc = ROOT / "node_modules" / ".bin" / "tsc"
    if not tsc.exists():
        for s in ts:
            s.result = {"status": "skip", "reason": "tsc is not installed; run npm install"}
        return
    d = workdir / "ts"
    d.mkdir(parents=True, exist_ok=True)

    by_file: dict[str, list[Sample]] = {}
    for s in ts:
        by_file.setdefault(s.file, []).append(s)

    # module name -> the samples in it and the line each one starts on
    layout: dict[str, list[tuple[int, Sample]]] = {}
    for i, (rel, group) in enumerate(sorted(by_file.items())):
        mod = f"m{i:03d}"
        lines, placed = [], []
        for s in group:
            placed.append((len(lines) + 1, s))
            lines.extend(s.code.splitlines())
            lines.append("")
        (d / f"{mod}.ts").write_text("\n".join(lines))
        layout[mod] = placed

    client_src = ROOT.parent / "effGen" / "clients" / "typescript" / "src"
    (d / "tsconfig.json").write_text(json.dumps({
        "compilerOptions": {
            "noEmit": True, "target": "ES2022", "module": "ESNext",
            "moduleResolution": "bundler", "strict": False, "skipLibCheck": True,
            "allowJs": True, "lib": ["ES2022", "DOM"],
            # Every sample file is a module, so top-level await is legal — which
            # is how the page writes them.
            "moduleDetection": "force",
            "types": ["node"],
            "typeRoots": [str(ROOT / "node_modules" / "@types")],
            "baseUrl": ".",
            "paths": {"effgen-client": [str(client_src / "index.ts")]},
        },
        "include": ["*.ts"],
    }, indent=1))
    rc, out, err = _run([str(tsc), "-p", str(d)], d, 300)

    blamed: dict[str, list[str]] = collections.defaultdict(list)
    for m in re.finditer(r"^(m\d+)\.ts\((\d+),(\d+)\): (.*)$", out, re.M):
        mod, line = m.group(1), int(m.group(2))
        owner = None
        for start_line, s in layout.get(mod, []):
            if start_line <= line:
                owner = s
        if owner is not None:
            blamed[owner.id].append(f"line {line}: {m.group(4)}")
    for s in ts:
        if blamed.get(s.id):
            s.result = {"status": "fail", "stage": "tsc",
                        "stderr": "\n".join(blamed[s.id][:6])}
        else:
            s.result = {"status": "pass", "stage": "tsc"}


RUNNERS = {
    "python": run_python,
    "bash": run_bash,
    "json": run_json,
    "yaml": run_yaml,
    "toml": run_toml,
}


# --------------------------------------------------------------------------
# the local server some samples talk to
# --------------------------------------------------------------------------

def start_server(port: int):
    """Start `effgen serve` for the duration of the run.

    A dozen samples on `/docs/openai-api`, `/docs/api-server` and `/docs/clients`
    post to `http://127.0.0.1:8000` and read `EFFGEN_API_KEY`. Deferring them
    would mean the pages that document the server are the pages nothing checks,
    so the harness starts one and shuts it down again. It binds to loopback, so
    nothing outside this machine is contacted or exposed.
    """
    if shutil.which("effgen") is None:
        return None
    ENV.setdefault("EFFGEN_API_KEY", "effgen-snippet-harness-local")
    # Samples show several different keys — a literal, an environment variable,
    # none at all — because each is making a different point about the client.
    # Dev mode accepts all of them, so what is exercised is the request the page
    # shows rather than the harness's own credential.
    ENV["EFFGEN_DEV_MODE"] = "1"
    try:
        proc = subprocess.Popen(
            ["effgen", "serve", "--port", str(port), "--rate-limit", "0"],
            env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    import urllib.request
    for _ in range(60):
        if proc.poll() is not None:
            return None
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return proc
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    return None


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="extract, do not run")
    ap.add_argument("--per-file", action="store_true", help="print counts per file")
    ap.add_argument("--only", default=None, help="only samples whose path matches")
    ap.add_argument("--lang", default=None, help="only this language")
    ap.add_argument("--out", default=None, help="directory for the report")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-serve", action="store_true",
                    help="do not start a local effgen server for the run")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    samples = extract()
    policy = load_policy()
    entries = policy.get("entries", {})
    for s in samples:
        s.policy = entries.get(s.digest)

    if args.only:
        samples = [s for s in samples if args.only in s.file]
    if args.lang:
        samples = [s for s in samples if s.language == args.lang]

    per_file = collections.Counter(s.file for s in samples)
    by_lang = collections.Counter(s.language for s in samples)
    print(f"extracted {len(samples)} samples from {len(per_file)} files")
    print(f"provider keys loaded: {KEYS_LOADED}")
    print("by language: " + ", ".join(f"{k}={v}" for k, v in by_lang.most_common()))
    if args.per_file or args.list:
        for f, n in sorted(per_file.items()):
            print(f"{n:5d}  {f}")
    if args.list:
        return 0

    out_dir = Path(args.out) if args.out else (ROOT / "snippet-results")
    out_dir.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="effgen-snippets-"))
    # `continues` blocks are joined to the sample above them in the same file.
    # A `continues` block carries on from the block above it — the block in the
    # same language, since a captured terminal session often sits between two
    # halves of one file and is not part of it.
    prev: dict[tuple[str, str], Sample] = {}
    for s in samples:
        key = (s.file, s.language)
        if s.continues and prev.get(key) is not None:
            s.code = prev[key].code.rstrip() + "\n\n" + s.code
        prev[key] = s

    typecheck(samples, work)

    todo = []
    for s in samples:
        if s.result is not None:
            continue
        if s.policy:
            s.result = {"status": "policy", "reason": s.policy.get("reason", ""),
                        "kind": s.policy.get("kind", "")}
            continue
        if s.unresolved:
            s.result = {"status": "fail", "stage": "extract",
                        "stderr": "unresolved interpolation: " + ", ".join(s.unresolved)}
            continue
        if s.language not in RUNNERS:
            s.result = {"status": "not-code", "reason": s.language}
            continue
        todo.append(s)

    # The pages address a local server on several ports: 8000 is "the effGen
    # server", and the others stand for "some other OpenAI-compatible server you
    # are pointing effGen at". Which ports those are is read out of the samples
    # rather than listed here, so a page that picks a new one is still covered.
    # One port is deliberately left alone: a sample that documents what happens
    # when nothing is listening needs nothing to be listening.
    wanted: set[int] = set()
    for t in todo:
        for port in re.findall(r"127\.0\.0\.1:(\d{2,5})", t.code):
            wanted.add(int(port))
    # Two kinds of port must be left alone: one a sample documents as dead, and
    # one a sample binds for itself. Occupying either would make the harness the
    # reason the sample fails.
    BINDS = ("HTTPServer(", "socketserver", "serve_forever", "uvicorn", "app.run(")
    reserved = {int(p) for t in todo for p in re.findall(r"127\.0\.0\.1:(\d{2,5})", t.code)
                if "nothing is listening" in t.code or "did not answer" in t.code
                or any(b in t.code for b in BINDS)}
    reserved |= {int(p) for t in todo if any(b in t.code for b in BINDS)
                 for p in re.findall(r"\b(?:port|PORT)\D{0,4}(\d{4,5})", t.code)}
    wanted -= reserved
    if any("EFFGEN_API_KEY" in t.code for t in todo):
        wanted.add(args.port)

    servers = []
    if not args.no_serve:
        for port in sorted(wanted):
            proc = start_server(port)
            print(f"local effgen server on {port}:", "up" if proc else "could NOT be started")
            if proc:
                servers.append(proc)

    print(f"running {len(todo)} samples")

    # One directory per page, and the page's samples run in it in the order they
    # appear. A page is written to be worked through from the top: the sample
    # that builds a knowledge base comes before the ones that search it, and the
    # file a later sample opens is written by an earlier one. Running them out of
    # order, or each in its own empty directory, tests something the page never
    # asked for.
    by_page: dict[str, list[Sample]] = {}
    for s in todo:
        by_page.setdefault(s.file, []).append(s)

    def work_page(item):
        rel, group = item
        d = work / re.sub(r"[^\w.-]", "_", rel)
        d.mkdir(parents=True, exist_ok=True)
        out = []
        for s in group:
            try:
                out.append((s, RUNNERS[s.language](s, d, args.timeout)))
            except Exception as exc:  # a broken runner must not look like a clean run
                out.append((s, {"status": "fail", "stage": "harness", "stderr": repr(exc)}))
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for group_results in pool.map(work_page, sorted(by_page.items())):
            for s, res in group_results:
                s.result = res
                mark = {"pass": "pass", "fail": "FAIL"}.get(res["status"], res["status"])
                print(f"  {mark:6s} {s.id} {s.language:10s} {s.brief()}", flush=True)

    counts = collections.Counter(s.result["status"] for s in samples)
    print()
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    failures = [s for s in samples if s.result["status"] == "fail"]
    if failures:
        print(f"\n{len(failures)} FAILURES")
        for s in failures:
            print(f"\n--- {s.id} {s.brief()} ({s.language}) [{s.result.get('stage')}]")
            print((s.result.get("stderr") or s.result.get("stdout") or "").strip()[-1200:])

    for proc in servers:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()

    report = out_dir / "E1-snippets-index.json"
    report.write_text(json.dumps([asdict(s) for s in samples], indent=1))
    print(f"\nindex written to {report}")
    shutil.rmtree(work, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
