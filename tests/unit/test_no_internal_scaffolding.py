"""Guard against internal build-process scaffolding leaking into shipped source.

Shipped code, tests, docs, configs and workflows describe *behavior*, not the
internal process used to build the project, and they describe that behavior
plainly — without self-congratulatory adjectives. This gate scans every tracked
text file and fails on two classes of text:

**(a) Internal process jargon** that should never ship:

* internal tracking IDs (e.g. ``VF5``, ``GA4``, ``RA-N3``, ``SEC1``, ``FN-2``,
  ``Audit-2 #x``) and per-finding IDs (e.g. ``E1-1``, ``E11-3``),
* internal milestone references (``Phase 7``, ``Phase-7``, ``Phase_7``,
  ``Phases 16/21/23``, ``build plan``, ``stabilization sprint``) — the
  separator between ``Phase`` and its number may be a space, hyphen, or
  underscore, and the singular and plural forms are both matched,
* author/process breadcrumbs (``this phase``, ``as per audit``,
  ``fixed in phase``, ``builder/verifier added``),
* names of internal planning artifacts (``findings report``, ``phase brief``,
  ``zero-ignore``, ``ask-before-commit``, ``AUDIT_REPORT``) and paths into the
  internal planning tree (``followups/x.md``, ``outputs/104_phase/…``), which
  read as live cross-references but name a gitignored path no reader outside
  the authoring checkout can follow,
* leftover debugging (``breakpoint()`` / ``pdb.set_trace()`` /
  ``ipdb.set_trace()`` / ``import pdb`` / a JavaScript ``debugger;``),
* unresolved placeholder markers (``TODO`` / ``FIXME`` / ``XXX`` / ``HACK`` /
  ``TBD``) and hand-written ship blockers (``DO NOT SHIP``, ``NOTE TO SELF``).

**(b) Editorializing self-praise** — the code describing its *own* behavior with
approving adjectives/adverbs. State what the code does, not how good it is at it:
say "reports the failure with a typed error", not "fails honestly"; say
"degrades to plain output when rich is absent", not "degrades gracefully". The
gated vocabulary is the set that is essentially always editorializing when it
describes the software's behavior: ``honest``/``honestly``/``honesty``,
``graceful``/``gracefully``, ``delightful``/``delightfully``,
``beautiful``/``beautifully``, ``elegant``/``elegantly``, ``robustly``,
``seamless``/``seamlessly``, ``effortless``/``effortlessly``, ``magically``,
``blazing``/``blazingly``, ``production-grade``, ``production-ready``,
``world-class``, ``battle-tested``, ``bulletproof``, ``state-of-the-art``,
``flawless``/``flawlessly``, ``rock-solid``, ``industrial-strength``,
``enterprise-grade``, ``military-grade``, ``best-in-class``, ``turnkey``,
``lightning-fast``, ``hassle-free``, ``painless``/``painlessly``, ``sleek``,
``gorgeous``, ``stunning``, ``pristine``, ``polished``, ``slick``,
``snappy``, ``buttery``/``butter-smooth``, ``silky``, ``supercharged``,
``intuitive``/``intuitively``, ``awesome``, ``amazing``, ``incredible``,
``superb``, ``impeccable``, ``exquisite``, ``masterful``, ``unparalleled``,
``top-notch``, ``game-changing``, ``revolutionary``, ``craftsmanship``,
``next-gen``, ``a joy to use``, and ``a breeze``.

Two entries carry a narrow carve-out for a genuine technical use:

* ``graceful`` is gated except in ``graceful shutdown`` — the standard name
  for the ASGI/uvicorn lifecycle in which in-flight requests finish before
  the process exits, which is what ``effgen.api.middleware`` implements;
* ``beautiful`` would otherwise fire on ``BeautifulSoup``, the HTML parser,
  because a CamelCase hump is split into words before matching. The two
  modules that import it are allowlisted on that literal.

``cleanly``/``properly``/``correctly`` are deliberately *not* machine-gated:
they carry a large volume of plainly factual technical uses ("imports cleanly",
"routes correctly", "properly configured") that a regex cannot separate from the
rare filler use, so blanket-gating them would force an unwieldy allowlist.
``first-class`` is excluded for the same reason: it is a term of art for a type
or value the language/framework supports natively, and separating that from the
promotional use ("a first-class experience") needs a human. ``cutting-edge`` and
``bleeding-edge`` are excluded too: they describe how recent a dependency or
branch is ("installs the bleeding-edge main branch") as often as they praise.
Those words are scrubbed by human review, not by this gate.

Both classes are matched against each line twice: as written, and with
identifiers broken into words, so a gated term hidden inside a function or class
name (``test_fails_gracefully``, ``TestExecutorHonesty``) is caught too.

A small, documented allowlist excuses the handful of *legitimate* occurrences
(an SSN-format mask, files that intentionally exclude the internal planning
directory, the CI meta-files that themselves grep for these markers). The scan
is intentionally narrow and self-tested so it cannot silently no-op: a planted
violation of every pattern must be caught (see the ``test_detector_catches_*``
tests).

Scanning is by suffix (see ``_SOURCE_SUFFIXES``) and covers every file type
effGen authors prose or markup in — JavaScript in all three module forms
(``.js``/``.mjs``/``.cjs``, the last two being how the site's build and lint
configuration is written), the bundled web surfaces
(``.html``/``.css``), the brand assets (``.svg``), and the scaffolding templates
the ``create-plugin`` command emits (``.tmpl``/``.tpl``), whose comments and
copy ship to users like any other source. A file is matched on *any* of its
dotted suffixes and on a variant name prefix, so the forms that park a real file
type behind another word — a parked workflow (``release.yml.disabled``), a
per-target image (``Dockerfile.sandbox``) — are scanned as what they are.

Data files are out of scope by design:

* ``.txt``/``.jsonl`` fixtures and golden files hold quoted third-party prose
  (paper abstracts) and recorded model output — neither is effGen describing
  itself, and gating them would mean allowlisting verbatim quotations;
* ``.json`` is dependency lockfiles, provider catalog snapshots and generated
  schemas — machine-written payloads with no authored prose;
* ``.sql`` is query fixtures for the prompt-library evals.

``build_plan/`` is deliberately *not* scanned — it is gitignored internal
scaffolding and is never shipped. ``CHANGELOG.md``/``NEWS.md`` and the
``website/`` tree are scanned for process jargon but *exempt from the
editorializing check*: the first two are an append-only historical record, and
the third is marketing page copy mirrored from the published site. Neither is
exempt from the jargon check — a milestone reference or a finding ID is a leak
wherever it lands, and most of all on a public page.

The scan reads tracked files *and* files that are untracked but not ignored, so
a violation in a file that has been written but not yet committed is caught on
the run that introduces it rather than on the push that first tracks it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files scanned by suffix (source/docs/config) plus these exact names.
_SOURCE_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".toml", ".cfg", ".ini",
    ".yaml", ".yml", ".sh", ".bash", ".ts", ".js", ".mjs", ".cjs", ".tsx", ".jsx",
    # Authored markup that ships to users: the self-contained web surfaces,
    # the brand assets, and the plugin scaffolding templates.
    ".html", ".css", ".svg", ".tmpl", ".tpl",
}
_SOURCE_NAMES = {".gitignore", ".dockerignore", ".gitattributes", "MANIFEST.in", "Dockerfile"}

# Names whose variant forms are authored the same way as the base name: a
# per-target container image is conventionally ``Dockerfile.<variant>``.
_SOURCE_NAME_PREFIXES = ("Dockerfile.",)

# The only directory never scanned: internal planning scaffolding, which is
# gitignored and never shipped. Bundled datasets need no directory rule — their
# payload files are ``.txt``/``.json``/``.jsonl``, already out of scope by
# suffix, while the scripts that fetch them are ordinary authored source.
_SKIP_DIR_PREFIXES = ("build_plan/",)

# ``website/`` holds the marketing site and its documentation app, kept as a
# mirror of the site they are published from so an update there is a straight
# copy. Its page copy is written to sell the project, so it is exempt from the
# editorializing check — rewording it here would put the repository and the live
# site out of step. It is *not* exempt from the process-jargon check: a
# milestone reference or a finding ID on a public marketing page is a leak
# whatever the page is for, and there is no editorial reason to allow one.
_EDITORIALIZING_EXEMPT_DIR_PREFIXES = ("website/",)

# This gate embeds every forbidden pattern as a literal, so it cannot scan
# itself without self-tripping.
_SKIP_FILES = {"tests/unit/test_no_internal_scaffolding.py"}

# Dated release narrative. Scanned for process jargon, but exempt from the
# editorializing-self-praise check: these are append-only historical records of
# what each past release said, so rewording them would falsify the record.
# README.md/README_PYPI.md are deliberately NOT exempt — they describe the
# project as it is today and are held to the same standard as source.
_EDITORIALIZING_EXEMPT_FILES = {
    "CHANGELOG.md", "NEWS.md",
}

# ── forbidden patterns: (a) internal process jargon ───────────────────────────
PATTERNS: dict[str, re.Pattern[str]] = {
    # internal tracking IDs from the build/audit process. ``BUG-12``/``ISSUE-7``
    # are the same shape from the same process — an internal finding number that
    # means nothing to a reader outside the tracker it was filed in.
    "internal-tracking-id": re.compile(
        r"\b(?:VF\d+|GA\d+|RA-[NC]\d+|SEC\d+|FN-\d+|E\d+-\d+|BUG-\d+|ISSUE-\d+)\b"
        r"|Audit-2 #"
    ),
    # internal milestone / planning references. The separator between "Phase"
    # and the number may be a space, hyphen, or underscore ("Phase 7",
    # "Phase-7", "Phase_7") — an earlier hyphenated breadcrumb slipped past the
    # space-only form. The plural is matched too: a comment citing a range or a
    # list ("Phases 16/21/23") is the same breadcrumb.
    # An ordinal development phase is the same breadcrumb with the number left
    # off ("a later phase", "previous phases"), so the ordinal forms are matched
    # too. Only the ordinals that place a phase on a build timeline are gated —
    # "every phase shown" and "a phased rollout" stay ordinary English.
    "milestone-reference": re.compile(
        r"\bPhases?[ _-]?\d+\b|\bbuild[ _-]?plan\b|stabilization sprint"
        r"|\b(?:later|earlier|previous|next|future) phases?\b",
        re.IGNORECASE,
    ),
    # author/process breadcrumbs. The role words (builder/verifier/explorer)
    # are only flagged next to a process verb — "the verifier" is also the
    # ordinary name for a hash-checking component, and "explorer" for a file
    # browser.
    "process-breadcrumb": re.compile(
        r"\bthis phase\b|\bas per (?:audit|the report)\b|\bfixed in phase\b|"
        r"\bthe report (?:asked|requested|wanted)\b|\bbuilder agent\b|"
        r"\b(?:builder|verifier|explorer) (?:added|fixed|confirmed|noted|reported)\b",
        re.IGNORECASE,
    ),
    # Names of internal planning artifacts. These describe how the project is
    # built, never what it does, so there is no legitimate shipped use.
    "planning-artifact": re.compile(
        r"\bfindings? report\b|\bphase brief\b|\bexplorer report\b|"
        r"\bphase evidence\b|"
        r"\bzero[ -]ignore\b|\bask[ -]before[ -]commit\b|AUDIT_REPORT",
        re.IGNORECASE,
    ),
    # Paths inside the internal planning tree. A comment that points a reader
    # at one of these is unreadable outside the checkout it was written in,
    # and the directory itself is gitignored. The slash is required, so the
    # ordinary English "follow-ups" and "the findings" are untouched.
    "planning-path": re.compile(
        r"\b(?:resolved_)?follow[ _-]?ups?/|\bfindings/phase|"
        r"\boutputs/\d+_phase\b|\bbuild_plan/",
        re.IGNORECASE,
    ),
    # leftover debugging, in Python and in the bundled JavaScript
    "debug-leftover": re.compile(
        r"breakpoint\(\)|(?:pdb|ipdb|pytest)\.set_trace\(|"
        r"^\s*import +i?pdb\b|(?<![\w.])debugger\s*;",
        re.MULTILINE,
    ),
    # unresolved placeholder markers and hand-written ship blockers
    "placeholder-marker": re.compile(
        r"\b(?:TODO|FIXME|XXX|HACK|TBD)\b|\bDO NOT SHIP\b|\bNOTE TO SELF\b"
    ),
}

# ── forbidden patterns: (b) editorializing self-praise ────────────────────────
# Vocabulary that is essentially always the code praising its own behavior.
# See the module docstring for why cleanly/properly/correctly are excluded.
EDITORIALIZING_PATTERNS: dict[str, re.Pattern[str]] = {
    "praise-honest": re.compile(r"\bhonest(?:ly|y)?\b", re.IGNORECASE),
    # "graceful shutdown" is the standard name for the ASGI lifecycle where
    # in-flight requests finish before the process exits, so it is carved out;
    # every other use of the word is the code praising itself.
    "praise-graceful": re.compile(r"\bgraceful(?:ly)?\b(?!\s+shutdown)", re.IGNORECASE),
    "praise-delightful": re.compile(r"\bdelightful(?:ly)?\b", re.IGNORECASE),
    "praise-beautiful": re.compile(r"\bbeautiful(?:ly)?\b", re.IGNORECASE),
    "praise-elegant": re.compile(r"\belegant(?:ly)?\b", re.IGNORECASE),
    "praise-robustly": re.compile(r"\brobustly\b", re.IGNORECASE),
    "praise-seamless": re.compile(r"\bseamless(?:ly)?\b", re.IGNORECASE),
    "praise-effortless": re.compile(r"\beffortless(?:ly)?\b", re.IGNORECASE),
    "praise-magically": re.compile(r"\bmagically\b", re.IGNORECASE),
    "praise-blazing": re.compile(r"\bblazing(?:ly)?\b", re.IGNORECASE),
    "praise-production-grade": re.compile(r"\bproduction[ -]grade\b", re.IGNORECASE),
    "praise-production-ready": re.compile(r"\bproduction[ -]ready\b", re.IGNORECASE),
    "praise-world-class": re.compile(r"\bworld[ -]class\b", re.IGNORECASE),
    "praise-battle-tested": re.compile(r"\bbattle[ -]tested\b", re.IGNORECASE),
    "praise-bulletproof": re.compile(r"\bbullet[ -]?proof\b", re.IGNORECASE),
    "praise-state-of-the-art": re.compile(r"\bstate[ -]of[ -]the[ -]art\b", re.IGNORECASE),
    "praise-flawless": re.compile(r"\bflawless(?:ly)?\b", re.IGNORECASE),
    "praise-rock-solid": re.compile(r"\brock[ -]solid\b", re.IGNORECASE),
    "praise-industrial-strength": re.compile(r"\bindustrial[ -]strength\b", re.IGNORECASE),
    "praise-enterprise-grade": re.compile(r"\benterprise[ -]grade\b", re.IGNORECASE),
    "praise-military-grade": re.compile(r"\bmilitary[ -]grade\b", re.IGNORECASE),
    "praise-best-in-class": re.compile(r"\bbest[ -]in[ -]class\b", re.IGNORECASE),
    "praise-turnkey": re.compile(r"\bturn[ -]?key\b", re.IGNORECASE),
    "praise-lightning-fast": re.compile(r"\blightning[ -]fast\b", re.IGNORECASE),
    "praise-hassle-free": re.compile(r"\bhassle[ -]free\b", re.IGNORECASE),
    "praise-painless": re.compile(r"\bpainless(?:ly)?\b", re.IGNORECASE),
    # Visual self-praise, aimed at the bundled web surfaces now in scope.
    "praise-sleek": re.compile(r"\bsleek\b", re.IGNORECASE),
    "praise-gorgeous": re.compile(r"\bgorgeous\b", re.IGNORECASE),
    "praise-stunning": re.compile(r"\bstunning\b", re.IGNORECASE),
    "praise-pristine": re.compile(r"\bpristine\b", re.IGNORECASE),
    "praise-polished": re.compile(r"\bpolished\b", re.IGNORECASE),
    "praise-slick": re.compile(r"\bslick\b", re.IGNORECASE),
    "praise-snappy": re.compile(r"\bsnappy\b", re.IGNORECASE),
    "praise-buttery": re.compile(r"\bbuttery\b|\bbutter[ -]smooth\b", re.IGNORECASE),
    "praise-silky": re.compile(r"\bsilky\b", re.IGNORECASE),
    "praise-supercharged": re.compile(r"\bsuper[ -]?charg(?:ed|es|ing)\b", re.IGNORECASE),
    "praise-intuitive": re.compile(r"\bintuitive(?:ly)?\b", re.IGNORECASE),
    # Unqualified superlatives about the software's quality.
    "praise-awesome": re.compile(r"\bawesome\b", re.IGNORECASE),
    "praise-amazing": re.compile(r"\bamazing(?:ly)?\b", re.IGNORECASE),
    "praise-incredible": re.compile(r"\bincredibl[ey]\b", re.IGNORECASE),
    "praise-superb": re.compile(r"\bsuperb\b", re.IGNORECASE),
    "praise-impeccable": re.compile(r"\bimpeccabl[ey]\b", re.IGNORECASE),
    "praise-exquisite": re.compile(r"\bexquisite(?:ly)?\b", re.IGNORECASE),
    "praise-masterful": re.compile(r"\bmasterful(?:ly)?\b", re.IGNORECASE),
    "praise-unparalleled": re.compile(r"\bunparalleled\b", re.IGNORECASE),
    "praise-top-notch": re.compile(r"\btop[ -]notch\b", re.IGNORECASE),
    "praise-game-changing": re.compile(r"\bgame[ -]chang(?:er|ing)\b", re.IGNORECASE),
    "praise-revolutionary": re.compile(r"\brevolutionar(?:y|ily)\b", re.IGNORECASE),
    "praise-craftsmanship": re.compile(r"\bcraftsmanship\b", re.IGNORECASE),
    # Short marketing form only: "next generation of <hardware>" is ordinary
    # prose and appears in recorded example output.
    "praise-next-gen": re.compile(r"\bnext[ -]gen\b", re.IGNORECASE),
    "praise-joy-to-use": re.compile(r"\ba joy to\b", re.IGNORECASE),
    "praise-a-breeze": re.compile(r"\ba breeze\b", re.IGNORECASE),
}

# ── documented allowlist of legitimate occurrences ────────────────────────────
# Each entry: (tracked path, substring that must appear on the matched line).
# A matched line is excused only if both the path and the substring match.
ALLOWLIST: list[tuple[str, str]] = [
    # An SSN-format mask used by the PII redactor, not a placeholder marker.
    ("effgen/guardrails/content.py", "XXX-XX-XXXX"),
    # Files that intentionally exclude the internal planning directory.
    (".gitignore", "build_plan"),
    ("MANIFEST.in", "build_plan"),
    ("deploy/docker/.dockerignore", "build_plan"),
    (".gitleaks.toml", "build_plan"),
    ("website/eslint.config.mjs", "build_plan"),
    # The sibling jargon-check test carries the forbidden words as test data.
    ("tests/unit/test_onboarding.py", "for bad in"),
    # CI meta-files that themselves grep for placeholder markers.
    (".github/workflows/pr-check.yml", "TODO"),
    ("CONTRIBUTING.md", "without a tracking issue"),
    # Historical release-notes entry (changelog is human narrative, not source).
    ("CHANGELOG.md", "ACP TODO"),
    # The dated release narrative cites the tracker ids the entries closed. The
    # changelog is append-only, so rewriting those lines would falsify the
    # record; new source is still held to the pattern.
    ("CHANGELOG.md", "BUG-0"),
    ("CHANGELOG.md", "ISSUE-0"),
    # A @font-face Unicode subset descriptor, not a tracking id. Splitting
    # identifiers on the CamelCase hump turns "U+1E00-1E9F" into "U+1 E00-1 E9F",
    # and "E00-1" then matches the E<n>-<n> form. The two lines are the Latin and
    # Latin-Extended subsets of the self-hosted faces the documentation renders in.
    ("website/effgen-docs/src/styles/globals.css", "unicode-range:"),
]

# Documented allowlist for the editorializing check: genuinely-legitimate domain
# uses of an otherwise-gated word. Same (path, substring) form as above.
EDITORIALIZING_ALLOWLIST: list[tuple[str, str]] = [
    # Prompt-template payload: the wording is an instruction sent to the model
    # about the artifact *it* should design, not a claim about effGen itself.
    # Changing it would change the template's output.
    (
        "effgen/prompts/library/domains/data/etl_plan_v1.py",
        "You are a senior data engineer.",
    ),
    # ``BeautifulSoup`` is the name of the HTML parser these two modules import.
    # Identifier splitting turns the CamelCase hump into "Beautiful Soup", so
    # the gated adjective matches a third-party library name.
    ("effgen/rag/ingest.py", "BeautifulSoup"),
    ("effgen/tools/builtin/url_fetch.py", "BeautifulSoup"),
]


def _is_allowed(rel_path: str, line: str, allowlist: list[tuple[str, str]]) -> bool:
    return any(p == rel_path and needle in line for p, needle in allowlist)


# Every pattern below is anchored on ``\b``, and ``_`` is a word character while
# a CamelCase hump is not a boundary at all — so a gated word sitting inside an
# identifier (``test_fails_gracefully``, ``TestExecutorHonesty``) would never
# match. Each line is therefore also searched in a form where identifiers are
# broken into their constituent words. Splitting on the hump rather than on
# every capital keeps acronyms intact, and words that merely *contain* a gated
# stem (``dishonest``) stay unsplit and so stay unmatched.
_CAMEL_HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _split_identifiers(line: str) -> str:
    return _CAMEL_HUMP.sub(" ", line).replace("_", " ")


def _scan(
    rel_path: str,
    text: str,
    patterns: dict[str, re.Pattern[str]],
    allowlist: list[tuple[str, str]],
) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every unjustified hit.

    The allowlist is matched against the line as written, so an entry keeps
    naming the real text of the line it excuses.
    """
    out: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _is_allowed(rel_path, line, allowlist):
            continue
        split = _split_identifiers(line)
        for name, pat in patterns.items():
            if pat.search(line) or (split != line and pat.search(split)):
                out.append((lineno, name, line.strip()))
    return out


def find_violations(rel_path: str, text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every unjustified jargon hit."""
    return _scan(rel_path, text, PATTERNS, ALLOWLIST)


def find_editorializing(rel_path: str, text: str) -> list[tuple[int, str, str]]:
    """Return ``(line_no, pattern_name, line)`` for every editorializing hit."""
    return _scan(rel_path, text, EDITORIALIZING_PATTERNS, EDITORIALIZING_ALLOWLIST)


def _is_scanned_path(rel: str) -> bool:
    """Whether a tracked path is one this gate reads.

    A file is matched on *any* of its dotted suffixes and on a variant name
    prefix, so the forms that park a real file type behind another word — a
    parked workflow (``release.yml.disabled``), a per-target image
    (``Dockerfile.sandbox``) — are scanned as what they are.
    """
    if not rel or rel in _SKIP_FILES or rel.startswith(_SKIP_DIR_PREFIXES):
        return False
    name = rel.rsplit("/", 1)[-1]
    return (
        any(s in _SOURCE_SUFFIXES for s in Path(rel).suffixes)
        or name in _SOURCE_NAMES
        or name.startswith(_SOURCE_NAME_PREFIXES)
    )


def _tracked_source_files() -> list[str]:
    """Every source file this gate reads: tracked, plus staged-to-be.

    ``git ls-files`` alone leaves a hole. A file that has been written but not
    yet committed is invisible to it, so a violation in a brand-new file passes
    every local run and only fails on the push that first tracks the file —
    which is the run where it is most expensive to find. ``--others
    --exclude-standard`` adds exactly those files: untracked and *not* ignored.
    Ignored paths stay out, so gitignored scratch space (``build_plan/`` and
    anything a contributor has excluded) is unaffected.
    """
    listings = []
    for args in (["ls-files"], ["ls-files", "--others", "--exclude-standard"]):
        try:
            listings.append(subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True, text=True, check=True,
            ).stdout)
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("git not available / not a git checkout")
    seen: dict[str, None] = {}
    for out in listings:
        for rel in out.splitlines():
            if _is_scanned_path(rel):
                seen[rel] = None
    return list(seen)


def test_tracked_source_has_no_internal_scaffolding():
    violations: list[str] = []
    for rel in _tracked_source_files():
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, name, line in find_violations(rel, text):
            violations.append(f"{rel}:{lineno} [{name}] {line}")
    assert not violations, (
        "Internal build-process scaffolding found in tracked source. Describe "
        "the behavior, not the process (or add a documented allowlist entry):\n  "
        + "\n  ".join(violations)
    )


def test_tracked_source_has_no_editorializing_self_praise():
    violations: list[str] = []
    for rel in _tracked_source_files():
        if rel in _EDITORIALIZING_EXEMPT_FILES:
            continue
        if rel.startswith(_EDITORIALIZING_EXEMPT_DIR_PREFIXES):
            continue
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, name, line in find_editorializing(rel, text):
            violations.append(f"{rel}:{lineno} [{name}] {line}")
    assert not violations, (
        "Editorializing self-praise found in tracked source. Describe what the "
        "code does plainly, not how well it does it (or add a documented "
        "EDITORIALIZING_ALLOWLIST entry for a genuine domain use):\n  "
        + "\n  ".join(violations)
    )


# ── proof corpus ──────────────────────────────────────────────────────────────
# One entry per pattern, so a pattern cannot be added without a line proving it
# fires. ``test_every_pattern_carries_a_proof_sample`` asserts the mapping stays
# exhaustive in both directions.
JARGON_SAMPLES: dict[str, list[str]] = {
    "internal-tracking-id": [
        "this is the VF5 regression case",
        "the E11-3 per-finding case",
        "fixes Audit-2 #42",
        "GA4 covered the catalog refresh",
        "RA-N3 and SEC1 and FN-2 all landed",
        "closes BUG-012 and ISSUE-7",
    ],
    "milestone-reference": [
        "see Phase 7 of the build plan",
        "a Phase-35 breadcrumb",
        "the Phase_18 guard",
        "additive aliases (Phases 16/21/23)",
        "carried over from the stabilization sprint",
        "a later phase will revisit this",
        "regression tests for previous phases",
    ],
    "process-breadcrumb": [
        "as per audit, this phase reworked it",
        "as per the report, the retry budget was raised",
        "the report asked for a clearer error here",
        "builder added the fallback path",
        "the verifier confirmed this on three families",
    ],
    "planning-artifact": [
        "see the findings report for the full list",
        "the phase brief asked for a --json flag",
        "carried by the explorer report",
        "the zero-ignore policy applies here",
        "held until the ask-before-commit gate",
        "recorded in AUDIT_REPORT_6",
        "the live proof lives in the phase evidence",
    ],
    "planning-path": [
        "the wider seam is followups/mixin_attr_defined_seam.md",
        "closed in resolved_followups/stop_sequences.md",
        "written up in findings/phase12_report.md",
        "the transcript is in outputs/104_phase/104-suite.txt",
        "see build_plan/preamble.md",
    ],
    "debug-leftover": [
        "    breakpoint()  # debug",
        "    pdb.set_trace()",
        "    ipdb.set_trace()",
        "import pdb",
        "  debugger;",
    ],
    "placeholder-marker": [
        "x = 1  # TODO clean this up",
        "# FIXME: revisit the timeout",
        "# HACK around the SDK bug",
        "# XXX not sure about this",
        "retry budget: TBD",
        "# DO NOT SHIP",
        "# NOTE TO SELF: rewrite this",
    ],
}


def test_detector_catches_planted_violation():
    """The jargon detector must actually fire — a no-op gate would pass forever."""
    for name, samples in JARGON_SAMPLES.items():
        for sample in samples:
            hits = find_violations("some/source.py", sample)
            assert any(n == name for _, n, _ in hits), (name, sample)


def test_detector_catches_a_path_into_the_internal_planning_tree():
    """A comment pointing at an internal planning file is a leak.

    It reads as a live cross-reference but names a gitignored path, so nobody
    outside the authoring checkout can follow it. The slash is what makes it a
    path: ordinary prose about follow-ups or findings stays legal.
    """
    for sample in (
        "# the wider mixin seam is followups/mixin_attr_defined_seam.md",
        "# see build_plan/preamble.md",
        "# transcript: outputs/104_phase/104-suite.txt",
    ):
        hits = find_violations("effgen/core/some_module.py", sample)
        assert any(n == "planning-path" for _, n, _ in hits), sample
    for legal in (
        "Known follow-ups are tracked as GitHub issues.",
        "the findings are reported through the returned envelope",
        "collects follow up questions from the user",
    ):
        assert not any(
            n == "planning-path" for _, n, _ in find_violations("s.py", legal)
        ), legal


def test_detector_catches_hyphenated_and_underscored_phase():
    """The milestone pattern must catch every ``Phase``/number separator."""
    for sample in ("a Phase-35 breadcrumb", "the Phase_18 guard", "Phase 8 catalog"):
        hits = find_violations("some/source.py", sample)
        assert hits and hits[0][1] == "milestone-reference", sample
    # A bare "phase" with no number is ordinary English, not a milestone ref.
    assert not any(
        n == "milestone-reference" for _, n, _ in find_violations("s.py", "a phased rollout")
    )


def test_detector_catches_report_and_role_breadcrumbs():
    """Breadcrumbs naming the internal report or an internal role must fire."""
    for sample in (
        "as per the report, the retry budget was raised",
        "the report asked for a clearer error here",
        "the report requested a --json flag",
        "builder added the fallback path",
        "the verifier confirmed this on three families",
        "explorer reported a cryptic traceback here",
    ):
        hits = find_violations("some/source.py", sample)
        assert any(n == "process-breadcrumb" for _, n, _ in hits), sample
    # The same role words in their ordinary technical sense must stay clean:
    # a hash verifier, a file explorer, a package builder.
    for ok in (
        "EFFGEN_VERIFY_HASHES=1 must call the verifier",
        "opens the path in the system file explorer",
        "the wheel builder writes to dist/",
    ):
        assert not any(
            n == "process-breadcrumb" for _, n, _ in find_violations("s.py", ok)
        ), ok


def test_detector_catches_gated_words_inside_identifiers():
    """A gated word hidden in a snake_case or CamelCase name must still fire."""
    snake = [
        ("def test_init_with_string_model_fails_gracefully(self):", "praise-graceful"),
        ("def test_empty_workflow_is_honest_failure():", "praise-honest"),
        ("def test_renders_beautifully_in_notebook():", "praise-beautiful"),
        ("production_ready = True", "praise-production-ready"),
        ("def test_trace_with_no_steps_is_graceful():", "praise-graceful"),
    ]
    camel = [
        ("class TestPublicCodeExecutorHonesty:", "praise-honest"),
        ("class SeamlessMigrationHelper:", "praise-seamless"),
        ("soup = BeautifulSoup(html)", "praise-beautiful"),
        ("class PristineStdoutGuard:", "praise-pristine"),
    ]
    for sample, expected in snake + camel:
        hits = find_editorializing("some/source.py", sample)
        assert any(n == expected for _, n, _ in hits), sample

    # Process jargon hidden in an identifier is caught the same way.
    for jargon in ("def test_phase_7_regression():", "def test_phases_16_regression():"):
        assert any(
            n == "milestone-reference" for _, n, _ in find_violations("s.py", jargon)
        ), jargon

    # A word that merely *contains* a gated stem is not split, so it stays clean.
    for ok in ("dishonest_input = True", "class Blazer:", "the production environment"):
        assert not find_editorializing("some/source.py", ok), ok


def test_gated_terms_match_only_on_a_word_boundary():
    """Every pattern is anchored, so a gated term inside a longer word is not a hit.

    This is the property that keeps the gate usable: without it ``dishonest``,
    ``Blazer``, ``polishing`` and ``metaphases`` would all be violations, and the
    allowlist would have to carry every ordinary English word that happens to
    contain a gated stem. Asserted in both directions — the bare term fires, the
    term with a letter glued to either side does not.
    """
    boundary_cases = [
        # (fires on its own, does not fire when it is part of a longer word)
        # ``dishonest`` stays unsplit (no separator), so it stays unmatched;
        # ``honesty_check`` is deliberately the other way — the underscore is a
        # separator, so the identifier is split and the gated word does fire.
        ("honest", ("dishonest",)),
        ("blazing", ("trailblazing",)),
        ("polished", ("unpolished",)),
        ("intuitive", ("counterintuitively",)),
        ("elegant", ("inelegant",)),
    ]
    for term, glued in boundary_cases:
        assert find_editorializing("s.py", f"a {term} result"), term
        for word in glued:
            assert not find_editorializing("s.py", f"a {word} result"), word

    jargon_boundary = [
        ("Phase 7", ("Phased 7", "Metaphase 7")),
        ("BUG-12", ("DEBUG-12", "BUG-12a")),
        ("a later phase", ("a later phased rollout",)),
        # ``aTODO`` is deliberately absent: the CamelCase hump splits it into
        # "a TODO", which is the identifier-splitting rule doing its job.
        ("TODO", ("TODOLIST",)),
    ]
    for term, glued in jargon_boundary:
        assert find_violations("s.py", f"# {term} note"), term
        for word in glued:
            assert not find_violations("s.py", f"# {word} note"), word


def test_allowlist_matches_the_line_as_written():
    """Splitting identifiers must not defeat an allowlist entry."""
    # ".gitignore" excuses the literal "build_plan"; the split form reads
    # "build plan", which the milestone pattern also matches.
    assert not find_violations(".gitignore", "build_plan/")
    assert find_violations("some/source.py", "build_plan/")


def test_scan_covers_authored_markup_suffixes():
    """The bundled web surfaces and scaffolding templates are in scope."""
    for suffix in (".html", ".css", ".svg", ".tmpl", ".tpl"):
        assert suffix in _SOURCE_SUFFIXES, suffix
    scanned = set(_tracked_source_files())
    assert "effgen/dashboard/static/index.html" in scanned
    assert "effgen/playground/static/index.html" in scanned
    assert "effgen/cli/_templates/plugin/tools.py.tmpl" in scanned
    # Recorded model output and third-party quotations stay out of scope.
    for suffix in (".txt", ".jsonl", ".json", ".sql"):
        assert suffix not in _SOURCE_SUFFIXES, suffix


def test_scan_covers_javascript_module_suffixes():
    """All three JavaScript module forms are read, not only ``.js``.

    A site's build, lint and codegen configuration is conventionally written as
    ``.mjs``/``.cjs``, and those files carry ordinary authored comments — so a
    scan that reads only ``.js`` leaves authored JavaScript unread.
    """
    for suffix in (".js", ".mjs", ".cjs"):
        assert suffix in _SOURCE_SUFFIXES, suffix
    assert "website/eslint.config.mjs" in set(_tracked_source_files())


def test_scan_covers_files_whose_type_is_not_the_last_suffix():
    """A parked workflow and a per-target image are scanned as what they are."""
    scanned = set(_tracked_source_files())
    assert ".github/workflows/release.yml.disabled" in scanned
    assert "deploy/sandbox/Dockerfile.sandbox" in scanned


def test_scan_reaches_a_file_that_is_written_but_not_yet_committed(tmp_path):
    """A new, uncommitted file is scanned — the hole that used to reach a push.

    Written into the repository (not ``tmp_path``, which git cannot see) and
    removed again, so the checkout is left as it was found.
    """
    planted = REPO_ROOT / "tests" / "unit" / "_scaffolding_gate_probe.py"
    assert not planted.exists(), "probe path is already in use"
    planted.write_text("# see Phase 9 of the build plan\n", encoding="utf-8")
    try:
        scanned = set(_tracked_source_files())
        rel = "tests/unit/_scaffolding_gate_probe.py"
        assert rel in scanned, "an untracked source file is invisible to the gate"
        assert find_violations(rel, planted.read_text(encoding="utf-8"))
    finally:
        planted.unlink()
    assert rel not in set(_tracked_source_files())


def test_scan_does_not_reach_an_ignored_file(tmp_path):
    """Ignored scratch space stays out, so a contributor's scratch file is safe."""
    scanned = set(_tracked_source_files())
    assert not any(r.startswith("build_plan/") for r in scanned)


def test_scan_covers_scripts_that_sit_beside_bundled_data():
    """Only the dataset payloads are out of scope, not the code next to them."""
    scanned = set(_tracked_source_files())
    assert "examples/data/download_arc.py" in scanned
    assert "examples/data/arc_easy_test.jsonl" not in scanned
    assert "examples/data/arc_easy_test.txt" not in scanned


PRAISE_SAMPLES: dict[str, list[str]] = {
    "praise-honest": [
        "# fails honestly with a typed error",
        "the honesty of the cost label",
        "an honest 400",
    ],
    "praise-delightful": ["a delightful first-run experience", "starts up delightfully"],
    "praise-elegant": ["an elegant fallback path", "resolves elegantly"],
    "praise-robustly": ["handles arbitrary input robustly"],
    "praise-seamless": ["a seamless upgrade", "swaps seamlessly"],
    "praise-effortless": ["scales effortlessly", "an effortless setup"],
    "praise-magically": ["the cache magically warms"],
    "praise-blazing": ["blazing throughput", "blazingly fast startup"],
    "praise-production-grade": ["a production-grade pipeline", "production grade RAG"],
    "praise-production-ready": ["a production-ready chart", "production ready adapter"],
    "praise-world-class": ["world-class ergonomics", "world class tracing"],
    "praise-battle-tested": ["battle-tested retries", "battle tested sandbox"],
    "praise-bulletproof": ["bulletproof auth", "bullet-proof parsing"],
    "praise-state-of-the-art": ["state-of-the-art routing", "state of the art reranking"],
    "praise-flawless": ["flawless recovery", "replays flawlessly"],
    "praise-rock-solid": ["rock-solid checkpoints", "rock solid streaming"],
    "praise-industrial-strength": ["industrial-strength queueing"],
    "praise-enterprise-grade": ["enterprise-grade RBAC", "enterprise grade audit log"],
    "praise-military-grade": ["military-grade encryption"],
    "praise-best-in-class": ["best-in-class latency", "best in class recall"],
    "praise-turnkey": ["a turnkey deployment", "turn-key onboarding"],
    "praise-lightning-fast": ["lightning-fast cold starts"],
    "praise-hassle-free": ["hassle-free upgrades"],
    "praise-painless": ["a painless migration", "migrates painlessly"],
    "praise-sleek": ["a sleek dashboard"],
    "praise-gorgeous": ["gorgeous trace rendering"],
    "praise-stunning": ["a stunning topology view"],
    "praise-pristine": ["keeps stdout pristine"],
    "praise-polished": ["a polished REPL"],
    "praise-slick": ["a slick palette picker"],
    "praise-snappy": ["snappy cold starts"],
    "praise-buttery": ["buttery scrolling", "butter-smooth streaming"],
    "praise-silky": ["silky redraws"],
    "praise-supercharged": ["supercharged retrieval", "super-charged batching"],
    "praise-intuitive": ["an intuitive API", "reads intuitively"],
    "praise-awesome": ["an awesome dashboard"],
    "praise-amazing": ["amazing throughput", "amazingly small"],
    "praise-incredible": ["incredible recall", "incredibly fast"],
    "praise-superb": ["superb ergonomics"],
    "praise-impeccable": ["impeccable defaults", "impeccably typed"],
    "praise-exquisite": ["exquisite trace output"],
    "praise-masterful": ["masterful orchestration"],
    "praise-unparalleled": ["unparalleled coverage"],
    "praise-top-notch": ["top-notch errors", "top notch docs"],
    "praise-game-changing": ["a game-changing router", "a game changer"],
    "praise-revolutionary": ["a revolutionary sandbox"],
    "praise-craftsmanship": ["the craftsmanship of the CLI"],
    "praise-next-gen": ["a next-gen planner", "next gen tooling"],
    "praise-joy-to-use": ["a joy to operate"],
    "praise-a-breeze": ["upgrades are a breeze"],
    "praise-graceful": ["a graceful fallback", "degrades gracefully"],
    "praise-beautiful": ["a beautiful report", "renders beautifully"],
}

# Prose that must stay clean: factual technical writing, terms of art, and words
# that merely share a stem with a gated one.
PRAISE_NON_SAMPLES: tuple[str, ...] = (
    "routes to the correct adapter",
    "the model imports cleanly",
    "properly typed",
    "the production environment variable",
    "class Blazer:",
    "artwork classification",
    "the enterprise directory endpoint",
    "flaws = validate(config)",
    "installs the bleeding-edge main branch",
    "in-flight requests finish before a graceful shutdown",
    "def _install_graceful_shutdown(app, timeout):",
    "power the next generation of laptops",
    "a phased rollout of the new router",
    "the joystick axis is ignored",
    "polish = compute_polish_ratio(x)",
    "supercharger = None",
    "dishonest_input = True",
    "first-class functions are supported",
)


def test_detector_catches_planted_editorializing():
    """The editorializing detector must actually fire on each gated word."""
    for name, samples in PRAISE_SAMPLES.items():
        for sample in samples:
            hits = find_editorializing("some/source.py", sample)
            assert any(n == name for _, n, _ in hits), (name, sample)
    for ok in PRAISE_NON_SAMPLES:
        assert not find_editorializing("some/source.py", ok), ok


def test_every_pattern_carries_a_proof_sample():
    """No pattern may ship without a line proving it fires, and vice versa.

    This is what keeps the gate from drifting into a no-op: adding a pattern
    without a sample fails here, and deleting a pattern while leaving its sample
    behind fails here too.
    """
    assert set(JARGON_SAMPLES) == set(PATTERNS), (
        "JARGON_SAMPLES and PATTERNS disagree: "
        f"unproven={sorted(set(PATTERNS) - set(JARGON_SAMPLES))}, "
        f"orphaned={sorted(set(JARGON_SAMPLES) - set(PATTERNS))}"
    )
    assert set(PRAISE_SAMPLES) == set(EDITORIALIZING_PATTERNS), (
        "PRAISE_SAMPLES and EDITORIALIZING_PATTERNS disagree: "
        f"unproven={sorted(set(EDITORIALIZING_PATTERNS) - set(PRAISE_SAMPLES))}, "
        f"orphaned={sorted(set(PRAISE_SAMPLES) - set(EDITORIALIZING_PATTERNS))}"
    )
    for name, samples in {**JARGON_SAMPLES, **PRAISE_SAMPLES}.items():
        assert samples, f"{name} has no proof sample"


def test_gated_vocabulary_covers_the_house_style_list():
    """Every word the house style forbids must have a live pattern.

    Spelled out as literal prose rather than as pattern names so that renaming
    or dropping a pattern cannot quietly shrink what the gate covers.
    """
    must_be_caught = [
        # process jargon
        "see Phase 12", "Phases 3-5 landed", "the build plan says", "build_plan/",
        "this phase reworked it", "as per the report", "fixed in phase 4",
        "a later phase will revisit it", "regression tests for previous phases",
        "the builder added a fallback", "finding E3-2", "E4-1 regression",
        "closes BUG-012", "closes ISSUE-7", "see the phase evidence",
        "TODO", "FIXME", "XXX", "HACK", "breakpoint()", "pdb.set_trace()",
        # self-praise
        "fails honestly", "an honest error", "the honesty of it",
        "degrades gracefully", "a graceful fallback",
        "a delightful CLI", "renders beautifully", "a beautiful report",
        "an elegant design", "resolves elegantly", "handles it robustly",
        "a seamless upgrade", "swaps seamlessly", "effortless setup",
        "scales effortlessly", "magically resolved", "blazing fast",
        "blazingly quick", "production-grade RAG", "production-ready server",
        "world-class docs", "battle-tested retries", "bulletproof parsing",
        "state-of-the-art routing",
    ]
    for sample in must_be_caught:
        assert find_violations("s.py", sample) or find_editorializing("s.py", sample), sample

    # Deliberately NOT machine-gated (documented in the module docstring): the
    # filler adverbs and "first-class", which a regex cannot separate from their
    # very common factual uses. Asserted so the exclusion stays a choice.
    for excluded in (
        "imports cleanly", "routes correctly", "properly configured",
        "first-class functions", "the bleeding-edge branch",
    ):
        assert not find_editorializing("s.py", excluded), excluded


def test_allowlist_entries_are_live_and_load_bearing():
    """Every allowlist entry must name a real line that would otherwise fail.

    A stale entry (file or line gone) or a redundant one (the line does not
    actually trip a pattern) is a hole in the gate, so both fail here.
    """
    tracked = set(_tracked_source_files())
    for allowlist, finder, label in (
        (ALLOWLIST, find_violations, "ALLOWLIST"),
        (EDITORIALIZING_ALLOWLIST, find_editorializing, "EDITORIALIZING_ALLOWLIST"),
    ):
        for path, needle in allowlist:
            if label == "EDITORIALIZING_ALLOWLIST":
                assert path not in _EDITORIALIZING_EXEMPT_FILES, (
                    f"{label}: {path} is already exempt; the entry is dead"
                )
            elif path in _EDITORIALIZING_EXEMPT_FILES:
                pass  # jargon is still scanned on the exempt release narrative
            assert path in tracked or path in _SKIP_FILES, (
                f"{label}: {path} is not a scanned tracked file any more"
            )
            text = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
            matching = [ln for ln in text.splitlines() if needle in ln]
            assert matching, f"{label}: no line in {path} contains {needle!r} any more"
            # Without the entry the line must fail, or the entry excuses nothing.
            assert any(
                finder("some/other.py", ln) for ln in matching
            ), f"{label}: {path} / {needle!r} excuses nothing — remove it"


def test_scan_set_is_broad_and_did_not_silently_collapse():
    """A scan that quietly stops seeing files would pass forever.

    Pins the floor on how much the gate looks at: the total, and at least one
    file under every top-level tree that holds authored text.
    """
    scanned = _tracked_source_files()
    assert len(scanned) > 1250, f"scan set shrank to {len(scanned)} files"
    for prefix in (
        "effgen/", "tests/", "docs/", "examples/", "scripts/", "deploy/",
        ".github/workflows/", "assets/", "clients/", "website/",
    ):
        assert any(r.startswith(prefix) for r in scanned), prefix
    # Every suffix and exact name the scanner claims to cover is either present
    # in the tree or accepted by the matcher for a path of that kind.
    for suffix in _SOURCE_SUFFIXES:
        assert _is_scanned_path(f"pkg/sample{suffix}"), suffix
    for name in _SOURCE_NAMES:
        assert _is_scanned_path(f"pkg/{name}"), name
    for prefix in _SOURCE_NAME_PREFIXES:
        assert _is_scanned_path(f"pkg/{prefix}variant"), prefix


def test_every_scanned_file_kind_is_actually_read():
    """Planting a violation in a file of each scanned kind must be detected.

    Suffixes with no tracked file today (``.rst``, ``.pyi``, ``.jsonl``-free
    configs) are proven at the path level: the scanner accepts the path, and the
    detector fires on the planted line for it.
    """
    planted_jargon = "# see Phase 9 of the build plan"
    planted_praise = "# a production-ready, seamless experience"
    kinds = (
        [f"pkg/sample{s}" for s in sorted(_SOURCE_SUFFIXES)]
        + [f"pkg/{n}" for n in sorted(_SOURCE_NAMES)]
        + [f"pkg/{p}variant" for p in _SOURCE_NAME_PREFIXES]
    )
    for rel in kinds:
        assert _is_scanned_path(rel), rel
        assert find_violations(rel, planted_jargon), rel
        assert find_editorializing(rel, planted_praise), rel


def test_skipped_trees_are_the_documented_ones_only():
    """The skip list must stay small and explicit."""
    assert _SKIP_DIR_PREFIXES == ("build_plan/",)
    assert _SKIP_FILES == {"tests/unit/test_no_internal_scaffolding.py"}
    assert _EDITORIALIZING_EXEMPT_FILES == {"CHANGELOG.md", "NEWS.md"}
    assert _EDITORIALIZING_EXEMPT_DIR_PREFIXES == ("website/",)
    # The exempt release narrative is still scanned for process jargon.
    for rel in _EDITORIALIZING_EXEMPT_FILES:
        assert _is_scanned_path(rel), rel


def test_marketing_site_is_exempt_from_praise_but_not_from_jargon():
    """``website/`` is page copy, so it may sell — but it may not leak process.

    The tree used to be skipped outright, which meant a milestone reference or a
    finding ID could sit on a public marketing page and no gate would see it.
    """
    scanned = set(_tracked_source_files())
    assert any(r.startswith("website/") for r in scanned), "website/ is not scanned"
    # A praise word on a marketing page is allowed...
    assert _EDITORIALIZING_EXEMPT_DIR_PREFIXES == ("website/",)
    # ...but the jargon scan still reads those files, and still fires on them.
    assert find_violations("website/components/Features.tsx", "shipped in Phase 12")


def test_editorializing_allowlist_is_scoped_to_its_file():
    """An allowlisted line is excused only on its own path."""
    line = "        \"You are a senior data engineer. Design a production-ready ETL pipeline.\\n\\n\""
    assert not find_editorializing(
        "effgen/prompts/library/domains/data/etl_plan_v1.py", line
    )
    assert find_editorializing("effgen/elsewhere.py", line)


def test_allowlist_excuses_legitimate_lines():
    # SSN mask is not flagged on its own file...
    assert not find_violations(
        "effgen/guardrails/content.py", "    # US Social Security Number: XXX-XX-XXXX"
    )
    # ...but the same text elsewhere is flagged.
    assert find_violations("effgen/elsewhere.py", "mask = 'XXX-XX-XXXX'")


def test_no_test_module_lets_the_repo_dotenv_override_the_environment():
    """``load_dotenv`` is always called with an explicit ``override=``.

    python-dotenv defaults to ``override=False``, but the argument is easy to
    leave off while meaning it, and one module that omits it lets the
    repository ``.env`` win over a key the operator exported on purpose — a run
    aimed at one account then silently bills another.

    Read from the syntax tree, not by pattern: the path argument is usually a
    ``Path(...)`` expression, and a text scan stops at its closing bracket
    before ever reaching the keyword.
    """
    import ast

    root = Path(__file__).resolve().parents[2]
    # An sdist or an exported copy is not a git checkout, and this rule is
    # about what the repository contains, so there is nothing to check there.
    try:
        listing = subprocess.run(
            ["git", "ls-files", "tests/"], cwd=root,
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout, so there is no tracked file list to scan")
    tracked = listing.stdout.split()

    offenders = []
    for name in tracked:
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = getattr(func, "id", None) or getattr(func, "attr", None)
            if called != "load_dotenv":
                continue
            # A bare call is the search walk, which takes no path to override.
            if not node.args and not node.keywords:
                continue
            if not any(kw.arg == "override" for kw in node.keywords):
                offenders.append(f"{name}:{node.lineno}")

    assert not offenders, (
        "load_dotenv without an explicit override= (use override=False):\n  "
        + "\n  ".join(offenders)
    )
