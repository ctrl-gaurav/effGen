"""
tests/security/test_vuln_audit.py

Validates that the current Python environment has no HIGH/CRITICAL CVEs in
core effGen dependencies by running pip-audit in JSON mode.

The test is intentionally strict about *core* dependencies (those declared
in ``[project.dependencies]`` in ``pyproject.toml``) and permissive about
optional extras such as ``vllm``, ``torch``, ``transformers``, and dev tools.

Requirements:
    pip install pip-audit

If pip-audit is not installed, the tests are skipped with a clear message.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

try:  # tomllib is stdlib on Python 3.11+; fall back to the tomli backport
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Paths & helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
OUTPUT_DIR = REPO_ROOT / "dist"

# Last full pip-audit sweep across core + [all]: verified 2026-07-21 — a fresh
# install resolves every non-exempt dependency to a patched release (e.g.
# starlette 1.3.1 clears CVE-2026-54282/54283, aiohttp 3.14.x, cryptography
# 49.x, python-multipart 0.0.32, tornado 6.5.7, pypdf 6.13.3 clears
# GHSA-jm82-fx9c-mx94, mcp 1.28.1 clears CVE-2026-59950, httplib2 0.32.0 clears
# PYSEC-2026-3444, Pillow 12.3.0 clears the imaging set), so this gate is green
# out of the box. Both lockfiles are recompiled from the same specifiers, so
# they resolve to the same patched releases.
#
# An entry below is exempt because the fix is unreachable from here, not
# because the advisory is unimportant. Re-check each one when its upstream
# moves.
#
# Packages excluded from the HIGH/CRITICAL gate:
#   - dev-only utilities (py, pytest-forked, wheel) — not in the built wheel
#   - optional extras whose fixed release is a major jump under evaluation
#     (vllm, torch, transformers, xgrammar)
_EXEMPT = {
    "pip",  # installer, not a shipped runtime dep
    "setuptools",  # build backend; the version in requirements-lock.txt is
    # whatever torch permits, so it advances when torch does
    "wheel",
    "py",
    "pytest-forked",
    "vllm",
    "torch",
    "transformers",
    "xgrammar",
    "diskcache",  # no fix available upstream (CVE-2025-69872, pickle serialization)
    "chromadb",  # optional [vector-db] extra; no upstream fix for CVE-2026-45829
    "stanza",  # optional transitive dep of argostranslate, which requires exactly
    # stanza==1.10.1 — so 1.12.2 (the release that clears PYSEC-2026-3075) cannot
    # be installed until argostranslate widens that pin. The earlier
    # CVE-2026-54499 (unsafe pickle in a torch.load fallback) has no fix at all.
    # effgen never loads untrusted stanza checkpoints.
    "nltk",  # optional [eval] extra (also transitive via rouge_score), not a core
    # dependency; no upstream fix for CVE-2026-12243 (3.9.4 is the latest release).
    # The CVE is a path traversal reachable only when an attacker controls the
    # resource name passed to nltk.data.load()/find(); effgen passes only the
    # fixed corpus names "wordnet"/"omw-1.4", never an attacker-supplied path.
    "accelerate",  # CVE-2026-69112 affects every release through 1.14.0, which is
    # the latest on PyPI, so there is no version to upgrade to. The advisory is a
    # path traversal in load_checkpoint_in_model / load_checkpoint_and_dispatch,
    # which fail to sanitise weight_map entries in a sharded checkpoint index.
    # effgen never calls either function; it reaches accelerate only through
    # transformers from_pretrained(device_map=...), so the path is reachable only
    # by loading a checkpoint the user chose to trust. Drop this entry as soon as
    # a fixed accelerate ships.
}

def _pip_audit_runnable() -> bool:
    """True if ``pip_audit`` is runnable by *this* interpreter.

    We check ``sys.executable -m pip_audit`` rather than ``shutil.which`` so
    the skip decision matches how :func:`_run_pip_audit` actually invokes the
    tool. A ``pip-audit`` binary on ``PATH`` may belong to a different
    interpreter (e.g. ``~/.local/bin``) whose module is not importable here.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--version"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


_PIP_AUDIT = _pip_audit_runnable()
skip_no_audit = pytest.mark.skipif(
    not _PIP_AUDIT,
    reason="pip-audit not runnable by this interpreter; install with: pip install pip-audit",
)


def _run_pip_audit() -> list[dict]:
    """Run pip-audit --format json and return the list of dependency dicts."""
    result = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--format", "json", "--progress-spinner", "off"],
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(result.stdout).get("dependencies", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def _get_core_dep_names() -> set[str]:
    """Return the normalised names of core (non-optional) runtime deps."""
    with open(PYPROJECT, "rb") as fh:
        cfg = tomllib.load(fh)
    names: set[str] = set()
    for dep in cfg.get("project", {}).get("dependencies", []):
        # Strip markers and version specs
        name = dep.split(";")[0].strip()
        for sep in (">=", "<=", "==", "!=", ">", "<", "["):
            name = name.split(sep)[0]
        names.add(name.strip().lower().replace("-", "_").replace(".", "_"))
    return names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipAuditInstalled:
    """pip-audit must be importable/runnable."""

    def test_pip_audit_importable(self):
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("pip-audit not installed; run: pip install pip-audit")
        version = result.stdout.strip() or result.stderr.strip()
        assert version, "pip-audit --version returned no output"


@skip_no_audit
class TestNoHighCriticalInCoreEnv:
    """Core effGen deps must have no HIGH/CRITICAL CVEs.

    pip-audit does not expose severity in its JSON output (it uses OSV data
    which often lacks CVSS scores).  We treat *any* CVE in a non-exempt core
    dep as blocking, consistent with the CI gate in deps-audit.yml.
    """

    def test_no_blocking_vulns_in_core_deps(self):
        """Run pip-audit and assert no CVEs in core effGen dependencies."""
        deps = _run_pip_audit()
        assert deps, "pip-audit returned no dependency data — something went wrong"

        blocking: list[tuple[str, str, str]] = []
        for dep in deps:
            name = (dep.get("name") or "").lower()
            if name in _EXEMPT:
                continue
            for vuln in dep.get("vulns", []):
                blocking.append((name, dep.get("version", "?"), vuln.get("id", "?")))

        if blocking:
            lines = "\n".join(f"  {n}=={v}: {vid}" for n, v, vid in blocking)
            pytest.fail(
                f"pip-audit found {len(blocking)} vulnerability/ies in core deps:\n{lines}\n"
                "Upgrade the affected packages or add them to _EXEMPT with justification."
            )

    def test_pip_audit_output_saved(self, tmp_path):
        """pip-audit JSON output can be saved to the outputs directory."""
        out_file = OUTPUT_DIR / "pip-audit.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--format",
                "json",
                "--output",
                str(out_file),
                "--progress-spinner",
                "off",
            ],
            capture_output=True,
            text=True,
        )
        assert out_file.exists(), f"pip-audit did not create output at {out_file}"
        with open(out_file) as fh:
            data = json.load(fh)
        assert "dependencies" in data, "pip-audit JSON must have a 'dependencies' key"
        assert len(data["dependencies"]) > 10, "Expected >10 dependencies in audit output"

    def test_total_dep_count_reasonable(self):
        """Environment must have a reasonable number of packages (sanity check)."""
        deps = _run_pip_audit()
        n = len(deps)
        assert n >= 20, f"Only {n} packages found — pip-audit may have failed silently"


@skip_no_audit
class TestExemptPackagesKnownVulns:
    """Document known exemptions so they are explicit and reviewable."""

    def test_known_exempt_packages_documented(self):
        """All entries in _EXEMPT must have a rationale (documented here)."""
        rationale = {
            "pip": "packaging toolchain, not a runtime dependency of effgen",
            "setuptools": "packaging toolchain, not a runtime dependency of effgen",
            "wheel": "packaging toolchain, not a runtime dependency of effgen",
            "py": "dev-only (required by pytest-forked); not shipped in production wheel",
            "pytest-forked": "dev-only test utility; not shipped in production wheel",
            "vllm": "optional extra [vllm]; upstream has known issues, not in core",
            "torch": "optional extra; upstream PyTorch CVEs in non-default codepaths",
            "transformers": "optional extra; upstream HuggingFace CVEs in conversion scripts",
            "xgrammar": "optional extra (vllm dependency); segfault in nested syntax",
            "diskcache": "no upstream fix available for CVE-2025-69872 (pickle); "
                         "effgen does not use pickle-deserialization paths",
            "chromadb": "optional [vector-db] extra (not a core dependency); no "
                        "upstream fix for CVE-2026-45829",
            "stanza": "optional transitive dep via argostranslate (not a core "
                      "dependency); no upstream fix for CVE-2026-54499 (unsafe "
                      "pickle in torch.load fallback); effgen never loads untrusted "
                      "stanza checkpoints",
            "nltk": "optional [eval] extra (also transitive via rouge_score), not "
                    "a core dependency; no upstream fix for CVE-2026-12243 (3.9.4 is "
                    "latest); the path-traversal requires an attacker-controlled "
                    "resource name to nltk.data.load()/find(), and effgen only passes "
                    "the fixed corpus names 'wordnet'/'omw-1.4'",
            "accelerate": "CVE-2026-69112 affects every release through 1.14.0, the "
                          "latest on PyPI, so there is no version to upgrade to; the "
                          "path traversal is in load_checkpoint_in_model / "
                          "load_checkpoint_and_dispatch, which effgen never calls, and "
                          "accelerate is reached only through transformers "
                          "from_pretrained(device_map=...), so it needs a checkpoint "
                          "the user chose to trust",
        }
        missing = _EXEMPT - set(rationale.keys())
        assert not missing, (
            f"Packages in _EXEMPT without a documented rationale: {missing}. "
            "Add a comment explaining why the CVE is acceptable."
        )
        # All documented packages must be in _EXEMPT too
        undocumented_removed = set(rationale.keys()) - _EXEMPT
        # This is informational only — docs may lead the code
        if undocumented_removed:
            import warnings

            warnings.warn(
                f"Rationale documents {undocumented_removed} which are not in _EXEMPT "
                "(may be stale documentation)",
                stacklevel=2,
            )

    def test_ci_exempt_list_matches_test_exempt(self):
        """The CI gate's EXEMPT_PACKAGES must be a subset of this module's _EXEMPT.

        The two lists are maintained separately (one in the workflow YAML, one
        here).  If they diverge, a CVE could pass CI but fail the test or vice
        versa.  Every package the CI gate exempts must also be exempt here.
        """
        import re

        workflow = REPO_ROOT / ".github" / "workflows" / "deps-audit.yml"
        if not workflow.exists():
            pytest.skip("deps-audit.yml not present")

        text = workflow.read_text()
        # Grab the EXEMPT_PACKAGES = { ... } literal from the embedded python block
        m = re.search(r"EXEMPT_PACKAGES\s*=\s*\{(.*?)\}", text, re.DOTALL)
        assert m, "Could not find EXEMPT_PACKAGES in deps-audit.yml"
        ci_exempt = set(re.findall(r'"([^"]+)"', m.group(1)))
        assert ci_exempt, "EXEMPT_PACKAGES in CI workflow parsed as empty"

        extra_in_ci = ci_exempt - _EXEMPT
        assert not extra_in_ci, (
            f"CI deps-audit.yml exempts packages not in test _EXEMPT: {extra_in_ci}. "
            "Keep the two exemption lists in sync."
        )
