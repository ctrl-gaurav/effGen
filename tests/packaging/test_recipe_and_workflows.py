"""Offline distribution-hygiene checks: conda recipe, constraints, workflows.

These are lint-level guards (no network, no install) that keep the release-
adjacent assets from silently drifting:

* the conda recipe must source the *real* package version (not the 0.0.0 that a
  metadata-free ``setup.py`` shim yields via ``load_setup_py_data``), and its
  ``run:`` dependencies must track ``pyproject.toml``;
* the per-CUDA torch ``constraints-*.txt`` files must exist and pin torch so an
  extras install can't replace a driver-compatible build;
* every GitHub workflow must parse, avoid end-of-life action versions, and the
  real-model smoke workflows must reference model ids that exist in the shipped
  catalog snapshots (the registry is the source of truth).
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = REPO_ROOT / "conda-recipe" / "meta.yaml"
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
DATA_DIR = REPO_ROOT / "effgen" / "models" / "_data"

# conda-forge package name -> pip/pyproject distribution name.
_CONDA_TO_PYPI = {"pytorch": "torch"}


def _package_version() -> str:
    text = (REPO_ROOT / "effgen" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    assert m, "could not find __version__ in effgen/__init__.py"
    return m.group(1)


def _pyproject_core_dep_names() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    names = set()
    for spec in data["project"]["dependencies"]:
        name = re.split(r"[<>=!~;\[\s]", spec, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def _recipe_run_deps() -> list[str]:
    """Extract the ``requirements: run:`` list items from the (jinja) meta.yaml."""
    lines = RECIPE.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    in_run = False
    for line in lines:
        if re.match(r"^\s{2}run:\s*$", line):
            in_run = True
            continue
        if in_run:
            item = re.match(r"^\s{4}-\s+(.*\S)\s*$", line)
            if item:
                out.append(item.group(1))
                continue
            # a non-list, non-blank line ends the run block
            if line.strip() and not line.startswith(" " * 5):
                break
    return out


# --------------------------------------------------------------------------- #
# conda recipe
# --------------------------------------------------------------------------- #

def test_recipe_does_not_use_setup_py_data():
    """Guard against the 0.0.0 regression: version must come from the package."""
    raw = RECIPE.read_text(encoding="utf-8")
    # Strip jinja comments ({# ... #}) so prose mentioning the old call doesn't count.
    text = re.sub(r"\{#.*?#\}", "", raw, flags=re.DOTALL)
    assert "load_setup_py_data" not in text, (
        "meta.yaml uses load_setup_py_data, which yields no version from the "
        "metadata-free setup.py shim and silently packages 0.0.0"
    )
    assert "effgen/__init__.py" in text and "__version__" in text, (
        "meta.yaml should source the version from effgen/__init__.py"
    )


def test_recipe_renders_real_version():
    """If conda-build is present, the rendered version must be the real one."""
    pytest.importorskip("conda_build")
    from conda_build.config import Config
    from conda_build.metadata import MetaData

    meta = MetaData(str(RECIPE.parent), config=Config())
    assert meta.version() == _package_version()
    assert meta.name() == "effgen"


def test_dockerfile_version_arg_tracks_package():
    """The image's ``ARG VERSION`` default must equal the package version.

    It is baked into ``org.opencontainers.image.version`` at build time, so a
    stale default mislabels every default-built image. Guard it the same way the
    conda recipe version is guarded.
    """
    dockerfile = REPO_ROOT / "deploy" / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    args = re.findall(r"^ARG VERSION=(\S+)\s*$", text, flags=re.MULTILINE)
    assert args, "no `ARG VERSION=` default found in deploy/docker/Dockerfile"
    version = _package_version()
    assert all(v == version for v in args), (
        f"deploy/docker/Dockerfile ARG VERSION defaults {args} must all equal "
        f"the package version {version!r}"
    )


@pytest.mark.skipif(tomllib is None, reason="needs tomllib/tomli")
def test_recipe_run_deps_track_pyproject():
    """Every conda run dep (minus python) must map to a pyproject core dep."""
    core = _pyproject_core_dep_names()
    stray = []
    for dep in _recipe_run_deps():
        name = re.split(r"[<>=!~\s]", dep, maxsplit=1)[0].strip().lower()
        if name == "python":
            continue
        mapped = _CONDA_TO_PYPI.get(name, name)
        if mapped not in core:
            stray.append(f"{dep} (-> {mapped})")
    assert not stray, (
        "conda recipe run: deps not present in pyproject core dependencies "
        f"(drift): {stray}"
    )


# --------------------------------------------------------------------------- #
# torch constraints
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", [
    "constraints-cu124.txt",
    "constraints-cu128.txt",
    "constraints-cu130.txt",
    "constraints-cpu.txt",
])
def test_constraints_files_pin_torch(name):
    path = REPO_ROOT / name
    assert path.exists(), f"missing constraints file: {name}"
    body = path.read_text(encoding="utf-8")
    assert re.search(r"^torch==", body, re.MULTILINE), f"{name} must pin torch=="
    assert "download.pytorch.org/whl/" in body, (
        f"{name} should carry an --extra-index-url for the pytorch wheels"
    )


# --------------------------------------------------------------------------- #
# GitHub workflows
# --------------------------------------------------------------------------- #

def test_workflows_parse_as_yaml():
    yaml = pytest.importorskip("yaml")
    assert WORKFLOWS, "no workflow files found"
    for wf in WORKFLOWS:
        with wf.open() as fh:
            list(yaml.safe_load_all(fh))  # raises on malformed YAML


def test_workflows_avoid_eol_actions():
    """No end-of-life action major versions (the common silent-failure source)."""
    eol = ("actions/upload-artifact@v3", "actions/download-artifact@v3",
           "actions/checkout@v2", "actions/checkout@v1",
           "actions/setup-python@v2", "actions/setup-python@v3")
    offenders = []
    for wf in WORKFLOWS:
        text = wf.read_text(encoding="utf-8")
        for bad in eol:
            if bad in text:
                offenders.append(f"{wf.name}: {bad}")
    assert not offenders, f"EOL GitHub Action versions in workflows: {offenders}"


def test_workflow_model_ids_exist_in_catalog():
    """Real-model smoke workflows must reference ids present in the catalog."""
    def catalog_ids(provider: str) -> set[str]:
        data = json.loads((DATA_DIR / f"{provider}.json").read_text())
        models = data.get("models", data) if isinstance(data, dict) else data
        return {m.get("id") for m in models}

    cerebras = catalog_ids("cerebras")
    groq = catalog_ids("groq")

    # ids hard-coded in the smoke workflows (kept in lockstep with the bundled catalog)
    assert "gpt-oss-120b" in cerebras, "nightly/cerebras-quick references a dead cerebras id"
    assert "openai/gpt-oss-20b" in groq, "groq smoke references a dead groq id"
    assert "openai/gpt-oss-120b" in groq, "groq fallback references a dead groq id"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
