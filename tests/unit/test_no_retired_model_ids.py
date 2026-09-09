"""Guard the shipped surfaces against model ids their provider has retired.

A provider can withdraw a model id at any time. When it does, every place the
id is written down goes stale at once: the adapter default, the bundled
catalog, CLI help, error messages, doc examples and the quick starts in the
READMEs. Those quick starts are copy-paste commands, so a retired id there is a
404 the first time somebody tries the project.

The ids below were each confirmed retired against the provider's own live
listing on the date given. The check is a plain text scan: a retired id must
not appear anywhere in the shipped tree.

Adding a row here is what makes a retirement stick. Retire an id, replace its
uses, then record it — after that the gate keeps it from drifting back in
through a doc example or a copied snippet.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Retired ids, with the provider and the date the retirement was confirmed
#: against that provider's live model listing.
RETIRED_MODEL_IDS: dict[str, tuple[str, str]] = {
    "llama-3.1-8b-instant": ("groq", "2026-09-07"),
    "llama-3.3-70b-versatile": ("groq", "2026-09-07"),
}

#: The surfaces a user reads or runs. `tests/` is scanned too: a test that
#: still asks for a retired id fails against the live provider.
SCANNED_ROOTS = (
    "effgen", "docs", "tests", "scripts", "tools", ".github",
    # Repo-level material a reader runs from a checkout: the sample configs
    # the config loader's own docstrings point at, the runnable examples, and
    # the deployment manifests.
    "configs", "examples", "deploy",
)
SCANNED_FILES = ("README.md", "README_PYPI.md")

_SUFFIXES = {
    ".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".cfg", ".ini", ".sh", ".ts", ".tsx", ".js", ".jsx", ".html",
}

# A handful of places name a retired id deliberately, because they are the
# *record* of the retirement rather than an offer of the model. They are listed
# with the reason so the exemption stays reviewable.
RETIREMENT_RECORDS: dict[str, str] = {
    # This module is the list itself.
    "tests/unit/test_no_retired_model_ids.py":
        "the retired-id table and the gate's own probe",
    # The snapshot-hash anchor requires each re-cut to say what it dropped.
    "tests/unit/test_coding_suitability.py":
        "the bundled-snapshot hash anchor records why groq.json was re-cut",
}


def _tracked_text_files() -> list[str]:
    """Every tracked or newly added text file under the scanned surfaces.

    Untracked-but-added files are included: a brand new doc page that names a
    retired id is exactly the case this gate exists to catch, and it would not
    yet be in ``git ls-files``.
    """
    seen: set[str] = set()
    for args in (["ls-files"], ["ls-files", "--others", "--exclude-standard"]):
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, check=True,
        ).stdout
        seen.update(p for p in out.splitlines() if p)

    keep = []
    for rel in sorted(seen):
        if rel in RETIREMENT_RECORDS:
            continue
        if not (rel.startswith(SCANNED_ROOTS) or rel in SCANNED_FILES):
            continue
        if Path(rel).suffix not in _SUFFIXES:
            continue
        keep.append(rel)
    return keep


@pytest.mark.parametrize("retired_id", sorted(RETIRED_MODEL_IDS))
def test_no_shipped_surface_names_a_retired_model_id(retired_id):
    provider, confirmed_on = RETIRED_MODEL_IDS[retired_id]
    pattern = re.compile(re.escape(retired_id))

    hits: list[str] = []
    for rel in _tracked_text_files():
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:100]}")

    assert not hits, (
        f"{provider} retired {retired_id!r} (confirmed {confirmed_on}); it still "
        f"appears in {len(hits)} place(s):\n  " + "\n  ".join(hits[:20])
    )


def test_the_bundled_catalog_offers_no_retired_id():
    """The snapshot a fresh install reads must not advertise a dead model."""
    import json

    for provider in {p for p, _ in RETIRED_MODEL_IDS.values()}:
        snap = json.loads(
            (REPO_ROOT / "effgen" / "models" / "_data" / f"{provider}.json").read_text()
        )
        offered = {m["id"] for m in snap["models"]}
        retired = {mid for mid, (p, _) in RETIRED_MODEL_IDS.items() if p == provider}
        assert not (offered & retired), f"{provider}.json still offers {offered & retired}"


def test_the_groq_default_is_a_model_the_catalog_still_carries():
    """The adapter default has to name something the provider still serves.

    The bundled snapshot is re-cut from the live listing, so an id that
    survives in it is one the provider answered for.
    """
    import json

    from effgen.models.groq_models import GROQ_DEFAULT_MODEL

    snap = json.loads(
        (REPO_ROOT / "effgen" / "models" / "_data" / "groq.json").read_text()
    )
    assert GROQ_DEFAULT_MODEL in {m["id"] for m in snap["models"]}


def test_every_exemption_is_a_file_that_exists():
    """A stale exemption would silently widen the gate."""
    for rel in RETIREMENT_RECORDS:
        assert (REPO_ROOT / rel).is_file(), rel


def test_the_gate_catches_a_planted_retired_id(tmp_path, monkeypatch):
    """The scan finds a fresh violation rather than only passing by luck."""
    planted = REPO_ROOT / "docs" / "_retired_id_gate_probe.md"
    planted.write_text("Try `effgen run hi -m groq:llama-3.1-8b-instant`.\n")
    try:
        with pytest.raises(AssertionError, match="llama-3.1-8b-instant"):
            test_no_shipped_surface_names_a_retired_model_id("llama-3.1-8b-instant")
    finally:
        planted.unlink()
