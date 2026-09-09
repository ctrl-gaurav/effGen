"""Provider-registry isolation: no test may change what a later test sees.

Two things are covered here:

* the registry's own contract — ``reset()`` returns it to the state effGen
  starts in, ``clear()`` empties it, and ``snapshot()``/``restore()`` undo edits
  made inside a provider record (a tripped circuit breaker, an added or removed
  model, a cleared capability set), including in the catalog dict the adapter
  module reads;
* the suite-wide guarantee — the autouse fixture in ``tests/conftest.py``
  reinstalls the catalog before every test, so a test that empties or edits the
  registry cannot make a later one fail.

The last three tests run pytest in a subprocess, so the guarantee is checked in
a real multi-module session rather than asserted about one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from effgen.models.registry import ProviderRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]

BUILTIN_PROVIDERS = {
    "anthropic", "cerebras", "fireworks", "gemini", "groq",
    "hf", "openai", "replicate", "together",
}

pytestmark = pytest.mark.unit


class _FakeAdapter:
    pass


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------

def test_reset_restores_the_builtin_providers():
    """``reset()`` leaves a usable registry, not an empty one."""
    ProviderRegistry.clear()
    assert ProviderRegistry.list_providers() == []

    ProviderRegistry.reset()

    assert BUILTIN_PROVIDERS.issubset(set(ProviderRegistry.list_providers()))
    provider, adapter_cls, info = ProviderRegistry.lookup("openai:gpt-5-nano")
    assert provider == "openai"
    assert adapter_cls is not None
    assert "context" in info


def test_reset_drops_circuit_breaker_and_bulkhead_state():
    breaker = ProviderRegistry.get_circuit_breaker("openai", failure_threshold=1)
    breaker.on_failure()
    assert breaker.state.value == "open"

    ProviderRegistry.reset()

    assert ProviderRegistry.reliability_stats()["openai"]["circuit_breaker"] is None
    assert ProviderRegistry.get_circuit_breaker("openai").state.value == "closed"


def test_clear_leaves_the_registry_empty():
    ProviderRegistry.clear()

    assert ProviderRegistry.list_providers() == []
    with pytest.raises(KeyError):
        ProviderRegistry.lookup("gpt-5-nano")


def test_clear_reports_an_empty_registry_when_a_model_is_loaded():
    """The load error names the empty registry, not a bad provider prefix."""
    from effgen.models.model_loader import load_model

    ProviderRegistry.clear()
    with pytest.raises(ValueError) as excinfo:
        load_model("openai:gpt-5-nano")

    message = str(excinfo.value)
    assert "provider registry is empty" in message
    assert "ProviderRegistry.reset()" in message


def test_register_builtins_is_idempotent():
    ProviderRegistry.register_builtins()
    providers = ProviderRegistry.list_providers()
    index_sizes = {mid: len(p) for mid, p in ProviderRegistry._model_index.items()}

    ProviderRegistry.register_builtins()

    assert ProviderRegistry.list_providers() == providers
    assert {mid: len(p) for mid, p in ProviderRegistry._model_index.items()} == index_sizes


def test_register_builtins_reports_the_builtin_providers():
    """The returned list is the same whether or not they were registered already."""
    assert BUILTIN_PROVIDERS.issubset(set(ProviderRegistry.register_builtins()))

    ProviderRegistry.clear()

    assert BUILTIN_PROVIDERS.issubset(set(ProviderRegistry.register_builtins()))


def test_restore_undoes_registrations_and_removals():
    snapshot = ProviderRegistry.snapshot()
    ProviderRegistry.register("fake_isolation_provider", _FakeAdapter, {"fake-1": {"context": 8}})
    ProviderRegistry._providers.pop("groq")

    ProviderRegistry.restore(snapshot)

    assert "fake_isolation_provider" not in ProviderRegistry.list_providers()
    assert "fake_isolation_provider" not in ProviderRegistry._model_index.get("fake-1", [])
    assert "groq" in ProviderRegistry.list_providers()


def test_restore_undoes_edits_made_inside_a_provider_record():
    snapshot = ProviderRegistry.snapshot()
    record = ProviderRegistry.get_provider_info("openai")
    record["models"]["fake-isolation-model"] = {"context": 8}
    record["models"]["gpt-5-nano"]["context"] = 1
    record["capabilities"].clear()
    record["env_keys"].append("FAKE_ISOLATION_KEY")

    ProviderRegistry.restore(snapshot)

    restored = ProviderRegistry.get_provider_info("openai")
    assert "fake-isolation-model" not in restored["models"]
    assert restored["models"]["gpt-5-nano"]["context"] > 1
    assert restored["capabilities"]
    assert restored["env_keys"] == ["OPENAI_API_KEY"]


def test_restore_puts_the_catalog_back_where_the_adapter_reads_it():
    """Adapters resolve models from their own catalog dict, not from the registry.

    An entry added to or removed from a provider record therefore has to be put
    back in that dict, or the adapter keeps rejecting a model the registry
    serves.
    """
    from effgen.models.groq_models import GROQ_MODELS

    snapshot = ProviderRegistry.snapshot()
    catalog = ProviderRegistry.get_provider_info("groq")["models"]
    assert catalog is GROQ_MODELS
    catalog.pop("openai/gpt-oss-20b")
    catalog["fake-isolation-model"] = {"context": 8}
    GROQ_MODELS["openai/gpt-oss-120b"]["context"] = 1

    ProviderRegistry.restore(snapshot)

    assert "openai/gpt-oss-20b" in GROQ_MODELS
    assert "fake-isolation-model" not in GROQ_MODELS
    assert GROQ_MODELS["openai/gpt-oss-120b"]["context"] > 1
    assert ProviderRegistry.get_provider_info("groq")["models"] is GROQ_MODELS


def test_restore_drops_a_tripped_circuit_breaker():
    snapshot = ProviderRegistry.snapshot()
    breaker = ProviderRegistry.get_circuit_breaker("openai", failure_threshold=1)
    breaker.on_failure()
    assert breaker.state.value == "open"

    ProviderRegistry.restore(snapshot)

    assert ProviderRegistry.get_circuit_breaker("openai").state.value == "closed"


def test_the_same_snapshot_can_be_restored_repeatedly():
    snapshot = ProviderRegistry.snapshot()
    for _ in range(3):
        ProviderRegistry.get_provider_info("openai")["models"].pop("gpt-5-nano", None)
        ProviderRegistry.clear()
        ProviderRegistry.restore(snapshot)
        assert "gpt-5-nano" in ProviderRegistry.get_provider_info("openai")["models"]


# ---------------------------------------------------------------------------
# Suite-wide guarantee
# ---------------------------------------------------------------------------

def test_pollutes_the_registry_for_whatever_runs_next():
    """Leave the registry in the worst state a test could leave it in.

    Whichever test runs after this one must be unaffected; the check itself is
    :func:`test_registry_starts_from_the_full_catalog`, which the subprocess
    tests below pin to run right after this one.
    """
    ProviderRegistry.get_circuit_breaker("openai", failure_threshold=1).on_failure()
    ProviderRegistry.get_provider_info("openai")["capabilities"].clear()
    ProviderRegistry.clear()

    assert ProviderRegistry.list_providers() == []


def test_registry_starts_from_the_full_catalog():
    """Every test begins with the built-in providers and no reliability state."""
    assert BUILTIN_PROVIDERS.issubset(set(ProviderRegistry.list_providers()))
    assert ProviderRegistry.get_capabilities("openai")
    stats = ProviderRegistry.reliability_stats()
    assert all(
        rec["circuit_breaker"] is None and rec["bulkhead"] is None
        for rec in stats.values()
    ), f"reliability state carried over: {stats}"


def test_every_warn_once_record_is_cleared_between_tests():
    """No module-level "already said this" flag may escape the reset fixture.

    A heads-up that fires at most once per process is remembered in a
    module-level flag or set. If the fixture in ``tests/conftest.py`` does not
    clear one of them, a test that asserts that message passes alone and fails
    behind whichever test tripped the flag first. Scanning the source rather
    than listing the modules here means a newly added record has to be handled,
    not merely noticed.
    """
    import ast

    from tests.conftest import _WARN_ONCE_RECORDS

    covered = {(module, attr) for module, attr in _WARN_ONCE_RECORDS}
    found: set[tuple[str, str]] = set()

    for path in sorted((REPO_ROOT / "effgen").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        for node in tree.body:  # module level only
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            else:
                continue
            value = node.value
            is_flag = isinstance(value, ast.Constant) and isinstance(value.value, bool)
            is_set = isinstance(value, ast.Set) or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "set"
            )
            if not (is_flag or is_set):
                continue
            for name in names:
                if any(word in name.lower() for word in ("warn", "notified", "announced")):
                    found.add((module, name))

    missing = found - covered
    assert not missing, (
        "one-shot warning records not cleared between tests: "
        f"{sorted(missing)} — add them to _WARN_ONCE_RECORDS in tests/conftest.py"
    )
    stale = covered - found
    assert not stale, f"_WARN_ONCE_RECORDS names records that no longer exist: {sorted(stale)}"


# ---------------------------------------------------------------------------
# The guarantee, checked in a real multi-module session
# ---------------------------------------------------------------------------

def _run_pytest(*args: str, reverse: bool = False) -> subprocess.CompletedProcess:
    """Run pytest in a subprocess with the collection order pinned."""
    env = dict(os.environ, EFFGEN_TEST_REVERSE_ORDER="1" if reverse else "0")
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:randomly", "-p", "no:cacheprovider",
         "--no-header", "-q", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )


@pytest.mark.slow
def test_a_polluting_test_does_not_affect_the_next_one():
    result = _run_pytest(
        f"{Path(__file__).relative_to(REPO_ROOT)}::test_pollutes_the_registry_for_whatever_runs_next",
        f"{Path(__file__).relative_to(REPO_ROOT)}::test_registry_starts_from_the_full_catalog",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
def test_a_module_that_empties_the_registry_does_not_affect_a_later_module():
    """The model tests empty the registry in their own fixture by design."""
    result = _run_pytest(
        "tests/models/test_auth_check.py",
        f"{Path(__file__).relative_to(REPO_ROOT)}::test_registry_starts_from_the_full_catalog",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.slow
def test_reverse_order_env_var_reverses_collection():
    """``EFFGEN_TEST_REVERSE_ORDER=1`` is what the reversed CI pass relies on."""
    target = str(Path(__file__).relative_to(REPO_ROOT))
    forward = _run_pytest(target, "--collect-only")
    assert forward.returncode == 0, forward.stdout + forward.stderr

    reversed_run = _run_pytest(target, "--collect-only", reverse=True)
    assert reversed_run.returncode == 0, reversed_run.stdout + reversed_run.stderr

    def _ids(out: str) -> list[str]:
        return [line for line in out.splitlines() if line.startswith(target)]

    forward_ids = _ids(forward.stdout)
    assert len(forward_ids) > 1
    assert _ids(reversed_run.stdout) == list(reversed(forward_ids))
