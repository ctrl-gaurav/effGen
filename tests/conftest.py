"""
Shared test fixtures for the effGen test suite.
"""

import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pytest

# Ensure effgen and the test-support package are importable before anything below
# reads them.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests._harness import (  # noqa: E402
    hermetic,
    lane_timing,
    optional_deps,
    provider_unavailable,
)

# Remove the ambient state of this machine when the run asked for it. This happens
# before the .env load and before the temporary cost/run directories below, so what
# the suite sets for itself survives and what the machine happened to export does not.
hermetic.activate()

# Suppress ImportWarning from optional dependencies before importing effgen
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Secret scrubbing for traceback locals (defense in depth: even if a future
# dev re-enables --showlocals, real API key values must not land in test logs).
# ---------------------------------------------------------------------------

_SECRET_ENV_PATTERNS = (
    "API_KEY", "API_TOKEN", "SECRET", "TOKEN", "PASSWORD",
)


def _collect_secret_values() -> tuple[str, ...]:
    values = []
    for name, val in os.environ.items():
        if not val or len(val) < 8:
            continue
        if any(p in name.upper() for p in _SECRET_ENV_PATTERNS):
            values.append(val)
    # Sort longest-first so substrings don't shadow longer matches
    return tuple(sorted(set(values), key=len, reverse=True))


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_makereport(item, call):
    """Report an absent optional dependency as a skip rather than a failure.

    A contributor who installs ``.[dev]`` sees ~94 red tests on a correct
    checkout, all of them a tool whose optional extra was never installed. The
    library behaves correctly in every one — it raises a typed, actionable
    ``ImportError`` naming the extra — so the failure says only that the feature
    is not installed, which is a reason to skip.

    The rule is kept narrow on purpose (see ``tests/_harness/optional_deps.py``):
    only a package the project itself declares in an *optional* extra qualifies,
    and only the two shapes that mean "not installed". An environment with the
    extras present converts nothing.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when not in ("setup", "call") or not report.failed:
        return
    rendered = str(report.longrepr)
    missing = optional_deps.absent_optional_dependency(rendered)
    if missing is not None:
        reason = (
            f"Skipped: optional dependency '{missing}' is not installed "
            f"(pip install '{missing}' or the extra that carries it)"
        )
    else:
        # The same idea for a provider that would not serve the call: a spent
        # quota or a provider outage is an account/provider state, not a defect.
        # Narrow on purpose — a bare connection error is NOT converted, because
        # that shape is usually local misconfiguration. See provider_unavailable.
        # Never under tests/unit: those never reach a provider, so any provider
        # error text there is simulated — a fault server, a fixture — and is the
        # very thing the test is asserting about. Converting one to a skip would
        # hide a regression in effGen's own error handling.
        if "/tests/unit/" in _norm_path(item):
            return
        unavailable = provider_unavailable.provider_unavailable_reason(rendered)
        if unavailable is None:
            return
        detail = provider_unavailable.first_provider_sentence(rendered)
        reason = f"Skipped: {unavailable}{f' — {detail}' if detail else ''}"
    report.outcome = "skipped"
    report.longrepr = (str(item.fspath), item.location[1] or 0, reason)
    # Clear any xfail marking by REMOVING the attribute, never by setting it to
    # None. pytest decides a report is an xfail with `hasattr(report,
    # "wasxfail")` and then calls `.startswith()` on the value, so a None left
    # here raises inside the terminal reporter — an INTERNALERROR that ends the
    # whole session with exit 3 and discards several thousand passing results.
    # It only fires on the verbose reporting path, so a quiet local run looks
    # clean and CI does not.
    try:
        del report.wasxfail
    except AttributeError:
        # Either it was never set, or it is inherited rather than owned by this
        # instance; both mean there is nothing on the report to mislead pytest.
        pass


def pytest_exception_interact(node, call, report):
    """Redact known secret values from rendered traceback text."""
    secrets = _collect_secret_values()
    if not secrets:
        return
    longrepr = getattr(report, "longrepr", None)
    if longrepr is None:
        return
    text = str(longrepr)
    redacted = text
    for s in secrets:
        redacted = redacted.replace(s, "***REDACTED***")
    if redacted != text:
        # Replace longrepr with the scrubbed string so terminal + xml reporters
        # both see the redacted version.
        report.longrepr = redacted

# Load test credentials without printing or inspecting values. Project-local
# values are useful for live integration tests; user-level values remain a
# fallback for developer machines. A run with the ambient environment removed
# loads neither, so the key-gated tests skip the way they do on a clean runner.
if not hermetic.is_active():
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent.parent / ".env", override=False)
        load_dotenv(Path.home() / ".effgen" / ".env", override=False)
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Isolate the cost-tracker SQLite DB from the user's real ~/.effgen/costs.sqlite
# during tests. Many unit tests record large fixture values (1M tokens etc.)
# and those must not pollute the user's actual cost history.
# ---------------------------------------------------------------------------
_TEST_COST_DB_DIR: str | None = None
if "EFFGEN_COST_DB" not in os.environ:
    _TEST_COST_DB_DIR = tempfile.mkdtemp(prefix="effgen_test_costs_")
    os.environ["EFFGEN_COST_DB"] = str(Path(_TEST_COST_DB_DIR) / "costs.sqlite")
    # Also point the budget config away from the developer's real
    # ~/.effgen/budget.json. Otherwise a host that has configured a daily budget
    # makes cost-tracker tests (which record large fixture spends) trip the real
    # budget and raise BudgetExceededError. The empty temp path reads as "no
    # budget configured", which is the state a clean CI runner sees.
    os.environ["EFFGEN_BUDGET_CONFIG"] = str(Path(_TEST_COST_DB_DIR) / "budget.json")

# Run history is durable too: keep test runs out of the user's real
# ~/.effgen/runs, and keep the user's stored runs out of test assertions.
if "EFFGEN_RUN_HISTORY_DIR" not in os.environ:
    if _TEST_COST_DB_DIR is None:
        _TEST_COST_DB_DIR = tempfile.mkdtemp(prefix="effgen_test_costs_")
    os.environ["EFFGEN_RUN_HISTORY_DIR"] = str(Path(_TEST_COST_DB_DIR) / "runs")


def pytest_configure(config):
    """Record when the session started and register the opt-in timing plugin."""
    config._effgen_session_start = time.time()
    lane_timing.maybe_register(config)


def pytest_report_header(config):
    """Say so when the run is not seeing this machine's ambient state."""
    _ = config
    line = hermetic.report_line()
    return [line] if line else []


def pytest_sessionfinish(session, exitstatus):
    """Remove the isolated cost DB and temporary home created for this session."""
    _ = (session, exitstatus)
    if _TEST_COST_DB_DIR:
        shutil.rmtree(_TEST_COST_DB_DIR, ignore_errors=True)
    hermetic.cleanup()

from effgen.core.agent import Agent, AgentConfig
from effgen.tools.builtin import Calculator, DateTimeTool, JSONTool, TextProcessingTool
from tests.fixtures.mock_models import MockModel, MockStreamingModel, MockToolCallingModel

# ---------------------------------------------------------------------------
# ProviderRegistry isolation (suite order-independence).
#
# The provider registry is a process-wide singleton, and tests edit it: they
# register fake providers, empty it, trip a provider's circuit breaker, or
# adjust a capability set. None of that may change what a later test sees, and
# a test that only fails after some other test ran is expensive to diagnose.
#
# This autouse fixture reinstalls a full copy of the provider catalog at the
# START of every test, so each one begins from the same state no matter what
# ran before it. Restoring at setup (rather than teardown) also makes it
# independent of fixture teardown order: a test that deliberately wants an
# empty registry still gets one, because its own fixture runs after this one.
# ProviderRegistry.snapshot()/restore() copy the per-provider records, so edits
# made *inside* a record — a tripped circuit breaker, an added or removed model,
# a cleared capability set — do not survive either; a model catalog is put back
# in the dict the adapter module reads, so the two cannot drift apart.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _provider_registry_snapshot():
    """Canonical copy of the provider catalog, captured once per session."""
    from effgen.models.registry import ProviderRegistry

    ProviderRegistry.register_builtins()
    return ProviderRegistry.snapshot()


@pytest.fixture(autouse=True)
def _restore_provider_registry(_provider_registry_snapshot):
    """Restore the full ProviderRegistry before every test (order-independence)."""
    from effgen.models.registry import ProviderRegistry

    ProviderRegistry.restore(_provider_registry_snapshot)
    yield


# ---------------------------------------------------------------------------
# "Warn once per process" records (suite order-independence).
#
# Several heads-up messages are emitted at most once per process, keyed on what
# they are about, so a long run does not repeat them. The record of what has
# already been said is module-level state: whichever test triggers the message
# first sees it, and a test that asserts the same message later sees nothing.
# Clearing the records before every test makes each one observe the warnings it
# actually triggers, whatever ran before it.
# ---------------------------------------------------------------------------

_WARN_ONCE_RECORDS = (
    ("effgen.core._compat", "_warned_drift"),
    ("effgen.core.agent_generation", "_reasoning_budget_warned"),
    ("effgen.models._adapter_utils", "_reasoning_only_warned"),
    ("effgen.models._adapter_utils", "_bpe_unavailable_warned"),
    ("effgen.core.agent_runtime", "_tool_output_injection_gap_warned"),
    ("effgen.core.agent_runtime", "_message_protocol_unavailable_warned"),
    ("effgen.models._catalog", "_WARNED"),
    ("effgen.models._cost", "_UNPRICED_BUDGET_WARNED"),
    ("effgen.presets.registry", "_tool_overhead_warned"),
    ("effgen.server.auth", "_AUTH_UNCONFIGURED_WARNED"),
    ("effgen.server.auth", "_DEV_MODE_WARNED"),
    ("effgen.server.auth", "_UNVERIFIED_DECODE_WARNED"),
    ("effgen.ui.theme", "_warned_unknown"),
    ("effgen.gpu.cuda_compat", "_warned"),
    ("effgen.security.sandbox", "_subprocess_fallback_warned"),
    ("effgen.security.sandbox", "_confinement_degraded_warned"),
)

# Records of something the process learned once and reuses, cleared for the same
# reason: a test that asserts the first-time behavior must not depend on whether
# an earlier test already taught the process.
_LEARNED_ONCE_RECORDS = (
    ("effgen.core.agent_generation", "_reasoning_stream_models"),
    ("effgen.core.agent_runtime", "_MESSAGE_PROTOCOL_PROBE"),
)


@pytest.fixture(autouse=True)
def _reset_warn_once_records():
    """Let every test see the one-time warnings it triggers itself."""
    for module_name, attr in _WARN_ONCE_RECORDS + _LEARNED_ONCE_RECORDS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        record = getattr(module, attr, None)
        if isinstance(record, set | dict):
            record.clear()
        elif isinstance(record, bool):
            setattr(module, attr, False)
    yield


@pytest.fixture(autouse=True)
def _color_decision_comes_from_the_code(monkeypatch):
    """Clear the color-forcing env vars for every test.

    Tests that assert what the terminal surfaces emit when they are not
    attached to a TTY (no ANSI escapes when piped, ``NO_COLOR`` honored) read
    the output rich actually produced. Rich also consults ``FORCE_COLOR`` and
    ``CLICOLOR_FORCE``, which many terminals, CI runners and editor consoles
    export; with either set, a piped run keeps its colors and those assertions
    fail for a reason outside the code under test. Clearing them here makes the
    outcome depend only on what the CLI decides.
    """
    for var in ("FORCE_COLOR", "CLICOLOR_FORCE"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _keep_the_empty_home():
    """Put the temporary home back after a test that reassigned it.

    Only does anything when the run has the ambient environment removed. A test that
    sets ``HOME`` through ``monkeypatch`` is restored by pytest; one that assigns
    ``os.environ`` directly is restored here, so the test after it still cannot reach
    the developer's real home.
    """
    if not hermetic.is_active():
        yield
        return
    expected = str(hermetic.home())
    yield
    if os.environ.get("HOME") != expected:
        os.environ["HOME"] = expected
        os.environ["USERPROFILE"] = expected


@pytest.fixture(autouse=True)
def _restore_server_auth_env():
    """Restore EFFGEN_API_KEY after every test (order-independence).

    ``effgen serve`` mints an ephemeral key into ``os.environ`` when no auth is
    configured. A test that drives the serve command in-process (with
    ``uvicorn.run`` stubbed) therefore leaves that key in the process
    environment. Because the server reads ``EFFGEN_API_KEY`` when it builds its
    auth middleware, a leaked key silently switches later server tests into
    API-key mode and makes them reject valid JWTs with 401. Snapshot the value
    at setup and restore it at teardown so serve-invoking tests cannot pollute
    the ones that run after them.
    """
    _sentinel = object()
    _prev = os.environ.get("EFFGEN_API_KEY", _sentinel)
    yield
    if _prev is _sentinel:
        os.environ.pop("EFFGEN_API_KEY", None)
    else:
        os.environ["EFFGEN_API_KEY"] = _prev


#: The variables that decide where an OpenAI-protocol call is addressed. Kept
#: here rather than imported so this guard works even if the models package
#: cannot be imported in a reduced install.
_ENDPOINT_ENV_VARS = ("EFFGEN_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")


@pytest.fixture(autouse=True)
def _restore_endpoint_env():
    """Restore the endpoint overrides after every test (order-independence).

    These decide the address of every OpenAI-protocol call, and one left behind
    redirects every live cell that runs after it for the rest of the session.
    A blank value is the damaging shape: effGen reads it as "no override", the
    OpenAI SDK reads it as an address and sends the request to ``''``, and what
    the run reports is a connection error advising you to check the provider's
    status page. One test leaking the project template's blanks that way turned
    45 later live cells red in a single suite run, none of which named the
    cause.
    """
    _sentinel = object()
    _prev = {name: os.environ.get(name, _sentinel) for name in _ENDPOINT_ENV_VARS}
    yield
    for name, was in _prev.items():
        if was is _sentinel:
            os.environ.pop(name, None)
        else:
            os.environ[name] = was


# ---------------------------------------------------------------------------
# Mock Model Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    """A simple mock model that returns 'Final Answer: 42'."""
    return MockModel(responses=[
        "Thought: I know the answer.\nFinal Answer: 42"
    ])


@pytest.fixture
def mock_tool_model():
    """Mock model that calls calculator then gives final answer."""
    return MockToolCallingModel(tool_sequence=[
        {
            "thought": "I need to calculate this.",
            "action": "calculator",
            "action_input": '{"expression": "2 + 2"}',
        },
        {
            "thought": "I now know the final answer.",
            "action": "Final Answer",
            "action_input": "The answer is 4.",
        },
    ])


@pytest.fixture
def mock_streaming_model():
    """Mock model for streaming tests."""
    return MockStreamingModel(responses=[
        "Thought: Let me think.\nFinal Answer: Hello, world!"
    ])


# ---------------------------------------------------------------------------
# Tool Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calculator():
    return Calculator()


@pytest.fixture
def datetime_tool():
    return DateTimeTool()


@pytest.fixture
def json_tool():
    return JSONTool()


@pytest.fixture
def text_tool():
    return TextProcessingTool()


# ---------------------------------------------------------------------------
# Agent Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_agent(mock_model):
    """Agent with no tools."""
    config = AgentConfig(
        name="test-basic",
        model=mock_model,
        tools=[],
        max_iterations=3,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


@pytest.fixture
def tool_agent(mock_tool_model, calculator):
    """Agent with calculator tool."""
    config = AgentConfig(
        name="test-tool",
        model=mock_tool_model,
        tools=[calculator],
        max_iterations=5,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


@pytest.fixture
def multi_tool_agent(mock_model, calculator, datetime_tool, json_tool):
    """Agent with multiple tools."""
    config = AgentConfig(
        name="test-multi",
        model=mock_model,
        tools=[calculator, datetime_tool, json_tool],
        max_iterations=5,
        enable_memory=False,
        enable_sub_agents=False,
    )
    return Agent(config=config)


# ---------------------------------------------------------------------------
# Temp Directory Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp(prefix="effgen_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# GPU Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def free_gpu_id():
    """Find a free GPU with minimal memory usage."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        # Find GPU with least memory usage
        min_mem = float("inf")
        best_gpu = 0
        for i in range(torch.cuda.device_count()):
            mem_used = torch.cuda.memory_allocated(i)
            if mem_used < min_mem:
                min_mem = mem_used
                best_gpu = i
        return best_gpu
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Fixtures directory path
# ---------------------------------------------------------------------------

@pytest.fixture
def fixtures_dir():
    """Path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def knowledge_base_dir(fixtures_dir):
    """Path to the knowledge base fixtures."""
    return fixtures_dir / "knowledge_base"


# ---------------------------------------------------------------------------
# GPU test isolation
# ---------------------------------------------------------------------------
#
# e2e tests load models with bitsandbytes 4-bit quantization. The bnb kernels
# leave CUDA state that historically caused the integration streaming tests
# (TextIteratorStreamer) to deadlock when they ran AFTER e2e in the same
# pytest session. Two-part defense:
#
#   1. Reorder collection so all tests/e2e/* items run LAST, after unit +
#      integration. This way any CUDA-state corruption happens after
#      everything downstream has already completed.
#   2. Between each module we force a CUDA cleanup via an autouse
#      module-scoped fixture. Cheap when CUDA is absent; essential when
#      GPU tests and streaming tests coexist.


# Directory -> marker auto-application. Keeps the marker surface authoritative
# (so `-m security` / `-m deployment` actually select the right suites and map to
# CI jobs) without sprinkling `pytestmark` into dozens of files. Explicit markers
# on individual tests still apply and stack on top of these.
_DIR_MARKERS = {
    "/tests/security/": "security",
    "/tests/deploy/": "deployment",
}


def _norm_path(item) -> str:
    return str(item.fspath).replace("\\", "/")


def pytest_collection_modifyitems(config, items):
    """Auto-mark by directory and run tests/e2e/* last.

    e2e tests load models with bitsandbytes 4-bit quantization; their CUDA state
    historically deadlocked the integration streaming tests when they ran first,
    so e2e is sorted to the end of the session.

    Setting ``EFFGEN_TEST_REVERSE_ORDER=1`` reverses collection order first, so
    every module runs before the modules it normally follows. That is a cheap,
    deterministic way to surface a test that only passes because of what ran
    before it; the order-independence CI lane uses it alongside the shuffled
    runs. The e2e bucket still sorts to the end.
    """
    if os.environ.get("EFFGEN_TEST_REVERSE_ORDER") == "1":
        items.reverse()

    for item in items:
        p = _norm_path(item)
        for needle, marker in _DIR_MARKERS.items():
            if needle in p and marker not in {m.name for m in item.iter_markers()}:
                item.add_marker(getattr(pytest.mark, marker))

    def _bucket(item):
        p = _norm_path(item)
        if "/tests/e2e/" in p:
            return 2
        if "/tests/integration/" in p:
            return 1
        return 0

    items.sort(key=_bucket)


@pytest.fixture(autouse=True, scope="module")
def _cuda_state_hygiene():
    """Flush CUDA caches between modules to reduce state leakage."""
    yield
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
