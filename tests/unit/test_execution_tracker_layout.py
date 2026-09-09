"""Layout of the ``effgen.core.execution_tracker`` split.

The tracker is one facade plus four concerns:

* ``execution_tracker_events`` — the event id sequence and the four data types
  (``EventType``, ``ExecutionEvent``, ``ExecutionStatus``, ``ExecutionNode``);
* ``execution_tracker_state`` — event intake, parent assignment, the tree and
  the live status snapshot;
* ``execution_tracker_metrics`` — the summary, the performance metric groups,
  bottlenecks and the critical path;
* ``execution_tracker_render`` — display formatting and trace export;
* ``execution_tracker`` — the composed ``ExecutionTracker`` class, which keeps
  ``__init__`` and ``__repr__``.

Five invariants keep that split safe:

* every member has exactly one owning module, and is defined exactly once across
  the class's MRO — a second definition would silently shadow the first;
* ``__init__`` and ``__repr__` stay in the class body, so the object's state is
  declared in one place, next to the class the mixins are composed into;
* the facade stays the single import path: the set of names it exposes is
  unchanged and each is the same object as on its owner;
* imports go one way only, so the graph stays acyclic;
* ``ExecutionStatus`` keeps meaning two different things — the tracker's status
  snapshot in ``effgen.core``, and the sandbox's in ``effgen``. Both resolve to
  what they always resolved to.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

import effgen.core.execution_tracker as facade
from effgen.core.execution_tracker import ExecutionTracker

PACKAGE = "effgen.core"
FACADE = "execution_tracker"

#: Every module the split is made of, the facade last.
SPLIT_MODULES = (
    "execution_tracker_events",
    "execution_tracker_state",
    "execution_tracker_metrics",
    "execution_tracker_render",
    FACADE,
)

#: The mixins, in MRO order after the class itself.
MIXINS = (
    ("execution_tracker_state", "ExecutionTrackerStateMixin"),
    ("execution_tracker_metrics", "ExecutionTrackerMetricsMixin"),
    ("execution_tracker_render", "ExecutionTrackerRenderMixin"),
)

#: Which module owns each name.
OWNERS = {
    # the data model
    "_EVENT_SEQ": "execution_tracker_events",
    "_EVENT_SEQ_LOCK": "execution_tracker_events",
    "_next_event_id": "execution_tracker_events",
    "EventType": "execution_tracker_events",
    "ExecutionEvent": "execution_tracker_events",
    "ExecutionStatus": "execution_tracker_events",
    "ExecutionNode": "execution_tracker_events",
    # the composed class
    "ExecutionTracker": FACADE,
}

#: Which module's class body defines each tracker member.
MEMBER_OWNERS = {
    # state
    "add_listener": "execution_tracker_state",
    "remove_listener": "execution_tracker_state",
    "track_event": "execution_tracker_state",
    "_STRUCTURAL_EVENTS": "execution_tracker_state",
    "_assign_parent": "execution_tracker_state",
    "_update_state": "execution_tracker_state",
    "_update_tree": "execution_tracker_state",
    "_latest_running_tool": "execution_tracker_state",
    "get_live_status": "execution_tracker_state",
    "generate_execution_tree": "execution_tracker_state",
    "get_trace": "execution_tracker_state",
    "clear": "execution_tracker_state",
    # metrics
    "get_summary": "execution_tracker_metrics",
    "get_performance_metrics": "execution_tracker_metrics",
    "_calculate_timing_metrics": "execution_tracker_metrics",
    "_calculate_throughput_metrics": "execution_tracker_metrics",
    "_calculate_resource_usage": "execution_tracker_metrics",
    "_calculate_efficiency_metrics": "execution_tracker_metrics",
    "_identify_bottlenecks": "execution_tracker_metrics",
    "get_critical_path": "execution_tracker_metrics",
    # rendering and export
    "format_for_display": "execution_tracker_render",
    "_format_rich": "execution_tracker_render",
    "_format_markdown": "execution_tracker_render",
    "_format_plain": "execution_tracker_render",
    "_get_event_icon": "execution_tracker_render",
    "export_trace": "execution_tracker_render",
    "_generate_html_trace": "execution_tracker_render",
    # the class keeps its own construction and repr
    "__init__": FACADE,
    "__repr__": FACADE,
}

#: What the facade itself declares.
FACADE_DEFINITIONS = {"ExecutionTracker"}

#: The names the facade exposed as a single file, pinned. The mixins are
#: deliberately not among them: they are composed into the class and unbound.
EXPOSED_NAMES = {
    "Any", "Enum", "EventType", "ExecutionEvent", "ExecutionNode", "ExecutionStatus",
    "ExecutionTracker", "_EVENT_SEQ", "_EVENT_SEQ_LOCK", "_next_event_id", "annotations",
    "dataclass", "datetime", "field", "itertools", "json", "threading", "time",
}

#: What ``effgen.core`` re-exports from the facade.
PACKAGE_RE_EXPORTS = (
    "EventType",
    "ExecutionEvent",
    "ExecutionNode",
    "ExecutionStatus",
    "ExecutionTracker",
)

#: What each module may import from within the split.
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "execution_tracker_events": set(),
    "execution_tracker_state": {"execution_tracker_events"},
    "execution_tracker_metrics": {"execution_tracker_events"},
    "execution_tracker_render": {"execution_tracker_events"},
    FACADE: set(SPLIT_MODULES) - {FACADE},
}

#: The public-surface anchor, and the module ``effgen.ExecutionStatus`` means.
#: 1.0.0 took this from 206 to 219: the OpenAI-compatible adapter and its
#: BackendUnreachableError, the middleware surface, the tool-call records, and
#: the compaction strategies. 1.1.0 took it to 234: AgentThread, the Step
#: protocol, and the seven kinds of step a run's conversation is made of.
#: Growing it is a deliberate act — move the number in the same commit that
#: widens the surface.
TOP_LEVEL_API_SIZE = 234
SANDBOX_STATUS_MODULE = "effgen.execution.sandbox"

#: No module in the split may grow past this.
MAX_LINES = 500


def module(name: str):
    return importlib.import_module(f"{PACKAGE}.{name}")


def source_path(name: str) -> Path:
    return Path(inspect.getsourcefile(module(name)))


def tree_of(name: str) -> ast.Module:
    return ast.parse(source_path(name).read_text(encoding="utf-8"))


def declared_names(name: str) -> set[str]:
    """Names the module declares in its own source — every import excluded."""
    names: set[str] = set()
    for node in tree_of(name).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def own_definitions(name: str) -> set[str]:
    """Names the module binds itself — re-exports from a sibling excluded."""
    names = declared_names(name)
    for node in tree_of(name).body:
        if isinstance(node, ast.ImportFrom):
            tail = (node.module or "").split(".")[-1]
            if tail in SPLIT_MODULES:
                continue
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def defined_members(name: str) -> set[str]:
    """Every tracker member the module defines in a class body."""
    found: set[str] = set()
    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    found.add(item.name)
                elif isinstance(item, ast.Assign):
                    found.update(t.id for t in item.targets if isinstance(t, ast.Name))
    return found


def _import_module_argument(node: ast.Call) -> str | None:
    """The literal module name a dynamic import call names, if it is one."""
    func = node.func
    dynamic = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
        isinstance(func, ast.Name) and func.id in ("import_module", "__import__")
    )
    if not dynamic or not node.args:
        return None
    first = node.args[0]
    return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None


def imported_split_modules(name: str) -> set[str]:
    """The modules of this split that *name* imports, however it spells them.

    All four spellings count: ``from .execution_tracker_x import y``,
    ``from effgen.core import execution_tracker_x`` (which names the module in
    the alias rather than the module path),
    ``import effgen.core.execution_tracker_x`` and a dynamic
    ``importlib.import_module("effgen.core.execution_tracker_x")``.
    """
    found: set[str] = set()

    def note(dotted: str) -> None:
        tail = dotted.split(".")[-1]
        if tail in SPLIT_MODULES:
            found.add(tail)

    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                note(node.module)
            for alias in node.names:
                if alias.name in SPLIT_MODULES:
                    found.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
        elif isinstance(node, ast.Call):
            dotted = _import_module_argument(node)
            if dotted:
                note(dotted)
    return found


# ---------------------------------------------------------------------------
# One owner per name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("member", "owner"), sorted(OWNERS.items()))
def test_each_name_is_defined_by_exactly_one_module(member, owner):
    assert member in own_definitions(owner), f"{owner} does not define {member}"
    others = [m for m in SPLIT_MODULES if m != owner and member in own_definitions(m)]
    assert not others, f"{member} is also defined in {others}"


@pytest.mark.parametrize(("member", "owner"), sorted(MEMBER_OWNERS.items()))
def test_each_member_is_defined_by_exactly_one_class(member, owner):
    assert member in defined_members(owner), f"{owner} does not define {member}"
    others = [m for m in SPLIT_MODULES if m != owner and member in defined_members(m)]
    assert not others, f"{member} is also defined in {others}"


def test_facade_declares_only_the_composed_class():
    assert declared_names(FACADE) == FACADE_DEFINITIONS


# ---------------------------------------------------------------------------
# The composed class
# ---------------------------------------------------------------------------


def test_mro_is_the_class_then_its_three_mixins():
    expected = [ExecutionTracker] + [getattr(module(m), c) for m, c in MIXINS] + [object]
    assert list(ExecutionTracker.__mro__) == expected


@pytest.mark.parametrize("member", sorted(MEMBER_OWNERS))
def test_each_member_is_defined_once_across_the_mro(member):
    # ``object`` supplies a default ``__init__``/``__repr__``; the question here
    # is whether two classes of the split define the same member.
    holders = [
        c.__name__ for c in ExecutionTracker.__mro__ if c is not object and member in vars(c)
    ]
    assert len(holders) == 1, f"{member} is defined in {holders}"


def test_construction_and_repr_stay_in_the_class_body():
    assert "__init__" in vars(ExecutionTracker)
    assert "__repr__" in vars(ExecutionTracker)


def test_structural_events_is_reachable_from_the_class():
    state_mixin = module("execution_tracker_state").ExecutionTrackerStateMixin
    assert isinstance(ExecutionTracker._STRUCTURAL_EVENTS, frozenset)
    assert "_STRUCTURAL_EVENTS" in vars(state_mixin)
    assert ExecutionTracker._STRUCTURAL_EVENTS is state_mixin._STRUCTURAL_EVENTS


# ---------------------------------------------------------------------------
# The facade stays the single import path
# ---------------------------------------------------------------------------


def test_the_facade_exposes_exactly_the_names_it_always_exposed():
    assert {n for n in dir(facade) if not n.startswith("__")} == EXPOSED_NAMES


@pytest.mark.parametrize(("member", "owner"), sorted(OWNERS.items()))
def test_facade_re_exports_every_moved_name_as_the_same_object(member, owner):
    assert hasattr(facade, member), f"{member} no longer resolves on the facade"
    assert getattr(facade, member) is getattr(module(owner), member), (
        f"{member} on the facade is a different object from {owner}.{member}"
    )


@pytest.mark.parametrize("name", PACKAGE_RE_EXPORTS)
def test_the_package_still_re_exports_the_same_object(name):
    package = importlib.import_module(PACKAGE)
    assert name in package.__all__, f"{name} dropped out of {PACKAGE}.__all__"
    assert getattr(package, name) is getattr(facade, name)


def test_execution_status_still_means_two_different_things():
    import effgen

    assert len(effgen.__all__) == TOP_LEVEL_API_SIZE
    assert "ExecutionStatus" in effgen.__all__
    assert effgen.ExecutionStatus.__module__ == SANDBOX_STATUS_MODULE
    assert effgen.ExecutionStatus is not facade.ExecutionStatus


# ---------------------------------------------------------------------------
# Imports go one way only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_split_modules_only_import_downwards(name):
    allowed = ALLOWED_IMPORTS[name]
    assert imported_split_modules(name) <= allowed, (
        f"{name} imports {sorted(imported_split_modules(name) - allowed)}"
    )


def test_no_sibling_imports_the_facade():
    for name in SPLIT_MODULES:
        if name == FACADE:
            continue
        assert FACADE not in imported_split_modules(name), f"{name} imports the facade"


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_no_module_in_the_split_grows_past_the_cap(name):
    lines = len(source_path(name).read_text(encoding="utf-8").splitlines())
    assert lines <= MAX_LINES, f"{name} is {lines} lines"
