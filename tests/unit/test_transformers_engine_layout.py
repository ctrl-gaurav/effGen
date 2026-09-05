"""Layout of the ``effgen.models.transformers_engine`` split.

The local Transformers engine is one facade plus five concerns:

* ``transformers_engine_support`` — cache/offline detection and special-token
  stripping, the only module here that needs neither torch nor transformers;
* ``transformers_engine_placement`` — device-map resolution and CUDA-fault recovery;
* ``transformers_engine_sampling`` — generation-config assembly and chat templates;
* ``transformers_engine_loading`` — weight loading, quantization and unloading;
* ``transformers_engine_generation`` — blocking and batched generation;
* ``transformers_engine_streaming`` — token-by-token streaming;
* ``transformers_engine`` — the engine class, its constructor and its capability
  reports.

Five invariants keep that split safe:

* every name has exactly one owning module, so a helper cannot drift back into
  the facade or be duplicated into a second concern;
* every module-level name the facade published stays importable from the facade
  and is the same object, so ``effgen.models.transformers_engine`` remains the
  single import path;
* imports go one way only — ``_support`` imports no sibling and no sibling
  imports the facade — so the import graph stays acyclic;
* the four generation entry points stay bound in the engine class's own
  ``__dict__``. ``BaseModel.__init_subclass__`` instruments them by reading that
  dict, so a name that resolves only through a base class silently loses the
  pre-call budget check, the latency and cost stamping, and the credential
  redaction on a load failure;
* every module whose scope imports torch or transformers repeats the guard that
  turns a missing local-inference extra into a message naming the install.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

# The facade imports torch at module scope, and a module-scope ImportError here
# interrupts collection for the whole directory — so an install without the
# local-inference extra could not run the unit lane at all. The layout these
# cases describe is only meaningful where the engine can be imported.
pytest.importorskip("torch", reason="the local Transformers engine needs torch")

import effgen.models.transformers_engine as facade  # noqa: E402
from effgen.models.transformers_engine import TransformersEngine  # noqa: E402

PACKAGE = "effgen.models"
FACADE = "transformers_engine"

#: Every module the split is made of, the facade last.
SPLIT_MODULES = (
    "transformers_engine_support",
    "transformers_engine_placement",
    "transformers_engine_sampling",
    "transformers_engine_loading",
    "transformers_engine_generation",
    "transformers_engine_streaming",
    FACADE,
)

#: The module-level names ``_support`` owns. All of them stay importable from the
#: facade, which is where the rest of the tree reaches for them.
SUPPORT_NAMES = (
    "GPUPlacementError",
    "ModelNotCachedError",
    "_offline_mode_active",
    "_list_cached_model_repos",
    "_is_cache_miss_error",
    "_reraise_if_classified",
    "_TOOL_CALL_DELIMITERS",
    "_COMMON_END_MARKERS",
    "_strip_special_tokens_keep_tool_calls",
    "_quiet_model_load",
)

#: Which module owns each engine member. A member defined in two places, or in
#: the wrong one, is caught here.
OWNERS = {
    # device placement and CUDA-fault recovery
    "_is_cuda_device_side_assert": "transformers_engine_placement",
    "_effective_device_map": "transformers_engine_placement",
    "_last_logits_look_valid": "transformers_engine_placement",
    "_probe_auto_device_map_logits": "transformers_engine_placement",
    "_apply_cuda_device_map_pin_fallback": "transformers_engine_placement",
    "_ensure_device_map_viable_before_sampling": "transformers_engine_placement",
    "_maybe_retry_after_cuda_assert": "transformers_engine_placement",
    "_resolve_placement": "transformers_engine_placement",
    "_free_vram_gb": "transformers_engine_placement",
    # generation config, sampling kwargs and chat templates
    "_eos_token_ids": "transformers_engine_sampling",
    "_create_generation_config": "transformers_engine_sampling",
    "_HF_GEN_PARAMS": "transformers_engine_sampling",
    "_seed_sampling": "transformers_engine_sampling",
    "_sanitize_generation_kwargs": "transformers_engine_sampling",
    "_fold_into_generation_config": "transformers_engine_sampling",
    "_apply_chat_template": "transformers_engine_sampling",
    # weights, quantization and teardown
    "_assemble_model_kwargs": "transformers_engine_loading",
    "_from_pretrained_with_flash_fallback": "transformers_engine_loading",
    "_load_model_weights": "transformers_engine_loading",
    "_drop_model_weights": "transformers_engine_loading",
    "load": "transformers_engine_loading",
    "_create_quantization_config": "transformers_engine_loading",
    "_get_max_length": "transformers_engine_loading",
    "unload": "transformers_engine_loading",
    # generation
    "generate": "transformers_engine_generation",
    "generate_batch": "transformers_engine_generation",
    "count_tokens": "transformers_engine_generation",
    "generate_stream": "transformers_engine_streaming",
    # the facade keeps construction and the capability reports
    "__init__": FACADE,
    "get_context_length": FACADE,
    "get_max_batch_size": FACADE,
    "supports_tool_calling": FACADE,
    "tool_call_support": FACADE,
}

#: The facade defines the engine class and its logger, and nothing else.
FACADE_DEFINITIONS = {"TransformersEngine", "logger"}

#: What each module may import from within the split.
ALLOWED_IMPORTS: dict[str, set[str]] = {
    "transformers_engine_support": set(),
    "transformers_engine_placement": set(),
    "transformers_engine_sampling": set(),
    "transformers_engine_loading": {"transformers_engine_support"},
    "transformers_engine_generation": {"transformers_engine_support"},
    "transformers_engine_streaming": {"transformers_engine_support"},
    FACADE: set(SPLIT_MODULES) - {FACADE},
}

#: The four names ``BaseModel.__init_subclass__`` instruments by reading the
#: engine class's own ``__dict__``.
INSTRUMENTED = ("load", "generate", "generate_batch", "generate_stream")

#: Members that must stay unbound helpers rather than becoming instance methods.
STATIC_METHODS = ("_free_vram_gb", "_is_cuda_device_side_assert")

#: Everything a caller can reach on the engine class, inherited members included.
#: Splitting the file may move where a member is defined; it may not add, drop or
#: rename one.
#:
#: ``build_assistant_message`` / ``build_tool_result_message`` were added on
#: 2026-08-13 — on ``BaseModel``, so every adapter and engine inherits them and
#: a multi-turn tool loop is portable across providers. Recorded here rather
#: than silently absorbed, because this list is the anchor that says the surface
#: only changes on purpose.
ENGINE_MEMBERS = (
    "_HF_GEN_PARAMS",
    "_abc_impl",
    "_apply_chat_template",
    "_apply_cuda_device_map_pin_fallback",
    "_assemble_model_kwargs",
    "_create_generation_config",
    "_create_quantization_config",
    "_drop_model_weights",
    "_effective_device_map",
    "_ensure_device_map_viable_before_sampling",
    "_eos_token_ids",
    "_fold_into_generation_config",
    "_free_vram_gb",
    "_from_pretrained_with_flash_fallback",
    "_get_max_length",
    "_is_cuda_device_side_assert",
    "_last_logits_look_valid",
    "_load_model_weights",
    "_maybe_retry_after_cuda_assert",
    "_probe_auto_device_map_logits",
    "_resolve_placement",
    "_sanitize_generation_kwargs",
    "_seed_sampling",
    "build_assistant_message",
    "build_tool_result_message",
    "count_tokens",
    "generate",
    "generate_batch",
    "generate_stream",
    "get_context_length",
    "get_max_batch_size",
    "get_metadata",
    "get_total_cost",
    "is_loaded",
    "load",
    "reset_cost",
    "streams_tool_calls",
    # Inherited from BaseModel, which answers False: the local engine has no
    # request layer that could enforce a tool choice, so it advertises that it
    # cannot require a call rather than sending one the runtime would ignore.
    "supports_forced_tool_call",
    "supports_tool_calling",
    "tool_call_support",
    "unload",
    "validate_prompt",
)

#: No module in the split may grow past this.
MAX_LINES = 500


def module(name: str):
    return importlib.import_module(f"{PACKAGE}.{name}")


def source_path(name: str) -> Path:
    return Path(inspect.getsourcefile(module(name)))


def tree_of(name: str) -> ast.Module:
    return ast.parse(source_path(name).read_text(encoding="utf-8"))


def top_level_names(name: str) -> set[str]:
    """Every name the module binds at module scope, imports excluded."""
    names: set[str] = set()
    for node in tree_of(name).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def defined_members(name: str) -> set[str]:
    """Every engine member the module *defines* in a class body.

    A class-body assignment of one name to another class's attribute — the
    ``load = TransformersLoadingMixin.load`` re-bind that keeps
    ``__init_subclass__`` able to see the four entry points — is a binding, not a
    definition. Counting it as one would report every instrumented name as
    defined twice.
    """
    found: set[str] = set()
    for node in ast.walk(tree_of(name)):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    found.add(item.name)
                elif isinstance(item, ast.Assign) and not isinstance(item.value, ast.Attribute):
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

    All four spellings count: ``from .transformers_engine_x import y``,
    ``from effgen.models import transformers_engine_x`` (which names the module
    in the alias rather than in the module path), ``import
    effgen.models.transformers_engine_x``, and a dynamic
    ``importlib.import_module("effgen.models.transformers_engine_x")``.
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


def module_scope_imports(name: str) -> set[str]:
    """The top-level packages the module imports at module scope.

    Imports inside a function body are deliberately excluded: keeping the heavy
    ones there is what the guard test is about.
    """
    found: set[str] = set()
    for node in tree_of(name).body:
        nodes = [node]
        if isinstance(node, ast.Try):
            nodes = list(ast.walk(node))
        for sub in nodes:
            if isinstance(sub, ast.Import):
                found.update(a.name.split(".")[0] for a in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module:
                found.add(sub.module.split(".")[0])
    return found


# ---------------------------------------------------------------------------
# One owner per name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("member", "owner"), sorted(OWNERS.items()))
def test_each_member_is_defined_by_exactly_one_module(member, owner):
    assert member in defined_members(owner), f"{owner} does not define {member}"
    others = [m for m in SPLIT_MODULES if m != owner and member in defined_members(m)]
    assert not others, f"{member} is also defined in {others}"


@pytest.mark.parametrize("name", SUPPORT_NAMES)
def test_support_names_are_owned_by_the_support_module(name):
    assert name in top_level_names("transformers_engine_support")
    others = [
        m for m in SPLIT_MODULES if m != "transformers_engine_support" and name in top_level_names(m)
    ]
    assert not others, f"{name} is also defined in {others}"


def test_facade_defines_only_the_engine_class_and_its_logger():
    assert top_level_names(FACADE) == FACADE_DEFINITIONS


# ---------------------------------------------------------------------------
# The facade stays the single import path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SUPPORT_NAMES)
def test_facade_re_exports_every_support_name_as_the_same_object(name):
    assert hasattr(facade, name), f"{name} no longer imports from the facade"
    assert getattr(facade, name) is getattr(module("transformers_engine_support"), name)


def test_engine_class_is_the_same_object_everywhere():
    import effgen
    import effgen.models

    assert effgen.TransformersEngine is TransformersEngine
    assert effgen.models.TransformersEngine is TransformersEngine


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


# ---------------------------------------------------------------------------
# The instrumentation seam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", INSTRUMENTED)
def test_generation_entry_points_stay_in_the_engine_class_dict(name):
    """``BaseModel.__init_subclass__`` reads ``cls.__dict__``, not the MRO.

    A name that resolves only through a base class is never wrapped, and the
    budget check, the latency/cost stamping and the load-failure redaction
    disappear with no error raised.
    """
    assert name in TransformersEngine.__dict__, (
        f"{name} resolves through the MRO instead of the engine class's own __dict__"
    )
    assert getattr(TransformersEngine.__dict__[name], "__effgen_timed__", False), (
        f"{name} is not instrumented"
    )


def test_constructor_stays_in_the_facade_class_body():
    """``__init__`` is the one method here that calls zero-argument ``super()``.

    That compiles to a lookup of an implicit ``__class__`` cell bound to the
    class whose body lexically contains it, so on a mixin it raises
    ``TypeError: super(type, obj): obj must be an instance or subtype of type``.
    """
    assert "__init__" in TransformersEngine.__dict__
    cells = TransformersEngine.__init__.__closure__ or ()
    assert any(c.cell_contents is TransformersEngine for c in cells), (
        "__init__ no longer carries the __class__ cell of TransformersEngine"
    )


@pytest.mark.parametrize("name", STATIC_METHODS)
def test_unbound_helpers_stay_static_methods(name):
    assert isinstance(inspect.getattr_static(TransformersEngine, name), staticmethod)


def test_engine_is_concrete():
    assert not inspect.isabstract(TransformersEngine)


def test_engine_member_surface_is_unchanged():
    """The set a caller sees, wherever each member is now defined."""
    seen = {n for n in dir(TransformersEngine) if not n.startswith("__")}
    assert seen == set(ENGINE_MEMBERS)


# ---------------------------------------------------------------------------
# The torch guard travels with the imports it explains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_torch_importing_modules_name_the_install(name):
    """A direct import of a sibling on an install without the extras must say so.

    Without the guard the reader gets ``No module named 'torch'`` instead of the
    message naming the extra to install.
    """
    heavy = {"torch", "transformers"} & module_scope_imports(name)
    source = source_path(name).read_text(encoding="utf-8")
    if not heavy:
        assert "missing_torch_error" not in source, f"{name} guards an import it does not make"
        return
    assert "missing_torch_error(\"transformers\")" in source, (
        f"{name} imports {sorted(heavy)} at module scope without the guard"
    )


@pytest.mark.parametrize("name", SPLIT_MODULES)
def test_no_module_in_the_split_grows_past_the_cap(name):
    lines = len(source_path(name).read_text(encoding="utf-8").splitlines())
    assert lines <= MAX_LINES, f"{name} is {lines} lines"
