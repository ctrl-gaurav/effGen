"""CLI UX regression tests.

Covers: provider validation, quiet-by-default logging, registry-backed
`models list`/`info`, `--json` outputs, doctor system report and exit code,
examples-dir discovery, all-preset acceptance, and centralized `.env`
discovery. No live model calls here — live behavior (`run --provider`,
`doctor --live`) is exercised via end-to-end CLI smokes.
"""
from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pytest

from effgen.cli import _main


# --------------------------------------------------------------------------- #
# Provider validation
# --------------------------------------------------------------------------- #
def test_resolve_provider_valid():
    assert _main.resolve_provider_name("groq") == ("groq", None)
    assert _main.resolve_provider_name("OpenAI") == ("openai", None)


def test_resolve_provider_alias():
    assert _main.resolve_provider_name("google") == ("gemini", None)
    assert _main.resolve_provider_name("huggingface") == ("hf", None)


def test_resolve_provider_none():
    assert _main.resolve_provider_name(None) == (None, None)


def test_resolve_provider_typo_suggests():
    resolved, err = _main.resolve_provider_name("grok")
    assert resolved is None
    assert err is not None
    assert "groq" in err  # fuzzy suggestion
    assert "grok" in err


def test_loader_rejects_unknown_provider():
    from effgen.models.model_loader import ModelLoader
    with pytest.raises(ValueError, match="Unknown provider"):
        ModelLoader().load_model("some-model", provider="grok")


# --------------------------------------------------------------------------- #
# Quiet-by-default logging
# --------------------------------------------------------------------------- #
def test_setup_logging_default_is_warning():
    _main.setup_logging(verbose=False, quiet=False)
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_verbose_is_debug():
    _main.setup_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_quiet_is_error():
    _main.setup_logging(quiet=True)
    assert logging.getLogger().level == logging.ERROR
    # restore a sane default for the rest of the suite
    _main.setup_logging(verbose=False)


# --------------------------------------------------------------------------- #
# Registry-backed models list / info
# --------------------------------------------------------------------------- #
def _cli():
    return _main.CLIInterface()


def test_models_list_json_is_registry_backed(capsys):
    args = SimpleNamespace(provider=None, free=False, tools=False, output_json=True)
    code = _cli()._models_list(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert "providers" in data and "local_cache" in data
    # The known 9 providers are present and registry-derived, not yaml.
    assert {"openai", "groq", "cerebras"} <= set(data["providers"])
    assert data["providers"]["openai"]["count"] > 0


def test_models_list_no_bogus_empty_config_ids(capsys):
    # The old empty-config fallback printed mistral-7b/llama-2-7b/
    # gemma-7b. Those must never appear now.
    args = SimpleNamespace(provider=None, free=False, tools=False, output_json=False)
    _cli()._models_list(args)
    out = capsys.readouterr().out
    for bogus in ("mistral-7b", "llama-2-7b", "gemma-7b"):
        assert bogus not in out


def test_models_info_known(capsys):
    args = SimpleNamespace(name="gpt-5-nano", output_json=True)
    code = _cli()._models_info(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert data["id"] == "gpt-5-nano"
    assert data["provider"] == "openai"


def test_models_info_unknown_suggests_and_exits_nonzero(capsys):
    args = SimpleNamespace(name="gpt-5-nanoo", output_json=False)
    code = _cli()._models_info(args)
    out = capsys.readouterr().out
    assert code == 1
    assert "not found" in out.lower()


# --------------------------------------------------------------------------- #
# Local-cache awareness (models info / list)
# --------------------------------------------------------------------------- #
def _fake_local(entries):
    """Build a fake _local_cached_models payload."""
    return [
        {"id": e["id"], "size_gb": e.get("size_gb", 1.0),
         "path": e.get("path", "/tmp/x"), "complete": e.get("complete", True)}
        for e in entries
    ]


def test_models_info_local_aware_when_not_in_catalog(capsys, monkeypatch):
    # A bare HF id that's downloaded locally but absent from the cloud catalog
    # must describe the local copy, not say "not found".
    cli = _cli()
    monkeypatch.setattr(cli, "_local_cached_models",
                        lambda: _fake_local([{"id": "meta-llama/Llama-3.2-3B-Instruct",
                                              "size_gb": 6.0}]))
    monkeypatch.setattr(cli, "_local_model_context_window", lambda path: 131072)
    args = SimpleNamespace(name="meta-llama/Llama-3.2-3B-Instruct", output_json=True)
    code = cli._models_info(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert data["local"]["cached"] is True
    assert "transformers" in data["local"]["engines"]
    assert data["local"]["context_window"] == 131072


def test_models_info_engine_prefixed_cached_resolves_local(capsys, monkeypatch):
    # "transformers:<cached repo>" resolves to the local view, not a catalog miss.
    cli = _cli()
    monkeypatch.setattr(cli, "_local_cached_models",
                        lambda: _fake_local([{"id": "Qwen/Qwen2.5-1.5B-Instruct",
                                              "size_gb": 2.9}]))
    monkeypatch.setattr(cli, "_local_model_context_window", lambda path: 32768)
    args = SimpleNamespace(name="transformers:Qwen/Qwen2.5-1.5B-Instruct", output_json=True)
    code = cli._models_info(args)
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["engine"] == "transformers"
    assert data["id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert data["local"]["cached"] is True


def test_models_info_engine_prefixed_uncached_reports_cache_miss(capsys, monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, "_local_cached_models",
                        lambda: _fake_local([{"id": "Qwen/Qwen2.5-1.5B-Instruct"}]))
    args = SimpleNamespace(name="transformers:meta-llama/Not-Cached-99B", output_json=True)
    code = cli._models_info(args)
    data = json.loads(capsys.readouterr().out)
    assert code == 1
    assert data["engine"] == "transformers"
    assert data["local"] is None
    assert "Qwen/Qwen2.5-1.5B-Instruct" in data["cached_models"]


def test_models_status_json_shape(capsys):
    args = SimpleNamespace(output_json=True)
    code = _cli()._models_status(args)
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert set(data) >= {"cuda_available", "gpus", "loaded_models", "capability_profiles"}
    assert isinstance(data["gpus"], list)
    assert isinstance(data["loaded_models"], list)
    for gpu in data["gpus"]:
        assert {"index", "name", "total_gb", "used_gb", "free_gb"} <= set(gpu)


def test_models_info_includes_local_block_alongside_cloud(capsys, monkeypatch):
    # An id present in BOTH the catalog and the local cache shows the cloud row
    # AND a local-copy block (non-JSON path), so the local copy isn't hidden.
    cli = _cli()
    monkeypatch.setattr(cli, "_local_cached_models",
                        lambda: _fake_local([{"id": "gpt-5-nano", "size_gb": 2.0}]))
    monkeypatch.setattr(cli, "_local_model_context_window", lambda path: 4096)
    args = SimpleNamespace(name="gpt-5-nano", output_json=True)
    code = cli._models_info(args)
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["provider"] == "openai"  # cloud row preserved
    assert data["local"] is not None and data["local"]["cached"] is True


# --------------------------------------------------------------------------- #
# Cross-provider catalog browser (models browse)
# --------------------------------------------------------------------------- #
def _browse_args(**overrides):
    base = {
        "search": None, "provider": None, "free": False, "tools": False,
        "vision": False, "audio": False, "min_context": None,
        "max_price_in": None, "max_price_out": None, "sort": "provider",
        "desc": False, "limit": None, "offset": 0, "include_local": False,
        "output_json": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_models_browse_json_spans_all_providers(capsys):
    code = _cli()._models_browse(_browse_args(output_json=True))
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["count"] > 100  # the full catalog, not one provider
    provs = {m["provider"] for m in data["models"]}
    assert {"openai", "groq", "together"} <= provs
    # Every record carries the fields a browser needs, incl. is_priced.
    rec = data["models"][0]
    for f in ("id", "provider", "context_window", "price_in_per_1m",
              "supports_vision", "supports_audio", "is_priced"):
        assert f in rec


def test_models_browse_search_filters(capsys):
    code = _cli()._models_browse(_browse_args(search="gpt-5-nano", output_json=True))
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["count"] >= 1
    assert all("gpt-5-nano" in m["id"] for m in data["models"])


def test_models_browse_vision_and_min_context(capsys):
    args = _browse_args(vision=True, min_context=128000, output_json=True)
    data = json.loads(_run_json(_cli()._models_browse, args, capsys))
    assert data["count"] >= 1
    assert all(m["supports_vision"] and m["context_window"] >= 128000
               for m in data["models"])


def test_models_browse_sort_price_out_ascending(capsys):
    args = _browse_args(sort="price-out", output_json=True, limit=40)
    data = json.loads(_run_json(_cli()._models_browse, args, capsys))
    priced = [m["price_out_per_1m"] for m in data["models"]
              if m["price_out_per_1m"] is not None]
    assert priced == sorted(priced)  # non-decreasing; None sorts last


def test_models_browse_max_price_excludes_unpriced(capsys):
    # A price ceiling must exclude unpriced rows, not treat them as free/cheap.
    args = _browse_args(max_price_in=0.1, output_json=True)
    data = json.loads(_run_json(_cli()._models_browse, args, capsys))
    for m in data["models"]:
        assert m["price_in_per_1m"] is not None
        assert m["price_in_per_1m"] <= 0.1


def test_models_browse_paging(capsys):
    first = json.loads(_run_json(_cli()._models_browse,
                                 _browse_args(limit=5, output_json=True), capsys))
    assert len(first["models"]) == 5
    second = json.loads(_run_json(_cli()._models_browse,
                                  _browse_args(limit=5, offset=5, output_json=True), capsys))
    assert first["count"] == second["count"]
    assert first["models"][0]["id"] != second["models"][0]["id"]


def test_models_browse_piped_is_complete(capsys, monkeypatch):
    # Non-terminal output shows full ids and prices — no width truncation.
    cli = _cli()
    if cli.console is not None:
        monkeypatch.setattr(type(cli.console), "is_terminal", property(lambda self: False))
    cli._models_browse(_browse_args(search="qwen2.5-7b-instruct-turbo"))
    out = capsys.readouterr().out
    assert "Qwen/Qwen2.5-7B-Instruct-Turbo" in out  # full id, not truncated


def test_models_browse_narrow_terminal_keeps_id_visible(capsys, monkeypatch):
    # A narrow terminal must not hide the model id: the nine-column rich table
    # would collapse the id column at ~80 cols, so browse renders the complete
    # aligned plain-text table instead. The id and both prices stay present.
    cli = _cli()
    if cli.console is not None:
        monkeypatch.setattr(type(cli.console), "is_terminal", property(lambda self: True))
        monkeypatch.setattr(type(cli.console), "width", property(lambda self: 80))
    cli._models_browse(_browse_args(search="gpt-5-nano"))
    out = capsys.readouterr().out
    assert "gpt-5-nano" in out           # id present, not collapsed to nothing
    assert "$0.05" in out or "$0" in out  # a price cell present


def test_models_browse_json_verified_on_populated(capsys):
    # The advertised provenance field carries the provider snapshot date, not a
    # null (mirrors the human view and the dashboard catalog payload).
    data = json.loads(_run_json(_cli()._models_browse,
                                _browse_args(limit=5, output_json=True), capsys))
    assert data["models"]
    assert all(m["verified_on"] for m in data["models"])


def _run_json(fn, args, capsys):
    capsys.readouterr()  # clear
    fn(args)
    return capsys.readouterr().out


def test_models_info_surfaces_multi_provider(capsys):
    # A bare id served by more than one provider lists the alternatives.
    args = SimpleNamespace(name="Qwen/Qwen2.5-7B-Instruct", output_json=True)
    code = _cli()._models_info(args)
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    also = data.get("also_available", [])
    provs = {data["provider"]} | {v["provider"] for v in also}
    assert {"together", "hf"} <= provs
    for v in also:
        assert v["provider"] != data["provider"]


def test_models_info_single_provider_has_empty_alternatives(capsys):
    args = SimpleNamespace(name="gpt-5-nano", output_json=True)
    _cli()._models_info(args)
    data = json.loads(capsys.readouterr().out)
    assert data["also_available"] == []


def test_models_list_flags_incomplete_download(capsys, monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, "_local_cached_models", lambda: _fake_local([
        {"id": "Qwen/Qwen2.5-1.5B-Instruct", "size_gb": 2.9, "complete": True},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "size_gb": 0.01, "complete": False},
    ]))
    args = SimpleNamespace(provider=None, free=False, tools=False, output_json=False)
    cli._models_list(args)
    out = capsys.readouterr().out
    assert "incomplete" in out.lower()
    # the "ready" count excludes the incomplete one.
    assert "(1 ready)" in out


def test_local_cached_models_carries_complete_flag():
    # Against the real HF cache on this host: every entry exposes a bool 'complete'.
    for entry in _cli()._local_cached_models():
        assert isinstance(entry.get("complete"), bool)
        assert "id" in entry and "size_gb" in entry


# --------------------------------------------------------------------------- #
# --json outputs
# --------------------------------------------------------------------------- #
def test_tools_list_json(capsys):
    args = SimpleNamespace(output_json=True, category=None)
    code = _cli()._tools_list(args)
    out = capsys.readouterr().out
    data = json.loads(out)
    assert code == 0
    assert isinstance(data, list) and len(data) > 10
    assert {"name", "category", "description"} <= set(data[0])


def test_an_unmatched_category_is_not_reported_as_an_empty_registry(capsys):
    """"No tools registered" describes the registry, not the filter.

    The same command without the filter lists every tool one line later, so
    the two messages contradicted each other and sent the reader looking for a
    broken install.
    """
    args = SimpleNamespace(output_json=False, category="no-such-category")
    code = _cli()._tools_list(args)
    out = capsys.readouterr().out

    assert code == 0, "an empty result is not an error"
    assert "No tools registered" not in out
    assert "no-such-category" in out
    # The valid values are what the reader needs next.
    assert "computation" in out


# --------------------------------------------------------------------------- #
# Doctor system report
# --------------------------------------------------------------------------- #
def test_doctor_system_report_has_cuda_and_vllm():
    report = _main._doctor_system_report(include_pip_check=False)
    assert "torch.cuda.is_available()" in report
    assert "vLLM" in report
    assert "torch" in report


def test_doctor_reliability_report_empty_for_untouched_provider():
    from effgen.models.registry import ProviderRegistry

    provider = "_test_doctor_reliability_untouched"
    ProviderRegistry._providers.setdefault(provider, {})
    try:
        report = _main._doctor_reliability_report()
        assert provider not in report
    finally:
        ProviderRegistry._providers.pop(provider, None)


def test_doctor_reliability_report_shows_open_circuit():
    from effgen.models.registry import ProviderRegistry

    provider = "_test_doctor_reliability_open_circuit"
    ProviderRegistry._providers.setdefault(provider, {})
    try:
        cb = ProviderRegistry.get_circuit_breaker(provider, failure_threshold=1, recovery_timeout=30.0)
        cb.on_failure()
        assert cb.state.value == "open"

        report = _main._doctor_reliability_report()
        assert provider in report
        assert report[provider]["circuit_breaker"]["state"] == "open"
        assert report[provider]["bulkhead"] is None
    finally:
        ProviderRegistry._providers.pop(provider, None)


def test_doctor_reliability_report_shows_bulkhead_state():
    from effgen.models.registry import ProviderRegistry

    provider = "_test_doctor_reliability_bulkhead"
    ProviderRegistry._providers.setdefault(provider, {})
    try:
        bh = ProviderRegistry.get_bulkhead(provider, max_concurrency=3, queue_size=3)
        with bh.acquire():
            report = _main._doctor_reliability_report()
        assert provider in report
        assert report[provider]["bulkhead"]["active"] == 1
        assert report[provider]["bulkhead"]["max_concurrency"] == 3
    finally:
        ProviderRegistry._providers.pop(provider, None)


def test_doctor_exit_code_is_format_independent():
    # A keyed provider whose live probe failed must yield exit 1 regardless of
    # whether output is JSON or the human table.
    failed = {"openai": {"available": True, "live": {"ok": False}}}
    assert _main._doctor_exit_code(failed, live=True) == 1
    # No live probe requested → always 0 even if a key is missing.
    assert _main._doctor_exit_code(failed, live=False) == 0
    # All keyed providers usable → 0; a missing-key provider never fails the run.
    ok = {
        "openai": {"available": True, "live": {"ok": True}},
        "anthropic": {"available": False},
    }
    assert _main._doctor_exit_code(ok, live=True) == 0


# --------------------------------------------------------------------------- #
# Examples directory discovery
# --------------------------------------------------------------------------- #
def test_find_examples_dir_via_env(tmp_path, monkeypatch):
    ex = tmp_path / "examples"
    (ex / "basic").mkdir(parents=True)
    (ex / "basic" / "hello.py").write_text("def main():\n    pass\n")
    monkeypatch.setenv("EFFGEN_EXAMPLES_DIR", str(ex))
    found = _main.CLIInterface._find_examples_dir()
    assert found == ex


# --------------------------------------------------------------------------- #
# All 9 presets accepted
# --------------------------------------------------------------------------- #
def test_all_presets_accepted_by_run_parser():
    from effgen.presets import list_presets
    parser = _main.create_parser()
    all_presets = set(list_presets())
    for preset in all_presets:
        ns = parser.parse_args(["run", "task", "--preset", preset])
        assert ns.preset == preset
    # the previously-rejected four must be in there
    assert {"rag", "media", "multimodal", "notify"} <= all_presets


# --------------------------------------------------------------------------- #
# .env discovery is centralized + documented
# --------------------------------------------------------------------------- #
def test_env_search_paths_include_override_and_home(monkeypatch):
    # EFFGEN_NO_DOTENV short-circuits the walk to an empty list, so clear it:
    # the search order must be asserted regardless of the ambient environment.
    monkeypatch.delenv("EFFGEN_NO_DOTENV", raising=False)
    monkeypatch.setenv("EFFGEN_DOTENV", "/tmp/custom.env")
    paths = [str(p) for p in _main._env_search_paths()]
    assert "/tmp/custom.env" == paths[0]
    assert any(p.endswith("/.effgen/.env") for p in paths)
    # walks up from cwd, so it includes a ./.env candidate
    assert any(p.endswith("/.env") for p in paths)


def test_env_no_dotenv_flag_skips_the_walk_entirely(monkeypatch):
    monkeypatch.setenv("EFFGEN_NO_DOTENV", "1")
    monkeypatch.setenv("EFFGEN_DOTENV", "/tmp/custom.env")  # must not matter
    assert _main._env_search_paths() == []
    assert _main.load_env_files() == []


def test_env_dotenv_none_is_equivalent_to_no_dotenv(monkeypatch):
    monkeypatch.delenv("EFFGEN_NO_DOTENV", raising=False)
    monkeypatch.setenv("EFFGEN_DOTENV", "none")
    assert _main._env_search_paths() == []


def test_env_dotenv_walk_runs_by_default(monkeypatch):
    monkeypatch.delenv("EFFGEN_NO_DOTENV", raising=False)
    monkeypatch.delenv("EFFGEN_DOTENV", raising=False)
    assert _main._env_search_paths() != []


def test_load_env_files_skips_filesystem_when_disabled(monkeypatch, tmp_path):
    """A .env that would otherwise be picked up must not be loaded."""
    env_file = tmp_path / ".env"
    env_file.write_text("EFFGEN_TEST_DOTENV_PROBE=leaked\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EFFGEN_TEST_DOTENV_PROBE", raising=False)
    monkeypatch.setenv("EFFGEN_NO_DOTENV", "1")
    try:
        _main.load_env_files()
        assert "EFFGEN_TEST_DOTENV_PROBE" not in os.environ
    finally:
        os.environ.pop("EFFGEN_TEST_DOTENV_PROBE", None)


# --------------------------------------------------------------------------- #
# The library exposes the CLI's zero-config .env discovery: effgen.load_env()
# --------------------------------------------------------------------------- #
def test_toplevel_load_env_is_exported_and_callable():
    import effgen

    assert "load_env" in effgen.__all__
    assert callable(effgen.load_env)


def test_toplevel_load_env_loads_a_nearby_dotenv(monkeypatch, tmp_path):
    """A script/notebook user gets the CLI's key discovery from load_env()."""
    import effgen

    (tmp_path / ".env").write_text("EFFGEN_TEST_LOADENV_PROBE=from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EFFGEN_TEST_LOADENV_PROBE", raising=False)
    monkeypatch.delenv("EFFGEN_NO_DOTENV", raising=False)
    monkeypatch.delenv("EFFGEN_DOTENV", raising=False)
    try:
        loaded = effgen.load_env()
        assert any(p.endswith(".env") for p in loaded)
        assert os.environ.get("EFFGEN_TEST_LOADENV_PROBE") == "from_dotenv"
    finally:
        os.environ.pop("EFFGEN_TEST_LOADENV_PROBE", None)


def test_toplevel_load_env_never_overrides_real_env(monkeypatch, tmp_path):
    """A value already in the environment wins over the file."""
    import effgen

    (tmp_path / ".env").write_text("EFFGEN_TEST_LOADENV_PROBE=from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EFFGEN_TEST_LOADENV_PROBE", "from_real_env")
    monkeypatch.delenv("EFFGEN_NO_DOTENV", raising=False)
    monkeypatch.delenv("EFFGEN_DOTENV", raising=False)
    effgen.load_env()
    assert os.environ["EFFGEN_TEST_LOADENV_PROBE"] == "from_real_env"


def test_toplevel_load_env_respects_no_dotenv(monkeypatch, tmp_path):
    import effgen

    (tmp_path / ".env").write_text("EFFGEN_TEST_LOADENV_PROBE=leaked\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EFFGEN_TEST_LOADENV_PROBE", raising=False)
    monkeypatch.setenv("EFFGEN_NO_DOTENV", "1")
    try:
        assert effgen.load_env() == []
        assert "EFFGEN_TEST_LOADENV_PROBE" not in os.environ
    finally:
        os.environ.pop("EFFGEN_TEST_LOADENV_PROBE", None)


# --------------------------------------------------------------------------- #
# "Did you mean?" on mistyped subcommands and choice-based options
# --------------------------------------------------------------------------- #
def test_unknown_subcommand_suggests_and_exits_2(capsys):
    parser = _main.create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["rnu"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "unknown command 'rnu'" in err
    assert "Did you mean 'run'?" in err
    assert "Available commands:" in err


def test_mistyped_preset_value_suggests_and_exits_2(capsys):
    parser = _main.create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["run", "task", "--preset", "codng"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid value 'codng' for --preset" in err
    assert "Did you mean 'coding'?" in err


def test_mistyped_completion_value_suggests(capsys):
    parser = _main.create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--completion", "bsh"])
    err = capsys.readouterr().err
    assert "Did you mean 'bash'?" in err


# --------------------------------------------------------------------------- #
# prompts list --json alias and chat --system-prompt/--persona
# --------------------------------------------------------------------------- #

def test_prompts_list_json_is_alias_for_format_json():
    parser = _main.create_parser()
    args = parser.parse_args(["prompts", "list", "--json"])
    assert args.list_format == "json"
    # --format json still works and the default stays table.
    assert parser.parse_args(["prompts", "list"]).list_format == "table"
    assert parser.parse_args(["prompts", "list", "--format", "markdown"]).list_format == "markdown"


def test_run_accepts_system_prompt_and_persona_alias():
    parser = _main.create_parser()
    persona = "You are Grum, a gruff dwarf blacksmith."
    assert parser.parse_args(["run", "hi", "--system-prompt", persona]).system_prompt == persona
    assert parser.parse_args(["run", "hi", "--persona", persona]).system_prompt == persona
    assert parser.parse_args(["run", "hi"]).system_prompt is None


def test_run_and_chat_accept_max_tokens():
    parser = _main.create_parser()
    assert parser.parse_args(["run", "hi", "--max-tokens", "2000"]).max_tokens == 2000
    assert parser.parse_args(["run", "hi"]).max_tokens is None
    assert parser.parse_args(["chat", "--max-tokens", "2000"]).max_tokens == 2000
    assert parser.parse_args(["chat"]).max_tokens is None


def test_chat_accepts_system_prompt_and_persona_alias():
    parser = _main.create_parser()
    persona = "You are a patient Socratic tutor."
    assert parser.parse_args(["chat", "--system-prompt", persona]).system_prompt == persona
    assert parser.parse_args(["chat", "--persona", persona]).system_prompt == persona
    # Absent by default (so a plain chat keeps the default assistant persona).
    assert parser.parse_args(["chat"]).system_prompt is None


def test_batch_accepts_system_prompt_and_persona_alias():
    parser = _main.create_parser()
    persona = "Translate into formal European French (vous); keep {placeholders} verbatim."
    args = parser.parse_args(["batch", "-i", "in.jsonl", "--system-prompt", persona])
    assert args.system_prompt == persona
    args = parser.parse_args(["batch", "-i", "in.jsonl", "--persona", persona])
    assert args.system_prompt == persona
    assert parser.parse_args(["batch", "-i", "in.jsonl"]).system_prompt is None


def test_run_chat_batch_accept_guardrails_flag():
    parser = _main.create_parser()
    assert parser.parse_args(["run", "hi", "--guardrails", "phi"]).guardrails == "phi"
    assert parser.parse_args(["run", "hi"]).guardrails is None
    assert parser.parse_args(["chat", "--guardrails", "strict"]).guardrails == "strict"
    assert parser.parse_args(["chat"]).guardrails is None
    args = parser.parse_args(["batch", "-i", "in.jsonl", "--guardrails", "standard"])
    assert args.guardrails == "standard"
    assert parser.parse_args(["batch", "-i", "in.jsonl"]).guardrails is None


def test_batch_accepts_short_input_output_flags():
    parser = _main.create_parser()
    args = parser.parse_args(["batch", "-i", "in.jsonl", "-o", "out.jsonl"])
    assert args.input == "in.jsonl"
    assert args.output == "out.jsonl"
    # Long forms still work.
    args = parser.parse_args(["batch", "--input", "in.jsonl", "--output", "out.jsonl"])
    assert args.input == "in.jsonl"
    assert args.output == "out.jsonl"


def test_batch_accepts_excel_bom_flag():
    parser = _main.create_parser()
    assert parser.parse_args(["batch", "-i", "in.jsonl", "--excel"]).excel_bom is True
    assert parser.parse_args(["batch", "-i", "in.jsonl", "--bom"]).excel_bom is True
    assert parser.parse_args(["batch", "-i", "in.jsonl"]).excel_bom is False


# --------------------------------------------------------------------------- #
# Headless --json contract: run + the CI-gate commands
# --------------------------------------------------------------------------- #

def test_run_accepts_json_flag():
    parser = _main.create_parser()
    assert parser.parse_args(["run", "2+2", "--json"]).output_json is True
    assert parser.parse_args(["run", "2+2"]).output_json is False


def test_automation_commands_accept_json_flag():
    parser = _main.create_parser()
    assert parser.parse_args(["eval", "--suite", "math", "--json"]).output_json is True
    assert parser.parse_args(
        ["compare", "--models", "x", "--suite", "math", "--json"]
    ).output_json is True
    assert parser.parse_args(["workflow", "run", "w.yaml", "--json"]).output_json is True
    assert parser.parse_args(["workflow", "validate", "w.yaml", "--json"]).output_json is True
    assert parser.parse_args(["sessions", "list", "--json"]).output_json is True


def test_sessions_list_json_is_parseable(capsys):
    args = SimpleNamespace(session_command="list", output_json=True)
    code = _main._handle_sessions_command(args, _cli())
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert "sessions" in data and "sessions_dir" in data
    assert isinstance(data["sessions"], list)


def test_sessions_list_json_emits_raw_utf8_not_escaped(capsys, tmp_path, monkeypatch):
    # A locale-labeled agent name (e.g. a French persona session) must come
    # back as readable UTF-8 on --json, not \uXXXX escapes.
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path))
    (tmp_path / "s1.json").write_text(
        json.dumps({
            "session_id": "s1",
            "agent_name": "professeur-français",
            "messages": [],
        }),
        encoding="utf-8",
    )
    args = SimpleNamespace(session_command="list", output_json=True)
    code = _main._handle_sessions_command(args, _cli())
    raw = capsys.readouterr().out
    assert code == 0
    assert "professeur-français" in raw
    assert "\\u" not in raw


def test_eval_suite_list_json(capsys):
    args = SimpleNamespace(
        suite="list", model=None, preset=None, scoring="contains",
        threshold=0.5, difficulty=None, max_cases=None, output_json=True,
    )
    code = _main._handle_eval_command(args, _cli())
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    names = {d["name"] for d in data}
    assert {"math", "tool_use", "reasoning"} <= names


# --------------------------------------------------------------------------- #
# `eval` CI-gate wiring: --fail-under, --compare-baseline, agent cleanup
# --------------------------------------------------------------------------- #

def _eval_args(suite_path, **overrides):
    base = {
        "suite": str(suite_path), "model": None, "preset": None, "scoring": "contains",
        "threshold": 0.5, "fail_under": 0.5, "temperature": None, "baseline_dir": None,
        "save_baseline": False, "compare_baseline": False, "output": None,
        "difficulty": None, "max_cases": None, "output_json": False, "quiet": True,
        "no_animation": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _write_suite(tmp_path, expected="right answer"):
    suite_file = tmp_path / "cases.jsonl"
    suite_file.write_text(json.dumps({"query": "q", "expected": expected}) + "\n")
    return suite_file


def test_eval_fail_under_gates_exit_code(tmp_path, monkeypatch, capsys):
    """A suite scoring below --fail-under must exit non-zero even though the
    old hardcoded 50% gate would have passed a 0% run too — this specifically
    proves --fail-under (not --threshold) drives the exit code."""
    from tests.fixtures.mock_models import MockModel

    monkeypatch.setattr(
        "effgen.models.load_model",
        lambda *a, **k: MockModel(responses=["Thought: done\nFinal Answer: nope"]),
    )
    suite_file = _write_suite(tmp_path, expected="right answer")
    args = _eval_args(suite_file, fail_under=0.9)
    code = _main._handle_eval_command(args, _cli())
    capsys.readouterr()
    assert code == 1


def test_eval_fail_under_passes_when_accuracy_meets_gate(tmp_path, monkeypatch, capsys):
    from tests.fixtures.mock_models import MockModel

    monkeypatch.setattr(
        "effgen.models.load_model",
        lambda *a, **k: MockModel(responses=["Thought: done\nFinal Answer: right answer"]),
    )
    suite_file = _write_suite(tmp_path, expected="right answer")
    args = _eval_args(suite_file, fail_under=0.9)
    code = _main._handle_eval_command(args, _cli())
    capsys.readouterr()
    assert code == 0


def test_eval_compare_baseline_regression_fails_build(tmp_path, monkeypatch, capsys):
    """The core CI-gate bug: a detected blocking regression must fail the
    build (non-zero exit) even when the run's own accuracy still clears
    --fail-under."""
    from effgen.eval.evaluator import SuiteResults
    from effgen.eval.regression import RegressionTracker
    from tests.fixtures.mock_models import MockModel

    baseline_dir = tmp_path / "baselines"
    RegressionTracker(baselines_dir=baseline_dir).save_baseline(
        "cases", SuiteResults(suite_name="cases", accuracy=1.0), version="0.1.0",
    )
    monkeypatch.setattr(
        "effgen.models.load_model",
        lambda *a, **k: MockModel(responses=["Thought: done\nFinal Answer: wrong"]),
    )
    suite_file = _write_suite(tmp_path, expected="right answer")
    args = _eval_args(
        suite_file, compare_baseline=True, baseline_dir=str(baseline_dir), fail_under=0.0,
    )
    code = _main._handle_eval_command(args, _cli())
    out = capsys.readouterr().out
    assert code == 1
    assert "REGRESSION DETECTED" in out


def test_eval_compare_baseline_no_regression_uses_fail_under_gate(tmp_path, monkeypatch, capsys):
    """When the baseline comparison finds no regression, the exit code falls
    back to the --fail-under accuracy gate."""
    from effgen.eval.evaluator import SuiteResults
    from effgen.eval.regression import RegressionTracker
    from tests.fixtures.mock_models import MockModel

    baseline_dir = tmp_path / "baselines"
    RegressionTracker(baselines_dir=baseline_dir).save_baseline(
        "cases", SuiteResults(suite_name="cases", accuracy=1.0), version="0.1.0",
    )
    monkeypatch.setattr(
        "effgen.models.load_model",
        lambda *a, **k: MockModel(responses=["Thought: done\nFinal Answer: right answer"]),
    )
    suite_file = _write_suite(tmp_path, expected="right answer")
    args = _eval_args(
        suite_file, compare_baseline=True, baseline_dir=str(baseline_dir), fail_under=0.5,
    )
    code = _main._handle_eval_command(args, _cli())
    capsys.readouterr()
    assert code == 0


def test_eval_save_baseline_uses_resolved_suite_name_not_raw_path(tmp_path, monkeypatch, capsys):
    """A custom-dataset suite argument with directory separators must not
    crash --save-baseline (the baseline is keyed on the resolved stem)."""
    from tests.fixtures.mock_models import MockModel

    monkeypatch.setattr(
        "effgen.models.load_model",
        lambda *a, **k: MockModel(responses=["Thought: done\nFinal Answer: right answer"]),
    )
    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)
    suite_file = nested / "cases.jsonl"
    suite_file.write_text(json.dumps({"query": "q", "expected": "right answer"}) + "\n")
    baseline_dir = tmp_path / "baselines"
    args = _eval_args(suite_file, save_baseline=True, baseline_dir=str(baseline_dir))
    code = _main._handle_eval_command(args, _cli())
    capsys.readouterr()
    assert code == 0
    assert (baseline_dir / "eval_baseline_cases.json").exists()


def test_eval_closes_agent_after_run(tmp_path, monkeypatch, capsys):
    """`eval` must release its agent like `compare` does, instead of relying
    on garbage collection (which logs a warning on every run)."""
    from effgen.core.agent import Agent
    from tests.fixtures.mock_models import MockModel

    closed = []
    original_close = Agent.close

    def _tracking_close(self):
        closed.append(True)
        return original_close(self)

    monkeypatch.setattr(Agent, "close", _tracking_close)
    monkeypatch.setattr(
        "effgen.models.load_model",
        lambda *a, **k: MockModel(responses=["Thought: done\nFinal Answer: right answer"]),
    )
    suite_file = _write_suite(tmp_path, expected="right answer")
    _main._handle_eval_command(_eval_args(suite_file), _cli())
    capsys.readouterr()
    assert closed == [True]


def test_eval_suite_help_mentions_custom_dataset_path():
    """`eval --help` must document the custom-dataset path the same way
    `compare --help` already does (same underlying `_resolve_eval_suite`)."""
    parser = _main.create_parser()
    eval_help = parser._subparsers._group_actions[0].choices["eval"].format_help()
    compare_help = parser._subparsers._group_actions[0].choices["compare"].format_help()
    assert "jsonl/.json test cases" in eval_help
    assert "jsonl/.json test cases" in compare_help


def test_compare_accepts_no_animation_and_temperature_flags():
    parser = _main.create_parser()
    args = parser.parse_args([
        "compare", "--models", "x,y", "--suite", "math",
        "--no-animation", "--temperature", "0",
    ])
    assert args.no_animation is True
    assert args.temperature == 0.0


# --------------------------------------------------------------------------- #
# `eval`/`compare --provider` and `prompts run/eval -m` shorthand
# --------------------------------------------------------------------------- #

def test_eval_accepts_model_and_provider_flags_together():
    """The natural `-m X --provider Y` pattern (as on `run`/`chat`) must parse
    on `eval` instead of failing with a bare argparse error."""
    parser = _main.create_parser()
    args = parser.parse_args([
        "eval", "--suite", "math", "-m", "gpt-5-nano", "--provider", "openai",
    ])
    assert args.model == "gpt-5-nano"
    assert args.provider == "openai"


def test_compare_accepts_provider_flag():
    parser = _main.create_parser()
    args = parser.parse_args([
        "compare", "--models", "x,y", "--suite", "math", "--provider", "openai",
    ])
    assert args.provider == "openai"


def test_prompts_run_and_eval_accept_m_shorthand():
    parser = _main.create_parser()
    run_args = parser.parse_args(["prompts", "run", "some.prompt", "-m", "gpt-5-nano"])
    assert run_args.model == "gpt-5-nano"
    eval_args = parser.parse_args(["prompts", "eval", "-m", "gpt-5-nano"])
    assert eval_args.model == "gpt-5-nano"


def test_eval_provider_flag_is_forwarded_to_load_model(monkeypatch, tmp_path, capsys):
    """A typo'd --provider must fail fast (mirrors `run`); a valid one must
    reach `load_model` so a bare id resolves against the intended provider."""
    from tests.fixtures.mock_models import MockModel

    seen: dict = {}

    def _fake_load_model(model_name, *a, **k):
        seen["model_name"] = model_name
        seen["provider"] = k.get("provider")
        return MockModel(responses=["Thought: done\nFinal Answer: right answer"])

    monkeypatch.setattr("effgen.models.load_model", _fake_load_model)
    suite_file = _write_suite(tmp_path, expected="right answer")
    args = _eval_args(suite_file, model="gpt-5-nano", provider="openai")
    code = _main._handle_eval_command(args, _cli())
    capsys.readouterr()
    assert code == 0
    assert seen["provider"] == "openai"

    # An unknown provider fails fast with a suggestion, before any model load.
    seen.clear()
    args = _eval_args(suite_file, model="gpt-5-nano", provider="not-a-real-provider")
    code = _main._handle_eval_command(args, _cli())
    err = capsys.readouterr()
    assert code == 2
    assert seen == {}  # load_model never called
    assert "not-a-real-provider" in (err.out + err.err) or "Unknown provider" in (err.out + err.err)


def test_compare_provider_flag_applies_only_to_bare_ids(monkeypatch, tmp_path, capsys):
    """--provider is a fallback for ids with no explicit `provider:` prefix of
    their own; an id that already carries a prefix keeps it."""
    from tests.fixtures.mock_models import MockModel

    seen: list = []

    def _fake_load_model(model_name, *a, **k):
        seen.append((model_name, k.get("provider")))
        return MockModel(responses=["Thought: done\nFinal Answer: right answer"])

    monkeypatch.setattr("effgen.models.load_model", _fake_load_model)
    suite_file = _write_suite(tmp_path, expected="right answer")
    args = SimpleNamespace(
        models="groq:openai/gpt-oss-20b,gpt-5-nano", suite=str(suite_file),
        scoring="contains", threshold=0.5, temperature=None, preset=None,
        difficulty=None, max_cases=None, optimize="accuracy", output_json=False,
        output=None, provider="openai", quiet=True, no_animation=True,
    )
    code = _main._handle_compare_command(args, _cli())
    capsys.readouterr()
    assert code == 0
    assert ("groq:openai/gpt-oss-20b", None) in seen
    assert ("gpt-5-nano", "openai") in seen


def test_workflow_validate_json(capsys, tmp_path):
    wf = tmp_path / "wf.yaml"
    wf.write_text(
        "name: t\nnodes:\n  - id: a\n    task: x\n  - id: b\n    task: y\n"
        "    depends_on: [a]\n"
    )
    args = SimpleNamespace(workflow_command="validate", file=str(wf), output_json=True)
    code = _main._handle_workflow_command(args, _cli())
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["valid"] is True
    assert data["nodes"] == 2
    assert data["execution_order"] == ["a", "b"]


# --------------------------------------------------------------------------- #
# `run --json` still emits a JSON envelope when the agent never gets built
# --------------------------------------------------------------------------- #
def test_run_json_on_construction_failure_emits_json_envelope(capsys):
    # An unknown provider prefix fails before any AgentResponse exists (no
    # network call happens), so this needs no live key. --json must still put
    # one parseable object on stdout, not leave it empty on a nonzero exit.
    parser = _main.create_parser()
    args = parser.parse_args([
        "run", "hi", "-m", "bogus-provider:not-a-real-model", "--json", "-q",
    ])
    code = _cli().run_agent(args)
    out = capsys.readouterr().out
    assert code == 1
    data = json.loads(out)
    assert data["success"] is False
    assert "bogus-provider" in data["error"]["message"]


# --------------------------------------------------------------------------- #
# `run -c config.json` forwards a "guardrails" key to AgentConfig, and warns
# (rather than silently dropping) any other recognized-but-unwired key.
# --------------------------------------------------------------------------- #
def test_run_config_file_guardrails_key_reaches_agent_config(tmp_path, monkeypatch):
    captured = {}

    class _StubAgent:
        def __init__(self, config, session_id=None):
            captured["guardrails"] = config.guardrails
            raise RuntimeError("stop before any model call")

    monkeypatch.setattr(_main, "Agent", _StubAgent)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"guardrails": "phi"}))
    parser = _main.create_parser()
    args = parser.parse_args(["run", "hi", "-c", str(config_path), "-m", "does-not-matter"])
    code = _cli().run_agent(args)
    assert code == 1
    assert captured["guardrails"] == "phi"


def test_run_guardrails_flag_overrides_config_file(tmp_path, monkeypatch):
    captured = {}

    class _StubAgent:
        def __init__(self, config, session_id=None):
            captured["guardrails"] = config.guardrails
            raise RuntimeError("stop before any model call")

    monkeypatch.setattr(_main, "Agent", _StubAgent)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"guardrails": "phi"}))
    parser = _main.create_parser()
    args = parser.parse_args([
        "run", "hi", "-c", str(config_path), "--guardrails", "strict", "-m", "does-not-matter",
    ])
    _cli().run_agent(args)
    assert captured["guardrails"] == "strict"


def test_warn_unapplied_config_keys_flags_unwired_but_valid_fields():
    warnings = []
    stub_cli = SimpleNamespace(print_warning=lambda msg: warnings.append(msg))
    # "guardrails" is applied; "top_p" is a real AgentConfig field `run` never
    # reads from a config file; "unknown_key" isn't a field at all.
    _main._warn_unapplied_config_keys(
        {"guardrails": "phi", "top_p": 0.5, "unknown_key": 1}, stub_cli,
    )
    assert len(warnings) == 1
    assert "top_p" in warnings[0]
    assert "guardrails" not in warnings[0]
    assert "unknown_key" not in warnings[0]


def test_warn_unapplied_config_keys_silent_when_nothing_unwired():
    warnings = []
    stub_cli = SimpleNamespace(print_warning=lambda msg: warnings.append(msg))
    _main._warn_unapplied_config_keys(
        {"guardrails": "standard", "temperature": 0.2, "not_a_field": True}, stub_cli,
    )
    assert warnings == []


# --------------------------------------------------------------------------- #
# `run --stream` with a file output is refused before any model is loaded
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ["-o", "--card"])
def test_run_stream_with_a_file_output_is_refused(tmp_path, capsys, flag):
    # Streaming assembles no result object, so the file would never be written.
    # The refusal happens before the model is resolved, so this needs no key.
    target = tmp_path / ("out.json" if flag == "-o" else "card.html")
    parser = _main.create_parser()
    args = parser.parse_args(["run", "hi", "--stream", flag, str(target)])

    code = _cli().run_agent(args)

    assert code == 1
    assert not target.exists()
    message = capsys.readouterr().out
    assert "--stream" in message and flag.lstrip("-") in message.replace("/", " ")


# --------------------------------------------------------------------------- #
# `compare` terminal view: a failed model is named, with why
# --------------------------------------------------------------------------- #
def test_comparison_tables_name_the_failure_behind_an_error_row(capsys):
    """The terminal shows ERROR in the metric tables; the reason is printed
    beneath them so a reader is not left with a bare label."""
    from effgen.eval.comparison import ComparisonMatrix, ModelScore

    matrix = ComparisonMatrix(
        scores=[
            ModelScore(model_name="good", suite_name="mini", accuracy=1.0,
                       avg_latency=0.5, total_tokens=10, avg_cost_usd=0.001),
            ModelScore(model_name="missing", suite_name="mini",
                       error="ModelNotFoundError: no such model"),
            ModelScore(model_name="flaky", suite_name="mini", accuracy=0.5,
                       avg_latency=0.7, error_count=2),
        ],
        recommendations={"mini": "good"},
    )

    _main._render_comparison_tables(_cli(), matrix)

    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "missing" in out and "did not run" in out
    assert "no such model" in out
    assert "flaky" in out and "2 case(s) failed to run" in out


def test_compare_counts_contenders_not_the_judge(monkeypatch, capsys, tmp_path):
    """`--judge` names a grader, not a contender, so the run reports the number
    of models being compared."""
    suite = tmp_path / "suite.json"
    suite.write_text('[{"query": "2+2?", "expected_output": "4"}]', encoding="utf-8")

    from tests.fixtures.mock_models import MockModel

    monkeypatch.setattr(
        "effgen.models.load_model",
        lambda name, **kw: MockModel(responses=["Thought: done\nFinal Answer: 4"] * 8),
    )
    parser = _main.create_parser()
    args = parser.parse_args([
        "compare", "--suite", str(suite), "--models", "a,b",
        "--scoring", "llm_judge", "--judge", "c",
    ])

    _main._handle_compare_command(args, _cli())

    out = capsys.readouterr().out
    assert "Comparing 2 models" in out
    assert "Grading every model's answers with c." in out
