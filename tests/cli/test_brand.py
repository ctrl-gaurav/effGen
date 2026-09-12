"""Tests for the terminal brand identity: logo, glyphs, and the bare-command landing.

Covers the visual assets (:mod:`effgen.ui.brand`), the canonical glyph table and
its ASCII fallback (:mod:`effgen.ui.palette`), the named brand palette and its
lockstep with the dashboard values, and the landing's gating, edge cases, and the
map from a quick-action key to the command it dispatches. The byte-compatibility
contract (``--version``, bare piped invocation, a piped subcommand) is checked as
a subprocess so the non-TTY output is proven unchanged and free of the landing.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import types

import pytest

from effgen import __version__
from effgen.ui import brand, palette

# SGR sequence and, within it, a color parameter (basic 30-49/90-107 or 38/48).
_SGR = re.compile(r"\x1b\[([0-9;]*)m")


def _has_color(text: str) -> bool:
    for params in _SGR.findall(text):
        codes = params.split(";")
        i = 0
        while i < len(codes):
            c = codes[i]
            if c in ("38", "48"):  # extended color: rest of the run is a color spec
                return True
            if c.isdigit():
                n = int(c)
                if 30 <= n <= 49 or 90 <= n <= 107:
                    return True
            i += 1
    return False


class _EncStringIO(io.StringIO):
    """A writable text buffer whose ``encoding`` can be pinned (StringIO's is fixed)."""

    def __init__(self, encoding: str = "utf-8"):
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._encoding


def _stream(encoding: str | None):
    """A read-only stand-in exposing just ``.encoding`` for supports_unicode()."""
    return types.SimpleNamespace(encoding=encoding)


class _FakeConsole:
    """A minimal console standing in for rich in landing dispatch tests."""

    def __init__(self, scripted: list[str], *, encoding: str = "utf-8"):
        self.file = _stream(encoding)
        self.width = 80
        self.is_terminal = True
        self.no_color = False
        self.color_system = "truecolor"
        self._scripted = list(scripted)
        self.prints: list[str] = []

    def print(self, *args, **kwargs):
        self.prints.append(" ".join(str(a) for a in args))

    def input(self, prompt: str = "") -> str:
        if not self._scripted:
            raise EOFError
        return self._scripted.pop(0)


# --------------------------------------------------------------------------- #
# Logo banner assets
# --------------------------------------------------------------------------- #
class TestBanner:
    def test_banner_fits_the_box(self):
        rows = brand.LOGO_BANNER.split("\n")
        assert len(rows) <= 6
        assert brand.BANNER_WIDTH <= 64

    def test_banner_is_a_rectangle(self):
        rows = brand.LOGO_BANNER.split("\n")
        assert len({len(r) for r in rows}) == 1

    def test_ascii_banner_has_same_shape_and_no_blocks(self):
        uni = brand.LOGO_BANNER.split("\n")
        asc = brand.LOGO_BANNER_ASCII.split("\n")
        assert len(uni) == len(asc)
        assert all(len(a) == len(u) for a, u in zip(asc, uni))
        assert not any(ch in brand.LOGO_BANNER_ASCII for ch in "▀▄█")

    def test_ascii_banner_encodes_as_ascii(self):
        brand.LOGO_BANNER_ASCII.encode("ascii")  # must not raise


# --------------------------------------------------------------------------- #
# supports_unicode + wordmark
# --------------------------------------------------------------------------- #
class TestUnicodeAndWordmark:
    def test_supports_unicode_true_for_utf8(self):
        assert palette.supports_unicode(_stream("utf-8")) is True

    def test_supports_unicode_false_for_ascii(self):
        assert palette.supports_unicode(_stream("ascii")) is False

    def test_supports_unicode_false_without_encoding(self):
        assert palette.supports_unicode(_stream(None)) is False

    def test_supports_unicode_reads_through_a_live_region_proxy(self):
        """While a live region is up, stdout is a proxy carrying no encoding."""
        proxy = types.SimpleNamespace(rich_proxied_file=_stream("utf-8"))
        assert palette.supports_unicode(proxy) is True
        assert palette.glyph("success", proxy) == palette.GLYPHS["success"]

    def test_wordmark_plain_and_versioned(self):
        s = _stream("utf-8")
        assert brand.wordmark(stream=s) == "⚡ effGen"
        assert brand.wordmark("1.2.3", stream=s) == "⚡ effGen v1.2.3"

    def test_wordmark_ascii_fallback(self):
        s = _stream("ascii")
        assert "⚡" not in brand.wordmark(stream=s)
        assert brand.wordmark(stream=s).startswith(">")


# --------------------------------------------------------------------------- #
# logo_renderable across color contexts
# --------------------------------------------------------------------------- #
def _render_logo(**console_kwargs) -> str:
    from rich.console import Console

    buf = _EncStringIO("utf-8")
    console = Console(file=buf, force_terminal=True, width=console_kwargs.pop("width", 80),
                      **console_kwargs)
    console.print(brand.logo_renderable(console=console))
    return buf.getvalue()


def _strip(text: str) -> str:
    return _SGR.sub("", text)


class TestLogoRenderable:
    def test_truecolor_has_color(self):
        out = _render_logo(color_system="truecolor")
        assert _has_color(out)
        assert "█" in out

    def test_no_color_has_no_color_codes(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        out = _render_logo(no_color=True, color_system="truecolor")
        assert not _has_color(out)

    def test_narrow_width_uses_wordmark(self):
        out = _render_logo(color_system="truecolor", width=24)
        # The one-line wordmark, not the multi-row banner.
        assert "effGen" in _strip(out)
        assert out.count("\n") <= 1

    def test_ascii_stream_uses_ascii_banner(self):
        from rich.console import Console

        buf = _EncStringIO("ascii")
        console = Console(file=buf, force_terminal=True, width=80, color_system="truecolor")
        console.print(brand.logo_renderable(console=console))
        out = buf.getvalue()
        assert "#" in out
        assert "█" not in out

    def test_monochrome_theme_has_no_color(self, monkeypatch):
        monkeypatch.setenv("EFFGEN_THEME", "monochrome")
        out = _render_logo(color_system="truecolor")
        assert not _has_color(out)


# --------------------------------------------------------------------------- #
# Named brand palette + glyph table
# --------------------------------------------------------------------------- #
class TestBrandPalette:
    def test_brand_colors_lockstep_with_dashboard(self):
        values = set(palette.DASHBOARD_DARK.values()) | set(palette.DASHBOARD_LIGHT.values())
        for hexval in palette.BRAND_COLORS.values():
            assert hexval in values, f"{hexval} drifted from the dashboard palette"

    def test_gradient_stops_per_theme(self):
        assert palette.brand_gradient_stops("monochrome") == ()
        assert palette.brand_gradient_stops("light")[0] == palette.BRAND_COLORS["bolt-indigo-dark"]
        stops = palette.brand_gradient_stops("default")
        assert stops[0] == palette.BRAND_COLORS["bolt-indigo"]
        assert stops[-1] == palette.BRAND_COLORS["bolt-cyan"]

    def test_named_ramp_present_for_each_theme(self):
        for name in ("default", "high-contrast", "light"):
            assert len(palette.brand_named_ramp(name)) == 2
        assert palette.brand_named_ramp("monochrome") == ()


class TestGlyphs:
    def test_glyph_byte_identical_on_utf8(self):
        s = _stream("utf-8")
        for name, value in palette.GLYPHS.items():
            assert palette.glyph(name, s) == value

    def test_glyph_ascii_fallback_is_encodable(self):
        s = _stream("ascii")
        for name in palette.GLYPHS:
            palette.glyph(name, s).encode("ascii")  # must not raise
        assert palette.glyph("success", s) == "+"
        assert palette.glyph("error", s) == "x"

    def test_status_glyphs_are_covered_by_the_table(self):
        # Every glyph the status map uses also exists in the canonical table so
        # the two presentations of a state cannot diverge.
        assert set(palette.STATUS_GLYPH.values()) <= set(palette.GLYPHS.values())


# --------------------------------------------------------------------------- #
# Landing gating + dispatch map
# --------------------------------------------------------------------------- #
from effgen.cli import landing  # noqa: E402
from effgen.cli._main import create_parser  # noqa: E402


class _Cli:
    def __init__(self, console):
        self.console = console


class TestLandingGating:
    def _args(self, **kw):
        import argparse

        ns = argparse.Namespace(command=None, quiet=False)
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    def test_not_shown_when_command_set(self, monkeypatch):
        monkeypatch.setattr(landing._onboarding, "is_interactive", lambda: True)
        console = _FakeConsole([])
        assert landing.should_show(_Cli(console), self._args(command="run")) is False

    def test_not_shown_when_quiet(self, monkeypatch):
        monkeypatch.setattr(landing._onboarding, "is_interactive", lambda: True)
        console = _FakeConsole([])
        assert landing.should_show(_Cli(console), self._args(quiet=True)) is False

    def test_not_shown_when_not_interactive(self, monkeypatch):
        monkeypatch.setattr(landing._onboarding, "is_interactive", lambda: False)
        console = _FakeConsole([])
        assert landing.should_show(_Cli(console), self._args()) is False

    def test_not_shown_when_console_not_terminal(self, monkeypatch):
        monkeypatch.setattr(landing._onboarding, "is_interactive", lambda: True)
        console = _FakeConsole([])
        console.is_terminal = False
        assert landing.should_show(_Cli(console), self._args()) is False

    def test_shown_when_interactive_terminal(self, monkeypatch):
        monkeypatch.setattr(landing._onboarding, "is_interactive", lambda: True)
        console = _FakeConsole([])
        assert landing.should_show(_Cli(console), self._args()) is True


class TestLandingDispatch:
    @pytest.mark.parametrize(
        "key,expected",
        [
            ("c", "chat"),
            ("chat", "chat"),
            ("q", "quickstart"),
            ("d", "doctor"),
            ("m", "models"),
        ],
    )
    def test_key_reaches_intended_command(self, key, expected, monkeypatch, tmp_path):
        monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
        parser = create_parser()
        console = _FakeConsole([key])
        seen: list = []

        def fake_dispatch(args, cli, prs):
            seen.append(args.command)
            return 0

        rc = landing.run(_Cli(console), parser, argparse_ns(), fake_dispatch)
        assert rc == 0
        assert seen == [expected]

    def test_enter_reaches_wizard(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
        parser = create_parser()
        console = _FakeConsole([""])
        seen: list = []
        orig = argparse_ns()

        def fake_dispatch(args, cli, prs):
            seen.append(args.command)
            return 0

        landing.run(_Cli(console), parser, orig, fake_dispatch)
        # Enter dispatches the original (command=None) args → wizard path.
        assert seen == [None]

    def test_exit_returns_zero_without_dispatch(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
        parser = create_parser()
        console = _FakeConsole(["x"])
        called = []
        rc = landing.run(_Cli(console), parser, argparse_ns(), lambda *a: called.append(1) or 0)
        assert rc == 0
        assert called == []

    def test_invalid_then_exit(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EFFGEN_HOME", str(tmp_path))
        parser = create_parser()
        console = _FakeConsole(["zzz", "x"])
        rc = landing.run(_Cli(console), parser, argparse_ns(), lambda *a: 0)
        assert rc == 0
        assert any("Unknown choice" in p for p in console.prints)


def argparse_ns():
    import argparse

    return argparse.Namespace(command=None, quiet=False, verbose=False)


# --------------------------------------------------------------------------- #
# Byte-compatibility contract (non-TTY) — proven as a subprocess.
# --------------------------------------------------------------------------- #
_EFFGEN = shutil.which("effgen")
needs_cli = pytest.mark.skipif(_EFFGEN is None, reason="effgen console script not on PATH")


@needs_cli
class TestNonTtyByteContract:
    def test_version_is_frozen(self):
        out = subprocess.run([_EFFGEN, "--version"], capture_output=True, text=True)
        assert out.stdout == f"effGen {__version__}\n"
        assert out.returncode == 0

    def test_bare_piped_prints_only_the_hint(self):
        from effgen.cli.commands.run import NO_TERMINAL_HINT

        out = subprocess.run([_EFFGEN], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True)
        # Nothing on stdout, the whole hint on stderr, and none of the wizard
        # steps rendered on the way to it.
        assert out.stdout == ""
        assert out.stderr == NO_TERMINAL_HINT + "\n"
        assert out.returncode == 2
        assert "Interactive Setup" not in out.stderr
        assert "Reasoning Style" not in out.stderr

    def test_bare_piped_has_no_landing(self):
        out = subprocess.run([_EFFGEN], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True)
        combined = out.stdout + out.stderr
        # The branded landing is TTY-only and must not appear here.
        assert "What next?" not in combined
        assert "⚡ effGen" not in combined

    def test_piped_subcommand_has_no_landing(self):
        out = subprocess.run([_EFFGEN, "presets"], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True)
        assert out.returncode == 0
        assert "What next?" not in out.stdout


def test_public_surface_anchor_unchanged():
    import effgen

    # 1.0.0 added 17 names: the OpenAI-compatible adapter and its
    # BackendUnreachableError, the middleware surface, the tool-call records,
    # the compaction strategies, and the workflow checkpoint stores. 1.1.0
    # added 9 more: AgentThread, the Step protocol, and the seven kinds of
    # step a run's conversation is made of. Growing this number is a
    # deliberate act — update it in the same commit that widens the surface,
    # and say why.
    assert len(effgen.__all__) == 245


class TestLandingRendersEachLineAsOneSpan:
    """A styled landing line carries one colour run, not a highlighted mosaic.

    Rich recolours numbers, brackets and other repr-ish tokens inside a line it
    is already styling, which splits ``effGen v0.3.2`` into three differently
    coloured runs and bolds the parentheses in an action description. The chat
    banner already turns that off; the landing is the first surface a user sees.
    """

    def _render_landing(self) -> str:
        from effgen.cli import landing
        from effgen.ui.theme import get_console

        buf = _EncStringIO("utf-8")
        console = get_console(file=buf, force_terminal=True, width=100,
                              color_system="truecolor")
        landing._render(console)
        return buf.getvalue()

    def _styled_lines(self, out: str) -> list[str]:
        # Skip the logo: it is a per-column gradient by design.
        body = out.split("\n")
        start = max(
            (i for i, ln in enumerate(body) if "effGen v" in _strip(ln)), default=0
        )
        return [ln for ln in body[start:] if _SGR.search(ln)]

    def test_the_version_line_is_not_split_by_the_number_highlighter(self):
        out = self._render_landing()
        version_line = next(ln for ln in out.split("\n") if "effGen v" in _strip(ln))
        # One opening sequence and one reset — a single span over the whole line.
        assert len(_SGR.findall(version_line)) == 2, version_line
        assert f"effGen v{__version__}" in _strip(version_line)

    def test_the_environment_line_is_not_split(self):
        out = self._render_landing()
        env_line = next(ln for ln in out.split("\n") if "Python " in _strip(ln))
        assert len(_SGR.findall(env_line)) == 2, env_line

    def test_an_action_line_styles_only_its_key(self):
        out = self._render_landing()
        wizard = next(ln for ln in out.split("\n") if "[Enter]" in _strip(ln))
        # The key is styled; the description that follows carries no styling of
        # its own, so parentheses in it are not bolded.
        assert len(_SGR.findall(wizard)) == 2, wizard
        assert "(interactive wizard)" in _strip(wizard)

    def test_every_styled_landing_line_is_a_single_span(self):
        for line in self._styled_lines(self._render_landing()):
            assert len(_SGR.findall(line)) == 2, line
