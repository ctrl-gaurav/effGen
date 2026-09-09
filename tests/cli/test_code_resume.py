"""A coding session that survives a restart.

``effgen code --session-id ID`` continues a stored session: the conversation is
recalled by the session store, and the coding state — where the run was working,
under which permissions, on which files — is recorded beside it and adopted
under rules that never widen what a run may do.

The model is scripted in-process, so the store, the restore rules and the turn
bookkeeping are what is measured.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from effgen.cli.code.engine import CODING_METADATA_KEY, CodeEngine
from effgen.cli.code.permissions import PermissionMode
from effgen.cli.code.repl import CodeREPL
from effgen.core.session import Session


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Keep every session this module writes inside the test's own directory."""
    monkeypatch.setenv("EFFGEN_HOME", str(tmp_path / "home"))
    yield


def _args(**over):
    base = {
        "task": None, "print_task": None, "model": "fake:model", "provider": None,
        "workspace": None, "plan_only": False, "auto_edit": False,
        "assume_yes": False, "undo": False, "max_iterations": None,
        "temperature": None, "max_tokens": None, "output_json": False,
        "quiet": True, "session_id": None, "review": None, "review_files": None,
        "no_animation": True,
    }
    base.update(over)
    return SimpleNamespace(**base)


class _Cli:
    """The narrowest CLI surface the session's view methods use."""

    console = None
    _human_to_stderr = True

    def print_line(self, text="", style=None):
        self.lines.append(text)

    def print_header(self, text):
        self.lines.append(text)

    def print(self, text=""):
        self.lines.append(str(text))

    def _human(self):
        return None

    def _human_stream(self):
        import sys

        return sys.stderr

    def __init__(self):
        self.lines = []


def _repl(workspace, **over):
    repl = CodeREPL(_Cli(), _args(workspace=str(workspace), **over))
    repl.interactive = over.pop("interactive", False)
    return repl


def _stored(session_id):
    return Session.load_or_create(session_id)


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------

def test_the_coding_block_is_written_beside_the_conversation(tmp_path):
    engine = CodeEngine(
        model="gpt-5-nano", provider="openai", workspace=tmp_path,
        mode=PermissionMode.AUTO_EDIT,
    )
    session = Session.load_or_create("s1")
    session.add_user_message("hello")
    session.add_assistant_message("hi")
    engine._agent = SimpleNamespace(session=session)
    engine.save_coding_state(files_in_context=["a.py"], files_written=["b.py"])

    reread = _stored("s1")
    assert len(reread.messages) == 2
    block = reread.metadata[CODING_METADATA_KEY]
    assert block["workspace"] == str(tmp_path)
    assert block["permission_mode"] == "auto-edit"
    assert block["files_in_context"] == ["a.py"]
    assert block["files_written"] == ["b.py"]
    assert block["model"] == "gpt-5-nano"
    assert block["provider"] == "openai"
    assert block["updated_at"]


def test_saving_the_block_needs_no_session(tmp_path):
    engine = CodeEngine(model="m", workspace=tmp_path, mode=PermissionMode.PLAN)
    engine._agent = SimpleNamespace(session=None)
    engine.save_coding_state()  # no session attached: nothing to do, no error


def test_the_session_listing_still_reads_a_coding_session(tmp_path):
    from effgen.core.session import SessionManager

    engine = CodeEngine(model="m", workspace=tmp_path, mode=PermissionMode.PLAN)
    session = Session.load_or_create("listed")
    session.add_user_message("q")
    session.add_assistant_message("a")
    session.metadata["model"] = "m"
    engine._agent = SimpleNamespace(session=session)
    engine.save_coding_state(files_in_context=["a.py"])

    sessions, unreadable = SessionManager().scan()
    assert unreadable == []
    rows = [s for s in sessions if s["session_id"] == "listed"]
    assert rows, "a coding session must still be listed"
    assert rows[0]["messages"] == 2
    assert rows[0]["model"] == "m"


# --------------------------------------------------------------------------
# The restore rules
# --------------------------------------------------------------------------

def _write_state(session_id, **block):
    session = Session.load_or_create(session_id)
    session.add_user_message("earlier question")
    session.add_assistant_message("earlier answer")
    session.metadata[CODING_METADATA_KEY] = block
    session.save()
    return session


def test_resume_restores_the_files_in_context_and_the_written_paths(tmp_path):
    (tmp_path / "kept.py").write_text("x = 1\n", encoding="utf-8")
    _write_state(
        "r1", workspace=str(tmp_path), permission_mode="ask",
        files_in_context=["kept.py"], files_written=["out.py"], model="m",
    )
    repl = _repl(tmp_path, session_id="r1", model=None, quiet=False)
    repl._adopt_session_state("r1")

    assert repl.context_files == ["kept.py"]
    assert repl.session_files == ["out.py"]
    assert any("Resumed 'r1' (1 turn(s), 1 file(s) in context)" in line for line in repl.cli.lines)


def test_a_pinned_file_that_is_gone_is_dropped_and_named(tmp_path):
    _write_state(
        "r2", workspace=str(tmp_path), files_in_context=["gone.py"], model="m",
    )
    repl = _repl(tmp_path, session_id="r2", model=None, quiet=False)
    repl._adopt_session_state("r2")

    assert repl.context_files == []
    assert any("gone.py" in line for line in repl.cli.lines)


def test_a_different_workspace_is_reported_and_not_adopted(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    _write_state("r3", workspace=str(other), model="m")
    repl = _repl(tmp_path, session_id="r3", model=None, quiet=False)
    repl._adopt_session_state("r3")

    assert repl.workspace == tmp_path.resolve()
    assert any(str(other) in line for line in repl.cli.lines)


def test_a_stored_yes_mode_is_not_restored_without_a_terminal(tmp_path):
    _write_state("r4", workspace=str(tmp_path), permission_mode="yes", model="m")
    repl = _repl(tmp_path, session_id="r4", model=None, quiet=False)
    assert repl.interactive is False
    repl._adopt_session_state("r4")

    assert repl.mode is PermissionMode.PLAN
    assert any("this run uses plan" in line for line in repl.cli.lines)


def test_a_stored_mode_is_restored_on_a_terminal(tmp_path):
    _write_state("r5", workspace=str(tmp_path), permission_mode="auto-edit", model="m")
    repl = _repl(tmp_path, session_id="r5", model=None, quiet=False)
    repl.interactive = True
    repl.mode, repl.mode_explicit = PermissionMode.ASK, False
    repl._adopt_session_state("r5")

    assert repl.mode is PermissionMode.AUTO_EDIT


def test_a_named_permission_flag_wins_over_the_stored_one(tmp_path):
    _write_state("r6", workspace=str(tmp_path), permission_mode="yes", model="m")
    repl = _repl(tmp_path, session_id="r6", model=None, quiet=False)
    repl.interactive = True
    repl.mode, repl.mode_explicit = PermissionMode.PLAN, True
    repl._adopt_session_state("r6")

    assert repl.mode is PermissionMode.PLAN


def test_the_stored_model_is_used_when_none_was_named(tmp_path):
    _write_state("r7", workspace=str(tmp_path), model="gpt-5-nano", provider="openai")
    repl = _repl(tmp_path, session_id="r7", model=None, quiet=False)
    repl._adopt_session_state("r7")

    assert repl.model_id == "gpt-5-nano"
    assert repl.provider == "openai"


def test_a_named_model_wins_over_the_stored_one(tmp_path):
    _write_state("r8", workspace=str(tmp_path), model="gpt-5-nano", provider="openai")
    repl = _repl(tmp_path, session_id="r8", model="groq:openai/gpt-oss-20b", quiet=False)
    repl._adopt_session_state("r8")

    assert repl.model_id == "groq:openai/gpt-oss-20b"


def test_an_unknown_id_starts_a_new_session(tmp_path):
    repl = _repl(tmp_path, session_id="fresh", model=None, quiet=False)
    repl._adopt_session_state("fresh")

    assert repl.context_files == []
    assert any("Starting a new session 'fresh'" in line for line in repl.cli.lines)


# --------------------------------------------------------------------------
# The stored turn
# --------------------------------------------------------------------------

def _turn_repl(tmp_path, session_id):
    """A REPL wired to a session, poised at the start of one turn."""
    (tmp_path / "app.py").write_text("print('hi')\n", encoding="utf-8")
    engine = CodeEngine(model="m", workspace=tmp_path, mode=PermissionMode.PLAN)
    session = Session.load_or_create(session_id)
    engine._agent = SimpleNamespace(session=session)

    repl = _repl(tmp_path, session_id=session_id, quiet=False)
    repl.engine = engine
    repl.context_files = ["app.py"]
    repl._last_task = "add a docstring"
    repl._stored_before = len(session.messages)
    return repl, session


_DONE = SimpleNamespace(success=True, answer="done")


def test_a_turn_the_agent_wrote_back_keeps_the_line_that_was_typed(tmp_path):
    """The blocking path stores the composed task; the typed line replaces it."""
    repl, session = _turn_repl(tmp_path, "t1")
    composed = repl._compose_task("add a docstring")
    assert composed != "add a docstring"
    session.add_user_message(composed)      # what the agent stores
    session.add_assistant_message("done")
    repl._record_turn(_DONE)
    repl._save_coding_state()

    reread = _stored("t1")
    assert len(reread.messages) == 2
    assert reread.messages[0]["content"] == "add a docstring"
    assert reread.messages[0]["metadata"]["context_files"] == ["app.py"]
    assert reread.metadata[CODING_METADATA_KEY]["files_in_context"] == ["app.py"]


def test_a_streamed_turn_is_recorded_rather_than_lost(tmp_path):
    """The streamed path writes nothing back, so the session records the turn.

    Without this a resumed session carries its coding state and none of the
    conversation, which is the one thing "resume" has to mean.
    """
    repl, _session = _turn_repl(tmp_path, "t2")
    repl._record_turn(_DONE)
    repl._save_coding_state()

    reread = _stored("t2")
    assert [(m["role"], m["content"]) for m in reread.messages] == [
        ("user", "add a docstring"), ("assistant", "done"),
    ]
    assert reread.messages[0]["metadata"]["context_files"] == ["app.py"]


def test_a_turn_is_recorded_once(tmp_path):
    """Two consecutive turns with the same answer are two stored turns."""
    repl, session = _turn_repl(tmp_path, "t3")
    repl._record_turn(_DONE)
    repl._stored_before = len(session.messages)
    repl._last_task = "again"
    repl._record_turn(_DONE)

    reread = _stored("t3")
    assert [m["content"] for m in reread.messages] == [
        "add a docstring", "done", "again", "done",
    ]


def test_a_turn_that_failed_is_not_recorded(tmp_path):
    repl, _session = _turn_repl(tmp_path, "t4")
    repl._record_turn(SimpleNamespace(success=False, answer="Generation failed: 429"))
    assert _stored("t4").messages == []


def test_resuming_a_stored_session_does_not_duplicate_its_turns(tmp_path):
    _write_state("d1", workspace=str(tmp_path), model="m")
    assert len(_stored("d1").messages) == 2

    recorded = []

    class _Memory:
        def get_messages(self):
            return []

        def add_user_message(self, text):
            recorded.append(("user", text))

        def add_assistant_message(self, text):
            recorded.append(("assistant", text))

    agent = SimpleNamespace(
        short_term_memory=_Memory(), reset_memory=lambda: recorded.clear(),
        session=None, _session_id=None,
    )
    engine = CodeEngine(model="m", workspace=tmp_path, mode=PermissionMode.PLAN)
    engine._agent = agent
    repl = _repl(tmp_path, quiet=False)
    repl.engine = engine
    repl._cmd_session("d1")

    assert repl.session_id == "d1"
    assert recorded == [("user", "earlier question"), ("assistant", "earlier answer")]
    assert len(_stored("d1").messages) == 2, "the stored turns must not be appended to"


def test_a_new_id_still_starts_persisting_the_live_conversation(tmp_path):
    class _Memory:
        def get_messages(self):
            from effgen.memory.short_term import Message, MessageRole

            return [
                Message(role=MessageRole.USER, content="live question"),
                Message(role=MessageRole.ASSISTANT, content="live answer"),
            ]

        def add_user_message(self, text):
            pass

        def add_assistant_message(self, text):
            pass

    agent = SimpleNamespace(
        short_term_memory=_Memory(), reset_memory=lambda: None,
        session=None, _session_id=None,
    )
    engine = CodeEngine(model="m", workspace=tmp_path, mode=PermissionMode.PLAN)
    engine._agent = agent
    repl = _repl(tmp_path, quiet=False)
    repl.engine = engine
    repl._cmd_session("brand-new")

    stored = _stored("brand-new")
    assert [m["content"] for m in stored.messages] == ["live question", "live answer"]
    assert stored.metadata[CODING_METADATA_KEY]["workspace"] == str(tmp_path)


def test_session_with_no_argument_names_the_way_back(tmp_path):
    repl = _repl(tmp_path, quiet=False)
    repl.session_id = "abc"
    repl._cmd_session("")
    assert any("effgen code --session-id abc" in line for line in repl.cli.lines)


def test_the_session_is_attached_only_on_the_first_engine_build(tmp_path, monkeypatch):
    built = []

    def _fake_build(self):
        built.append(self.session_id)
        self._agent = SimpleNamespace(session=None, tools={}, config=None)
        return self._agent

    monkeypatch.setattr(CodeEngine, "build_agent", _fake_build)
    repl = _repl(tmp_path, session_id="once")
    first = repl._build_engine()
    repl._build_engine(carry_from=first)

    assert built == ["once", None]
