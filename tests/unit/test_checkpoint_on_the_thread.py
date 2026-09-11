"""What a saved run carries, and what resuming one gets back.

A checkpoint, a session turn, a run record and a debug trace each held a run's
progress as the transcript string it rendered to. They hold the run's steps now,
versioned by the thread's own schema, so resuming gets the conversation the run
had rather than a reading of its text. A checkpoint written before the steps
were kept still resumes: it is reconstructed from its transcript, and the losses
that costs are named here and asserted, not left to be discovered.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from effgen.core.agent import Agent, AgentConfig
from effgen.core.checkpoint import Checkpoint, CheckpointManager
from effgen.core.multimodal import image_from
from effgen.core.session import Session
from effgen.core.thread import (
    ActionStep,
    AgentThread,
    ObservationStep,
    SystemStep,
    TaskStep,
    ThoughtStep,
)
from effgen.models.base import BaseModel, GenerationResult, ModelType, TokenCount
from effgen.tools.base_tool import (
    BaseTool,
    ParameterSpec,
    ParameterType,
    ToolCategory,
    ToolMetadata,
)

#: The checkpoint the v1.0.1 release itself wrote, from a two-call run: an agent
#: from that release was run against a checkpoint directory and the file it
#: produced was kept verbatim. The reader is therefore pinned by a file an older
#: effGen really wrote, not by a hand-typed one that might agree with the reader
#: by construction.
V101_CHECKPOINT = Path(__file__).parent / "compat" / "checkpoint_v1_0_1.json"

#: A one-pixel PNG, so a test needs no fixture file on disk.
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
    b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class Calc(BaseTool):
    """A deterministic tool, so a turn's call has one right answer."""

    def __init__(self) -> None:
        super().__init__(metadata=ToolMetadata(
            name="calculator",
            description="Evaluate an arithmetic expression.",
            category=ToolCategory.COMPUTATION,
            parameters=[ParameterSpec(
                name="expression", type=ParameterType.STRING,
                description="the expression", required=True,
            )],
        ))

    async def _execute(self, **kwargs: Any) -> str:
        expression = str(kwargs.get("expression", ""))
        return "391" if "17" in expression else "399"


class ReadsThePrompt(BaseModel):
    """Answers from what the prompt already contains, not from a turn counter.

    A model that replays a fixed script cannot say whether a resumed run and an
    uninterrupted one agree, because the resumed one restarts the script. This
    one decides from the conversation it is shown, which is what a real model at
    temperature 0 does and what makes the two runs comparable.
    """

    def __init__(self, *, native: bool = False) -> None:
        super().__init__(model_name="reads-the-prompt", model_type=ModelType.OPENAI)
        self._is_loaded = True
        self.native = native
        self.prompts: list[Any] = []

    def load(self) -> None:
        self._is_loaded = True

    def unload(self) -> None:
        self._is_loaded = False

    def supports_tool_calling(self) -> bool:
        return self.native

    def tool_call_support(self) -> str:
        return "api" if self.native else "none"

    def streams_tool_calls(self) -> bool:
        return False

    def supports_conversation(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    def generate(self, prompt: Any, config: Any = None, **kwargs: Any) -> GenerationResult:
        self.prompts.append(prompt)
        seen = str(prompt)
        if "399" in seen:
            text = "Thought: I have what I need.\nFinal Answer: 399"
        elif "391" in seen:
            text = (
                "Thought: Now I add the offset.\nAction: calculator\n"
                'Action Input: {"expression": "391 + 8"}'
            )
        else:
            text = (
                "Thought: I should multiply the two numbers.\nAction: calculator\n"
                'Action Input: {"expression": "17 * 23"}'
            )
        return GenerationResult(
            text=text, tokens_used=5, finish_reason="stop", model_name=self.model_name,
        )

    def generate_stream(self, prompt: Any, config: Any = None, **kwargs: Any):
        yield self.generate(prompt, config, **kwargs).text

    def count_tokens(self, text: str) -> TokenCount:
        return TokenCount(count=len(str(text).split()), model_name=self.model_name)

    def get_context_length(self) -> int:
        return 32768


TASK = "Multiply 17 by 23 and add 8."


def _agent(**config: Any) -> Agent:
    config.setdefault("max_iterations", 4)
    return Agent(AgentConfig(
        name="ledger", model=ReadsThePrompt(), tools=[Calc()],
        raise_on_error=False, **config,
    ))


def _checkpoints(directory: Path) -> list[dict[str, Any]]:
    """Every periodic checkpoint in *directory*, oldest first."""
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name == "latest.json":
            continue
        with open(directory / name) as fh:
            out.append(json.load(fh))
    out.sort(key=lambda d: (d.get("iteration", 0), d.get("created_at", "")))
    return out


@pytest.fixture
def quiet_history(monkeypatch, tmp_path):
    """Keep run records and sessions out of the developer's real state dir."""
    monkeypatch.setenv("EFFGEN_RUN_HISTORY_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("EFFGEN_SESSIONS_DIR", str(tmp_path / "sessions"))
    return tmp_path


# --------------------------------------------------------------- AC-1: it carries them
class TestACheckpointCarriesTheRunsSteps:
    def test_a_periodic_checkpoint_stores_the_thread(self, tmp_path, quiet_history):
        agent = _agent()
        try:
            agent.run(TASK, checkpoint_dir=str(tmp_path), checkpoint_interval=1)
        finally:
            agent.close()
        saved = _checkpoints(tmp_path)
        assert saved, "the run wrote no periodic checkpoint"
        assert all(cp["thread"]["steps"] for cp in saved[1:]), (
            "a checkpoint taken after the first turn recorded no steps"
        )

    def test_the_stored_thread_carries_its_schema_version(self, tmp_path, quiet_history):
        from effgen.core.thread import THREAD_SCHEMA_VERSION

        agent = _agent()
        try:
            agent.run(TASK, checkpoint_dir=str(tmp_path), checkpoint_interval=1)
        finally:
            agent.close()
        for cp in _checkpoints(tmp_path):
            assert cp["thread"]["version"] == THREAD_SCHEMA_VERSION

    def test_the_transcript_is_written_beside_the_steps(self, tmp_path, quiet_history):
        """So a build that only knows the transcript can still read this file."""
        agent = _agent()
        try:
            agent.run(TASK, checkpoint_dir=str(tmp_path), checkpoint_interval=1)
        finally:
            agent.close()
        for cp in _checkpoints(tmp_path)[1:]:
            rendered = AgentThread.from_dict(cp["thread"]).to_text()
            assert cp["scratchpad"] == rendered

    def test_the_final_checkpoint_stores_the_thread(self, tmp_path, quiet_history):
        agent = _agent()
        try:
            response = agent.run(TASK, checkpoint_dir=str(tmp_path))
        finally:
            agent.close()
        final = CheckpointManager(str(tmp_path)).load(
            response.metadata["checkpoint_id"]
        )
        assert final.thread["steps"], "the final checkpoint recorded no steps"
        assert final.to_thread().to_text() == response.metadata["thread"].to_text()


# --------------------------------------------- AC-3: a 1.1.0 checkpoint round-trips
class TestA110CheckpointRoundTripsLosslessly:
    def _saved(self, tmp_path, quiet_history) -> tuple[Checkpoint, AgentThread]:
        agent = _agent()
        try:
            response = agent.run(TASK, checkpoint_dir=str(tmp_path), checkpoint_interval=1)
        finally:
            agent.close()
        before = response.metadata["thread"]
        path = tmp_path / "roundtrip.json"
        cp = CheckpointManager.snapshot_agent(
            agent, task=TASK, iteration=response.iterations, thread=before,
        )
        cp.checkpoint_id = "roundtrip"
        path.write_text(json.dumps(cp.to_dict(), indent=2))
        return Checkpoint.from_dict(json.loads(path.read_text())), before

    def test_through_a_file_the_transcript_is_byte_identical(self, tmp_path, quiet_history):
        restored, before = self._saved(tmp_path, quiet_history)
        assert restored.to_thread().to_text() == before.to_text()

    def test_through_a_file_the_steps_are_equal(self, tmp_path, quiet_history):
        restored, before = self._saved(tmp_path, quiet_history)
        assert restored.to_thread().to_dict() == before.to_dict()

    def test_a_call_id_and_an_arguments_mapping_survive(self, tmp_path, quiet_history):
        """The two things reading a transcript back cannot recover."""
        thread = AgentThread(steps=[
            ThoughtStep(text="I will look it up."),
            ActionStep(
                tool="calculator",
                arguments={"expression": "17 * 23", "precision": 2},
                raw='{"expression": "17 * 23", "precision": 2}',
                call_id="call_abc123",
            ),
            ObservationStep(text="391", call_id="call_abc123"),
        ])
        cp = Checkpoint(
            checkpoint_id="c", agent_name="a", task=TASK, iteration=1,
            thread=thread.to_dict(), scratchpad=thread.to_text(),
        )
        back = Checkpoint.from_dict(json.loads(json.dumps(cp.to_dict()))).to_thread()
        action = back.actions()[0]
        assert action.call_id == "call_abc123"
        assert action.arguments == {"expression": "17 * 23", "precision": 2}


# ------------------------------------------- AC-2: a 1.0.x checkpoint still resumes
class TestA10xCheckpointStillResumes:
    def _fixture(self) -> dict[str, Any]:
        return json.loads(V101_CHECKPOINT.read_text())

    def test_the_fixture_is_what_v1_0_1_wrote(self):
        """A fixture carrying a thread would prove nothing about the old shape."""
        data = self._fixture()
        assert "thread" not in data
        assert data["scratchpad"].startswith("\nThought:")
        assert data["tool_calls"] == 2

    def test_it_loads_without_raising(self):
        cp = Checkpoint.from_dict(self._fixture())
        assert cp.task == "Multiply 17 by 23 and add 8."
        assert cp.iteration == 3

    def test_the_reconstruction_renders_the_transcript_byte_for_byte(self):
        data = self._fixture()
        assert Checkpoint.from_dict(data).to_thread().to_text() == data["scratchpad"]

    def test_the_reconstruction_recovers_the_calls_and_their_answers(self):
        thread = Checkpoint.from_dict(self._fixture()).to_thread()
        assert [a.tool for a in thread.actions()] == ["calculator", "calculator"]
        answers = [o.text for o in thread.observations()]
        assert answers[0] == "391"
        # The nudge the run wrote after the second answer carries no marker, so
        # reading the text back folds it into that answer. It renders to the
        # same bytes; it is one step where the run had two.
        assert answers[1].startswith("399\n[You have the answer")

    def test_a_run_resumed_from_it_completes(self, tmp_path, quiet_history):
        directory = tmp_path / "ckpt"
        directory.mkdir()
        data = self._fixture()
        (directory / f"{data['checkpoint_id']}.json").write_text(json.dumps(data))
        agent = _agent()
        try:
            response = agent.resume(
                checkpoint_id=data["checkpoint_id"], checkpoint_dir=str(directory),
            )
        finally:
            agent.close()
        assert response.success
        assert "399" in response.output

    def test_what_the_reconstruction_loses(self):
        """Named here so the loss is a documented property, not a surprise.

        A call id is renamed from the step's position; an input the model did
        not write as JSON comes back whole under one key instead of as the
        arguments it stood for; a line the framework injected after an
        observation comes back as part of that observation; the step the run
        ended on is gone; and the frame the run was asked in — persona, tool
        contract, earlier turns, the task itself — was never in the transcript
        and does not return.
        """
        thread = Checkpoint.from_dict(self._fixture()).to_thread()
        action = thread.actions()[0]
        assert action.call_id == "effgen-1", "a provider's own call id is not in the text"
        assert action.raw == '{"expression": "17 * 23"}'
        assert not [s for s in thread.steps if isinstance(s, SystemStep | TaskStep)]
        # The injected line after the last observation is inside it, not beside it.
        assert not [s for s in thread.steps if s.to_dict()["kind"] == "nudge"]
        assert "Final Answer:" in thread.observations()[-1].text
        # And the run's own answer step, with the reason it stopped, is not there.
        assert not [s for s in thread.steps if s.to_dict()["kind"] == "answer"]

    def test_an_input_the_model_did_not_write_as_json_loses_its_argument_names(self):
        """The one loss the text really does cost an argument.

        An input rendered as JSON comes back as JSON — types, nesting and all —
        so the loss is narrower than "arguments become a string": it is an input
        that was never JSON, which comes back whole under a single key.
        """
        thread = AgentThread()
        thread.extend([
            ThoughtStep(text="work it out"),
            ActionStep(tool="calculator", arguments={"n": 17, "of": {"a": [1, 2]}},
                       raw='{"n": 17, "of": {"a": [1, 2]}}', call_id="provider-x"),
            ObservationStep(text="391", call_id="provider-x"),
        ])
        back = AgentThread.from_scratchpad(thread.to_text())
        assert back.actions()[0].arguments == {"n": 17, "of": {"a": [1, 2]}}

        prose = AgentThread()
        prose.extend([
            ThoughtStep(text="work it out"),
            ActionStep(tool="calculator", arguments={"expression": "17 * 23"}, raw="17 * 23"),
        ])
        recovered = AgentThread.from_scratchpad(prose.to_text()).actions()[0]
        assert "expression" not in recovered.arguments
        assert list(recovered.arguments.values()) == ["17 * 23"]

    def test_the_reconstruction_is_logged(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="effgen.core._compat"):
            Checkpoint.from_dict(self._fixture()).to_thread()
        assert any(
            "rebuilt a thread from a flat transcript" in r.message for r in caplog.records
        )


# ---------------------------------------------------- AC-4: resuming changes nothing
class TestAResumedRunAgreesWithAnUninterruptedOne:
    def _uninterrupted(self, quiet_history) -> Any:
        agent = _agent()
        try:
            return agent.run(TASK)
        finally:
            agent.close()

    def _resumed(self, tmp_path, quiet_history, *, at: int) -> Any:
        """Run until *at* iterations, then continue from that checkpoint."""
        directory = tmp_path / "ckpt"
        directory.mkdir()
        first = _agent(max_iterations=at)
        try:
            first.run(TASK, checkpoint_dir=str(directory), checkpoint_interval=1)
        finally:
            first.close()
        saved = [cp for cp in _checkpoints(directory) if cp["thread"]["steps"]]
        assert saved, "nothing to resume from"
        second = _agent()
        try:
            return second.resume(
                checkpoint_id=saved[-1]["checkpoint_id"], checkpoint_dir=str(directory),
            )
        finally:
            second.close()

    @pytest.mark.parametrize("at", [1, 2])
    def test_the_answer_is_the_same(self, tmp_path, quiet_history, at):
        whole = self._uninterrupted(quiet_history)
        resumed = self._resumed(tmp_path, quiet_history, at=at)
        assert resumed.output == whole.output

    def test_the_resumed_run_does_not_repeat_the_work_it_had_done(
        self, tmp_path, quiet_history
    ):
        resumed = self._resumed(tmp_path, quiet_history, at=2)
        assert resumed.iterations < self._uninterrupted(quiet_history).iterations

    def test_resuming_is_logged_with_the_step_count(self, tmp_path, quiet_history, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="effgen.core.agent_loop"):
            self._resumed(tmp_path, quiet_history, at=2)
        assert any("resumed a run from" in r.message for r in caplog.records)


# ------------------------------------------------------------- the other three stores
class TestTheSessionCarriesTheThread:
    def test_the_reply_carries_the_runs_steps(self, quiet_history):
        session = Session(session_id="s1")
        agent = _agent()
        try:
            agent.run(TASK, session=session)
        finally:
            agent.close()
        reply = session.messages[-1]
        assert reply["role"] == "assistant"
        assert reply["metadata"]["thread"]["steps"]

    def test_the_question_does_not(self, quiet_history):
        """One copy of the conversation per turn, not two."""
        session = Session(session_id="s2")
        agent = _agent()
        try:
            agent.run(TASK, session=session)
        finally:
            agent.close()
        assert "thread" not in session.messages[0]["metadata"]

    def test_last_thread_returns_it(self, quiet_history):
        session = Session(session_id="s3")
        agent = _agent()
        try:
            response = agent.run(TASK, session=session)
        finally:
            agent.close()
        assert session.last_thread().to_text() == response.metadata["thread"].to_text()

    def test_a_session_written_before_this_gives_an_empty_thread(self):
        session = Session(session_id="s4")
        session.add_message("user", "hello")
        session.add_message("assistant", "hi")
        assert session.last_thread().steps == []

    def test_it_survives_a_save_and_a_load(self, quiet_history, tmp_path):
        session = Session(session_id="s5")
        agent = _agent()
        try:
            response = agent.run(TASK, session=session)
        finally:
            agent.close()
        session.save()
        back = Session.load("s5")
        assert back.last_thread().to_text() == response.metadata["thread"].to_text()


class TestTheRunStoreCarriesTheThreadsShape:
    def test_a_record_names_the_steps_the_run_took(self, quiet_history):
        from effgen.observability import run_log

        agent = _agent()
        try:
            agent.run(TASK)
        finally:
            agent.close()
        record = run_log.get_recent_runs()[0]
        assert record["thread_steps"] >= 3
        assert "action" in record["thread_kinds"]
        assert record["thread_version"] == 1

    def test_the_record_does_not_grow_a_transcript(self, quiet_history):
        """This store is a bounded ring of previews; the text lives elsewhere."""
        from effgen.observability import run_log

        agent = _agent()
        try:
            agent.run(TASK)
        finally:
            agent.close()
        record = run_log.get_recent_runs()[0]
        assert all(not isinstance(v, dict) for v in record.values())
        assert "17 * 23" not in json.dumps(record)


class TestTheDebugTraceCarriesTheThread:
    def test_each_iteration_stores_the_steps(self, quiet_history):
        agent = _agent()
        try:
            response = agent.run(TASK, debug=True)
        finally:
            agent.close()
        trace = response.metadata["debug_trace"]
        assert trace.iterations
        for iteration in trace.iterations:
            assert iteration.thread_snapshot["steps"]
            assert (
                AgentThread.from_dict(iteration.thread_snapshot).to_text()
                == iteration.scratchpad_snapshot
            )

    def test_the_trace_document_carries_them(self, quiet_history):
        agent = _agent()
        try:
            response = agent.run(TASK, debug=True)
        finally:
            agent.close()
        first = response.metadata["debug_trace"].iterations[0].to_dict()
        assert first["thread"]["steps"]


# --------------------------------------------------- shapes this was not written for
class TestShapesThisWasNotWrittenFor:
    def test_a_checkpoint_from_a_run_that_used_a_tool(self, tmp_path, quiet_history):
        agent = _agent()
        try:
            agent.run(TASK, checkpoint_dir=str(tmp_path), checkpoint_interval=1)
        finally:
            agent.close()
        thread = Checkpoint.from_dict(_checkpoints(tmp_path)[-1]).to_thread()
        assert [a.tool for a in thread.actions()] == ["calculator", "calculator"]
        assert thread.unanswered_calls() == []

    def test_a_checkpoint_from_a_run_carrying_a_picture(self, tmp_path, quiet_history):
        agent = _agent()
        try:
            agent.run(
                TASK, inputs=[image_from(PNG, mime="image/png")],
                checkpoint_dir=str(tmp_path), checkpoint_interval=1,
            )
        finally:
            agent.close()
        saved = _checkpoints(tmp_path)
        assert saved, "the run wrote no checkpoint"
        restored = Checkpoint.from_dict(saved[-1]).to_thread()
        assert restored.to_dict() == AgentThread.from_dict(
            saved[-1]["thread"]
        ).to_dict()

    def test_a_checkpoint_taken_between_a_call_and_its_answer(self):
        """The unanswered call comes back as one, so the resumed run can close it."""
        thread = AgentThread(steps=[
            ThoughtStep(text="I will look it up."),
            ActionStep(tool="calculator", arguments={"expression": "17 * 23"},
                       raw='{"expression": "17 * 23"}', call_id="call_mid"),
        ])
        cp = Checkpoint(
            checkpoint_id="mid", agent_name="a", task=TASK, iteration=1,
            thread=thread.to_dict(), scratchpad=thread.to_text(),
        )
        back = Checkpoint.from_dict(json.loads(json.dumps(cp.to_dict()))).to_thread()
        assert [a.call_id for a in back.unanswered_calls()] == ["call_mid"]

    def test_a_checkpoint_written_by_a_nested_agent(self, tmp_path, quiet_history):
        """An inner agent's run writes its own file into the same directory."""
        directory = tmp_path / "ckpt"
        directory.mkdir()
        outer = Agent(AgentConfig(
            name="outer", model=ReadsThePrompt(), tools=[Calc()],
            max_iterations=2, raise_on_error=False,
        ))
        inner = Agent(AgentConfig(
            name="inner", model=ReadsThePrompt(), tools=[Calc()],
            max_iterations=2, raise_on_error=False,
        ))
        try:
            inner.run(TASK, checkpoint_dir=str(directory), checkpoint_interval=1)
            outer.run(TASK, checkpoint_dir=str(directory), checkpoint_interval=1)
        finally:
            inner.close()
            outer.close()
        saved = _checkpoints(directory)
        names = {cp["agent_name"] for cp in saved}
        assert names == {"inner", "outer"}
        assert len({cp["checkpoint_id"] for cp in saved}) == len(saved)
        for cp in saved:
            assert Checkpoint.from_dict(cp).to_thread().to_text() == cp["scratchpad"]


class TestASnapshotWithoutAThreadStillWorks:
    def test_no_thread_given(self):
        cp = CheckpointManager.snapshot_agent(
            object(), task=TASK, iteration=1, scratchpad="\nThought: hi",
        )
        assert cp.thread == {}
        assert cp.to_thread().to_text() == "\nThought: hi"

    def test_a_value_that_is_not_a_thread_is_refused_loudly(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="effgen.core.checkpoint"):
            cp = CheckpointManager.snapshot_agent(
                object(), task=TASK, iteration=1, thread="not a thread",
            )
        assert cp.thread == {}
        assert any("neither a thread nor thread data" in r.message for r in caplog.records)
