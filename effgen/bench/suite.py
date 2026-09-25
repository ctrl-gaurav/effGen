"""Bench suites: the small file a user writes to describe a measurement.

A suite names the tasks, the tools the agent may use, how an answer is scored,
how many tasks to run and the seed. It is YAML or JSON::

    name: arithmetic
    seed: 42
    n: 3
    tools: [calculator]
    scorer: number
    agent:
      system_prompt: "Use the calculator. End with 'Answer: <number>'."
      temperature: 0.1
      max_iterations: 5
    tasks:
      - id: crates
        input: "A crate holds 24 apples. How many apples are in 17 crates?"
        expected: 408

Tasks can instead come from a JSON Lines (or JSON list) file next to the suite,
with ``tasks_file:`` and, when its keys are named differently, ``fields:``.

Loading is strict: an unknown key, a task with no input, a duplicate id or an
``n`` larger than the task list is an error, because a typo that silently fell
back to a default would measure a different suite from the one the file names.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AGENT_SETTINGS", "BenchSuite", "BenchTask", "SuiteError", "load_suite", "scorer_needs_expected",
]

#: Agent settings a suite may fix. Every other ``AgentConfig`` field keeps the
#: bench default, so two suites differ only in what they write down.
AGENT_SETTINGS = ("system_prompt", "temperature", "top_p", "max_tokens", "max_iterations")

_TOP_LEVEL = (
    "name", "description", "seed", "n", "tools", "scorer", "agent", "model",
    "base_url", "tasks", "tasks_file", "fields",
)
_TASK_KEYS = ("id", "input", "expected", "context")


class SuiteError(ValueError):
    """A suite file that cannot be run as written."""


@dataclass(frozen=True)
class BenchTask:
    """One task: the prompt the agent receives and the answer it is scored against.

    Attributes:
        id: A name unique within the suite; runs are paired on it.
        input: The text handed to ``Agent.run``.
        expected: What the scorer compares the answer with.
        context: Optional text placed after the input, separated by a blank line.
    """

    id: str
    input: str
    expected: Any = None
    context: str = ""

    @property
    def prompt(self) -> str:
        """The text the agent receives: the input, then the context if any."""
        return f"{self.input}\n\n{self.context}" if self.context else self.input

    def to_dict(self) -> dict[str, Any]:
        """The task as plain data."""
        return {"id": self.id, "input": self.input, "expected": self.expected,
                "context": self.context}


@dataclass
class BenchSuite:
    """A loaded suite, with the tasks already selected by ``n`` and ``seed``.

    Attributes:
        name: The suite's name (the file stem when the file names none).
        path: The file it was read from, or ``None`` for an in-memory suite.
        tasks: The tasks that will run, in file order.
        total_tasks: How many tasks the file holds before ``n`` is applied.
        scorer: The scorer specification, as written.
        tools: Tool names resolved from the tool registry.
        n: The requested number of tasks, or ``None`` for all of them.
        seed: The seed: which tasks ``n`` picks, and the sampling seed handed
            to the model. ``None`` picks the first ``n`` and sends no seed.
        agent: Agent settings from :data:`AGENT_SETTINGS`.
        model: A default model id; the command line overrides it.
        base_url: A default endpoint; the command line overrides it.
        description: Free text.
    """

    name: str
    tasks: list[BenchTask]
    scorer: Any = "exact"
    tools: list[str] = field(default_factory=list)
    n: int | None = None
    seed: int | None = None
    agent: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    base_url: str | None = None
    description: str = ""
    path: Path | None = None
    total_tasks: int = 0

    def fingerprint(self) -> str:
        """A hash of the selected tasks and the scorer.

        Two runs with the same fingerprint scored the same answers against the
        same expectations; agent settings and the model are left out, so two
        configurations of one suite share it.
        """
        payload = json.dumps(
            {"tasks": [t.to_dict() for t in self.tasks], "scorer": self.scorer},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def describe(self) -> dict[str, Any]:
        """The suite as recorded in a run document."""
        return {
            "name": self.name,
            "path": str(self.path) if self.path else None,
            "fingerprint": self.fingerprint(),
            "tasks": len(self.tasks),
            "total_tasks": self.total_tasks,
            "n": self.n,
            "seed": self.seed,
            "scorer": self.scorer,
            "tools": list(self.tools),
            "agent": dict(self.agent),
        }


def _read_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SuiteError(f"{path} is not valid JSON ({exc}). Fix the syntax and run again.") from exc
    else:
        import yaml

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SuiteError(f"{path} is not valid YAML ({exc}). Fix the syntax and run again.") from exc
    if not isinstance(data, dict):
        raise SuiteError(f"{path} holds a {type(data).__name__}, not a suite. Use a mapping of keys, as in `effgen bench init`.")
    return data


def _read_task_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SuiteError(f"tasks_file {path} was not found. Check the path; it is relative to the suite file.")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SuiteError(f"{path} is not valid JSON ({exc}). Fix the syntax and run again.") from exc
        records = data if isinstance(data, list) else [data]
    else:
        records = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SuiteError(f"Line {number} of {path} is not valid JSON ({exc}). Fix that line and run again.") from exc
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise SuiteError(f"Record {number} of {path} is not an object. Make every record a JSON object.")
    return records


def _as_int(name: str, value: Any, *, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SuiteError(f"{name} is {value!r}, not an integer. Set it to a whole number.")
    if value < minimum:
        raise SuiteError(f"{name} is {value}, below {minimum}. Set it to at least {minimum}.")
    return value


def _build_tasks(records: list[dict[str, Any]], fields: dict[str, str]) -> list[BenchTask]:
    tasks: list[BenchTask] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        mapped = {key: record.get(fields.get(key, key)) for key in _TASK_KEYS}
        task_id = mapped["id"]
        task_id = f"task-{index + 1}" if task_id is None else str(task_id)
        where = f"task {task_id!r}"
        text = mapped["input"]
        if not isinstance(text, str) or not text.strip():
            raise SuiteError(f"{where} has no input text. Give it a non-empty '{fields.get('input', 'input')}'.")
        if task_id in seen:
            raise SuiteError(f"{where} is a duplicate id. Give every task a unique id.")
        seen.add(task_id)
        context = mapped["context"]
        tasks.append(BenchTask(
            id=task_id, input=text, expected=mapped["expected"],
            context="" if context is None else str(context),
        ))
    return tasks


def _select(tasks: list[BenchTask], n: int | None, seed: int | None) -> list[BenchTask]:
    if n is None or n == len(tasks):
        return list(tasks)
    if n > len(tasks):
        raise SuiteError(f"n={n} but the suite holds only {len(tasks)} tasks. Lower n to {len(tasks)} or add tasks.")
    if seed is None:
        return list(tasks[:n])
    keep = sorted(random.Random(seed).sample(range(len(tasks)), n))
    return [tasks[i] for i in keep]


def load_suite(path: str | Path, *, n: int | None = None, seed: int | None = None) -> BenchSuite:
    """Read a suite file and select its tasks.

    Args:
        path: A ``.yaml``, ``.yml`` or ``.json`` suite file.
        n: Overrides the file's ``n``.
        seed: Overrides the file's ``seed``.

    Returns:
        The suite, with ``tasks`` narrowed to the ``n`` tasks that will run.

    Raises:
        SuiteError: The file is missing, malformed, or asks for something the
            loader does not know.
    """
    from .scoring import validate_scorer

    source = Path(path)
    if not source.is_file():
        raise SuiteError(f"The suite file {source} was not found. Check the path, or create one with `effgen bench init`.")
    data = _read_mapping(source)
    unknown = sorted(set(data) - set(_TOP_LEVEL))
    if unknown:
        raise SuiteError(
            f"{source} has unknown key(s) {', '.join(unknown)}. Use only the supported keys: {', '.join(_TOP_LEVEL)}."
        )

    fields = data.get("fields") or {}
    if not isinstance(fields, dict) or any(k not in _TASK_KEYS for k in fields):
        raise SuiteError(f"'fields' is not a mapping of task keys. Map only the supported keys ({', '.join(_TASK_KEYS)}) to record keys.")
    if "tasks" in data and "tasks_file" in data:
        raise SuiteError("The suite gives both 'tasks' and 'tasks_file'. Remove one of them.")
    if "tasks_file" in data:
        records = _read_task_records((source.parent / str(data["tasks_file"])).resolve())
    else:
        listed = data.get("tasks")
        if not isinstance(listed, list):
            raise SuiteError(f"{source} has no task list. Give 'tasks' as a list, or use 'tasks_file'.")
        records = listed
        for number, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise SuiteError(f"Task {number} is not a mapping. Give it 'input' and 'expected' keys.")
            extra = sorted(set(record) - set(_TASK_KEYS))
            if extra:
                raise SuiteError(
                    f"Task {number} has unknown key(s) {', '.join(extra)}. Use only the supported keys: "
                    f"{', '.join(_TASK_KEYS)}."
                )
    tasks = _build_tasks(records, {str(k): str(v) for k, v in fields.items()})
    if not tasks:
        raise SuiteError(f"{source}: the suite has no tasks. Add at least one task.")

    tools = data.get("tools") or []
    if isinstance(tools, str):
        tools = [tools]
    if not isinstance(tools, list) or not all(isinstance(t, str) and t for t in tools):
        raise SuiteError("'tools' is not a list of tool names. Use a list, as `effgen tools list` names them.")

    agent = data.get("agent") or {}
    if not isinstance(agent, dict):
        raise SuiteError("'agent' is not a mapping. Use a mapping of agent settings.")
    bad = sorted(set(agent) - set(AGENT_SETTINGS))
    if bad:
        raise SuiteError(
            f"'agent' has settings it does not know (unknown: {', '.join(bad)}). Use only the supported settings: {', '.join(AGENT_SETTINGS)}."
        )

    scorer = data.get("scorer", "exact")
    validate_scorer(scorer, base_dir=source.parent)
    if scorer_needs_expected(scorer):
        missing = [t.id for t in tasks if t.expected is None]
        if missing:
            raise SuiteError(
                f"The scorer compares with 'expected', which is missing on: {', '.join(missing[:5])}. Add 'expected' to those tasks."
            )

    wanted_n = _as_int("n", n if n is not None else data.get("n"), minimum=1)
    wanted_seed = _as_int("seed", seed if seed is not None else data.get("seed"), minimum=0)
    selected = _select(tasks, wanted_n, wanted_seed)

    return BenchSuite(
        name=str(data.get("name") or source.stem),
        description=str(data.get("description") or ""),
        tasks=selected,
        total_tasks=len(tasks),
        scorer=scorer,
        tools=list(tools),
        n=wanted_n,
        seed=wanted_seed,
        agent=dict(agent),
        model=data.get("model"),
        base_url=data.get("base_url"),
        path=source.resolve(),
    )


def scorer_needs_expected(scorer: Any) -> bool:
    """Whether *scorer* compares an answer with the task's ``expected`` value."""
    return not (isinstance(scorer, dict) and "callable" in scorer)
