"""Result assembly for :class:`effgen.core.agent.Agent`.

Stamps a finished run onto its :class:`~effgen.core.agent_response.AgentResponse`
— the task, model, provider and start time a result document is read by — builds
the metadata a stored session turn carries, and records the run in the history
store behind ``effgen runs`` and the dashboard. Mixed into :class:`Agent`; this
module imports nothing from ``agent.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .agent_runtime import _safe_float_or_none, _safe_int_or_none
from .ledger import attach as _attach_ledger

if TYPE_CHECKING:
    from .agent_response import AgentResponse

# Result assembly logs under the agent module's logger name, so one filter
# covers a run and the record written for it.
logger = logging.getLogger("effgen.core.agent")


def _turn_thread(response: Any) -> dict[str, Any]:
    """The run's steps as session-turn metadata, or nothing when it kept none."""
    thread = (getattr(response, "metadata", None) or {}).get("thread")
    to_dict = getattr(thread, "to_dict", None)
    if callable(to_dict):
        return {"thread": to_dict()}
    return {"thread": dict(thread)} if isinstance(thread, dict) else {}


class AgentResultMixin:
    """Run identity, session-turn metadata, and the run-history record."""

    def _stamp_run_identity(
        self,
        response: AgentResponse,
        *,
        task: Any,
        started_at: str,
        ledger: Any = None,
    ) -> AgentResponse:
        """Record what the run was, on the response, and return it.

        A result document is read long after the run, often by someone who did
        not start it, so it carries the task, the model and provider that
        answered it, and when it started.

        *ledger* is the run's ledger recorder. Every way out of a run passes
        through here, so this is where the ledger is closed and its final form
        written to ``metadata["ledger"]``.

        The run's latency is set here from the closed ledger, so
        ``response.execution_time`` is the wall time the caller waited — the
        session save, the final checkpoint and the telemetry written after the
        answer included — and equals ``metadata["ledger"]["wall_s"]``, which
        the ledger splits into model, tool, caller, child and framework time.
        """
        closed = _attach_ledger(response, ledger, close=True)
        if closed is not None:
            _report_run_wall(response, closed.wall_s)
        if response.task is None and isinstance(task, str):
            response.task = task
        if response.model is None:
            response.model = getattr(self, "model_name", None)
        if response.provider is None:
            response.provider = self._resolve_provider(response)
        if response.started_at is None:
            response.started_at = started_at
        return response

    def _resolve_provider(self, response: AgentResponse) -> str | None:
        """Name the provider that served *response*, or ``None`` if none did.

        The caller may name a provider on the config and an adapter may report
        one in the run metadata, but a ``provider:model`` id carries it without
        either — so fall back to the adapter that answered the run. Local
        engines have no provider and stay unset.
        """
        metadata = response.metadata or {}
        provider = metadata.get("provider") or getattr(self.config, "provider", None)
        if not provider and getattr(self, "model", None) is not None:
            from effgen.models.base import _provider_of

            provider = _provider_of(self.model)
        return str(provider) if provider else None

    def _session_turn_metadata(
        self,
        response: AgentResponse,
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Model, tokens, cost and latency to stamp on a stored session turn."""
        metadata = response.metadata or {}
        meta: dict[str, Any] = {
            "model": str(getattr(self, "model_name", None) or "unknown"),
            "run_id": run_id,
            "stop_reason": getattr(response, "stop_reason", None),
            "latency_ms": round(response.execution_time * 1000, 1) if response.execution_time else None,
        }
        provider = metadata.get("provider") or getattr(self.config, "provider", None)
        if provider:
            meta["provider"] = str(provider)
        prompt_tokens = _safe_int_or_none(
            metadata.get("prompt_tokens", metadata.get("input_tokens"))
        )
        completion_tokens = _safe_int_or_none(
            metadata.get("completion_tokens", metadata.get("output_tokens"))
        )
        if prompt_tokens is not None:
            meta["prompt_tokens"] = prompt_tokens
        if completion_tokens is not None:
            meta["completion_tokens"] = completion_tokens
        if response.tokens_used:
            meta["tokens_used"] = response.tokens_used
        cost = _safe_float_or_none(metadata.get("cost_usd", metadata.get("cost")))
        if cost is not None:
            meta["cost_usd"] = cost
        return {k: v for k, v in meta.items() if v is not None}

    def _save_session_turn(
        self,
        session: Any,
        task: Any,
        output: Any,
        response: AgentResponse | None,
        *,
        run_id: str | None = None,
    ) -> None:
        """Append one answered turn to *session* and save it.

        Each message is stamped with the model, token counts, cost and latency
        the turn was answered with (:meth:`_session_turn_metadata`), so a stored
        conversation can be reviewed turn by turn. *response* is ``None`` for a
        turn that produced no response object (a tool-free stream), which is
        stamped with the model alone.
        """
        if response is not None:
            turn_meta = self._session_turn_metadata(response, run_id=run_id)
        else:
            turn_meta = {"model": str(getattr(self, "model_name", None) or "unknown")}
            if run_id:
                turn_meta["run_id"] = run_id
        session.add_message("user", task, **turn_meta)
        # The reply carries the run's own steps as well, so a later turn can
        # continue from the conversation the run had rather than from a reading
        # of its text. The question does not: one copy per turn is the record.
        session.add_message(
            "assistant",
            output,
            **turn_meta,
            **_turn_thread(response),
        )
        if turn_meta.get("model"):
            session.metadata["model"] = turn_meta["model"]
        session.metadata.setdefault("agent_name", getattr(self, "name", None))
        try:
            session.save()
        except Exception as _e:
            logger.warning("Failed to save session: %s", _e)

    def _record_dashboard_run(
        self,
        response: AgentResponse,
        *,
        error: str | None = None,
        task: str | None = None,
        ledger: Any = None,
    ) -> None:
        """Record the run in the history store read by `effgen runs` and the dashboard.

        With the run's *ledger* the record also carries its model and tool call
        counts, cached prompt tokens and the split of its time.
        """
        try:
            from effgen.observability.run_log import record_run

            spent: dict[str, Any] = {}
            if ledger is not None:
                spent = {
                    "llm_calls": ledger.llm_calls,
                    "tool_calls": ledger.tool_calls,
                    "cached_input_tokens": ledger.cached_input_tokens,
                    "cache_write_tokens": ledger.cache_write_tokens,
                    "model_wait_s": round(ledger.model_wait_s, 6),
                    "tool_wait_s": round(ledger.tool_wait_s, 6),
                    "framework_s": round(ledger.framework_s, 6),
                }
            metadata = response.metadata or {}
            cost = metadata.get("cost_usd", metadata.get("cost"))
            output_tokens = metadata.get("output_tokens", metadata.get("completion_tokens"))
            if output_tokens is None and response.tokens_used:
                output_tokens = response.tokens_used
            input_tokens = metadata.get("input_tokens", metadata.get("prompt_tokens"))
            provider = self._resolve_provider(response)
            record_run(
                model=str(getattr(self, "model_name", None) or "unknown"),
                input_tokens=_safe_int_or_none(input_tokens),
                output_tokens=_safe_int_or_none(output_tokens),
                duration_s=response.execution_time,
                cost_usd=_safe_float_or_none(cost),
                # The store bounds the message itself, marking a cut with an
                # ellipsis — a stop or classification message is a sentence or
                # two and reaches the history file intact.
                error=error if error is not None else (None if response.success else response.output),
                stop_reason=getattr(response, "stop_reason", None),
                outcome=getattr(response, "outcome", None),
                run_id=metadata.get("run_id"),
                task=task,
                output=response.output if response.success else None,
                provider=provider,
                session_id=self._session_id,
                agent=self.name,
                thread=metadata.get("thread"),
                **spent,
            )
        except Exception:  # noqa: BLE001 - run history must not break runs
            logger.debug("Run history logging failed", exc_info=True)


def _report_run_wall(response: AgentResponse, wall_s: float) -> None:
    """Report *wall_s* as the run's latency on *response*.

    ``latency_ms`` / ``duration_s`` in the metadata mirror ``execution_time``
    when they were derived from it; a value a path set for itself is kept.
    """
    earlier = float(response.execution_time or 0.0)
    metadata = response.metadata if isinstance(response.metadata, dict) else None
    mirrored = metadata is not None and (
        "duration_s" not in metadata or metadata.get("duration_s") == round(earlier, 4)
    )
    response.execution_time = wall_s
    if mirrored and metadata is not None and "duration_s" in metadata:
        metadata["latency_ms"] = round(wall_s * 1000.0, 1)
        metadata["duration_s"] = round(wall_s, 4)
    logger.debug(
        "run latency: reported the run's wall %.4fs (%.4fs before post-run work)",
        wall_s, earlier,
    )
