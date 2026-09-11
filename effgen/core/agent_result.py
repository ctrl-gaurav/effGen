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

if TYPE_CHECKING:
    from .agent_response import AgentResponse

# Result assembly logs under the agent module's logger name, so one filter
# covers a run and the record written for it.
logger = logging.getLogger("effgen.core.agent")


class AgentResultMixin:
    """Run identity, session-turn metadata, and the run-history record."""

    def _stamp_run_identity(
        self,
        response: AgentResponse,
        *,
        task: Any,
        started_at: str,
    ) -> AgentResponse:
        """Record what the run was, on the response, and return it.

        A result document is read long after the run, often by someone who did
        not start it, so it carries the task, the model and provider that
        answered it, and when it started.
        """
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

    def _record_dashboard_run(
        self,
        response: AgentResponse,
        *,
        error: str | None = None,
        task: str | None = None,
    ) -> None:
        """Record the run in the history store read by `effgen runs` and the dashboard."""
        try:
            from effgen.observability.run_log import record_run

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
            )
        except Exception:  # noqa: BLE001 - run history must not break runs
            logger.debug("Run history logging failed", exc_info=True)
