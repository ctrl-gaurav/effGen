"""What a streamed run needs beyond the loop itself.

The streamed tool loop used to live here, as a second copy of the loop in
:mod:`effgen.core.agent_react`. Both are now :mod:`effgen.core.agent_loop`,
which decides the prompt frame, the sampling settings, the guards and the
terminal contract once for every path, and hands a turn to an emitter that
either collects it or streams it.

What is left here is the pair of things that are genuinely about *rendering* a
streamed run: whether this agent's next turn can have its tool calls dispatched
while it streams, and the record the finished stream leaves behind.

Two rules still decide what may be shown, and when. They are enforced by
:class:`~effgen.core.agent_loop._AnswerStream`, re-exported below because it is
where a reader of this module expects to find it:

* **The answer-commit rule.** A turn's text is held back until the turn can no
  longer become a tool call. Once the adapter has recorded a tool call the text
  is delivered as a ``thought`` and never enters the answer; once a text delta
  has arrived with no call declared the turn is committed to answering and every
  later delta is passed through as it arrives.
* **Sanitize before emitting.** Only the settled prefix of
  :func:`~effgen.core.agent_runtime.sanitize_final_answer` applied to the text so
  far is emitted, so what reaches the screen is what the answer ends up being.

Together they give the invariant a consumer can rely on: for a stream that
answered, joining the ``answer`` events reproduces
``last_stream_response.output`` exactly. A stream that stopped at its iteration
cap, or one whose model wrote its call out as text, reports the typed outcome in
``output`` the same way ``run()`` does — that text is a ``status`` event, not an
answer delta.

This module imports nothing from ``agent.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .agent_loop import _AnswerStream  # noqa: F401 - kept where readers find it
from .agent_response import AgentResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .agent_config import AgentConfig

logger = logging.getLogger(__name__)


class AgentNativeStreamMixin:
    """Whether a streamed turn can dispatch calls, and the record it leaves."""

    if TYPE_CHECKING:
        # Contributed by the sibling mixins this one is combined with on
        # :class:`~effgen.core.agent.Agent`. Declared for the type checker only
        # — at run time they arrive through the MRO, and these statements do
        # not execute.
        model: Any
        tools: dict[str, Any]
        config: AgentConfig
        _tool_calling_strategy: Any

        def _has_native_tools(self) -> bool: ...

        def _has_gemini_native_tools(self) -> bool: ...

        def _model_tool_call_support(self) -> str: ...

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------
    def _can_stream_native_tools(self) -> bool:
        """True when this agent's next streamed turn can dispatch native calls.

        This asks one question — *can a turn's tool calls be dispatched while it
        streams?* — and it is no longer how a loop is chosen: every run takes
        the same loop and is prompted the same way. It decides only whether the
        turn's text may reach the consumer as it is written, and a presentation
        layer reads it to know whether to render tool events.

        Every condition has to hold: the agent carries ordinary tools, no
        provider-side native tool is attached (those have their own run paths),
        the strategy asks for native calling, the definitions travel through the
        provider's tool-calling API rather than a chat template, the adapter
        records the calls it streams, and no custom system-prompt template owns
        the prompt. A turn that fails any of them is accumulated and delivered
        whole, which is what a text-only model's turn has always been.
        """
        if not self.tools or self.model is None:
            return False
        if self.config.system_prompt_template:
            return False
        try:
            if self._has_native_tools() or self._has_gemini_native_tools():
                return False
        except Exception:  # noqa: BLE001 - an unreadable tool set is not eligible
            logger.debug("Native-tool probe failed", exc_info=True)
            return False
        if self._tool_calling_strategy.name not in ("native", "hybrid"):
            return False
        if self._model_tool_call_support() != "api":
            return False
        probe = getattr(self.model, "streams_tool_calls", None)
        try:
            return bool(probe and probe())
        except Exception:  # noqa: BLE001 - a capability probe never breaks a run
            logger.debug("streams_tool_calls probe failed", exc_info=True)
            return False

    @property
    def last_stream_response(self) -> AgentResponse | None:
        """The record of the most recent streamed run, or ``None``.

        Set once the iterator is exhausted, and shaped exactly like the
        :class:`~effgen.core.agent_response.AgentResponse` the same task would
        have produced through :meth:`run` — ``output``, ``success``,
        ``stop_reason``, ``iterations``, ``tool_calls``, ``tokens_used``,
        ``execution_time`` and the ``reason`` / ``error`` / ``partial`` /
        ``thread`` / ``prompt_protocol`` metadata — so a caller that renders a
        turn can report it without running the task again. It is ``None`` after
        a stream that never entered the loop, which is a tool-free stream.
        """
        return getattr(self, "_last_stream_response", None)
