"""
Top-level error types for the effGen framework.

These are raised at runtime when a required system dependency or backend
is unavailable, so callers can catch them specifically and display helpful
messages rather than cryptic import errors.

Also holds the two helpers every error message in the framework composes
itself with: :func:`quote_for_message`, applied to text quoted from somewhere
else so it is redacted and cut to a bounded length, and
:func:`with_next_step`, which joins what happened to what to do about it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from effgen.core.agent_response import AgentResponse, PartialResult

#: How much of a value produced elsewhere an error message quotes. A provider
#: echoes the rejected request back in its error body, a parser reports the
#: input it choked on, and an SDK wraps both — so an unbounded quote buries the
#: reason for the failure under kilobytes of payload in a terminal panel, a log
#: line, or an API envelope. Long enough to carry a provider's own sentence,
#: short enough to stay readable.
MESSAGE_ECHO_LIMIT = 240


def abbreviate_for_message(value: object, limit: int = MESSAGE_ECHO_LIMIT) -> str:
    """Return *value* as text, cut to *limit* characters and marked if cut.

    Every error message that quotes text it did not produce — a provider error
    body, a parser complaint, a rejected input — passes it through here first,
    so no single message can grow without bound.

    Args:
        value: The value to quote. Anything not already a string is rendered
            with ``str()``.
        limit: How many characters to keep before cutting.

    Returns:
        The text unchanged when it fits, otherwise the first *limit*
        characters followed by an ellipsis and the full length.
    """
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… ({len(text)} characters)"


def redact_for_message(value: object) -> str:
    """Return *value* as text with any credential in it replaced by a marker.

    Upstream error bodies echo parts of the request back, and some include the
    submitted credential. Text that reaches an error message goes through here
    so a key cannot travel into a log, a terminal panel or an API envelope.

    Redaction never replaces the failure: if the redactor cannot be loaded, the
    original text is returned rather than an error about redacting it.

    Args:
        value: The value to redact. Anything not already a string is rendered
            with ``str()``.

    Returns:
        The text with every recognized credential replaced by a marker.
    """
    text = str(value)
    if not text:
        return text
    try:
        # Imported here so the error types stay importable before the
        # observability stack is built.
        from effgen.observability.redact import get_redactor

        return get_redactor().scrub(text)
    except Exception:  # noqa: BLE001 - redaction must not replace the error
        return text


def quote_for_message(value: object, limit: int = MESSAGE_ECHO_LIMIT) -> str:
    """Return *value* ready to quote in an error message: redacted, then bound.

    The single call every error message makes on text it did not produce. The
    order matters — redacting first means a credential near the cut point is
    replaced rather than half-printed.

    Args:
        value: The value to quote.
        limit: How many characters to keep before cutting.

    Returns:
        The redacted text, cut to *limit* characters and marked if cut.
    """
    return abbreviate_for_message(redact_for_message(value), limit)


#: Terminators that already end a sentence, so no period is added after them.
_SENTENCE_END = ".!?…"


def with_next_step(statement: object, next_step: str) -> str:
    """Join what happened to what to do about it, as two sentences.

    The statement half is usually text from somewhere else — a provider error
    body, a parser complaint — which may or may not punctuate itself. Joining
    by hand produces either ``"...no.. Check"`` or ``"...no Check"`` depending
    on the upstream, so every message that carries guidance joins here instead.

    Args:
        statement: What happened. Rendered with ``str()`` and stripped of
            trailing whitespace and any separator left dangling at the end.
        next_step: What the reader can do about it, already a sentence.

    Returns:
        The two joined by a single space, with exactly one terminator between
        them. When *statement* is empty, *next_step* alone.
    """
    head = str(statement).rstrip()
    while head and head[-1] in ",;:":
        head = head[:-1].rstrip()
    if not head:
        return next_step
    if head[-1] not in _SENTENCE_END:
        head += "."
    return f"{head} {next_step}"


class EffGenError(Exception):
    """Base class for all effGen-specific errors."""


class MissingSystemDependency(EffGenError):
    """Raised when a required system-level binary is not found on PATH.

    Attributes:
        dependency: Name of the missing binary (e.g. "tesseract").
        install_instructions: Per-OS install guidance.
    """

    _DEFAULT_HINT = (
        "Install it with the system package manager (apt, brew, choco or "
        "conda), then re-run."
    )

    def __init__(self, dependency: str, install_instructions: str = "") -> None:
        self.dependency = dependency
        self.install_instructions = install_instructions
        msg = f"System dependency '{dependency}' is not installed or not on PATH."
        if install_instructions:
            msg += f"\n\n{quote_for_message(install_instructions)}"
        msg += f"\n\n{self._DEFAULT_HINT}"
        super().__init__(msg)


class OCRBackendUnavailable(EffGenError):
    """Raised when no OCR backend (Tesseract or OCR.space) is available.

    Attributes:
        tried_backends: List of backend names that were checked.
    """

    _DEFAULT_INSTALL = (
        "Install Tesseract:\n"
        "  Ubuntu/Debian : sudo apt install tesseract-ocr\n"
        "  macOS         : brew install tesseract\n"
        "  Windows       : choco install tesseract\n"
        "  conda         : conda install -c conda-forge tesseract\n\n"
        "Or set OCR_SPACE_API_KEY to use the OCR.space cloud fallback."
    )

    def __init__(
        self,
        tried_backends: list[str] | None = None,
        extra: str = "",
    ) -> None:
        self.tried_backends = tried_backends or []
        msg = "No OCR backend is available."
        if self.tried_backends:
            msg += f" Tried: {', '.join(self.tried_backends)}."
        if extra:
            msg += f"\n\n{quote_for_message(extra)}"
        msg += f"\n\n{self._DEFAULT_INSTALL}"
        super().__init__(msg)


class NoVisionProviderAvailable(EffGenError):
    """Raised when ImageCaptionTool cannot find a vision-capable provider.

    Attributes:
        tried_providers: List of provider names that were checked.
    """

    _DEFAULT_HINT = (
        "Configure a vision-capable provider:\n"
        "  OpenAI  : set OPENAI_API_KEY  (gpt-4o supports vision)\n"
        "  Google  : set GOOGLE_API_KEY  (gemini-2.5-flash-lite supports vision)\n"
        "  Replicate: set REPLICATE_API_TOKEN\n\n"
        "Then pass the provider when creating ImageCaptionTool or use "
        "the model router with Capability.vision."
    )

    def __init__(
        self,
        tried_providers: list[str] | None = None,
        extra: str = "",
    ) -> None:
        self.tried_providers = tried_providers or []
        msg = "No vision-capable provider is configured."
        if self.tried_providers:
            msg += f" Tried: {', '.join(self.tried_providers)}."
        if extra:
            msg += f"\n\n{quote_for_message(extra)}"
        msg += f"\n\n{self._DEFAULT_HINT}"
        super().__init__(msg)


class CorruptDocumentError(EffGenError):
    """Raised when a document file is malformed or cannot be parsed.

    Attributes:
        doc_type: The document type that failed (e.g. 'pdf', 'docx', 'xlsx').
        detail: Additional detail from the underlying parser.
    """

    def __init__(self, doc_type: str, detail: str = "") -> None:
        self.doc_type = doc_type
        self.detail = detail
        msg = f"Cannot parse {doc_type.upper()} document - file may be corrupt or invalid."
        if detail:
            msg += f"\nDetail: {quote_for_message(detail)}"
        msg += (
            "\n\nOpen the file in a reader to confirm it is intact, re-export "
            "it, or pass a different document."
        )
        super().__init__(msg)


class CorruptStateError(EffGenError):
    """Raised when a persisted session or checkpoint file cannot be read.

    The file is named explicitly so a user can find and fix (or delete) it
    instead of seeing a raw ``JSONDecodeError`` stack trace.

    Attributes:
        kind: What was being loaded (e.g. 'session', 'checkpoint').
        path: The file path that failed to parse.
        detail: Underlying parser detail.
    """

    def __init__(self, kind: str, path: str, detail: str = "") -> None:
        self.kind = kind
        self.path = path
        self.detail = detail
        msg = f"Cannot read {kind} file '{path}' - it is corrupt, truncated, or not valid JSON."
        if detail:
            msg += f"\nDetail: {quote_for_message(detail)}"
        msg += "\n\nFix: inspect the file, restore a backup, or delete it to start fresh."
        super().__init__(msg)


class MissingCredentialsError(EffGenError):
    """Raised when required credentials (env vars) are absent for a tool.

    Attributes:
        tool_name: The tool that needs credentials.
        missing_vars: List of env var names that are missing.
    """

    _DEFAULT_HINT = (
        "Set {names} in the environment or in the project .env file, then "
        "re-run."
    )

    def __init__(self, tool_name: str, missing_vars: list[str], hint: str = "") -> None:
        self.tool_name = tool_name
        self.missing_vars = missing_vars
        names = ", ".join(missing_vars)
        msg = (
            f"{tool_name} requires credentials that are not configured: "
            f"{names}."
        )
        if hint:
            msg += f"\n\n{quote_for_message(hint)}"
        msg += f"\n\n{self._DEFAULT_HINT.format(names=names)}"
        super().__init__(msg)


class CapabilityNotSupportedError(EffGenError):
    """Raised when an adapter cannot handle a requested capability (e.g. vision, audio).

    Attributes:
        capability: The Capability enum value that is not supported.
        provider: Name of the provider/adapter that raised this.
        hint: Optional guidance on which provider to use instead.
    """

    _DEFAULT_HINT = (
        "Choose a model that declares this capability — run "
        "`effgen models list --capability <name>` to see which do."
    )

    def __init__(self, capability: object, provider: str = "", hint: str = "") -> None:
        capability_name = getattr(capability, "value", str(capability))
        self.capability = capability_name
        self.provider = provider
        msg = f"Capability '{capability_name}' is not supported"
        if provider:
            msg += f" by provider '{provider}'"
        msg += "."
        if hint:
            msg += f"\n\n{quote_for_message(hint)}"
        msg += f"\n\n{self._DEFAULT_HINT}"
        super().__init__(msg)


class InvalidMultimodalContent(EffGenError):
    """Raised when multimodal content (image, audio, video) fails validation.

    Attributes:
        part_type: The part type that failed (e.g. 'image', 'audio', 'video_frames').
        reason: Description of the validation failure.
    """

    def __init__(self, part_type: str, reason: str = "") -> None:
        self.part_type = part_type
        self.reason = reason
        msg = f"Invalid {part_type} content"
        if reason:
            msg += f": {quote_for_message(reason)}"
        super().__init__(
            with_next_step(
                msg,
                f"Check that the {part_type} source is reachable and in a "
                "supported format.",
            )
        )


class AudioBackendUnavailable(EffGenError):
    """Raised when no audio transcription backend is available.

    Attributes:
        tried_backends: List of backend names that were checked.
    """

    _DEFAULT_INSTALL = (
        "Install local transcription support:\n"
        "  pip install 'effgen[audio]'\n"
        "  Ubuntu/Debian : sudo apt install ffmpeg\n"
        "  macOS         : brew install ffmpeg\n"
        "  Windows       : choco install ffmpeg\n"
        "  conda         : conda install -c conda-forge ffmpeg\n\n"
        "Or set HF_TOKEN to use the HuggingFace Inference fallback."
    )

    def __init__(
        self,
        tried_backends: list[str] | None = None,
        extra: str = "",
    ) -> None:
        self.tried_backends = tried_backends or []
        msg = "No audio transcription backend is available."
        if self.tried_backends:
            msg += f" Tried: {', '.join(self.tried_backends)}."
        if extra:
            msg += f"\n\n{quote_for_message(extra)}"
        msg += f"\n\n{self._DEFAULT_INSTALL}"
        super().__init__(msg)


class RunStoppedError(RuntimeError):
    """The loop stopped a run before the model wrote an answer.

    Raised by ``Agent.run()`` under the default ``raise_on_error=True`` when a
    run ends with ``outcome == "stopped"`` — the iteration cap, a repeated tool
    call, a tool that reproduced its own result, an empty final answer after
    tools ran, or a tool that failed on its own side (``tool_failed``: a
    connection or timeout error, a service error, missing credentials). The run
    never produced an answer, so there is nothing to return in its place.

    It subclasses :class:`RuntimeError`, which is what the iteration cap has
    always raised, so ``except RuntimeError`` keeps catching it. What is new is
    that the run comes with it: :attr:`response` is the full response and
    :attr:`partial` is what the run had reached, so reading the progress is one
    ``except`` clause rather than a configuration change.

    ::

        try:
            print(agent.run(task).output)
        except RunStoppedError as exc:
            print(exc.stop_reason, exc.partial.text if exc.partial else "")

    Attributes:
        response: The :class:`~effgen.core.agent_response.AgentResponse` the run
            produced, with ``success=False``.
        stop_reason: What stopped the run, copied from the response.
        partial: What the run had reached, or ``None`` when it had reached
            nothing.
    """

    def __init__(self, response: "AgentResponse") -> None:
        super().__init__(response.output)
        self.response = response
        self.stop_reason: str | None = response.stop_reason
        self.partial: "PartialResult | None" = response.partial
