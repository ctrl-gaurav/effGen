"""What the framework tells a model about the tools it just attached.

Attaching a tool is something the framework does to a prompt; the model only
learns about it from whatever the prompt says. Saying nothing leaves a small
model to fold the whole problem into one call, or to skip the tool and answer
from memory. Saying the *wrong* thing is worse than silence: telling a model
that holds a code executor to work the task out first and use the tool to check
its own reasoning is an instruction to be the interpreter, and it stops calling
the executor at all.

So the text is chosen from what the tools *are* — their declared
:class:`~effgen.tools.base_tool.ToolCategory` — and from nothing else. Four
texts cover the eight categories:

``TOOL_CONTRACT_VERIFY``
    The tool checks work the model can do itself, so reasoning comes first and
    the tool confirms it.
``TOOL_CONTRACT_EXECUTE``
    The tool does work the model cannot do in its head, so the tool does the
    task and the model reads the result back.
``TOOL_CONTRACT_LOOKUP``
    The tool returns material an answer is derived from, so the passages are
    source material rather than the answer.
``TOOL_CONTRACT_GENERAL``
    The default. It is what an unmapped category, a tool with no category and a
    mixed tool set all receive: a specific contract is only true because it is
    true of the tool, and on a mixed set none of the three is true of every tool
    held.

Every member of ``ToolCategory`` has an explicit entry in :data:`TOOL_CONTRACTS`
so a category added later fails the test that enumerates them rather than
silently inheriting a text written for something else. The four categories
mapped to the default say that no sharper wording has been measured for them.

Separately from *what* is said about a tool, there is *whether the tool has to
be used at all*, and that is :class:`ToolUsePolicy`. It has three values —
a call is ``REQUIRED``, the model decides (``AUTO``), or a run that already has
the answer should give it without calling (``SPARING``) — and, like the
contract, the default comes from the tools' declared categories through
:data:`TOOL_USE_POLICIES`. That is the same distinction the categories already
carry: skipping a code executor means reporting a result nothing produced, while
skipping a search means answering from what the model already knows, which is
sometimes right. ``AgentConfig.tool_use`` overrides it for a whole run, so
"always use this tool" and "only when you need it" are both things a caller can
say instead of properties of the tools they happened to attach.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from enum import Enum
from typing import Any

from ..tools.base_tool import ToolCategory

logger = logging.getLogger(__name__)


#: The default. Stated for a tool whose category carries no sharper wording, for
#: a tool that declares no category at all, and for a set of tools that map to
#: more than one contract.
TOOL_CONTRACT_GENERAL = (
    "Work through this task one step at a time. Before each tool call, say in "
    "one line what that step works out. Give a tool a single step, not the "
    "whole task at once, and use the result it returns rather than working it "
    "out again yourself. When every step is done, state the final answer."
)

#: Stated for tools that check arithmetic or symbolic work the model can also do
#: itself. Reasoning first, tool second: a model that hands the whole task to a
#: calculator loses the steps it would otherwise have written down.
TOOL_CONTRACT_VERIFY = (
    "First work the task out yourself, in your own words, step by step, and say "
    "what each step gives. Then use the tools to check the steps you are least "
    "sure of, one step per call, and correct yourself if a tool disagrees with "
    "you. Do not hand a tool the whole task at once, and do not skip the "
    "reasoning and call a tool instead. Finish by stating the final answer."
)

#: Stated for tools that run code or system commands — work the model cannot
#: carry out in its head. Describing what the code would print is not the same
#: as running it, and a model that does the former reports a result nothing
#: produced.
TOOL_CONTRACT_EXECUTE = (
    "Use the tools to do this task rather than working it out in your head. Say "
    "in one line what you are about to compute, call the tool to compute it, "
    "and read the answer off what it returns. Do not simulate the tool yourself "
    "and do not answer from a result you did not get back from it. If the tool "
    "errors or returns something unexpected, fix the input and call it again. "
    "Finish by stating the final answer."
)

#: Stated for tools that search or call an external service. What comes back is
#: material to answer from, and a run that returns a retrieved passage verbatim
#: has answered a different question from the one asked. Says nothing about
#: whether or how often to search — that is the caller's and the loop's.
TOOL_CONTRACT_LOOKUP = (
    "The tools bring back source material, not the answer. Answer the question "
    "yourself, in the form it asks for, from what they return, and do not "
    "return a passage as the answer. If what comes back does not answer the "
    "question, say so and name what is missing."
)


#: Which contract each declared tool category receives. Every member of
#: ``ToolCategory`` appears exactly once.
TOOL_CONTRACTS: dict[ToolCategory, str] = {
    ToolCategory.COMPUTATION: TOOL_CONTRACT_VERIFY,
    ToolCategory.CODE_EXECUTION: TOOL_CONTRACT_EXECUTE,
    ToolCategory.SYSTEM: TOOL_CONTRACT_EXECUTE,
    ToolCategory.INFORMATION_RETRIEVAL: TOOL_CONTRACT_LOOKUP,
    ToolCategory.EXTERNAL_API: TOOL_CONTRACT_LOOKUP,
    ToolCategory.FILE_OPERATIONS: TOOL_CONTRACT_GENERAL,
    ToolCategory.DATA_PROCESSING: TOOL_CONTRACT_GENERAL,
    ToolCategory.COMMUNICATION: TOOL_CONTRACT_GENERAL,
}

#: The short name each contract is logged under, so a run's log says which text
#: was selected without reprinting it.
CONTRACT_NAMES: dict[str, str] = {
    TOOL_CONTRACT_GENERAL: "general",
    TOOL_CONTRACT_VERIFY: "verify",
    TOOL_CONTRACT_EXECUTE: "execute",
    TOOL_CONTRACT_LOOKUP: "lookup",
}


class ToolUsePolicy(Enum):
    """Whether a run holding these tools has to call one.

    The contract above says what a tool is for; this says whether reaching for
    it is compulsory, free, or something to do only when the answer is not
    already in hand. The three are genuinely different asks and the difference
    is a property of the tool rather than of the task: answering without running
    a code executor reports a result nothing produced, while answering without
    searching is sometimes simply the right answer given faster.
    """

    #: A run holding these tools must actually call one. An answer written with
    #: no call is sent back once naming the tool, and the turn after it is
    #: required to call where the provider can enforce that.
    REQUIRED = "required"
    #: The tools are offered and the model decides. The framework asserts
    #: nothing either way, which is what every category but the executing ones
    #: gets.
    AUTO = "auto"
    #: The tools are offered and the framework adds that a run which already has
    #: the answer should give it without calling.
    SPARING = "sparing"


#: How compulsory each policy is, so a set of tools that declare different
#: policies can be resolved to one. A mixed set takes the strictest present, for
#: the same reason a mixed set takes the general contract: the weaker answer
#: would be false of a tool the model is holding, and telling a model it may
#: skip the executor it also holds is the expensive half of that mistake.
_POLICY_STRICTNESS: dict[ToolUsePolicy, int] = {
    ToolUsePolicy.SPARING: 0,
    ToolUsePolicy.AUTO: 1,
    ToolUsePolicy.REQUIRED: 2,
}

#: Which policy each declared tool category carries by default. Every member of
#: ``ToolCategory`` appears exactly once, so a category added later fails the
#: test that enumerates them rather than inheriting a default nobody chose.
#:
#: The executing categories are the only ones that require a call, which is what
#: the tree already did before the policy had a name. Everything else is
#: ``AUTO``: offering a calculator or a search tool and then insisting it be used
#: is a decision the framework has no basis for making on the caller's behalf.
TOOL_USE_POLICIES: dict[ToolCategory, ToolUsePolicy] = {
    ToolCategory.CODE_EXECUTION: ToolUsePolicy.REQUIRED,
    ToolCategory.SYSTEM: ToolUsePolicy.REQUIRED,
    ToolCategory.COMPUTATION: ToolUsePolicy.AUTO,
    ToolCategory.INFORMATION_RETRIEVAL: ToolUsePolicy.AUTO,
    ToolCategory.EXTERNAL_API: ToolUsePolicy.AUTO,
    ToolCategory.FILE_OPERATIONS: ToolUsePolicy.AUTO,
    ToolCategory.DATA_PROCESSING: ToolUsePolicy.AUTO,
    ToolCategory.COMMUNICATION: ToolUsePolicy.AUTO,
}

#: The one sentence :attr:`ToolUsePolicy.SPARING` adds, after whichever contract
#: is being stated. It is additive on purpose: the contract still says what the
#: tools are good for, and this says that having them does not oblige the model
#: to use them. ``REQUIRED`` and ``AUTO`` add nothing, so a run on either of them
#: sends the prompt it would have sent before the policy existed.
TOOL_USE_SPARING_NOTE = (
    "Calling a tool is optional. If you already know the answer, give it "
    "without calling anything, and call a tool only for the part you are not "
    "sure of."
)


def policy_for_category(category: Any) -> ToolUsePolicy:
    """Return the tool-use policy for one declared *category*.

    Anything that is not a known :class:`~effgen.tools.base_tool.ToolCategory`
    — ``None``, a plain string from a third-party tool, a member added after
    this mapping was written — answers :attr:`ToolUsePolicy.AUTO`. Failing to
    the loosest policy is the fail-closed direction here: an unrecognised tool
    is not assumed to be one the model may not skip.
    """
    try:
        policy = TOOL_USE_POLICIES.get(category)
    except TypeError:  # pragma: no cover - an unhashable category
        policy = None
    return policy or ToolUsePolicy.AUTO


def select_tool_use_policy(tools: Iterable[Any]) -> ToolUsePolicy:
    """Return the policy the tools an agent holds declare between them.

    One policy across the set is that policy; a set that declares more than one
    takes the strictest present (:data:`_POLICY_STRICTNESS`). An agent holding
    no tools answers :attr:`ToolUsePolicy.AUTO`, which adds nothing to a prompt
    that has no tools to talk about.
    """
    policies = [policy_for_category(_category_of(tool)) for tool in tools]
    if not policies:
        return ToolUsePolicy.AUTO
    return max(policies, key=_POLICY_STRICTNESS.__getitem__)


def coerce_tool_use_policy(value: Any) -> ToolUsePolicy | None:
    """Return *value* as a :class:`ToolUsePolicy`, or ``None`` for "unset".

    ``None`` means the caller stated nothing and the tools' declared categories
    decide. A member passes through; a string is matched against the member
    values (``"required"``, ``"auto"``, ``"sparing"``) case-insensitively, so a
    policy can be set wherever an agent is built from plain data -- a dict of
    overrides, a preset's keyword arguments -- without importing the enum.

    Raises:
        ValueError: If *value* is a string naming no policy, or a type that is
            neither. A silently ignored policy is worse than a refused one: the
            run would go out on a policy the caller did not choose.
    """
    if value is None or isinstance(value, ToolUsePolicy):
        return value
    if isinstance(value, str):
        try:
            return ToolUsePolicy(value.strip().lower())
        except ValueError:
            pass
    names = ", ".join(repr(p.value) for p in ToolUsePolicy)
    raise ValueError(
        f"unknown tool_use policy {value!r}. "
        f"Pass one of {names}, a ToolUsePolicy member, or None to take the "
        "policy from the tools' declared categories."
    )


def contract_for_category(category: Any) -> str:
    """Return the contract for one declared *category*.

    Anything that is not a known :class:`~effgen.tools.base_tool.ToolCategory`
    — ``None``, a plain string from a third-party tool, a member added after
    this mapping was written — gets :data:`TOOL_CONTRACT_GENERAL` rather than an
    exception or silence.
    """
    try:
        contract = TOOL_CONTRACTS.get(category)
    except TypeError:  # pragma: no cover - an unhashable category
        contract = None
    return contract or TOOL_CONTRACT_GENERAL


def select_tool_contract(tools: Iterable[Any]) -> str:
    """Return the contract for the tools an agent holds, or ``""`` for none.

    The tools' declared categories decide it. One distinct contract across the
    set is stated as it is; more than one is a mixed set and gets
    :data:`TOOL_CONTRACT_GENERAL`, because a specific contract asserts something
    about every tool the model holds and on a mixed set it would be false of
    some of them. Two tools of the same category are not mixed.

    Logs ``tool contract: <name>`` once per selection, which is once per prompt
    that carries one.
    """
    contracts = {contract_for_category(_category_of(tool)) for tool in tools}
    if not contracts:
        return ""
    contract = contracts.pop() if len(contracts) == 1 else TOOL_CONTRACT_GENERAL
    logger.info("tool contract: %s", CONTRACT_NAMES.get(contract, "general"))
    return contract


def is_execution_tool(tool: Any) -> bool:
    """True when *tool* does work the model cannot carry out in its head.

    This is :attr:`ToolUsePolicy.REQUIRED` asked of one tool. Reading it from
    the one policy table is what keeps "which tools must actually run" and
    "which tools are told to run" from disagreeing: a category whose policy
    requires a call is an execution tool here by construction, and the test that
    enumerates every member covers both.

    It is asked of a tool the agent holds, not of a task or a model, because the
    difference it marks is a property of the tool: describing what a code
    executor would have printed is not the same as running it, while answering
    without searching is a worse answer rather than an impossible one.

    Args:
        tool: A tool object, or anything carrying a ``category`` or a
            ``metadata.category``.

    Returns:
        bool: True for a tool in an execution category.
    """
    return policy_for_category(_category_of(tool)) is ToolUsePolicy.REQUIRED


def _category_of(tool: Any) -> Any:
    """The category *tool* declares, or ``None`` when it declares none.

    Reads the metadata a tool carries rather than requiring a
    :class:`~effgen.tools.base_tool.BaseTool`, so a duck-typed tool with no
    ``metadata`` at all is answerable instead of raising.
    """
    category = getattr(tool, "category", None)
    if category is None:
        category = getattr(getattr(tool, "metadata", None), "category", None)
    return category
