"""Unit tests for ``sanitize_final_answer`` — scaffolding-leak elimination.

The fixtures here are the real messy strings captured from small models (groq
``llama-3.1-8b``, OpenAI native web-search) plus the literal loop-bookkeeping
markers the ReAct scaffolding injects. The sanitizer must remove every internal
marker while leaving legitimate answer content (including markdown tables and
code) untouched.
"""

import pytest

from effgen.core.agent import sanitize_final_answer as sanitize

# --- Real leaked strings from the audit ------------------------------------

@pytest.mark.parametrize(
    "leaked, expected",
    [
        # OpenAI native-tool output: "Canberra\nFinal Answer: Canberra"
        ("Canberra\nFinal Answer: Canberra", "Canberra"),
        # Groq llama-3.1-8b multi-tool join with the bookkeeping marker embedded.
        (
            (
                "225 | 225\n[Tool results computed above. Continue or provide "
                "Final Answer:] | 450 | 225 | 675 | 0"
            ),
            "225 | 225 | 450 | 225 | 675 | 0",
        ),
        # Plain "Final Answer:" label.
        ("Final Answer: The capital is Paris.", "The capital is Paris."),
        ("Answer: 42", "42"),
        # Reasoning then a final-answer label — keep only the answer.
        ("Let me think. 15*15=225.\nFinal Answer: 225", "225"),
        # Trailing observation/thought bleed.
        ("Paris\nObservation: tool returned Paris", "Paris"),
        ("The result is 42.\nThought: I should double check", "The result is 42."),
        # Tool-echo fragment.
        ("[calculator({'expression': '15*15'})] → 225", "225"),
        # List-prefixed final-answer label.
        ("- Final Answer: done", "done"),
        # The label written in markdown emphasis. The closing marker used to be
        # left behind, and its stray "**" then paired with the next heading and
        # ran emphasis through the rest of the reply.
        ("**Answer:** \nThe product is 1,136,812.", "The product is 1,136,812."),
        ("**Final Answer:** 42", "42"),
        ("__Answer:__ 42", "42"),
        ("### Answer: 42", "42"),
        # An answer that is itself bold keeps both of its markers: only a
        # closing marker matching the label's own opening one is absorbed.
        ("Answer: **42**", "**42**"),
        # OpenAI native web-search replies sometimes end with a bare, dangling
        # "Final Answer:" label and nothing after it — strip the label, keep the
        # real answer that precedes it.
        (
            "The outage happened on July 19, 2024. ([blogs.microsoft.com](https://x))\n\nFinal Answer:",
            "The outage happened on July 19, 2024. ([blogs.microsoft.com](https://x))",
        ),
        ("Paris.\nFinal Answer: ", "Paris."),
        # The "you already have results from this tool" nudge the loop injects
        # when a tool has run twice — a small model (groq llama-3.1-8b) echoed it
        # straight into a pipe-joined answer. Must be stripped like the others.
        (
            (
                "79.96 | 86.3568 [You already have results from this tool. If you "
                "have enough information, respond now with 'Final Answer:'.] | 85.49"
            ),
            "79.96 | 86.3568 | 85.49",
        ),
        # The "you already computed this" nudge injected on a repeated action.
        (
            (
                "42 You already computed this. Please provide your final response "
                "using 'Final Answer:' now."
            ),
            "42",
        ),
    ],
)
def test_strips_scaffolding(leaked, expected):
    assert sanitize(leaked) == expected


# --- Leaked model tool-call syntax (Llama-style) -----------------------------

@pytest.mark.parametrize(
    "leaked, expected",
    [
        # Groq llama-3.1-8b emitted its tool call as text instead of routing it.
        (
            (
                'function=calculator>{"expression": "15*90", "operation": '
                '"calculate", "precision": 0}</function>'
            ),
            "",
        ),
        ('<function=calc>{"a": 1}</function>', ""),
        # Nested-object args must be matched whole, not leave a dangling "}".
        ('<function=calc>{"a": {"b": 1}}</function>', ""),
        (
            (
                '<function=calculator>{"args": {"expression": "1+1"}, "op": "x"}'
                "</function>"
            ),
            "",
        ),
        ("<tool_call>{\"name\": \"calc\"}</tool_call>", ""),
        # Real answer with a trailing stray tag — keep the answer, drop the tag.
        ("The total is 1350 </function>", "The total is 1350"),
        # Special tokens.
        ("Paris <|eot_id|>", "Paris"),
    ],
)
def test_strips_tool_call_syntax(leaked, expected):
    assert sanitize(leaked) == expected


# --- Leaked bare tool-call echo (no wrapping tag) ----------------------------

@pytest.mark.parametrize(
    "leaked, expected",
    [
        # Groq llama-3.1-8b-instant: a bare "tool_name {json}" prefix, no
        # <function=.../<tool_call> wrapper, before the real prose answer.
        (
            (
                'order_lookup {"order_id": "ORD-1001"} \n'
                'The order status is "shipped" and will arrive in 3-5 days.'
            ),
            'The order status is "shipped" and will arrive in 3-5 days.',
        ),
        ('calculator {"expression": "6*7"} 42', "42"),
        # Same-line, no newline separator.
        ('issue_refund {"order_id": "ORD-1001"} Refund not issued.', "Refund not issued."),
        # No separator at all between the name and its arguments — the shape a
        # model emits when the tool name and JSON are concatenated.
        ('mcp_p64_echo{"text": "world"}', ""),
        ('calculator{"expression": "6*7"}\nThe answer is 42.', "The answer is 42."),
        # Empty argument object, still a tool-call echo.
        ("issue_refund{}\nRefund issued.", "Refund issued."),
        # A model that begins a tag and abandons it leaves a stray "<" on the
        # same shape. Groq llama-3.1-8b-instant returned this as a whole answer.
        (
            '<wikipedia {"operation": "search", "query": "Eiffel Tower"}',
            "",
        ),
        (
            '<wikipedia {"operation": "search"} \nThe tower was completed in 1889.',
            "The tower was completed in 1889.",
        ),
        # Closed with a matching ">".
        ('<calculator {"expression": "1889+11"}>The answer is 1900.', "The answer is 1900."),
        ('<calculator{"expression": "6*7"}>42', "42"),
    ],
)
def test_strips_leading_untagged_tool_call_echo(leaked, expected):
    assert sanitize(leaked) == expected


@pytest.mark.parametrize(
    "text",
    [
        # Mid-sentence — must never be touched (only a *leading* echo is scaffolding).
        "The status is order_lookup {\"x\": 1} weird",
        # Capitalized "word {...}" is not a tool-name shape (tool names are
        # lowercase snake_case) and must be left alone.
        'Config {"debug": true} is enabled',
        # Without a separating space the brace block must look like a JSON
        # argument object, so a CSS rule or a brace block with bare keys stays.
        "body{color:red} is the CSS you want.",
        "config{alpha: 1} means something.",
        # An angle-bracketed opening only counts when the braces look like a
        # JSON argument object, so markup and templates are left alone.
        "<template {{ item }}> renders each row.",
        '<div class="x">hello</div>',
        "<section {alpha: 1}> is not a tool call.",
    ],
)
def test_leading_json_like_prose_not_corrupted(text):
    assert sanitize(text) == text



# --- Gemma 4 channel format (asymmetric <|channel> / <channel|>) --------------

@pytest.mark.parametrize(
    "leaked, expected",
    [
        # The answer follows the closed reasoning channel — keep only the answer.
        (
            "<|channel>thought\nlong english planning...\n<channel|>Paris is the capital.",
            "Paris is the capital.",
        ),
        # Japanese answer after the reasoning channel (captured from gemma-4-e4b).
        (
            "<|channel>thought\nreasoning\n<channel|>富士山の標高は3,776メートルです。",
            "富士山の標高は3,776メートルです。",
        ),
        # A leaked Gemma tool call in an answer must be removed whole.
        ('answer <|tool_call>call:arxiv{query: "x"}<tool_call|>', "answer"),
        # Truncated mid-reasoning (no close tag) must not surface as the answer.
        ("<|channel>thought\ncut off before finishing", ""),
    ],
)
def test_strips_gemma_channel_format(leaked, expected):
    assert sanitize(leaked) == expected


def test_gemma_sanitize_is_idempotent():
    raw = "<|channel>thought\nplan\n<channel|>富士山は約3,776mです。"
    once = sanitize(raw)
    assert sanitize(once) == once


# --- Idempotency --------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Canberra\nFinal Answer: Canberra",
        "225 | 225\n[Tool results computed above. Continue or provide Final Answer:] | 0",
        "Final Answer: 42",
        "Just a plain answer.",
    ],
)
def test_idempotent(text):
    once = sanitize(text)
    assert sanitize(once) == once


# --- Must NOT corrupt legitimate content -------------------------------------

def test_markdown_table_preserved():
    table = "| Name | Age |\n|------|-----|\n| Bob | 30 |"
    assert sanitize(table) == table


def test_code_block_with_pipe_preserved():
    code = "```python\nresult = a | b\n```"
    assert sanitize(code) == code


def test_plain_answer_unchanged():
    assert sanitize("The capital of France is Paris.") == "The capital of France is Paris."


def test_answer_mentioning_observation_word_unchanged():
    # "observation" as a normal word (not a line-anchored ReAct label) stays.
    text = "My observation is that the sky is blue."
    assert sanitize(text) == text


def test_ascii_arrow_in_prose_preserved():
    # The tool-echo scaffolding always uses the Unicode arrow; legitimate prose
    # with an ASCII "->" (e.g. a footnote reference) must not be mangled.
    text = "Step A [1] -> Step B"
    assert sanitize(text) == text


# --- Edge cases ---------------------------------------------------------------

def test_none_passthrough():
    assert sanitize(None) is None


def test_empty_passthrough():
    assert sanitize("") == ""


def test_non_string_passthrough():
    assert sanitize(123) == 123  # type: ignore[arg-type]


def test_only_label_returns_label_stripped_text():
    # If stripping a label would empty the string, keep the (stripped) content.
    # "Final Answer:" with nothing after → empty tail, label kept as-is content.
    assert sanitize("Final Answer:") == "Final Answer:"


# --- Drift guard: injection sites and the strip-list share one source --------

def test_every_injected_nudge_is_stripped():
    """Each loop nudge the ReAct scaffolding can inject must be sanitized away.

    The nudge strings are defined once in ``agent_runtime`` and referenced by
    the injection sites in ``agent_react``; this guards against a future nudge
    being added at an injection site without being added to the strip-list (the
    bug that let "[You already have results from this tool…]" reach a user).
    """
    import effgen.core.agent_runtime as rt

    nudges = [v for k, v in vars(rt).items() if k.startswith("NUDGE_")]
    assert nudges, "expected NUDGE_* constants to exist"
    for nudge in nudges:
        # Embedded mid-answer (the way a small model echoes it) must vanish.
        leaked = f"42 {nudge} done"
        assert nudge not in sanitize(leaked), f"nudge not stripped: {nudge!r}"


def test_every_unknown_tool_observation_is_stripped():
    """The unknown-tool observation is built per agent, so cover its shapes.

    It names the action the model wrote and lists the agent's tools, so it is
    not a fixed literal and cannot ride on the ``NUDGE_*`` list above. The
    observation is generated here rather than hardcoded, so rewording it
    without updating the strip pattern fails this test.
    """
    from effgen.core.agent_runtime import unknown_tool_observation

    cases = [
        ("weather_lookup", ["calculator"]),
        ("x", ["calculator", "wikipedia", "web_search"]),
        ('calculator | Action Input: {"expression": "2 + 2"}', ["calculator"]),
        ("it's a tool", ["calculator"]),
        ("", []),
    ]
    for action, tools in cases:
        observation = unknown_tool_observation(action, tools)
        cleaned = sanitize(f"42 {observation} done")
        assert "No tool named" not in cleaned, f"not stripped: {observation!r}"
        assert "The tools you can use are" not in cleaned
        assert "Use one of them" not in cleaned


def test_injection_sites_use_the_shared_nudges():
    """Every injection site must reference the shared NUDGE_* values.

    Reading the source keeps the files in sync: every literal a loop appends
    should be one of the named constants (so it is also on the strip-list). The
    sites are the blocking loop, the streamed loop, and the repeat policy they
    share.
    """
    import inspect

    import effgen.core.agent_react as ar
    import effgen.core.agent_runtime as rt
    import effgen.core.agent_stream_native as sn
    import effgen.core.agent_tool_loop as tl

    src = "\n".join(inspect.getsource(m) for m in (ar, tl, sn))
    for name in ("NUDGE_CONTINUE", "NUDGE_HAVE_ANSWER", "NUDGE_HAVE_RESULTS",
                 "NUDGE_ALREADY_COMPUTED", "NUDGE_NO_TOOLS", "NUDGE_NOT_USABLE",
                 "NUDGE_SEARCH_AGAIN"):
        assert getattr(rt, name, None), f"missing nudge constant {name}"
        assert name in src, f"no tool-loop module references {name}"

    # The unknown-tool observation is built, not a constant. Every loop must
    # build it from the one shared function, or the paths diverge on the same
    # model output.
    import effgen.core.agent_streaming as st

    assert callable(getattr(rt, "unknown_tool_observation", None))
    for module in (ar, st, sn):
        # The call, not the import — a module that keeps the import but writes
        # its own observation text is exactly the drift this guards.
        assert "unknown_tool_observation(" in inspect.getsource(module), (
            f"{module.__name__} no longer builds the shared unknown-tool observation"
        )


class TestXmlTaggedCallScaffolding:
    """A call written as nested tags never reaches the user as prose.

    Stripping the tags alone would leave the argument values standing in the
    answer, so ``<parameter=expression>4817 * 236</parameter>`` would read as a
    stray "4817 * 236" beside the real answer.
    """

    def test_a_wrapped_call_is_removed_whole(self):
        from effgen.core.agent_runtime import sanitize_final_answer

        text = (
            "<tool_call>\n<function=calculator>\n<parameter=operation>\ncalculate\n"
            "</parameter>\n<parameter=expression>\n4817 * 236\n</parameter>\n"
            "</function>\n</tool_call>\nThe product is 1136812."
        )
        assert sanitize_final_answer(text).strip() == "The product is 1136812."

    def test_the_attribute_spelling_is_removed_too(self):
        from effgen.core.agent_runtime import sanitize_final_answer

        text = (
            '<invoke name="calculator"><parameter name="expression">6*7</parameter>'
            "</invoke>\nIt is 42."
        )
        assert sanitize_final_answer(text).strip() == "It is 42."

    def test_a_construct_cut_short_leaves_no_tags(self):
        from effgen.core.agent_runtime import sanitize_final_answer

        out = sanitize_final_answer(
            "Working on it.\n<function=calculator>\n<parameter=expression>\n6*7"
        )
        assert "<" not in out and "parameter" not in out
        assert out.strip().startswith("Working on it.")

    def test_sanitizing_is_idempotent(self):
        from effgen.core.agent_runtime import sanitize_final_answer

        text = "<function=calculator><parameter=expression>6*7</parameter></function>42"
        once = sanitize_final_answer(text)
        assert sanitize_final_answer(once) == once

    def test_prose_about_parameters_is_left_alone(self):
        from effgen.core.agent_runtime import sanitize_final_answer

        text = "Pass the expression parameter to the calculator, e.g. 6*7 = 42."
        assert sanitize_final_answer(text) == text

    def test_a_mention_with_no_arguments_keeps_the_rest_of_the_answer(self):
        """Only a construct with argument tags may swallow the text after it.

        An answer that explains the syntax carries the opening tag and nothing
        else; dropping everything after the mention would delete the answer.
        """
        from effgen.core.agent_runtime import sanitize_final_answer

        out = sanitize_final_answer(
            "Templates write <function=NAME> to open a call. "
            "The rest of this sentence must survive."
        )
        assert "The rest of this sentence must survive." in out
