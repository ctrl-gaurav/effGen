import { FileText } from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  ApiTable,
  Callout,
  CodeBlock,
  DocPage,
  ParamTable,
  SeeAlso,
  Terminal,
} from '../components/docs';
import { siteData, version } from '../siteData';

const prompts = siteData.prompts;

/** The five prompting techniques, and what each one means. The counts beside
 *  them are read from the library, never written down. */
const VARIANTS: Array<[string, string]> = [
  ['zero_shot', 'Instructions only, no worked examples.'],
  ['structured', 'Asks for JSON in a stated shape, and the check validates it.'],
  ['cot', 'Chain of thought — the model is asked to reason step by step before answering.'],
  ['few_shot', 'Carries exemplars that set the style of the answer.'],
  ['tool', 'Written for an agent with tools, and tells it to retrieve before answering.'],
];

export default function Prompts() {
  return (
    <DocPage
      subtitle="The bundled templates, how they are rendered, and how to use one in an agent."
      icon={<FileText size={48} />}
    >
      <p>
        effGen ships {prompts.library} curated prompt templates across {prompts.domains.length}{' '}
        domains. Each one is a Python function that renders deterministically for fixed inputs,
        with a JSON schema for what it takes, a worked example, and a check its output is measured
        against.
      </p>

      <h2>Rendering one</h2>

      <CodeBlock filename="render.py" code={`from effgen.prompts.library import registry

prompt = registry.get("coding.docstring_fill.v1")
print(prompt.render(
    code="def add(a, b):\\n    return a + b",
    style="google",
)[:400])`} />

      <Terminal
        command="python render.py"
        output={`You are an expert Python technical writer.

Add Google-style docstrings to every function and class in the code below that is currently missing one.

Use this docstring format:
"""Short one-line summary.

Args:
    param_name (type): Description.

Returns:
    type: Description.

Raises:
    ExceptionType: When and why it is raised.
"""

Rules:
  - Infer parameter types and return types from usage`}
        caption={`Run against effGen 1.0.0; the first 400 characters of the rendered prompt.`}
      />

      <h2>What is in the library</h2>

      <ApiTable
        headers={['Domain', 'Templates']}
        rows={prompts.domains.map((domain) => [
          <Link to={`/prompts/gallery#domain-${domain}`}>{domain}</Link>,
          String(prompts.domain_counts[domain]),
        ])}
        caption={
          <>
            {prompts.library} templates in the domain library, out of {prompts.templates}{' '}
            registered templates in total — the rest are the base templates the agent itself
            builds prompts from. <Link to="/prompts/gallery">The gallery</Link> lists every one
            with its variables.
          </>
        }
      />

      <CodeBlock filename="registry.py" code={`from effgen.prompts.library import registry

prompts = registry.all()
print(len(prompts), "templates in", len(registry.domains()), "domains")

for prompt in registry.search(domain="research"):
    print(f"  {prompt.name:44} {prompt.variant}")`} />

      <Terminal command="python registry.py" output={`35 templates in 8 domains
  research.citation_extract.v1                 tool
  research.literature_review.v1.cot            cot
  research.literature_review.v1.zero_shot      zero_shot
  research.methodology_critique.v1             cot
  research.paper_summary.v1                    structured`} />

      <h3>Variants</h3>
      <p>
        Each prompting technique is its own named template rather than a switch, so what you asked
        for is what you get.
      </p>

      <ApiTable
        headers={['Variant', 'Count', 'What it means']}
        rows={VARIANTS.map(([name, blurb]) => [
          <code>{name}</code>,
          String(prompts.items.filter((p) => p.variant === name).length),
          blurb,
        ])}
        caption={`Derived from the ${prompts.library} library templates in effGen ${version}.`}
      />

      <h2>A template's schema</h2>

      <CodeBlock filename="schema.py" code={`from effgen.prompts.library import registry

prompt = registry.get("business.meeting_summary.v1")
print(prompt.name, "|", prompt.domain, "|", prompt.variant)
print(prompt.description)
schema = prompt.input_schema
for name, spec in schema["properties"].items():
    required = "required" if name in schema.get("required", []) else "optional"
    print(f"  {name:14} {spec.get('type'):8} {required:8} {spec.get('description', '')}")
print("fixture keys:", sorted(prompt.fixture))`} />

      <Terminal command="python schema.py" output={`business.meeting_summary.v1 | business | structured
Extract structured meeting summary as JSON: decisions, action_items (with owner, item, due), and risks. Input: transcript + meeting title + attendees.
  transcript     string   required Full or partial meeting transcript or notes.
  meeting_title  string   required Title or topic of the meeting.
  attendees      array    optional List of attendee names.
fixture keys: ['attendees', 'meeting_title', 'transcript']`} />

      <ParamTable
        nameLabel="Field"
        params={[
          { name: 'name', type: 'str', description: 'The dotted name — domain.purpose.version, sometimes with the variant on the end.' },
          { name: 'domain', type: 'str', description: 'One of the eight domains.' },
          { name: 'variant', type: 'str', description: 'zero_shot, cot, few_shot, tool or structured.' },
          { name: 'description', type: 'str', description: 'What the template does and what it takes.' },
          { name: 'template', type: 'Callable[..., str]', description: 'The function that renders it. `render(**inputs)` calls it.' },
          { name: 'input_schema', type: 'dict', description: 'A JSON schema for the inputs, which is what validate_input checks against.' },
          { name: 'fixture', type: 'dict', description: 'A worked set of inputs, used by `prompts render` and by the eval harness.' },
          { name: 'expected_shape', type: 'dict | None', description: 'How the output is checked: a JSON schema, a regex, or a function. None means it is not checked.' },
          { name: 'tags', type: 'list[str]', description: 'Labels for search.' },
        ]}
        caption={<><code>effgen.prompts.library.LibraryPrompt</code></>}
      />

      <h3>Checking the inputs before you render</h3>

      <CodeBlock filename="validate.py" code={`from effgen.prompts.library import registry

prompt = registry.get("business.meeting_summary.v1")

problems = prompt.validate_input({"meeting_title": "Q3 planning"})
for problem in problems:
    print(problem)

print("valid:", prompt.validate_input(prompt.fixture) == [])`} />

      <Terminal command="python validate.py" output={`'transcript' is a required property
valid: True`} />

      <Callout type="warning" title="render() does not validate">
        <p>
          <code>validate_input(inputs)</code> returns a list of problems — empty when there are
          none. It does not raise, and <code>render()</code> does not call it: a template whose
          function has defaults will happily render with a missing input. Call it first when the
          inputs come from somewhere you do not control.
        </p>
      </Callout>

      <h2>Using one with a model</h2>

      <CodeBlock filename="explain.py" code={`from effgen import Agent, AgentConfig
from effgen.prompts.library import registry

prompt = registry.get("data.sql_explain.v1")
rendered = prompt.render(
    sql="SELECT country, COUNT(*) FROM orders GROUP BY country",
    audience="business",
)

agent = Agent(AgentConfig(name="explainer", model="gpt-5-nano", provider="openai"))
print(agent.run(rendered).text)`} />

      <Terminal command="python explain.py" output={`1) One-sentence summary:
The query tries to show, for each country, how many orders were placed.

2) Step-by-step breakdown of each clause:
- FROM orders: The data is taken from the orders table in the given schema.
- SELECT country, COUNT(*): It attempts to pick the country value and count how many rows share that country.
- GROUP BY country: It groups the orders by country so there is one line per country with the total count.

Important note: In the provided schema, there is no country column in either orders or customers, so as written this query cannot run. To make it work you would need country data available in the orders table or by joining orders to a customers table that contains a country field, for example:
SELECT c.country, COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.country

3) What the result set looks like:
- Columns: country, count (the number of orders)
- Each row represents a country and the total number of orders from that country (e.g., United States, 150).`} maxLines={16} />

      <h2>From the command line</h2>

      <ApiTable
        headers={['Command', 'What it does']}
        rows={[
          [
            <code>effgen prompts list</code>,
            <>
              Every template. <code>--domain</code> and <code>--variant</code> filter;{' '}
              <code>--format table|json|markdown</code> (or <code>--json</code>) chooses the
              output.
            </>,
          ],
          [<code>effgen prompts show &lt;name&gt;</code>, 'Its description, tags, input schema and fixture.'],
          [
            <code>effgen prompts render &lt;name&gt;</code>,
            <>
              Render to stdout without calling a model. <code>-i/--input FILE</code> supplies the
              variables as JSON, validated against the schema; omit it to render the fixture.
            </>,
          ],
          [
            <code>effgen prompts run &lt;name&gt; -m &lt;model&gt;</code>,
            <>
              Render and send. <code>--max-tokens</code> and <code>--temperature</code> apply to
              that run.
            </>,
          ],
          [
            <code>effgen prompts eval</code>,
            <>
              Golden eval, no model needed. <code>--live -m &lt;model&gt;</code> adds a live pass,{' '}
              <code>--fail-under FRACTION</code> gates CI.
            </>,
          ],
          [
            <code>effgen prompts playground</code>,
            <>
              An interactive session for trying one — see{' '}
              <Link to="/prompts/authoring">Authoring templates</Link>.
            </>,
          ],
        ]}
      />

      <Terminal command="effgen prompts list --domain education" output={`                                 Prompt Library                                 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Name                      ┃ Domain    ┃ Variant   ┃ Description              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ education.explain_simply. │ education │ zero_shot │ Explain a concept for a  │
│ v1                        │           │           │ specific audience with   │
│                           │           │           │ one everyday analogy and │
│                           │           │           │ a comprehension check.   │
│                           │           │           │ Inputs: concept,         │
│                           │           │           │ audience.                │
│ education.lesson_plan.v1  │ education │ zero_shot │ Time-boxed lesson plan   │
│                           │           │           │ (hook → instruction →    │
│                           │           │           │ practice → assessment)   │
│                           │           │           │ for a single class.      │
│                           │           │           │ Inputs: topic,           │
│                           │           │           │ grade_level,             │
│                           │           │           │ duration_minutes,        │
│                           │           │           │ learning_objective.      │
│ education.quiz_generate.v │ education │ zero_shot │ Multiple-choice quiz     │
│ 1                         │           │           │ generator with an        │
│                           │           │           │ explained answer key.    │
│                           │           │           │ Inputs: topic,           │
│                           │           │           │ num_questions,           │
│                           │           │           │ difficulty               │
│                           │           │           │ (easy|medium|hard).      │
│ education.socratic_tutor. │ education │ zero_shot │ Socratic tutor that      │
│ v1                        │           │           │ guides a learner with    │
│                           │           │           │ one question at a time   │
│                           │           │           │ instead of revealing the │
│                           │           │           │ answer. Inputs: subject, │
│                           │           │           │ question, student_level. │
└───────────────────────────┴───────────┴───────────┴──────────────────────────┘

Total: 4 prompt(s)`} maxLines={22} />

      <Terminal command="effgen prompts show education.lesson_plan.v1" output={`
Prompt: education.lesson_plan.v1
  Domain:      education
  Variant:     zero_shot
  Description: Time-boxed lesson plan (hook → instruction → practice → 
assessment) for a single class. Inputs: topic, grade_level, duration_minutes, 
learning_objective.
  Tags:        education, lesson-plan, teaching, zero_shot

Input Schema:
  {
    "type": "object",
    "properties": {
      "topic": {
        "type": "string",
        "description": "The lesson topic, e.g. 'the water cycle'.",
        "minLength": 3
      },
      "grade_level": {
        "type": "string",
        "description": "Audience, e.g. '5th grade', 'intro undergraduate'.",
        "minLength": 2
      },
      "duration_minutes": {
        "type": "integer",
        "description": "Total class length in minutes.",
        "minimum": 5
      },
      "learning_objective": {
        "type": "string",
        "description": "What students should be able to do after the lesson.",
        "minLength": 5
      }
    },
    "required": [
      "topic",
      "grade_level",
      "duration_minutes",
      "learning_objective"
    ]
  }

Fixture:
  {
    "topic": "the water cycle",
    "grade_level": "5th grade",
    "duration_minutes": 45,
    "learning_objective": "students can name and describe evaporation, condensation, and precipitation"
  }

Rendered (fixture):
You are an experienced 5th grade teacher. Write a clear, time-boxed lesson plan for a 45-minute class on "the water cycle".

Learning objective: students can name and describe evaporation, condensation, and precipitation

Structure the plan with these sections, each with a time estimate that adds up to 45 minutes:
  1. Hook / warm-up
  2. Direct instruction (key ideas, in plain language)
  3. Guided practice (an activity students do together)
  4. Independent practice / check for understanding
  5. Wrap-up and a quick formative assessment

Also list the materials needed and one differentiation tip for students who need extra support. Write the lesson plan now:`} maxLines={26} />

      <h3>JSON, for a script</h3>

      <Terminal command="effgen prompts list --domain legal --json" output={`[
  {
    "name": "legal.clause_classify.v1",
    "domain": "legal",
    "variant": "zero_shot",
    "description": "Classify a contract clause by type and flag notable characteristics. Zero-shot. Includes mandatory legal disclaimer.",
    "tags": [
      "legal",
      "clause",
      "classification",
      "zero_shot"
    ]
  },
  {
    "name": "legal.contract_summarize.v1",
    "domain": "legal",
    "variant": "structured",
    "description": "Summarize a contract into structured JSON with parties, term, obligations, termination, and risks. Includes mandatory legal disclaimer.",
    "tags": [
      "legal",
      "contract",
      "structured",
      "json"
    ]
  },
  {
    "name": "legal.legal_research_brief.v1",
    "domain": "legal",
    "variant": "tool",
    "description": "Produce a structured legal research brief grounded in pre-retrieved sources. Tool-augmented. Includes mandatory legal disclaimer.",
    "tags": [
      "legal",
      "research",
      "brief",
      "tool"
    ]
  }
]`} maxLines={22} />

      <h2>Evaluating a template</h2>
      <p>
        Every template ships a fixture and a golden rendering, so a change to a template that
        alters its output is caught without a model call. The <code>--live</code> pass adds a real
        call and checks the output against <code>expected_shape</code>.
      </p>

      <CodeBlock filename="golden.py" code={`from effgen.prompts.library import PromptEval, registry

evaluator = PromptEval()
result = evaluator.eval_golden(registry.get("coding.docstring_fill.v1"))
print(result.passed, "|", result.message)`} />

      <Terminal
        command="python golden.py"
        output={`True | `}
        caption="A pass with nothing to report leaves the message empty."
      />

      <Terminal command="effgen prompts eval --domain education" output={`
Running golden eval...
Name                                          Kind     Status
----------------------------------------------------------------------
education.explain_simply.v1                   golden   PASS
education.lesson_plan.v1                      golden   PASS
education.quiz_generate.v1                    golden   PASS
education.socratic_tutor.v1                   golden   PASS
----------------------------------------------------------------------
Total: 4  Pass: 4  Fail: 0`} />

      <ApiTable
        headers={['expected_shape', 'How the output is checked', 'Templates']}
        rows={[
          [
            <code>{'{"type": "json", "schema": {...}}'}</code>,
            'Parsed as JSON and checked against the schema’s required keys.',
            String(prompts.items.filter((p) => p.check === 'json').length),
          ],
          [
            <code>{'{"type": "regex", "pattern": "..."}'}</code>,
            'Matched against a pattern.',
            String(prompts.items.filter((p) => p.check === 'regex').length),
          ],
          [
            <code>{'{"type": "callable", "fn": ...}'}</code>,
            'Handed to a function that returns True or a reason it failed.',
            String(prompts.items.filter((p) => p.check === 'function').length),
          ],
          [
            <code>None</code>,
            'Not checked — golden rendering only.',
            String(prompts.items.filter((p) => p.check === 'none').length),
          ],
        ]}
        caption={`Across the ${prompts.library} library templates.`}
      />

      <h2>What goes wrong</h2>

      <ApiTable
        headers={['What you see', 'What it means', 'What to do']}
        rows={[
          [
            <code>KeyError</code> ,
            'The registry has no template by that name.',
            <>
              <code>effgen prompts list</code> shows every name. Several carry the variant on the
              end — <code>research.literature_review.v1.cot</code>, not{' '}
              <code>research.literature_review.v1</code>.
            </>,
          ],
          [
            'A rendered prompt with a placeholder still in it',
            <>
              An input was missing and the template function had a default for it.
            </>,
            <>
              <code>validate_input(inputs)</code> before rendering. It lists what is missing.
            </>,
          ],
          [
            'An empty answer from `prompts run`',
            'A reasoning model spent the whole completion budget on internal reasoning and hit the cap.',
            <>
              Raise <code>--max-tokens</code>. The message says how many reasoning tokens were
              produced and that no answer was, rather than printing nothing.
            </>,
          ],
          [
            'A golden eval fails after you edited a template',
            'The rendering changed, which is what the golden file exists to catch.',
            <>
              If the change was intended, delete the golden file and re-run{' '}
              <code>effgen prompts eval</code> to regenerate it.
            </>,
          ],
          [
            'A live eval fails on a structured template',
            <>
              The model returned prose, or JSON missing a required key.
            </>,
            <>
              That is <code>expected_shape</code> working. A larger model, or a lower temperature,
              usually fixes it.
            </>,
          ],
          [
            'A rate limit during `prompts eval --live`',
            'The live pass makes one call per template.',
            <>
              <code>--delay</code> sets the seconds between calls; it defaults to 35 for exactly
              this reason.
            </>,
          ],
        ]}
      />

      <SeeAlso paths={['/prompts/gallery', '/prompts/authoring', '/agents']} />
    </DocPage>
  );
}
