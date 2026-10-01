"use client";

import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import {
  FiActivity,
  FiAlertTriangle,
  FiArrowRight,
  FiBox,
  FiCode,
  FiExternalLink,
  FiFileText,
  FiLayers,
  FiSave,
  FiTool,
  FiUsers,
} from "react-icons/fi";
import Container from "@/components/Container";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import ParamTable from "@/components/ui/ParamTable";
import CodeSample from "@/components/ui/CodeSample";
import RouteLink from "@/components/ui/RouteLink";
import { siteData, version } from "@/components/siteData";
import {
  DOCS_COMPACTION_URL,
  DOCS_MIDDLEWARE_URL,
  DOCS_MULTI_AGENT_URL,
  DOCS_RAG_URL,
  DOCS_REFERENCE_URL,
  DOCS_SESSIONS_URL,
  DOCS_TOOLS_URL,
  configField,
  configGroups,
  presetAccents,
  responseField,
  responseNotes,
  resumeRows,
  toolRoutes,
} from "./agentsData";
import { accentTextStyle } from "@/components/accentText";


// The samples on this page, held here rather than inline so each one is the
// file that was run, character for character. Every `output=` beside them in
// the JSX is that run's stdout, pasted from the transcript.

const SAMPLE_TOOL_CALLS = `from effgen import Agent, AgentConfig, tool

CATALOGUE = {"wall bracket": "KX-9", "shelf pin": "TR-2"}
STOCK = {"KX-9": 3400, "TR-2": 118}


@tool
def find_sku(product: str) -> str:
    """Look up the SKU for a product name."""
    return CATALOGUE[product.lower()]


@tool
def stock_level(sku: str) -> int:
    """How many units of a SKU are in the warehouse."""
    return STOCK[sku]


with Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[find_sku, stock_level],
    tool_calling_mode="react",
)) as agent:
    r = agent.run("How many wall brackets do we have in stock?")

print(r.output)
print()
print("calls:", r.tool_calls.total, "· names:", r.tool_calls.names)
for call in r.tool_calls:
    print(f"  iteration {call.iteration}  {call.name}({call.arguments})")
    print(f"    -> {call.result}   in {call.duration:.4f}s   error={call.error}")
print("failed:    ", len(r.tool_calls.failed))
print("find_sku:  ", r.tool_calls.by_name("find_sku").names)
print("compares as its own count:", r.tool_calls == r.tool_calls.total,
      "· int():", int(r.tool_calls))`;

const SAMPLE_TOOL_DECORATOR = `import asyncio
from effgen import Agent, AgentConfig, tool


@tool
def shipping_cost(weight_kg: float, express: bool = False) -> str:
    """Quote a shipping cost in euros for a parcel."""
    price = 4.50 + 1.75 * weight_kg + (9.0 if express else 0.0)
    return f"EUR {price:.2f}"


result = asyncio.run(shipping_cost.execute(weight_kg=2.5, express=True))
print("success:", result.success)
print("output: ", result.output)
print("error:  ", result.error)
print("took:    %.4fs" % result.execution_time)

with Agent(AgentConfig(model="openai:gpt-5-nano", tools=[shipping_cost])) as agent:
    r = agent.run("How much to send a 2.5 kg parcel by express?")
print(r.output)
print("calls:", r.tool_calls.names)`;

const SAMPLE_FROM_FUNCTION = `from effgen import Tool


def seat_count(rows: int, seats_per_row: int) -> int:
    """Count the seats in a rectangular block of a theatre."""
    return rows * seats_per_row


seats = Tool.from_function(seat_count, category="computation")
print(seats.name, "·", seats.description)
print(seats.metadata.category.value)
for parameter in seats.metadata.parameters:
    print(f"  {parameter.name}: {parameter.type.value} "
          f"(required={parameter.required}) — {parameter.description}")`;

const SAMPLE_LOGGING_MIDDLEWARE = `import logging

from effgen import Agent, AgentConfig, LoggingMiddleware, get_tool_registry

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("effgen.core.middleware").setLevel(logging.INFO)

clock = get_tool_registry().get_tool_sync("datetime")

with Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[clock],
    middleware=[LoggingMiddleware()],
)) as agent:
    r = agent.run("What is the current date in UTC?")

print(r.output)`;

const SAMPLE_APPROVAL = `from effgen import Agent, AgentConfig, ToolApprovalMiddleware, tool

asked = []


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


def approve(name: str, arguments: str) -> bool:
    asked.append(name)
    return False


with Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[issue_refund],
    middleware=[ToolApprovalMiddleware(approve, tools=["issue_refund"])],
)) as agent:
    r = agent.run("Refund order A-4471 for 20 euros.")

print("asked about:", asked)
for call in r.tool_calls:
    print(call.name, "->", call.result)`;

const SAMPLE_CUSTOM_MIDDLEWARE = `import time
from effgen import Agent, AgentConfig, AgentMiddleware, get_tool_registry


class TimeEachTool(AgentMiddleware):
    """Record how long every tool call took."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._started: dict[str, float] = {}

    def before_tool_call(self, ctx):
        self._started[ctx.tool_name] = time.perf_counter()

    def after_tool_call(self, ctx, result):
        started = self._started.pop(ctx.tool_name, None)
        if started is not None:
            self.timings[ctx.tool_name] = time.perf_counter() - started
        return result


timer = TimeEachTool()
clock = get_tool_registry().get_tool_sync("datetime")

with Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[clock],
    middleware=[timer],
)) as agent:
    agent.run("What is the current date in UTC?")

print({name: round(seconds, 4) for name, seconds in timer.timings.items()})`;

const SAMPLE_SESSION = `import os
import tempfile

os.environ["EFFGEN_SESSIONS_DIR"] = tempfile.mkdtemp()

from effgen import Agent, AgentConfig

with Agent(AgentConfig(model="openai:gpt-5-nano")) as agent:
    agent.run("My dog is named Pixel.", session="user-123")
    agent.run("My cat is named Mote.", session="user-456")

    print(agent.run("What is my pet's name? One word.", session="user-123").output)
    print(agent.run("What is my pet's name? One word.", session="user-456").output)`;

const SAMPLE_COMPACTION = `from effgen import Agent, AgentConfig, KeepFirstAndLast

with Agent(AgentConfig(
    model="openai:gpt-5-nano",
    compaction_strategy=KeepFirstAndLast(first=2, last=4),
    max_context_length=2048,
)) as agent:
    for line in ["The invoice number is INV-8842.",
                 "The customer is Marta Reyes.",
                 "The amount is 412 euros.",
                 "The due date is the 30th.",
                 "The purchase order is PO-19.",
                 "The shipping city is Porto."]:
        agent.run(line + " Reply with ok.")
    print(agent.run("What is the invoice number?").output)
    print("messages held:", len(agent.short_term_memory.get_messages()))
    print("tokens held:  ", agent.short_term_memory.get_token_count())`;

const SAMPLE_MEMORY = `from effgen import Agent, AgentConfig

with Agent(AgentConfig(model="openai:gpt-5-nano", enable_memory=True)) as agent:
    agent.run("The build server is called hawthorn. Reply with ok.")
    agent.run("It runs Ubuntu 24.04. Reply with ok.")
    print(agent.run("Which operating system does hawthorn run?").output)
    print("messages in memory:", len(agent.short_term_memory.get_messages()))
    print("tokens held:", agent.short_term_memory.get_token_count())`;

const SAMPLE_RAG = `import pathlib
import tempfile

from effgen import create_agent

docs = pathlib.Path(tempfile.mkdtemp())
(docs / "runbook.md").write_text(
    "# Runbook\\n\\n"
    "The nightly export starts at 02:15 UTC.\\n"
    "If it has not finished by 03:00, restart the exporter and page the on-call.\\n"
)
(docs / "contacts.md").write_text(
    "# Contacts\\n\\nThe on-call rota is in the operations calendar.\\n"
)

agent = create_agent("rag", "openai:gpt-5-nano", knowledge_base=str(docs))
r = agent.run("When does the nightly export start?")
print(r.output)
print("sources:", r.sources)
print("citations:", [c.source for c in r.citations])
agent.close()`;

const SAMPLE_TEAM = `from effgen import (Agent, AgentConfig, MultiAgentOrchestrator,
                    OrchestrationPattern, load_model)

model = load_model("openai:gpt-5-nano")
writer = Agent(AgentConfig(name="writer", model=model,
                           system_prompt="Write one plain sentence."))
editor = Agent(AgentConfig(name="editor", model=model,
                           system_prompt="Shorten what you are given. Reply with the sentence only."))

orchestrator = MultiAgentOrchestrator()
team = orchestrator.create_team(
    "copy", [writer, editor], pattern=OrchestrationPattern.SEQUENTIAL,
)
result = orchestrator.assign_task("Describe what a checkpoint is.", team)

print(result.output)
print("pattern:", result.pattern.value, "· agents:", len(result.agent_responses))
for stage in result.agent_responses:
    print(f"  {stage['agent_name']}: {stage['output'][:60]}"
          f" ({stage['tokens_used']} tokens, \${stage['cost_usd']:.6f})")
writer.close()
editor.close()`;

const SAMPLE_STRUCTURED = `import json

from effgen import Agent, AgentConfig

schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"},
        "construction_began": {"type": "integer"},
    },
    "required": ["city", "country", "construction_began"],
}

with Agent(AgentConfig(model="openai:gpt-5-nano")) as agent:
    r = agent.run("Where is the Alhambra, and in which year did its construction begin?",
                  output_schema=schema)

print(r.output)
print(json.loads(r.output)["construction_began"])
print("structured:", r.metadata["structured_output"])`;

const SAMPLE_STREAM = `from effgen import Agent, AgentConfig

with Agent(AgentConfig(model="openai:gpt-5-nano", enable_streaming=True)) as agent:
    chunks = []
    for chunk in agent.stream("Name the three primary colours, in one line."):
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print()

print("chunks:", len(chunks))
print("joined == answer:", "".join(chunks).strip() != "")`;

const SAMPLE_FAILURE = `from effgen import Agent, AgentConfig, tool


@tool
def read_ledger(account: str) -> str:
    """Read an account's ledger."""
    raise FileNotFoundError(f"no ledger for {account}")


with Agent(AgentConfig(
    model="openai:gpt-5-nano",
    tools=[read_ledger],
    max_iterations=2,
    raise_on_error=False,
)) as agent:
    r = agent.run("Read the ledger for account 55-2 and tell me the balance.")

print("success:", r.success, "· iterations:", r.iterations)
for call in r.tool_calls:
    print(call.name, "· ok:", call.ok, "· error:", call.error)
print("failed calls:", len(r.tool_calls.failed))
print("partial_output:", repr(r.metadata.get("partial_output"))[:120])`;

const SAMPLE_WORKFLOW = `import tempfile

from effgen import (Agent, AgentConfig, AgentMiddleware, FileCheckpointStore,
                    WorkflowDAG, WorkflowNode, load_model)


class FailOnce(AgentMiddleware):
    """Stand in for the process being killed part-way through."""

    def __init__(self) -> None:
        self.armed = True

    def before_run(self, ctx):
        if self.armed:
            self.armed = False
            raise RuntimeError("the box went away")
        return None


model = load_model("openai:gpt-5-nano")
short = "Reply with one short sentence."
research = Agent(AgentConfig(name="research", model=model, system_prompt=short))
draft = Agent(AgentConfig(name="draft", model=model, system_prompt=short))
breaker = FailOnce()
review = Agent(AgentConfig(name="review", model=model, system_prompt=short,
                           middleware=[breaker]))

store = FileCheckpointStore(tempfile.mkdtemp())


def build() -> WorkflowDAG:
    dag = WorkflowDAG("report")
    dag.add_node(WorkflowNode(id="research", agent=research))
    dag.add_node(WorkflowNode(id="draft", agent=draft))
    dag.add_node(WorkflowNode(id="review", agent=review))
    dag.connect("research", "draft")
    dag.connect("draft", "review")
    return dag


def show(label, result):
    print(f"{label}:", [(n["id"], n["status"], n["execution_time"])
                        for n in result.node_results], "success:", result.success)


show("first run ", build().run("Summarise what a DAG is.", checkpoint=store, run_id="q3"))
print("checkpoint:", "completed", sorted(store.load("q3").completed),
      "failed", sorted(store.load("q3").failed))
show("resumed   ", build().run("Summarise what a DAG is.", checkpoint=store, run_id="q3"))

for agent in (research, draft, review):
    agent.close()`;

const SECTION_DIVIDER = (
  <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-green-500/20 to-transparent" />
);

const api = siteData.api;

function Band({
  id,
  eyebrow,
  title,
  lede,
  tinted = false,
  children,
}: {
  id: string;
  eyebrow: string;
  title: React.ReactNode;
  lede?: React.ReactNode;
  tinted?: boolean;
  children: React.ReactNode;
}) {
  const { ref, inView } = useInView({ triggerOnce: true, threshold: 0.05 });

  return (
    <section
      id={id}
      className={`py-16 relative scroll-mt-24 ${tinted ? "bg-gray-50 dark:bg-[#030f07]" : ""}`}
      aria-labelledby={`${id}-heading`}
    >
      {SECTION_DIVIDER}
      <Container className="relative z-10">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="mb-10 max-w-3xl"
        >
          <span className="text-[10px] font-mono uppercase tracking-widest text-green-700 dark:text-green-400">
            {eyebrow}
          </span>
          <h2
            id={`${id}-heading`}
            className="mt-2 text-3xl md:text-4xl font-black text-gray-900 dark:text-white leading-tight"
          >
            {title}
          </h2>
          {lede && <p className="mt-4 text-gray-600 dark:text-gray-400 leading-relaxed">{lede}</p>}
        </motion.div>
        {children}
      </Container>
    </section>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-6">
      <h3 className="text-base font-bold text-gray-900 dark:text-white mb-2">{title}</h3>
      <div className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed space-y-3">
        {children}
      </div>
    </div>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return <code className="font-mono text-xs">{children}</code>;
}

const numeric = (value: number) => value.toLocaleString("en-US");

export default function AgentsView() {
  const { ref: heroRef, inView: heroInView } = useInView({ triggerOnce: true, threshold: 0.05 });

  const headline = [
    {
      value: numeric(siteData.public_names),
      label: "Names the package exports",
      accent: "#00ff88",
      icon: FiBox,
    },
    {
      value: String(siteData.tools.count),
      label: `Built-in tools, ${Object.keys(siteData.tools.categories).length} categories`,
      accent: "#00e5ff",
      icon: FiTool,
    },
    {
      value: String(siteData.presets.count),
      label: "Presets, each a configured agent",
      accent: "#a78bfa",
      icon: FiLayers,
    },
    {
      value: String(api.middleware_hooks.length),
      label: "Middleware points around the loop",
      accent: "#ffd700",
      icon: FiActivity,
    },
  ];

  return (
    <div className="min-h-screen bg-white dark:bg-[#020c08]">
      <Navbar />
      <main id="main">
        {/* Hero */}
        <section className="relative pt-32 pb-10 overflow-hidden">
          <div className="absolute inset-0 grid-pattern" />
          <Container className="relative z-10">
            <motion.div
              ref={heroRef}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="max-w-4xl"
            >
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-green-500/30 bg-green-500/5 text-green-700 dark:text-green-400 text-sm font-semibold mb-8">
                <FiCode size={14} />
                the Python library · {version}
              </span>
              <h1 className="text-5xl md:text-6xl font-black mb-6 text-gray-900 dark:text-white leading-tight">
                One class to configure,{" "}
                <span className="gradient-text">one call to run</span>
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 leading-relaxed max-w-3xl">
                <Mono>Agent(AgentConfig(...)).run(task)</Mono> is the whole surface for a
                first agent. Everything past that — {siteData.tools.count} tools and
                your own functions, hooks around every model and tool call, one agent
                serving many conversations, four ways to compact a history that no
                longer fits, retrieval, teams, and workflows that resume where they
                died — is a parameter on that same object.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <a
                  href="#response"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
                >
                  What a run gives back
                  <FiArrowRight size={15} />
                </a>
                <a
                  href={DOCS_REFERENCE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-green-500/50 font-semibold text-sm transition-colors"
                >
                  API reference
                  <FiExternalLink size={14} />
                </a>
              </div>
            </motion.div>
          </Container>
        </section>

        <section className="pb-12 relative">
          <Container className="relative z-10">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {headline.map((stat, idx) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={heroInView ? { opacity: 1, y: 0 } : {}}
                  transition={{ duration: 0.5, delay: idx * 0.08 }}
                  className="rounded-2xl p-5 bg-white dark:bg-gray-900/70 border border-gray-200 dark:border-gray-800 text-center"
                >
                  <stat.icon className="mx-auto mb-2" style={accentTextStyle(stat.accent)} size={20} />
                  <div className="text-2xl font-black mb-0.5" style={accentTextStyle(stat.accent)}>
                    {stat.value}
                  </div>
                  <div className="text-[10px] text-gray-600 dark:text-gray-400 font-semibold uppercase tracking-wider">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </Container>
        </section>

        {/* The artefact, above the fold */}
        <section className="pb-4 relative">
          <Container className="relative z-10">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <CodeSample
                language="python"
                accent="#00ff88"
                code={SAMPLE_TOOL_CALLS}
                output={`3,400 wall brackets in stock (SKU: KX-9).

calls: 3 · names: ['find_sku', 'find_sku', 'stock_level']
  iteration 1  find_sku({"product": "wall brackets"})
    -> Error executing tool 'find_sku': Tool execution failed: 'wall brackets'   in 0.0031s   error=Error executing tool 'find_sku': Tool execution failed: 'wall brackets'
  iteration 2  find_sku({"product": "wall bracket"})
    -> KX-9   in 0.0015s   error=None
  iteration 3  stock_level({"sku": "KX-9"})
    -> 3400   in 0.0015s   error=None
failed:     1
find_sku:   ['find_sku', 'find_sku']
compares as its own count: True · int(): 3`}
                outputLabel="what that run printed"
              />
              <div className="space-y-4">
                <h2 className="text-2xl font-black text-gray-900 dark:text-white leading-tight">
                  A run does not just say what it{" "}
                  <span className="gradient-text">answered</span>
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  Before 1.0.0, <Mono>tool_calls</Mono> was an integer: you could learn
                  that two calls happened and nothing else. It now carries the calls —
                  each one&rsquo;s name, the arguments the model chose, what came back,
                  how long it took, which turn it was on, and the error if it failed —
                  while still comparing and casting as the count, so code written
                  against the old integer keeps working.
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  That is the difference between an agent you can debug and one you have
                  to instrument first. This run took three turns: the model guessed a
                  product name the catalogue does not carry, the tool raised, it corrected
                  itself and looked the stock up — and all of that is readable from the
                  response afterwards.
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  <Mono>tool_calling_mode=&quot;react&quot;</Mono> is what makes it one
                  call per turn. In 1.0.0 a turn in which the model asks for several tools
                  at once recorded their names, and the fields beside them were filled in
                  only for a call the loop dispatched on its own turn; since 1.0.1 a batched
                  call records what it returned as well.
                </p>
              </div>
            </div>
          </Container>
        </section>

        {/* Configuring */}
        <Band
          id="config"
          eyebrow="Agent and AgentConfig"
          tinted
          title={
            <>
              Every decision an agent makes{" "}
              <span className="gradient-text">is a field on one object</span>
            </>
          }
          lede={
            <>
              <Mono>AgentConfig</Mono> carries {api.agent_config.length} fields, and{" "}
              <Mono>model</Mono> is the only one you have to supply. The rest have
              defaults that make a useful agent, and the ones below are what a first
              agent is usually written with. Every type and default in the tables is
              read out of the installed class.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <CodeSample
                language="python"
                accent="#00ff88"
                code={`from effgen import Agent, AgentConfig

with Agent(AgentConfig(
    model="openai:gpt-5-nano",
    system_prompt="Answer in one short sentence.",
)) as agent:
    r = agent.run("What is an agent loop?")

print(r.output)
print("success:", r.success, "· iterations:", r.iterations)
print("tokens:", r.tokens_used, "· cost: $%.6f" % r.metadata["cost_usd"])`}
                output={`An agent loop is the continuous cycle in which an agent perceives its environment, reasons and updates its internal state, selects and executes an action, and then observes the outcome to repeat the process.
success: True · iterations: 1
tokens: 322 · cost: $0.000122`}
                outputLabel="what that printed"
              />
              <Card title="The context manager is the point">
                <p>
                  An agent holds a model handle, an HTTP client and, on a local engine,
                  weights. <Mono>with</Mono> closes all of it at the end of the block;{" "}
                  <Mono>agent.close()</Mono> does the same thing where a{" "}
                  <Mono>with</Mono> does not fit, and <Mono>await agent.aclose()</Mono>{" "}
                  is the async form.
                </p>
                <p>
                  <Mono>run()</Mono> is synchronous. <Mono>run_async()</Mono> is the
                  coroutine, <Mono>stream()</Mono> yields the answer as it arrives,{" "}
                  <Mono>run_batch()</Mono> takes a list of tasks, and{" "}
                  <Mono>run_background()</Mono> hands the work to a worker thread and
                  returns a task id.
                </p>
              </Card>
            </div>
            <div className="space-y-8">
              {configGroups.map((group) => (
                <div key={group.id}>
                  <h3 className="text-base font-bold text-gray-900 dark:text-white">
                    {group.title}
                  </h3>
                  <p className="mt-1 mb-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {group.what}
                  </p>
                  <ParamTable
                    nameLabel="Parameter"
                    params={group.fields.map((name) => {
                      const field = configField(name);
                      return {
                        name: field.name,
                        type: field.type,
                        default: field.default ?? undefined,
                        required: field.required,
                        description: "",
                      };
                    })}
                  />
                </div>
              ))}
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Types and defaults from <Mono>AgentConfig</Mono> in effGen {version}. The
                other {api.agent_config.length - configGroups.flatMap((g) => g.fields).length}{" "}
                fields cover sub-agents, routing, multimodal input, prompt caching and the
                callbacks a human-in-the-loop run uses.
              </p>
            </div>
          </div>
        </Band>

        {/* The response */}
        <Band
          id="response"
          eyebrow="AgentResponse"
          title={
            <>
              What comes back is a record of the run,{" "}
              <span className="gradient-text">not just its answer</span>
            </>
          }
          lede={
            <>
              <Mono>run()</Mono> returns an <Mono>AgentResponse</Mono>. It is imported
              from <Mono>effgen.core.agent</Mono> rather than the top-level package,
              because you rarely name the type — you read fields off what you were
              handed.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div>
              <ParamTable
                nameLabel="Field"
                params={responseNotes.map((note) => {
                  const field = responseField(note.name);
                  return {
                    name: field.name,
                    type: field.type,
                    description: note.what,
                  };
                })}
                caption={`AgentResponse, effGen ${version}`}
              />
            </div>
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  One call, as a record
                </h3>
                <ParamTable
                  nameLabel="Attribute"
                  params={api.tool_call.map((field) => ({
                    name: field.name,
                    type: field.type,
                    description: "",
                  }))}
                  caption="ToolCall — one entry per dispatch, in the order they were made"
                />
              </div>
              <Card title="The list reads three ways">
                <p>
                  <Mono>tool_calls</Mono> is a list of those records.{" "}
                  <Mono>.names</Mono> is the call order with repeats kept,{" "}
                  <Mono>.failed</Mono> is the subset that reported an error,{" "}
                  <Mono>.by_name(&quot;calculator&quot;)</Mono> is the calls to one tool,
                  and <Mono>.total</Mono> is how many the run made.
                </p>
                <p>
                  It also compares and casts as that number, so{" "}
                  <Mono>if response.tool_calls:</Mono>,{" "}
                  <Mono>response.tool_calls == 2</Mono> and{" "}
                  <Mono>int(response.tool_calls)</Mono> all mean what they meant when the
                  field was an integer. <Mono>.to_list()</Mono> gives plain
                  dictionaries for JSON.
                </p>
              </Card>
            </div>
          </div>
        </Band>

        {/* Presets */}
        <Band
          id="presets"
          eyebrow="Presets"
          tinted
          title={
            <>
              {siteData.presets.count} agents that are{" "}
              <span className="gradient-text">already configured</span>
            </>
          }
          lede={
            <>
              A preset is an <Mono>AgentConfig</Mono> someone already wrote: a tool set,
              a system prompt, a temperature and an iteration cap chosen for one kind of
              work. <Mono>create_agent(preset, model)</Mono> builds it, and any field can
              still be overridden by keyword.
            </>
          }
        >
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {siteData.presets.items.map((preset) => (
              <article
                key={preset.name}
                className="relative rounded-xl bg-white dark:bg-gray-900/60 border border-gray-200 dark:border-gray-800 p-5"
              >
                <div
                  className="absolute left-0 top-5 bottom-5 w-0.5 rounded-full"
                  style={{ background: presetAccents[preset.name] }}
                />
                <div className="pl-4">
                  <code
                    className="text-sm font-mono font-bold"
                    style={accentTextStyle(presetAccents[preset.name])}
                  >
                    {preset.name}
                  </code>
                  <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {preset.description}
                  </p>
                  <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                    <div className="flex justify-between gap-2">
                      <dt className="text-gray-600 dark:text-gray-400">Tools</dt>
                      <dd className="font-mono text-gray-700 dark:text-gray-300">
                        {preset.tool_count}
                      </dd>
                    </div>
                    <div className="flex justify-between gap-2">
                      <dt className="text-gray-600 dark:text-gray-400">Schema cost</dt>
                      <dd className="font-mono text-gray-700 dark:text-gray-300">
                        ~{numeric(preset.approx_tokens_per_call)} tok
                      </dd>
                    </div>
                    <div className="flex justify-between gap-2">
                      <dt className="text-gray-600 dark:text-gray-400">Temperature</dt>
                      <dd className="font-mono text-gray-700 dark:text-gray-300">
                        {preset.temperature}
                      </dd>
                    </div>
                    <div className="flex justify-between gap-2">
                      <dt className="text-gray-600 dark:text-gray-400">Max turns</dt>
                      <dd className="font-mono text-gray-700 dark:text-gray-300">
                        {preset.max_iterations}
                      </dd>
                    </div>
                  </dl>
                </div>
              </article>
            ))}
          </div>

          <div className="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <CodeSample
              language="python"
              accent="#00ff88"
              code={`from effgen import create_agent, list_presets

for name, description in list_presets().items():
    print(f"{name:12} {description[:60]}")

agent = create_agent("math", "openai:gpt-5-nano")
r = agent.run("What is the 12th Fibonacci number?")
print()
print(r.output)
print("tools:", [t.name for t in agent.config.tools])
print("calls:", r.tool_calls.names)
agent.close()`}
              output={`math         Mathematical reasoning agent with Calculator and PythonREPL.
research     Research agent with WebSearch, URLFetch, Wikipedia, academic
coding       Coding agent with CodeExecutor, PythonREPL, FileOperations, 
general      General-purpose agent with a broad set of built-in tools, in
rag          Retrieval-Augmented Generation agent with hybrid search over
minimal      Minimal agent with no tools — direct model inference only.
multimodal   Multimodal agent for image, audio, and video understanding. 
notify       Notification agent that can send emails (SMTP), read email (
media        Media processing agent with AudioTranscribeTool (speech-to-t

144
tools: ['calculator', 'python_repl']
calls: ['calculator', 'calculator', 'calculator', 'python_repl']`}
              outputLabel="what that printed"
            />
            <div className="space-y-4">
              <Card title="The schema cost is why the list is short">
                <p>
                  Every tool a preset carries sends its JSON schema on every request. The{" "}
                  <Mono>general</Mono> preset&rsquo;s{" "}
                  {siteData.presets.items.find((p) => p.name === "general")?.tool_count} tools
                  cost about{" "}
                  {numeric(
                    siteData.presets.items.find((p) => p.name === "general")
                      ?.approx_tokens_per_call ?? 0,
                  )}{" "}
                  tokens of context before the task is even stated, which is most of a
                  small model&rsquo;s window.{" "}
                  <Mono>math</Mono> costs about{" "}
                  {numeric(
                    siteData.presets.items.find((p) => p.name === "math")
                      ?.approx_tokens_per_call ?? 0,
                  )}
                  . That figure is on every card above for the same reason it is in{" "}
                  <Mono>effgen presets</Mono>: it decides which models a preset fits.
                </p>
              </Card>
              <Card title="A domain instead of a preset">
                <p>
                  <Mono>create_agent(domain=LegalDomain())</Mono> builds the same object
                  from a knowledge domain instead — its system prompt, its tool names and
                  its guardrails. Legal, health, finance, science and tech ship;{" "}
                  <Mono>Domain</Mono> is the base class for one of your own.
                </p>
              </Card>
            </div>
          </div>
        </Band>

        {/* Tools */}
        <Band
          id="tools"
          eyebrow="Tools"
          title={
            <>
              A tool is a function the model may call,{" "}
              <span className="gradient-text">and there are four ways to supply one</span>
            </>
          }
          lede={
            <>
              Whichever route it arrives by, a tool is awaited as{" "}
              <Mono>await tool.execute(**kwargs)</Mono> and hands back a{" "}
              <Mono>ToolResult</Mono>. Inside an agent you never call it yourself — the
              loop does, and the record of that call lands in{" "}
              <Mono>response.tool_calls</Mono>.
            </>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
            {toolRoutes.map((route) => (
              <div
                key={route.title}
                className="relative rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5"
              >
                <div
                  className="absolute left-0 top-5 bottom-5 w-0.5 rounded-full"
                  style={{ background: route.accent }}
                />
                <div className="pl-4">
                  <h3 className="text-sm font-bold text-gray-900 dark:text-white">
                    {route.title}
                  </h3>
                  <p className="mt-1.5 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                    {route.what}
                  </p>
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  Your own function, decorated
                </h3>
                <CodeSample
                  language="python"
                  accent="#00e5ff"
                  code={SAMPLE_TOOL_DECORATOR}
                  output={`success: True
output:  EUR 17.88
error:   None
took:    0.0005s
EUR 17.88 for express shipping of a 2.5 kg parcel.

If you'd like, I can check the standard (non-express) rate or other options as well.
calls: ['shipping_cost']`}
                  outputLabel="what that printed"
                />
              </div>
              <div className="rounded-2xl border border-yellow-500/30 bg-yellow-500/[0.05] p-5">
                <div className="flex items-start gap-3">
                  <FiAlertTriangle
                    className="mt-0.5 shrink-0 text-yellow-600 dark:text-yellow-400"
                    size={18}
                  />
                  <div>
                    <h4 className="text-sm font-bold text-gray-900 dark:text-white">
                      <Mono>ToolResult</Mono> has no <Mono>data</Mono> field, and it is
                      not indexable
                    </h4>
                    <p className="mt-1.5 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                      Read what a tool returned from <Mono>result.output</Mono>, after
                      checking <Mono>result.success</Mono>. Arguments go in as keywords —{" "}
                      <Mono>
                        execute(operation=&quot;search&quot;, query=&quot;...&quot;)
                      </Mono>{" "}
                      — not as one dictionary. Both are worth stating plainly, because
                      the other shape has been written down often enough to look
                      plausible, and it raises.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  What a tool hands back
                </h3>
                <ParamTable
                  nameLabel="Attribute"
                  params={api.tool_result.map((field) => ({
                    name: field.name,
                    type: field.type,
                    description: "",
                  }))}
                  caption={`ToolResult, effGen ${version}`}
                />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  The schema is read from the signature
                </h3>
                <CodeSample
                  language="python"
                  accent="#a78bfa"
                  code={SAMPLE_FROM_FUNCTION}
                  output={`seat_count · Count the seats in a rectangular block of a theatre.
computation
  rows: integer (required=True) — The rows parameter.
  seats_per_row: integer (required=True) — The seats_per_row parameter.`}
                  outputLabel="what that printed"
                />
                <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  The name comes from the function, the description from the docstring
                  and the parameters from the type hints, so the schema the model sees
                  cannot drift from the function that runs.{" "}
                  <Mono>requires_approval=True</Mono> marks a tool with a real-world side
                  effect, which the approval gate below then holds.
                </p>
              </div>
              <RouteLink
                to="/docs/tools/gallery"
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
              >
                All {siteData.tools.count} built-in tools
                <FiArrowRight size={14} />
              </RouteLink>
            </div>
          </div>
        </Band>

        {/* Middleware */}
        <Band
          id="middleware"
          eyebrow="Middleware"
          tinted
          title={
            <>
              {api.middleware_hooks.length} places to put behaviour{" "}
              <span className="gradient-text">the framework does not ship</span>
            </>
          }
          lede={
            <>
              An approval gate, a cache, a redaction pass, a per-run spend cap, a trace
              exporter of your own. Subclass <Mono>AgentMiddleware</Mono>, override the
              hooks you need, and pass the instance in{" "}
              <Mono>AgentConfig(middleware=[...])</Mono>. Every hook has a default that
              does nothing, so overriding one leaves the other five as they were.
            </>
          }
        >
          <ParamTable
            nameLabel="Hook"
            params={api.middleware_hooks.map((hook) => ({
              name: hook.name,
              type: hook.signature,
              description: hook.what,
            }))}
            caption="AgentMiddleware — the order they fire in"
            className="mb-10"
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  One of your own
                </h3>
                <CodeSample
                  language="python"
                  accent="#00ff88"
                  code={SAMPLE_CUSTOM_MIDDLEWARE}
                  output={`{'datetime': 0.0007}`}
                  outputLabel="what that printed"
                />
              </div>
              <Card title="Editing, short-circuiting and ordering">
                <p>
                  A <Mono>before_</Mono> hook receives a context it may edit in place.
                  Return <Mono>None</Mono> and the call goes ahead with whatever the hook
                  left behind; return anything else and the real call does not happen —
                  the returned value is used as its result, and the matching{" "}
                  <Mono>after_</Mono> hook still runs. An <Mono>after_</Mono> hook returns
                  the result to use, so it can transform as well as observe.
                </p>
                <p>
                  <Mono>before_</Mono> hooks run in the order given and{" "}
                  <Mono>after_</Mono> hooks in reverse, so middleware nest the way context
                  managers do. A hook that raises is not caught, which is what lets a
                  refusing gate stop the run outright.{" "}
                  <Mono>run(..., middleware=[...])</Mono> adds to the configured list for
                  that one call. The list itself is held by a{" "}
                  <Mono>MiddlewareChain</Mono>, which is what walks it around each of the
                  three points.
                </p>
              </Card>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  The two that ship
                </h3>
                <div className="space-y-3 mb-6">
                  {api.middleware_shipped.map((middleware) => (
                    <div
                      key={middleware.name}
                      className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-4"
                    >
                      <code className="text-sm font-mono font-bold text-gray-900 dark:text-white">
                        {middleware.name}
                      </code>
                      <p className="mt-1 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                        {middleware.what}
                      </p>
                    </div>
                  ))}
                </div>
                <CodeSample
                  language="python"
                  accent="#ffd700"
                  code={SAMPLE_LOGGING_MIDDLEWARE}
                  output={`run start: What is the current date in UTC?
model call: openai:gpt-5-nano (attempt 1)
tool call: datetime({"operation": "now", "timezone": "UTC"})
model call: openai:gpt-5-nano (attempt 1)
run end: success=True tool_calls=1
{'datetime': '2026-08-22 09:51:41', 'date': '2026-08-22', 'time': '09:51:41', 'timezone': 'UTC', 'day_of_week': 'Saturday', 'week_number': 34, 'timestamp': 1787392301}`}
                  outputLabel="what that printed"
                />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  A refusal is reported to the model
                </h3>
                <CodeSample
                  language="python"
                  accent="#ff6b6b"
                  code={SAMPLE_APPROVAL}
                  output={`asked about: ['issue_refund']
issue_refund -> This call was not approved, so the tool did not run.`}
                  outputLabel="what that printed"
                />
                <p className="mt-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  The refusal reaches the model as the call&rsquo;s result, so the run
                  continues and can say what it was not allowed to do — rather than the
                  tool returning nothing and the answer being written as though the
                  refund had happened.
                </p>
              </div>
              <a
                href={DOCS_MIDDLEWARE_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
              >
                docs/guides/middleware.md
                <FiExternalLink size={13} />
              </a>
            </div>
          </div>
        </Band>

        {/* Conversations */}
        <Band
          id="conversations"
          eyebrow="Sessions, memory and compaction"
          title={
            <>
              One agent, many conversations,{" "}
              <span className="gradient-text">and a history that has to fit</span>
            </>
          }
          lede={
            <>
              An agent remembers its own conversation by default.{" "}
              <Mono>session_id=</Mono> on the constructor binds one conversation to the
              agent for its whole life; <Mono>session=</Mono> on <Mono>run()</Mono> names
              the conversation per call, which is what a server handling many people
              wants.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <CodeSample
                language="python"
                accent="#00ff88"
                code={SAMPLE_SESSION}
                output={`Pixel
Mote`}
                outputLabel="what that printed"
              />
              <Card title="What a session is, and where it lives">
                <p>
                  The run builds its prompt from that conversation&rsquo;s history and
                  appends the turn to it. The two conversations never see each other, and
                  the agent&rsquo;s own memory is untouched and restored when the call
                  ends — including when the run fails. Without <Mono>session=</Mono>,{" "}
                  <Mono>run()</Mono> uses the agent&rsquo;s own memory as before.
                </p>
                <p>
                  Sessions are JSON under <Mono>~/.effgen/sessions/</Mono>, written
                  atomically so a crash mid-write cannot leave a truncated file that
                  fails to load later. Every turn is stamped with the model, the token
                  counts, the cost and the latency it was answered with. A{" "}
                  <Mono>Session</Mono> object works wherever an id does, and{" "}
                  <Mono>effgen sessions</Mono> lists, reads, exports and cleans them up.
                </p>
              </Card>
              <CodeSample
                language="python"
                accent="#00e5ff"
                code={SAMPLE_MEMORY}
                output={`Ubuntu 24.04
messages in memory: 6
tokens held: 34`}
                outputLabel="what that printed"
              />
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  When the history stops fitting, something has to go
                </h3>
                <p className="mb-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  Which turns survive changes the answer more for a small model than for
                  a large one, and different tasks want different answers — so the choice
                  is a strategy rather than a fixed rule. Each one takes its thresholds
                  from the memory&rsquo;s own settings when you do not name them.
                </p>
                <div className="space-y-3">
                  {api.compaction.map((strategy) => (
                    <div
                      key={strategy.name}
                      className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-4"
                    >
                      <div className="flex flex-wrap items-baseline gap-2">
                        <code className="text-sm font-mono font-bold text-gray-900 dark:text-white">
                          {strategy.name}
                        </code>
                        <span className="text-[11px] font-mono text-gray-600 dark:text-gray-400">
                          {strategy.parameters
                            .map((parameter) => `${parameter.name}=${parameter.default}`)
                            .join(", ")}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                        {strategy.what}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
              <CodeSample
                language="python"
                accent="#a78bfa"
                code={SAMPLE_COMPACTION}
                output={`INV-8842
messages held: 14
tokens held:   68`}
                outputLabel="what that printed"
              />
              <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                The invoice number was in the first turn, which is the one{" "}
                <Mono>KeepFirstAndLast</Mono> is built to hold on to. A strategy can also
                be named as a string —{" "}
                <Mono>compaction_strategy=&quot;drop_oldest&quot;</Mono> — and{" "}
                <Mono>tokenizer=</Mono> measures the history in the units the
                model&rsquo;s window is actually measured in, rather than estimating four
                characters to a token.
              </p>
              <div className="flex flex-wrap gap-4">
                <a
                  href={DOCS_SESSIONS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  docs/guides/sessions-and-checkpoints.md
                  <FiExternalLink size={13} />
                </a>
                <a
                  href={DOCS_COMPACTION_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  docs/guides/context-compaction.md
                  <FiExternalLink size={13} />
                </a>
              </div>
            </div>
          </div>
        </Band>

        {/* Retrieval */}
        <Band
          id="knowledge"
          eyebrow="Retrieval"
          tinted
          title={
            <>
              Point an agent at documents,{" "}
              <span className="gradient-text">and the answer carries its sources</span>
            </>
          }
          lede={
            <>
              The <Mono>rag</Mono> preset takes a <Mono>knowledge_base=</Mono> and runs
              the pipeline behind it: ingestion, chunking, hybrid search, reranking and
              attribution. The response then carries <Mono>sources</Mono> and{" "}
              <Mono>citations</Mono> next to the answer, so a claim can be traced back to
              the passage it came from.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <CodeSample
              language="python"
              accent="#5eead4"
              code={SAMPLE_RAG}
              output={`The nightly export starts at 02:15 UTC. [1]
sources: ['/tmp/tmp8vuzhhe1/runbook.md', '/tmp/tmp8vuzhhe1/contacts.md']
citations: ['/tmp/tmp8vuzhhe1/runbook.md', '/tmp/tmp8vuzhhe1/contacts.md']`}
              outputLabel="what that printed"
            />
            <div className="space-y-4">
              <Card title="The pieces, when the preset is not the shape you want">
                <p>
                  <Mono>DocumentIngester</Mono> reads a directory — text, Markdown, JSON,
                  JSONL, CSV and HTML built in, PDF, DOCX and EPUB with their optional
                  dependencies — recursing and de-duplicating on a content hash.
                </p>
                <p>
                  Four chunkers cover what a flat splitter gets wrong:{" "}
                  <Mono>SemanticChunker</Mono> splits on meaning,{" "}
                  <Mono>CodeChunker</Mono> on functions and classes,{" "}
                  <Mono>TableChunker</Mono> keeps a table whole, and{" "}
                  <Mono>HierarchicalChunker</Mono> keeps a document&rsquo;s structure.
                </p>
                <p>
                  <Mono>HybridSearchEngine</Mono> combines dense and keyword retrieval;
                  the rerankers are a cross-encoder, a rule-based pass and a model as
                  judge. <Mono>CitationTracker</Mono> is what puts the <Mono>[1]</Mono>{" "}
                  in the answer above.
                </p>
              </Card>
              <Card title="Memory is a different thing">
                <p>
                  Memory is what this agent has said and heard. Retrieval is a corpus it
                  can look things up in. <Mono>ShortTermMemory</Mono> holds the
                  conversation, <Mono>LongTermMemory</Mono> keeps entries across runs with
                  an importance level, and <Mono>VectorMemoryStore</Mono> searches them by
                  similarity rather than by recency.
                </p>
              </Card>
              <a
                href={DOCS_RAG_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
              >
                docs/tutorials/rag-pipeline.md
                <FiExternalLink size={13} />
              </a>
            </div>
          </div>
        </Band>

        {/* Orchestration */}
        <Band
          id="orchestration"
          eyebrow="Teams and workflows"
          title={
            <>
              Several agents, and a pipeline that{" "}
              <span className="gradient-text">does not start again from the top</span>
            </>
          }
          lede={
            <>
              A team runs agents under one of {api.orchestration_patterns.length} patterns
              and aggregates what they produced. A <Mono>WorkflowDAG</Mono> is the
              explicit form: nodes, edges, independent nodes running in parallel, and a
              cycle rejected when the graph is built rather than when it runs.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  <FiUsers
                    className="inline mb-0.5 mr-1.5 text-green-700 dark:text-green-400"
                    size={15}
                  />
                  A team, and what each agent contributed
                </h3>
                <CodeSample
                  language="python"
                  accent="#00ff88"
                  code={SAMPLE_TEAM}
                  output={`A checkpoint is a saved state that can be restored later.
pattern: sequential · agents: 2
  writer: A checkpoint is a saved state or snapshot of a process, syst (436 tokens, $0.000168)
  editor: A checkpoint is a saved state that can be restored later. (514 tokens, $0.000190)`}
                  outputLabel="what that printed"
                />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  The patterns
                </h3>
                <ul className="flex flex-wrap gap-2">
                  {api.orchestration_patterns.map((pattern) => (
                    <li
                      key={pattern}
                      className="px-2.5 py-1 rounded-lg text-xs font-mono bg-gray-100 dark:bg-gray-900 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-800"
                    >
                      {pattern}
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  A team reports success only when at least one agent ran and every agent
                  that ran succeeded. On a failure it never echoes the input back as the
                  answer — a caller reading <Mono>.output</Mono> without checking{" "}
                  <Mono>.success</Mono> must not mistake the task for a result. Partial
                  work stays readable in <Mono>agent_responses</Mono>.
                </p>
                <a
                  href={DOCS_MULTI_AGENT_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  docs/tutorials/multi-agent.md
                  <FiExternalLink size={13} />
                </a>
              </div>
            </div>

            <div className="space-y-6">
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  <FiSave
                    className="inline mb-0.5 mr-1.5 text-green-700 dark:text-green-400"
                    size={15}
                  />
                  A workflow that resumes
                </h3>
                <p className="mb-4 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  Give <Mono>run()</Mono> a store and a run id. There is no separate
                  resume call: a run id the store has never seen starts from the
                  beginning, and one it knows continues. Below, the third node fails on
                  the first pass; running the same line again re-runs only that node, and
                  the two that finished are restored from the checkpoint at{" "}
                  <Mono>0.0s</Mono> without calling a model.
                </p>
                <CodeSample
                  language="python"
                  accent="#ffd700"
                  code={SAMPLE_WORKFLOW}
                  output={`first run : [('research', 'completed', 3.672), ('draft', 'completed', 3.817), ('review', 'failed', 0.002)] success: False
checkpoint: completed ['draft', 'research'] failed ['review']
resumed   : [('research', 'completed', 0.0), ('draft', 'completed', 0.0), ('review', 'completed', 4.392)] success: True`}
                  outputLabel="what that printed"
                />
              </div>
              <div>
                <h3 className="text-base font-bold text-gray-900 dark:text-white mb-3">
                  What resuming does with each node
                </h3>
                <ParamTable
                  nameLabel="State when the run stopped"
                  params={resumeRows.map((row) => ({
                    name: row.state,
                    description: row.onResume,
                  }))}
                  caption="Progress is saved after each topological level, atomically."
                />
              </div>
              <Card title="Where a checkpoint is kept">
                {api.checkpoint_stores.map((store) => (
                  <p key={store.name}>
                    <Mono>{store.name}</Mono> — {store.what}
                  </p>
                ))}
                <p>
                  The store holds run state, never the graph: agents own sockets, model
                  handles and credentials, none of which survive a process boundary.
                  Rebuild the same <Mono>WorkflowDAG</Mono> in the new process and hand it
                  the same run id. Resuming into a graph with different node ids raises
                  rather than mixing two workflows&rsquo; outputs, and re-running a
                  finished run replays its stored outputs without calling a model — which
                  makes a workflow idempotent under a job runner that retries.
                </p>
              </Card>
            </div>
          </div>
        </Band>

        {/* Output */}
        <Band
          id="output"
          eyebrow="Structured output and streaming"
          tinted
          title={
            <>
              Ask for a shape,{" "}
              <span className="gradient-text">or ask for it as it arrives</span>
            </>
          }
          lede={
            <>
              <Mono>output_schema=</Mono> validates the answer against a JSON Schema
              before the run reports success, so a caller that parses the result is not
              parsing prose that happens to look like JSON. <Mono>stream()</Mono> yields
              the answer as it is generated, with the intermediate steps delivered to
              callbacks rather than mixed into the text.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <CodeSample
              language="python"
              accent="#00e5ff"
              code={SAMPLE_STRUCTURED}
              output={`{"city":"Granada","country":"Spain","construction_began":1238}
1238
structured: True`}
              outputLabel="what that printed"
            />
            <div className="space-y-6">
              <CodeSample
                language="python"
                accent="#a78bfa"
                code={SAMPLE_STREAM}
                output={`Red, blue and yellow.
chunks: 6
joined == answer: True`}
                outputLabel="what that printed"
              />
              <Card title="What the stream contains">
                <p>
                  Iterating yields answer-text deltas, and joining every chunk
                  reconstructs the final answer — on the no-tool path and the tool path
                  alike. The loop&rsquo;s own scaffolding is never part of the text. On a
                  tool-using run the intermediate steps arrive through the{" "}
                  <Mono>on_thought</Mono>, <Mono>on_tool_call</Mono> and{" "}
                  <Mono>on_observation</Mono> callbacks, or as typed events with{" "}
                  <Mono>include_events=True</Mono>.
                </p>
              </Card>
            </div>
          </div>
        </Band>

        {/* Failure */}
        <Band
          id="failure"
          eyebrow="When it does not work"
          title={
            <>
              A tool that raised{" "}
              <span className="gradient-text">is on the record, not swallowed</span>
            </>
          }
          lede={
            <>
              A tool that fails does not take the run down: the error is reported to the
              model, which can try something else or say what it could not do. It is also
              kept on the call record — so a run that produced a plausible answer over a
              failed lookup can be found afterwards, instead of being indistinguishable
              from one that worked.
            </>
          }
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <CodeSample
              language="python"
              accent="#ff6b6b"
              code={SAMPLE_FAILURE}
              output={`success: True · iterations: 2
read_ledger · ok: False · error: Error executing tool 'read_ledger': Tool execution failed: no ledger for 55-2
failed calls: 1
partial_output: None`}
              outputLabel="what that printed"
            />
            <div className="space-y-4">
              <Card title="raise_on_error defaults to True since 1.0.0">
                <p>
                  One of 1.0.0&rsquo;s three breaking changes. Before, a run that
                  failed returned a response whose <Mono>success</Mono> was{" "}
                  <Mono>False</Mono> — and a caller that read <Mono>.output</Mono> without
                  checking got an error string treated as an answer. It now raises by
                  default. Pass <Mono>raise_on_error=False</Mono> for the old behaviour,
                  as the sample here does.
                </p>
              </Card>
              <Card title="A run that stops early says so">
                <p>
                  Hitting <Mono>max_iterations</Mono>, or a guard ending a run that
                  repeats itself, is not success. <Mono>success</Mono> is{" "}
                  <Mono>False</Mono>, <Mono>outcome</Mono> is{" "}
                  <Mono>&quot;stopped&quot;</Mono>, <Mono>stop_reason</Mono> names the
                  exit, and <Mono>.partial</Mono> carries what the run had reached — so
                  the work is not lost, and it is not mistaken for a finished answer.
                  Under the default <Mono>raise_on_error=True</Mono> the run raises{" "}
                  <Mono>RunStoppedError</Mono>, which carries the same three.
                </p>
              </Card>
              <Card title="An unreachable backend has no opt-out">
                <p>
                  A task that ran and failed is something you can inspect. A connection
                  that was refused is not. So a backend that never answered raises{" "}
                  <Mono>BackendUnreachableError</Mono> whatever{" "}
                  <Mono>raise_on_error</Mono> says — the third breaking change, and the
                  one that stops a whole batch completing against nothing and looking
                  healthy in the summary.
                </p>
                <RouteLink
                  to="/models"
                  className="inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  What that looks like
                  <FiArrowRight size={14} />
                </RouteLink>
              </Card>
            </div>
          </div>
        </Band>

        {/* Documentation */}
        <section className="py-16 relative">
          {SECTION_DIVIDER}
          <Container className="relative z-10">
            <div className="rounded-2xl border border-green-500/25 bg-green-500/[0.04] p-8 md:p-10 flex flex-col md:flex-row md:items-center gap-6 justify-between">
              <div className="max-w-2xl">
                <h2 className="text-2xl font-black text-gray-900 dark:text-white">
                  The reference for the <span className="gradient-text">library</span>
                </h2>
                <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  Every signature, every default and every exception, plus the guides
                  behind the sections above: middleware, sessions and checkpoints,
                  compaction, the tool registry, and the multi-agent patterns.
                </p>
                <a
                  href={DOCS_TOOLS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-flex items-center gap-1.5 text-sm font-semibold text-green-700 dark:text-green-400"
                >
                  <FiTool size={13} />
                  docs/tools/index.md — writing and registering tools
                  <FiExternalLink size={13} />
                </a>
              </div>
              <div className="flex shrink-0 flex-col gap-3">
                <a
                  href={DOCS_REFERENCE_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-green-500 hover:bg-green-400 text-black font-bold text-sm transition-colors"
                >
                  <FiFileText size={15} />
                  docs/api/reference.md
                  <FiExternalLink size={14} />
                </a>
                <RouteLink
                  to="/production"
                  className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl border border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-green-500/50 font-semibold text-sm transition-colors"
                >
                  Running it for real
                  <FiArrowRight size={14} />
                </RouteLink>
              </div>
            </div>
          </Container>
        </section>
      </main>
      <Footer />
    </div>
  );
}
