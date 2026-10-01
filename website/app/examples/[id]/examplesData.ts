// The six examples, keyed by the id in the URL.
//
// Each one pairs a script that ships in the framework repository with a short
// program built from it that was run on this machine, verbatim as shown, against
// the released package. `output` is that run's stdout, pasted — never edited to
// agree with the prose, and never written by hand.
//
// The data lives here rather than in the view so the route can enumerate the ids
// it generates for the static export, and so the teaser on the landing page can
// read the titles and accents from the same place the detail pages render.

export interface ExampleRun {
  /** The program, exactly as it was run. */
  code: string;
  /** What that run printed. */
  output: string;
  /** The model the run used. */
  model: string;
}

export interface Example {
  id: string;
  icon: string;
  title: string;
  subtitle: string;
  badge: string;
  accent: string;
  /** What the example does, in a paragraph. */
  description: string;
  /** What the run above demonstrates, one point each. */
  observations: string[];
  /** The tools it wires in, under the names the registry uses. */
  tools: string[];
  /** The script in the framework repository this is built from. */
  script: string;
  githubUrl: string;
  /** The command that runs the full script. */
  command: string;
  run: ExampleRun;
}

const REPO = "https://github.com/ctrl-gaurav/effGen/blob/main";

export const examples: Example[] = [
  {
    id: "code-assistant",
    icon: "💻",
    title: "Code assistant",
    subtitle: "Writes a program, runs it, and reports what it saw rather than what it expected.",
    badge: "Code execution",
    accent: "#00e5ff",
    description:
      "The point of a coding agent is not that it writes code — it is that it runs the code before telling you it works. This agent has two execution tools and an iteration budget: it writes a function, executes it on a real input, reads the output, and answers with what came back. If the code raises, it sees the traceback and tries again.",
    observations: [
      "The answer is the string the executed program printed, not the model's prediction of it.",
      "response.tool_calls carries every call the run made, including the ones that failed, so you can see which path the agent took.",
      "CodeExecutor runs in a sandbox: a container where one is available, and an unprivileged subprocess namespace otherwise. Code that exits non-zero returns success=False with the reason.",
    ],
    tools: ["python_repl", "code_executor"],
    script: "examples/tools/coding_agent.py",
    githubUrl: `${REPO}/examples/tools/coding_agent.py`,
    command: "effgen examples run tools/coding_agent",
    run: {
      model: "gemini:gemini-3.1-flash-lite",
      code: `from effgen import Agent, AgentConfig
from effgen.tools.builtin import CodeExecutor, PythonREPL

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    name="code-assistant",
    system_prompt=(
        "You are a coding assistant. Write the code, run it with a tool, "
        "read the output, and report the result you actually saw."
    ),
    tools=[PythonREPL(), CodeExecutor()],
    max_iterations=8,
))

response = agent.run(
    "Write a Python function that returns the longest palindromic substring "
    "of a string, run it on 'forgeeksskeegfor', and report what it printed."
)

print(response.text)
print()
for call in response.tool_calls:
    print(f"  {call.name} -> {'error' if call.error else 'ok'}")`,
      output: `geeksskeeg

  code_executor -> ok`,
    },
  },
  {
    id: "research-agent",
    icon: "🔍",
    title: "Research agent",
    subtitle: "Answers from what the search actually returned, and hands back the URLs.",
    badge: "Information retrieval",
    accent: "#a78bfa",
    description:
      "A search tool and an instruction not to answer from memory. What makes the result usable is not the prose: it is that response.sources carries the URLs the run retrieved, so you can check the answer against them. Since 1.0.0 these are separate fields — .sources is everything the search returned, .citations is what the answer referenced.",
    observations: [
      "The URLs are the ones the tool returned on this run. Run it tomorrow and they will be different, because the web is.",
      "WebSearch defaults to a backend that needs no API key, so this example runs with nothing configured.",
      "The system prompt is what keeps the model on the retrieved text. Without it, a model will happily answer from what it remembers.",
    ],
    tools: ["web_search"],
    script: "examples/web_retrieval/web_agent.py",
    githubUrl: `${REPO}/examples/web_retrieval/web_agent.py`,
    command: "effgen examples run web_retrieval/web_agent",
    run: {
      model: "gemini:gemini-3.1-flash-lite",
      code: `from effgen import Agent, AgentConfig
from effgen.tools.builtin import WebSearch

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    name="research-agent",
    system_prompt="Answer from what the search returned. Do not answer from memory.",
    tools=[WebSearch()],
    max_iterations=6,
))

response = agent.run("What is the Open-Meteo API, and does it need an API key?")

print(response.text)
print()
print("sources:", len(response.sources))
for url in response.sources[:3]:
    print(" ", url)`,
      output: `The Open-Meteo API is a service that provides access to historical, current, and forecasted weather data [1, 4]. It is designed to be easy to use and is a popular choice for projects requiring weather information [3, 4].

Regarding authentication, the Open-Meteo API does not require an API key for development or low-volume production use [2, 5]. It allows users to access its services without the need for registration, signup, or credit card information [3, 4]. However, users should review the current terms for heavy commercial workloads [2].

sources: 5
  https://freeapihub.com/apis/open-meteo-historical
  https://apideposu.com/en/blog/build-weather-widget-open-meteo-nextjs
  https://freeapi.watch/open-meteo/`,
    },
  },
  {
    id: "data-analysis",
    icon: "📊",
    title: "Data analysis",
    subtitle: "Reads a file, computes on it, and reports the number the tools produced.",
    badge: "Multi-tool pipeline",
    accent: "#00ff88",
    description:
      "Four tools that chain: file operations to read, a JSON tool to query and validate, a Python REPL to compute, and text processing to summarise. The agent picks which it needs. The value of running the sum through a REPL rather than asking the model for it is that arithmetic on a small model is where errors come from, and a tool does not guess.",
    observations: [
      "Two tool calls, in the order the agent chose them: read the file, then compute over what it read.",
      "FileOperations confines reads to an allowed directory. A path outside it is refused by name rather than read.",
      "The REPL session persists across calls within a run, so a variable defined in one call is available in the next.",
    ],
    tools: ["json_tool", "text_processing", "python_repl", "file_operations"],
    script: "examples/advanced/data_processing_agent.py",
    githubUrl: `${REPO}/examples/advanced/data_processing_agent.py`,
    command: "effgen examples run advanced/data_processing_agent",
    run: {
      model: "gemini:gemini-3.1-flash-lite",
      code: `import json
from pathlib import Path

from effgen import Agent, AgentConfig
from effgen.tools.builtin import FileOperations, JSONTool, PythonREPL, TextProcessingTool

Path("orders.json").write_text(json.dumps({
    "orders": [
        {"id": 1, "region": "emea", "total": 240.5},
        {"id": 2, "region": "amer", "total": 1204.0},
        {"id": 3, "region": "emea", "total": 87.25},
    ]
}))

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    name="data-analysis",
    system_prompt="Use the tools to read and compute. Report the numbers the tools returned.",
    tools=[JSONTool(), TextProcessingTool(), PythonREPL(), FileOperations()],
    max_iterations=8,
))

response = agent.run(
    "Read orders.json, and report the total value of the emea orders."
)

print(response.text)
print()
print("tool calls:", response.tool_calls.total)
for call in response.tool_calls:
    print(" ", call.name, "->", "error" if call.error else "ok")`,
      output: `The total value of the emea orders is 327.75.

tool calls: 2
  file_operations -> ok
  python_repl -> ok`,
    },
  },
  {
    id: "multi-agent",
    icon: "🤖",
    title: "Multi-agent pipeline",
    subtitle: "Three agents, each with one job, wired end to end.",
    badge: "Orchestration",
    accent: "#ff6b6b",
    description:
      "One agent restates the task, one solves it with a tool, one writes it up. Each has a narrow system prompt and only the tools it needs, which is what makes a small model reliable at its step. The framework also has a sub-agent router that decomposes a task on its own, and workflow DAGs with resumable checkpoints — but a pipeline you wired yourself is the version you can debug.",
    observations: [
      "A plain run() no longer fans out into sub-agents on its own — AgentConfig.mode defaults to SINGLE. Automatic decomposition is opt-in.",
      "The solver has a calculator and an instruction to give the number only. The arithmetic is the tool's, not the model's.",
      "Each step's output is just a string, so there is nothing to learn: the pipeline is three calls in a row.",
    ],
    tools: ["calculator"],
    script: "examples/advanced/multi_agent_pipeline.py",
    githubUrl: `${REPO}/examples/advanced/multi_agent_pipeline.py`,
    command: "effgen examples run advanced/multi_agent_pipeline",
    run: {
      model: "gemini:gemini-3.1-flash-lite",
      code: `from effgen import Agent, AgentConfig
from effgen.tools.builtin import Calculator

MODEL = "gemini:gemini-3.1-flash-lite"

analyst = Agent(AgentConfig(
    model=MODEL,
    name="analyst",
    system_prompt="Restate the task as one arithmetic question. Nothing else.",
))
solver = Agent(AgentConfig(
    model=MODEL,
    name="solver",
    system_prompt="Answer with the calculator. Give the number only.",
    tools=[Calculator()],
))
writer = Agent(AgentConfig(
    model=MODEL,
    name="writer",
    system_prompt="Write one sentence reporting the result to a manager.",
))

question = analyst.run("A team of 14 people each bill 37 hours at $145. What is the invoice?")
answer = solver.run(question.text)
summary = writer.run(f"Question: {question.text}\\nAnswer: {answer.text}")

print("analyst:", question.text.strip()[:120])
print("solver :", answer.text.strip())
print("writer :", summary.text.strip())`,
      output: `analyst: 14 * 37 * 145 = ?
solver : 75110
writer : The calculation of 14 * 37 * 145 results in a total of 75,110.`,
    },
  },
  {
    id: "weather-json-pipeline",
    icon: "🌤️",
    title: "Weather, with no API key",
    subtitle: "A live third-party API an agent can reach with nothing configured.",
    badge: "External API",
    accent: "#ffd700",
    description:
      "The weather tool goes to Open-Meteo, which needs no key, so this is the shortest path from a fresh install to an agent that calls something real. The JSON tool is there for when you want to reshape the response rather than read it. It is a small example, and the reason it is on this page is that it is the one you can run first.",
    observations: [
      "One tool call. The arguments are the model's, chosen from the tool's schema — a location string and an operation.",
      "The temperature, conditions and wind speed in the answer are the ones the API returned at the moment of the run.",
      "Nothing in this example reads an environment variable, so there is no key to get wrong.",
    ],
    tools: ["weather", "json_tool"],
    script: "examples/web_retrieval/weather_agent.py",
    githubUrl: `${REPO}/examples/web_retrieval/weather_agent.py`,
    command: "effgen examples run web_retrieval/weather_agent",
    run: {
      model: "gemini:gemini-3.1-flash-lite",
      code: `from effgen import Agent, AgentConfig
from effgen.tools.builtin import JSONTool, WeatherTool

agent = Agent(AgentConfig(
    model="gemini:gemini-3.1-flash-lite",
    name="weather-agent",
    system_prompt=(
        "Use the weather tool for current conditions. Report the temperature, "
        "the conditions and the wind speed."
    ),
    tools=[WeatherTool(), JSONTool()],
    max_iterations=6,
))

response = agent.run("What is the weather in Blacksburg, Virginia right now?")

print(response.text)
print()
print("tool calls:", response.tool_calls.total)
for call in response.tool_calls:
    print(" ", call.name, call.arguments)`,
      output: `The current weather in Blacksburg, Virginia is as follows:

*   **Temperature:** 25.1°C
*   **Conditions:** Partly cloudy
*   **Wind Speed:** 14.7 km/h

tool calls: 1
  weather {"location": "Blacksburg, Virginia", "operation": "current"}`,
    },
  },
  {
    id: "rag-knowledge-base",
    icon: "📚",
    title: "Retrieval over your own documents",
    subtitle: "One call builds the index; the answer comes back with the source it used.",
    badge: "RAG",
    accent: "#ff9500",
    description:
      "create_agent(\"rag\", model, knowledge_base=...) ingests the path you give it and wires a retrieval tool over it. Retrieval is hybrid — dense embeddings and BM25 — and the answer carries inline markers backed by response.citations, each naming the source and its relevance score. Ask it something the documents do not cover and it says so rather than inventing an answer.",
    observations: [
      "The preset refuses to build without a knowledge base, rather than succeeding over zero documents. Leave the argument out and it names the argument it needs.",
      "The [1] in the answer is backed by a Citation object: the source file, the chunk, the score and, for a PDF, the page.",
      "Ingestion reports what it skipped and why — a corrupt file, an empty one, a duplicate, an unsupported extension each have their own reason.",
    ],
    tools: ["retrieval"],
    script: "examples/web_retrieval/retrieval_agent.py",
    githubUrl: `${REPO}/examples/web_retrieval/retrieval_agent.py`,
    command: "effgen examples run web_retrieval/retrieval_agent",
    run: {
      model: "gemini:gemini-3.1-flash-lite",
      code: `from pathlib import Path

from effgen.presets import create_agent

Path("handbook.md").write_text(
    "# Support handbook\\n\\n"
    "Refunds are issued within 14 days of purchase.\\n"
    "Priority support answers within 4 business hours.\\n"
    "The free tier allows 60 API requests an hour.\\n"
)

agent = create_agent(
    "rag",
    "gemini:gemini-3.1-flash-lite",
    knowledge_base="handbook.md",
)

response = agent.run("How long do I have to ask for a refund?")

print(response.text)
print()
for citation in response.citations:
    print(f"[{citation.index}] {citation.source}  score={citation.relevance_score}")`,
      output: `You are eligible to request a refund within 14 days of your purchase [1].

[1] handbook.md  score=0.7`,
    },
  },
];

/** Keyed by the id in the URL, which is what the route and the teaser look up. */
export const examplesData: Record<string, Example> = Object.fromEntries(
  examples.map((example) => [example.id, example]),
);
