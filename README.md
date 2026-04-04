<div align="center">

<!-- Animated Header -->
<img src="assets/header.svg" alt="effGen" width="100%"/>

<br/>

<br/>

<!-- Badges -->
<a href="https://github.com/ctrl-gaurav/effGen/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/ctrl-gaurav/effGen/ci.yml?branch=main&style=for-the-badge&logo=github&label=CI" alt="CI"/></a>
<a href="https://arxiv.org/abs/2602.00887"><img src="https://img.shields.io/badge/arXiv-2602.00887-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/></a>
<a href="https://pypi.org/project/effgen/"><img src="https://img.shields.io/pypi/v/effgen.svg?style=for-the-badge&logo=pypi&logoColor=white&color=3775A9" alt="PyPI"/></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg?style=for-the-badge" alt="License"/></a>

<a href="https://pypi.org/project/effgen/"><img src="https://img.shields.io/pypi/dm/effgen.svg?style=for-the-badge&logo=pypi&logoColor=white&color=orange" alt="Downloads"/></a>
<a href="https://github.com/ctrl-gaurav/effGen"><img src="https://img.shields.io/github/stars/ctrl-gaurav/effGen?style=for-the-badge&logo=github&color=yellow" alt="Stars"/></a>
<a href="https://github.com/ctrl-gaurav/effGen/fork"><img src="https://img.shields.io/github/forks/ctrl-gaurav/effGen?style=for-the-badge&logo=github&color=blue" alt="Forks"/></a>

<!-- Quick Links -->
<a href="https://arxiv.org/abs/2602.00887"><img src="https://img.shields.io/badge/📄_Read_Paper-FF6B6B?style=for-the-badge" alt="Paper"/></a>
<a href="https://effgen.org/"><img src="https://img.shields.io/badge/🌐_Website-4ECDC4?style=for-the-badge" alt="Website"/></a>
<a href="https://effgen.org/docs/"><img src="https://img.shields.io/badge/📚_Documentation-45B7D1?style=for-the-badge" alt="Docs"/></a>
<a href="https://pypi.org/project/effgen/"><img src="https://img.shields.io/badge/📦_PyPI-96CEB4?style=for-the-badge" alt="PyPI"/></a>

<!-- Typing Animation -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=6C63FF&center=true&vCenter=true&multiline=true&repeat=true&width=600&height=80&lines=Build+AI+Agents+with+Small+Language+Models;Fast+%E2%80%A2+Efficient+%E2%80%A2+Powerful" alt="Typing SVG" />

</div>

---

## 📰 News & Updates

| | Date | Update |
|:---:|:---|:---|
| 🍎 | **4 Apr 2026** | **Apple Silicon + GUI Tools**: MLX & MLX-VLM backends for native Metal GPU acceleration, 6 Gradio GUI apps (Agent Builder, Chat, Agent Visualizer, Benchmark Dashboard, Tool Builder, Tool Tester), Python 3.14 support |
| 🔧 | **25 Mar 2026** | **v0.1.3 Released**: Verification hardening — smarter loop detection, "skip the tool" prompting, model-aware token counting, sub-agent depth limits, circuit breaker persistence. [See changelog](CHANGELOG.md#013---2026-03-25) |
| 🔧 | **12 Mar 2026** | **v0.1.2 Released**: Test-driven hardening — 10 example agents, 19 bug fixes, cross-model compatibility matrix (11 models, 73% pass rate). [See changelog](CHANGELOG.md#012---2026-03-12) |
| 🔒 | **6 Mar 2026** | **v0.1.1 Released**: Stabilization — fixed license/metadata consistency, improved error handling, added 6 examples, expanded test suite. [See changelog](CHANGELOG.md#011---2026-03-06) |
| 🎉 | **1 Mar 2026** | **v0.1.0 Released**: Major feature release — 14 built-in tools, agent presets, plugin system, real streaming, memory integration, ACP/MCP protocols, CI/CD, and comprehensive test suite. [See changelog](CHANGELOG.md#010---2026-03-01) |
| 🔧 | **3 Feb 2026** | **v0.0.2 Released**: vLLM backend fixes with automatic chat template support, GPU memory control, improved OOM error handling, and multi-model family compatibility |
| 📄 | **2 Feb 2026** | Preprint available: [EffGen: Enabling Small Language Models as Capable Autonomous Agents](https://arxiv.org/abs/2602.00887) |
| 🚀 | **31 Jan 2026** | Initial release of effGen framework **(v0.0.1)** |

---

## 🤔 What is effGen?

**effGen** transforms Small Language Models into powerful AI agents. While most frameworks require massive LLMs, effGen is **optimized from the ground up** for efficient, smaller models — delivering fast, capable agents without the compute overhead.

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

# Load a small but mighty model
model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")

# Create agent with tools
config = AgentConfig(
    name="math_agent",
    model=model,
    tools=[Calculator(), PythonREPL()]
)
agent = Agent(config=config)

# Run computation
result = agent.run("What is 24344 * 334?")
print(f"Answer: {result.output}")
```

---

## ⚡ Installation

> **Requires Python 3.10 or newer.** Tested on Python 3.10, 3.11, 3.12, 3.13, 3.14.

### 📦 From PyPI (Recommended)

```bash
pip install effgen
```

### 🍎 Apple Silicon (MLX — Recommended for Mac)

```bash
pip install effgen[mlx]          # Text models on Apple Silicon
pip install effgen[mlx-vlm]      # Vision-Language models on Apple Silicon
```

### 🚀 With vLLM for Faster Inference

```bash
pip install effgen[vllm]
```

### 🔧 From Source

```bash
git clone https://github.com/ctrl-gaurav/effGen.git
cd effGen

# Quick install
./install.sh

# Full install (includes vLLM + dev tools)
./install.sh --full

# Manual install
pip install -e .
```

---

## 🚀 Quick Start

### 💻 CLI Usage

```bash
# Run a task
effgen run "What is the capital of France?"

# Interactive chat
effgen chat

# Start API server
effgen serve --port 8000

# List available presets
effgen presets

# Check infrastructure health
effgen health

# Interactive wizard
effgen
```

### 🐍 Python API

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Load model
model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")

# Configure agent
config = AgentConfig(
    name="calculator_agent",
    model=model,
    tools=[Calculator()],
    system_prompt="You are a helpful math assistant."
)

# Create and run
agent = Agent(config=config)
result = agent.run("Calculate 15% tip on $85.50")
print(result.output)
```

### 🍎 Apple Silicon (MLX)

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Load MLX model — native Metal GPU, unified memory, no CPU-GPU transfer
model = load_model("LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit", engine="mlx")

config = AgentConfig(
    name="mlx_agent",
    model=model,
    tools=[Calculator()],
)
agent = Agent(config=config)
result = agent.run("What is sqrt(144) + 2^10?")
print(result.output)
```

---

## ✨ Features

<div align="center">

<table>
<tr>
<td align="center" width="14%">

**🧠**<br/>
SLM Optimized<br/>
<sub>For smaller models</sub>

</td>
<td align="center" width="14%">

**🍎**<br/>
Apple Silicon<br/>
<sub>MLX + Metal GPU</sub>

</td>
<td align="center" width="14%">

**🔧**<br/>
Tool Integration<br/>
<sub>MCP, A2A, ACP</sub>

</td>
<td align="center" width="14%">

**🖥️**<br/>
6 GUI Apps<br/>
<sub>Gradio dashboards</sub>

</td>
<td align="center" width="14%">

**👥**<br/>
Multi-Agent<br/>
<sub>Coordination</sub>

</td>
<td align="center" width="14%">

**💾**<br/>
Memory<br/>
<sub>Short & Long</sub>

</td>
<td align="center" width="14%">

**🔌**<br/>
Plugins<br/>
<sub>Extensible</sub>

</td>
</tr>
</table>

</div>

---

## 🎯 Agent Presets

Get started instantly with ready-to-use agent configurations:

```python
from effgen import load_model
from effgen.presets import create_agent

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# One-line agent creation
math_agent = create_agent("math", model)       # Calculator + PythonREPL
research_agent = create_agent("research", model) # WebSearch + URLFetch + Wikipedia
coding_agent = create_agent("coding", model)     # CodeExecutor + PythonREPL + FileOps + Bash
general_agent = create_agent("general", model)   # All 11 tools
minimal_agent = create_agent("minimal", model)   # Direct inference, no tools
```

```bash
# CLI preset support
effgen run --preset math "What is sqrt(144)?"
effgen run --preset research "Tell me about quantum computing"
```

---

## 🛠️ Built-in Tools (14)

<div align="center">

<table>
<tr>
<td align="center" width="14%">

**🔢**<br/>
Calculator<br/>
<sub>Math & Units</sub>

</td>
<td align="center" width="14%">

**🌐**<br/>
WebSearch<br/>
<sub>DuckDuckGo</sub>

</td>
<td align="center" width="14%">

**💻**<br/>
CodeExecutor<br/>
<sub>Sandboxed</sub>

</td>
<td align="center" width="14%">

**🐍**<br/>
PythonREPL<br/>
<sub>Interactive</sub>

</td>
<td align="center" width="14%">

**📁**<br/>
FileOps<br/>
<sub>Read/Write</sub>

</td>
<td align="center" width="14%">

**🔍**<br/>
Retrieval<br/>
<sub>RAG + BM25</sub>

</td>
<td align="center" width="14%">

**🎯**<br/>
AgenticSearch<br/>
<sub>ripgrep</sub>

</td>
</tr>
<tr>
<td align="center" width="14%">

**🖥️**<br/>
BashTool<br/>
<sub>Shell Cmds</sub>

</td>
<td align="center" width="14%">

**🌤️**<br/>
WeatherTool<br/>
<sub>Open-Meteo</sub>

</td>
<td align="center" width="14%">

**📋**<br/>
JSONTool<br/>
<sub>Query/Validate</sub>

</td>
<td align="center" width="14%">

**🕐**<br/>
DateTimeTool<br/>
<sub>Timezones</sub>

</td>
<td align="center" width="14%">

**📝**<br/>
TextProcessing<br/>
<sub>Regex/Count</sub>

</td>
<td align="center" width="14%">

**🔗**<br/>
URLFetch<br/>
<sub>Web Scrape</sub>

</td>
<td align="center" width="14%">

**📖**<br/>
Wikipedia<br/>
<sub>Free API</sub>

</td>
</tr>
</table>

</div>

---

## 📚 Examples

### 🖥️ GUI Applications (Gradio)

```bash
# Visual agent & tool development
python examples/agent_builder/app.py              # Agent Builder — configure, test, export agents (port 7860)
python examples/basic/chat_gui_mlx.py              # MLX Chat — streaming chat with Apple Silicon models (port 7860)
python examples/basic/agent_viz_mlx.py             # Agent Visualizer — step-by-step reasoning + code editor (port 7860)
python examples/basic/benchmark_gui.py             # Benchmark Dashboard — performance & eval leaderboard (port 7862)
python examples/basic/tool_builder_gui.py          # Tool Builder — visually create custom tools (port 7863)
python examples/basic/tool_tester_gui.py           # Tool Tester — browse, test, inspect all 14 tools (port 7864)
```

### 🍎 Apple Silicon (MLX)

```bash
python examples/basic/basic_agent_mlx.py           # Basic MLX agent with calculator
python examples/basic/chat_gui_mlx.py --autoload   # Chat GUI with auto model loading
python examples/basic/agent_viz_mlx.py --autoload   # Agent visualizer with auto model loading
```

### 🤖 Core Agent Examples

```bash
python examples/basic/qa_agent.py                  # Q&A agent (no tools)
python examples/basic/calculator_agent.py          # Math with Calculator + PythonREPL
python examples/tools/advanced_multi_tool_agent.py # 5 tools + fallback chains
python examples/tools/file_operations_agent.py     # File read/write/search
python examples/tools/coding_agent.py              # Code execution + iteration
python examples/advanced/conversational_agent.py   # Multi-turn memory
python examples/advanced/advanced_streaming_agent.py # Token streaming with callbacks
python examples/advanced/data_processing_agent.py  # JSON & data pipelines
python examples/advanced/multi_agent_pipeline.py   # Multi-agent orchestration
python examples/advanced/error_recovery_agent.py   # Error handling patterns
```

### ⚡ Quick-Start Examples

```bash
python examples/basic/basic_agent.py               # Basic agent (Transformers)
python examples/basic/basic_agent_vllm.py          # Basic agent (vLLM - 5-10x faster)
python examples/plugins_presets/preset_agents.py   # Ready-to-use agent presets
python examples/web_retrieval/streaming_agent.py   # Simple streaming
python examples/web_retrieval/memory_agent.py      # Simple multi-turn memory
python examples/tools/multi_tool_agent.py          # Simple multi-tool
python examples/web_retrieval/weather_agent.py     # Weather via Open-Meteo (free)
python examples/plugins_presets/plugin_example.py  # Custom tool plugins
python examples/web_retrieval/web_agent.py         # Web search agent
python examples/web_retrieval/retrieval_agent.py   # RAG-based retrieval
```

> 📊 See [examples/compatibility_matrix.md](examples/compatibility_matrix.md) for model compatibility across all agents.

<details>
<summary><b>📖 More Examples</b></summary>

### Multi-Tool Agent

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, WebSearch, PythonREPL

model = load_model("Qwen/Qwen2.5-3B-Instruct")

config = AgentConfig(
    name="research_agent",
    model=model,
    tools=[Calculator(), WebSearch(), PythonREPL()],
    system_prompt="You are a research assistant."
)

agent = Agent(config=config)
result = agent.run("Search for the population of Tokyo and calculate what percentage it is of Japan's total population")
```

### Streaming

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")
agent = Agent(config=AgentConfig(
    name="stream_demo", model=model,
    tools=[Calculator()], enable_streaming=True
))

for token in agent.stream("What is 2 + 2?"):
    print(token, end="", flush=True)
```

### Memory (Multi-Turn)

```python
agent = Agent(config=AgentConfig(
    name="memory_demo", model=model,
    tools=[], enable_memory=True
))

agent.run("My name is Alice and I'm working on quantum computing.")
result = agent.run("What's my name and what am I working on?")
# → "Your name is Alice and you're working on quantum computing."
```

### Retrieval Agent (RAG)

```python
from effgen.tools.builtin import Retrieval

retrieval_tool = Retrieval(knowledge_base_path="./docs")
config = AgentConfig(name="qa_agent", model=model, tools=[retrieval_tool])
agent = Agent(config=config)
result = agent.run("What does the documentation say about configuration?")
```

</details>

---

## 🤖 Multi-Model Support

effGen supports **5 inference backends** and is tested across 11+ model families:

| Backend | Platform | Best For |
|---------|----------|----------|
| **MLX** | Apple Silicon (M1/M2/M3/M4) | Native Metal GPU, unified memory, 4/8-bit quantization |
| **MLX-VLM** | Apple Silicon | Vision-Language models (Qwen2-VL, LLaVA, Phi-3 Vision, 30+ architectures) |
| **vLLM** | NVIDIA GPU | High-throughput batch inference |
| **Transformers** | Any (CPU/GPU) | Universal compatibility |
| **API** | Cloud | OpenAI, Anthropic, Google Gemini |

### Top Recommended Models

| Model | Size | Compatibility |
|-------|------|---------------|
| **LFM2.5-1.2B-Instruct-MLX-8bit** | 1.2B | Apple Silicon optimized, fast agentic |
| **Qwen2.5-1.5B-Instruct** | 1.5B | 10/10 agents pass |
| **Qwen2.5-3B-Instruct** | 3B | 10/10 agents pass (recommended default) |
| **Phi-4-mini-instruct** | 3.8B | 10/10 agents pass |
| Qwen3-1.7B | 1.7B | 9.5/10 |
| Qwen2.5-7B-Instruct | 7B | 9/10 |
| Llama-3.2-3B-Instruct | 3B | 8.5/10 |

> Full matrix with 11 models x 10 agents: [compatibility_matrix.md](examples/compatibility_matrix.md)

---

## 🔒 Security

<div align="center">

<table>
<tr>
<td align="center" width="33%">

**🐳**<br/>
Docker Sandbox<br/>
<sub>Isolated execution</sub>

</td>
<td align="center" width="33%">

**🛡️**<br/>
Input Validation<br/>
<sub>Auto sanitization</sub>

</td>
<td align="center" width="33%">

**⚡**<br/>
Rate Limiting<br/>
<sub>Configurable limits</sub>

</td>
</tr>
</table>

</div>

> 📋 For security policies and vulnerability reporting, see [SECURITY.md](SECURITY.md)

---

## 📖 Citation

If you use **effGen** in your research, please cite our paper:

```bibtex
@software{srivastava2026effgen,
      title={effGen: Enabling Small Language Models as Capable Autonomous Agents},
      author={Gaurav Srivastava and Aafiya Hussain and Chi Wang and Yingyan Celine Lin and Xuan Wang},
      year={2026},
      eprint={2602.00887},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.00887},
}
```

---

## 🔗 Links

<div align="center">

<a href="https://arxiv.org/abs/2602.00887"><img src="https://img.shields.io/badge/📄_Paper-arXiv:2602.00887-b31b1b?style=for-the-badge" alt="Paper"/></a>
<a href="https://effgen.org/"><img src="https://img.shields.io/badge/🌐_Website-effgen.org-4ECDC4?style=for-the-badge" alt="Website"/></a>
<a href="https://effgen.org/docs/"><img src="https://img.shields.io/badge/📚_Docs-effgen.org/docs-45B7D1?style=for-the-badge" alt="Docs"/></a>
<a href="https://pypi.org/project/effgen/"><img src="https://img.shields.io/badge/📦_PyPI-pypi.org/project/effgen-3775A9?style=for-the-badge" alt="PyPI"/></a>
<a href="https://github.com/ctrl-gaurav/effGen/issues"><img src="https://img.shields.io/badge/🐛_Issues-GitHub-red?style=for-the-badge" alt="Issues"/></a>

</div>

---

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

<div align="center">

<a href="https://effgen.org/docs/"><img src="https://img.shields.io/badge/🚀_Get_Started-FF6B6B?style=for-the-badge" alt="Get Started"/></a>
<a href="examples/"><img src="https://img.shields.io/badge/📚_Examples-4ECDC4?style=for-the-badge" alt="Examples"/></a>
<a href="https://arxiv.org/abs/2602.00887"><img src="https://img.shields.io/badge/📄_Paper-45B7D1?style=for-the-badge" alt="Paper"/></a>
<a href="https://github.com/ctrl-gaurav/effGen"><img src="https://img.shields.io/badge/⭐_Star_on_GitHub-yellow?style=for-the-badge" alt="GitHub"/></a>

**Made with ❤️ for the AI community**

<!-- Footer Wave -->
<img src="assets/footer.svg" alt="effGen footer" width="100%"/>

</div>
