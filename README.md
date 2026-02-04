<div align="center">

<!-- Animated Header -->
<img src="assets/header.svg" alt="effGen" width="100%"/>

<br/>

<!-- Badges -->
<a href="https://arxiv.org/abs/2602.00887"><img src="https://img.shields.io/badge/arXiv-2602.00887-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"/></a>
<a href="https://pypi.org/project/effgen/"><img src="https://img.shields.io/pypi/v/effgen.svg?style=for-the-badge&logo=pypi&logoColor=white&color=3775A9" alt="PyPI"/></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License"/></a>

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

### 📦 From PyPI (Recommended)

```bash
pip install effgen
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

**🔄**<br/>
Multi-Model<br/>
<sub>HF, OpenAI, etc.</sub>

</td>
<td align="center" width="14%">

**🔧**<br/>
Tool Integration<br/>
<sub>MCP, A2A, ACP</sub>

</td>
<td align="center" width="14%">

**🧩**<br/>
Task Decomp<br/>
<sub>Auto breakdown</sub>

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

**🔒**<br/>
Sandboxed<br/>
<sub>Docker exec</sub>

</td>
</tr>
</table>

</div>

---

## 🛠️ Built-in Tools

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
<sub>RAG Search</sub>

</td>
<td align="center" width="14%">

**🎯**<br/>
AgenticSearch<br/>
<sub>Exact Match</sub>

</td>
</tr>
</table>

</div>

---

## 📚 Examples

```bash
python examples/basic_agent.py      # Basic agent (Transformers backend)

python examples/basic_agent_vllm.py # Basic agent (vLLM backend - 5-10x faster)

python examples/web_agent.py        # Web search agent

python examples/retrieval_agent.py  # RAG-based retrieval

python examples/agentic_search_agent.py # Grep-based agentic search
```

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

### Retrieval Agent

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Retrieval

model = load_model("Qwen/Qwen2.5-1.5B-Instruct")
retrieval_tool = Retrieval(knowledge_base_path="./docs")

config = AgentConfig(
    name="qa_agent",
    model=model,
    tools=[retrieval_tool]
)

agent = Agent(config=config)
result = agent.run("What does the documentation say about configuration?")
```

</details>

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

MIT License — see [LICENSE](LICENSE) for details.

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
