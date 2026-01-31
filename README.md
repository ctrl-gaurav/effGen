<div align="center">

# effGen: Enabling Small Language Models as Capable Autonomous Agents

[![PyPI version](https://img.shields.io/pypi/v/effgen.svg)](https://pypi.org/project/effgen/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/effgen.svg)](https://pypi.org/project/effgen/)
[![GitHub stars](https://img.shields.io/github/stars/ctrl-gaurav/effGen?style=social)](https://github.com/ctrl-gaurav/effGen)

[**Website**](https://effgen.org/) | [**Documentation**](https://effgen.org/docs/) | [**Paper**](https://arxiv.org/) | [**PyPI**](https://pypi.org/project/effgen/)

</div>

---

## What is effGen?

**effGen** is a framework that makes Small Language Models (1B-7B parameters) work as powerful AI agents. While most agent frameworks require massive LLMs, effGen is optimized from the ground up for efficient, smaller models.

```python
from effgen import Agent, load_model

# Create an agent with any model
agent = Agent(
    model=load_model("Qwen/Qwen2.5-3B-Instruct"),
    tools=["calculator", "web_search", "code_executor"]
)

# Run tasks
result = agent.run("Search for the latest AI news and summarize the top 3 stories")
print(result.output)
```

---

## Installation

### From PyPI (Recommended)

```bash
pip install effgen
```

### With vLLM for faster inference

```bash
pip install effgen[vllm]
```

### From Source

```bash
git clone https://github.com/ctrl-gaurav/effGen.git
cd effGen

# Option 1: Quick install (recommended)
./install.sh

# Option 2: Quick install for CI (no animations)
./install.sh --quick

# Option 3: Full install (includes vLLM, dev tools)
./install.sh --full

# Option 4: Manual install
pip install -e .
```

---

## Quick Start

### CLI Usage

```bash
# Run a task
effgen run "What is the capital of France?"

# Interactive chat
effgen chat

# Start API server
effgen serve --port 8000
```

### Python API

```python
from effgen import Agent
from effgen.core.agent import AgentConfig

# Configure your agent
config = AgentConfig(
    name="MyAgent",
    model="Qwen/Qwen2.5-3B-Instruct",
    tools=["calculator", "web_search"],
    enable_memory=True
)

# Create and run
agent = Agent(config)
result = agent.run("Calculate 15% tip on $85.50")
```

---

## Features

| Feature | Description |
|---------|-------------|
| **SLM Optimized** | Prompt engineering and techniques designed for 1B-7B models |
| **Multi-Model** | Supports HuggingFace, OpenAI, Anthropic, Gemini, vLLM |
| **Tool Integration** | Built-in tools + MCP, A2A, ACP protocol support |
| **Task Decomposition** | Automatic breakdown of complex tasks |
| **Multi-Agent** | Coordinate multiple specialized agents |
| **Memory Systems** | Short-term and long-term memory |
| **Sandboxed Execution** | Safe code execution with Docker |

---

## Examples

See the [`examples/`](examples/) directory:

```bash
python examples/basic_agent.py   # Calculator agent with tools
python examples/web_agent.py     # Web search agent
```

---

## Security

effGen provides secure execution options:

- **Docker Sandbox**: Isolated code execution
- **Input Validation**: Automatic sanitization
- **Rate Limiting**: Configurable limits on tool usage

For security policies and vulnerability reporting, see [SECURITY.md](SECURITY.md).

---

## Citation

```bibtex
@software{effgen2026,
  title = {effGen: Enabling Small Language Models as Capable Autonomous Agents},
  author = {Srivastava, Gaurav and Hussain, Aafiya and Wang, Chi and Lin, Yingyan and Wang, Xuan},
  year = {2026},
  url = {https://github.com/ctrl-gaurav/effGen},
  version = {0.0.1}
}
```

---

## Links

- **Website**: [effgen.org](https://effgen.org/)
- **Documentation**: [effgen.org/docs](https://effgen.org/docs/)
- **PyPI**: [pypi.org/project/effgen](https://pypi.org/project/effgen/)
- **Issues**: [GitHub Issues](https://github.com/ctrl-gaurav/effGen/issues)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Get Started](https://effgen.org/docs/)** | **[Examples](examples/)** | **[GitHub](https://github.com/ctrl-gaurav/effGen)**

Made with care for the AI community

</div>
