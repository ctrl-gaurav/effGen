<div align="center">

# effGen: Enabling Small Language Models as Capable Autonomous Agents

[![arXiv](https://img.shields.io/badge/arXiv-2602.00887-b31b1b.svg)](https://arxiv.org/abs/2602.00887)
[![PyPI version](https://img.shields.io/pypi/v/effgen.svg)](https://pypi.org/project/effgen/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/effgen.svg)](https://pypi.org/project/effgen/)
[![GitHub stars](https://img.shields.io/github/stars/ctrl-gaurav/effGen?style=social)](https://github.com/ctrl-gaurav/effGen)

[**Paper**](https://arxiv.org/abs/2602.00887) | [**Website**](https://effgen.org/) | [**Documentation**](https://effgen.org/docs/) | [**PyPI**](https://pypi.org/project/effgen/)

</div>

---

## News & Releases

| Date | Update |
|------|--------|
| **Feb 2026** | v0.0.1 released with Retrieval and AgenticSearch tools |
| **Feb 2026** | Paper published on arXiv: [EffGen: Enabling Small Language Models as Capable Autonomous Agents](https://arxiv.org/abs/2602.00887) |
| **Feb 2026** | Initial release of effGen framework |

---

## What is effGen?

**effGen** is a framework that makes Small Language Models (1B-7B parameters) work as powerful AI agents. While most agent frameworks require massive LLMs, effGen is optimized from the ground up for efficient, smaller models.

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, PythonREPL

# Load model
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
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Load model
model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")

# Configure your agent
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

## Built-in Tools

| Tool | Description |
|------|-------------|
| `Calculator` | Mathematical calculations and unit conversions |
| `WebSearch` | Web search using DuckDuckGo (no API key required) |
| `CodeExecutor` | Safe code execution in sandboxed environment |
| `PythonREPL` | Interactive Python execution |
| `FileOperations` | File read/write/list operations |
| `Retrieval` | RAG-based semantic search over knowledge bases |
| `AgenticSearch` | Grep-based exact matching for precise retrieval |

---

## Examples

See the [`examples/`](examples/) directory:

```bash
python examples/basic_agent.py # Calculator agent with tools

python examples/web_agent.py # Web search agent

python examples/retrieval_agent.py # RAG-based retrieval agent

python examples/agentic_search_agent.py # Grep-based agentic search
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

If you use effGen in your research, please cite our paper:

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

## Links

- **Paper**: [arXiv:2602.00887](https://arxiv.org/abs/2602.00887)
- **Website**: [effgen.org](https://effgen.org/)
- **Documentation**: [effgen.org/docs](https://effgen.org/docs/)
- **PyPI**: [pypi.org/project/effgen](https://pypi.org/project/effgen/)
- **Issues**: [GitHub Issues](https://github.com/ctrl-gaurav/effGen/issues)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Get Started](https://effgen.org/docs/)** | **[Examples](examples/)** | **[Paper](https://arxiv.org/abs/2602.00887)** | **[GitHub](https://github.com/ctrl-gaurav/effGen)**

Made with care for the AI community

</div>
