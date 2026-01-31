# effGen Examples

Ready-to-run examples to get started with effGen.

## Examples

| Example | Description |
|---------|-------------|
| `basic_agent.py` | Calculator agent with code execution tools |
| `web_agent.py` | Web search agent using DuckDuckGo |

## Quick Start

```bash
# Run basic calculator example
python examples/basic_agent.py

# Run web search example
python examples/web_agent.py
```

## Configuration

Each example has a `DETAILED_LOGGING` flag at the top:
- Set to `True` for verbose debug output
- Set to `False` for minimal logging

## Requirements

- effGen installed (`pip install effgen`)
- GPU recommended (uses 4-bit quantization)
