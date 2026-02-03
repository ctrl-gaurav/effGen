# effGen Examples

Ready-to-run examples to get started with effGen.

## Examples

| Example | Description |
|---------|-------------|
| `basic_agent.py` | Calculator agent with code execution tools |
| `web_agent.py` | Web search agent using DuckDuckGo |
| `retrieval_agent.py` | Knowledge-base Q&A using embedding-based RAG |
| `agentic_search_agent.py` | Knowledge-base Q&A using grep-based exact matching |

## Quick Start

```bash
# Run basic calculator example
python examples/basic_agent.py

# Run web search example
python examples/web_agent.py

# Run retrieval examples (requires ARC dataset)
python examples/data/download_arc.py --output-dir examples/data
python examples/retrieval_agent.py
python examples/agentic_search_agent.py
```

## Retrieval Tools

effGen provides two powerful retrieval tools for knowledge-base access:

### Retrieval (RAG)
Embedding-based semantic search. Best for:
- Finding conceptually similar content
- Questions where exact keywords may not appear in the answer
- General knowledge retrieval

### AgenticSearch
Grep-based exact matching. Best for:
- Technical queries with specific terms
- Finding exact phrases, numbers, or formulas
- Large knowledge bases where indexing is impractical

## Configuration

Each example has a `DETAILED_LOGGING` flag at the top:
- Set to `True` for verbose debug output
- Set to `False` for minimal logging

## Requirements

- effGen installed (`pip install effgen`)
- GPU recommended (uses 4-bit quantization)
- For retrieval examples: `pip install datasets` (to download ARC dataset)
