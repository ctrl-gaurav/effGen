# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Streaming response support for real-time token generation
- Enhanced sub-agent execution with true multi-agent coordination
- Semantic similarity for improved response selection
- Advanced caching strategies and performance optimization

---

## [0.0.2] - 2026-02-03

### Added
- **Retrieval Tool**: RAG-based semantic search tool for knowledge base Q&A
- **Agentic Search Tool**: Grep-based exact match search with async support

### Fixed
- **vLLM Backend**: Fixed automatic chat template support for instruction-tuned models
- **GPU Memory Control**: Improved `gpu_memory_utilization` parameter handling
- **OOM Error Handling**: Better error messages and suggestions for CUDA out-of-memory errors
- **Tensor Parallel Auto-Selection**: Fixed auto-detection of tensor parallel size for small models (1.7B, 4B, etc.)
- **vLLM Cache Directory**: Resolved issues with vLLM cache directory handling

### Changed
- **Model Loader**: Improved small model detection for tensor parallel size selection
- **Version Management**: Consolidated `__version__` to single source in main `effgen/__init__.py`

### Compatibility
- Tested with multiple model families:
  - Qwen (Qwen3-1.7B, Qwen2.5-3B-Instruct)
  - Meta Llama (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct)
  - Microsoft Phi (Phi-4-mini-instruct)
  - HuggingFace SmolLM (SmolLM2-1.7B-Instruct, SmolLM3-3B)
  - Google Gemma (Gemma-3-4b-it)

---

## [0.0.1] - 2026-01-31

### Added

#### Core Framework
- **Agent System**: Complete agentic framework optimized for Small Language Models (1B-7B parameters)
- **Task Management**: Task and SubTask classes with priority levels and status tracking
- **Agent State**: Comprehensive state management for agent execution
- **ReAct Pattern**: Reasoning and Acting pattern implementation for structured problem-solving

#### Model Support
- **Multi-Backend Support**:
  - HuggingFace Transformers (local models)
  - vLLM (fast inference with 5-10x speedup)
  - OpenAI API adapter
  - Anthropic API adapter
  - Google Gemini API adapter
- **Model Loader**: Automatic model detection and loading with intelligent fallback
- **Generation Configuration**: Flexible configuration for temperature, tokens, sampling, etc.

#### Tool System
- **Built-in Tools**:
  - Calculator (basic math, conversions, financial calculations)
  - Web Search (DuckDuckGo integration with caching)
  - Code Executor (Python, JavaScript, Bash in sandboxed environment)
  - File Operations (read, write, list, search)
  - Python REPL (interactive Python execution)
- **Tool Registry**: Dynamic tool registration and discovery
- **Protocol Support**:
  - MCP (Model Context Protocol) - Official Anthropic SDK integration
  - A2A (Agent-to-Agent) protocol
  - ACP (Agent Communication Protocol)

#### Prompt Engineering
- **Template Manager**: Jinja2-based template system with versioning
- **Chain Manager**: Multi-step prompt chaining with conditional execution
- **Prompt Optimizer**: SLM-specific optimization techniques
- **Few-Shot Learning**: Dynamic example selection for improved performance

#### Memory Systems
- **Short-Term Memory**: Conversation history and context management
- **Long-Term Memory**: Persistent storage with importance-based retrieval
- **Vector Store**: Semantic search with FAISS, ChromaDB, and Qdrant support
- **Storage Backends**: JSON and SQLite storage options

#### Task Decomposition
- **Complexity Analysis**: Automatic task complexity assessment
- **Decomposition Engine**: Break complex tasks into manageable subtasks
- **Sub-Agent Manager**: Specialized sub-agents for different task types
- **Orchestrator**: Coordinate multi-agent execution with parallel/sequential strategies

#### GPU Management
- **GPU Allocator**: Intelligent GPU allocation with memory requirements
- **GPU Monitor**: Real-time monitoring of utilization, temperature, and power
- **Multi-GPU Support**: Automatic distribution across available GPUs

#### Code Execution
- **Sandboxed Execution**: Safe code execution with Docker containers
- **Code Validator**: Static analysis and security checks
- **Multiple Languages**: Support for Python, JavaScript, Bash, and more
- **Resource Limits**: Configurable CPU, memory, and timeout limits

#### Configuration
- **YAML Configuration**: Hierarchical configuration with validation
- **JSON Schema Validation**: Type-safe configuration with comprehensive schemas
- **Environment Variables**: Secure secret management with .env support
- **Cloud Secrets**: AWS Secrets Manager, HashiCorp Vault, Azure Key Vault integration

#### CLI Interface
- **Interactive Chat**: Real-time chat interface with rich formatting
- **One-Shot Execution**: Direct task execution from command line
- **API Server**: FastAPI-based REST API server
- **Web Agent**: Autonomous web browsing and interaction
- **Tool Management**: List, inspect, and test tools

#### Utilities
- **Logging System**: Rich, structured logging with multiple levels and formats
- **Metrics Tracking**: Performance metrics, token usage, and cost tracking
- **Error Handling**: Comprehensive error handling with retry logic
- **Async Support**: Full async/await support for concurrent operations

#### Examples & Documentation
- **Basic Agent Example**: Simple agent with calculator and web search
- **Web Agent Example**: Agent that can browse and extract information
- **Installation Script**: Interactive installer with animations
- **Security Policy**: Comprehensive security guidelines and vulnerability reporting

### Configuration Files
- `pyproject.toml`: Modern Python packaging with build system configuration
- `setup.py`: Traditional setuptools configuration for compatibility
- `.gitignore`: Comprehensive ignore patterns for Python, IDEs, and system files
- `requirements.txt`: Core dependencies with version specifications

### Package Metadata
- **License**: MIT License
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Development Status**: Alpha
- **Keywords**: ai, agents, llm, slm, language-models, tool-use, multi-agent

### Optional Dependencies
- `dev`: Development tools (pytest, black, isort, flake8, mypy)
- `vllm`: Fast inference engine
- `flash-attn`: Flash Attention for faster transformer inference
- `vector-db`: Vector database backends (FAISS, ChromaDB, Qdrant)
- `search`: Advanced search engines (Google, DuckDuckGo)
- `cloud-secrets`: Cloud secret management (AWS, Azure, Vault)
- `monitoring`: Experiment tracking (Weights & Biases, TensorBoard)
- `all`: All optional dependencies combined

### Entry Points
- `effgen`: Main CLI entry point
- `effgen-agent`: Agent-specific commands
- `effgen-web`: Web agent interface

---

## Version History

### Version Naming Convention
- **Major.Minor.Patch** (Semantic Versioning)
- **Major**: Breaking changes, major new features
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, minor improvements

### Release Schedule
- **Patch releases**: As needed for critical bugs
- **Minor releases**: Monthly feature updates
- **Major releases**: Quarterly for significant changes

---

## Links

- **GitHub**: https://github.com/ctrl-gaurav/effGen
- **PyPI**: https://pypi.org/project/effgen/
- **Documentation**: https://effgen.org/docs/
- **Issues**: https://github.com/ctrl-gaurav/effGen/issues

---

## Contributors

Thank you to all contributors who helped make effGen possible!

- Gaurav Srivastava (@ctrl-gaurav) - Creator and maintainer

---

[Unreleased]: https://github.com/ctrl-gaurav/effGen/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/ctrl-gaurav/effGen/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/ctrl-gaurav/effGen/releases/tag/v0.0.1
