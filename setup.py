"""
effGen: Framework for building agents with Small Language Models

A comprehensive framework that enables Small Language Models to function as
powerful agentic systems through tool integration, advanced prompt engineering,
and smart sub-agent decomposition.
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
version = {}
with open(os.path.join("effgen", "__init__.py"), "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version["__version__"] = line.split("=")[1].strip().strip('"').strip("'")
            break

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements, filtering out comments and empty lines
def read_requirements(filename):
    """Read requirements from file, filtering comments and empty lines."""
    requirements = []
    if not os.path.exists(filename):
        return requirements

    with open(filename, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            # Skip empty lines, comments, and -r references
            if line and not line.startswith("#") and not line.startswith("-r"):
                requirements.append(line)
    return requirements

install_requires = read_requirements("requirements.txt")
dev_requires = read_requirements("requirements-dev.txt")

setup(
    name="effgen",
    version=version.get("__version__", "0.0.1"),
    author="Gaurav Srivastava",
    author_email="gks@vt.edu",
    description="A comprehensive framework for building agents with Small Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ctrl-gaurav/effGen",
    project_urls={
        "Homepage": "https://effgen.org/",
        "Documentation": "https://effgen.org/docs/",
        "Source Code": "https://github.com/ctrl-gaurav/effGen",
        "Bug Tracker": "https://github.com/ctrl-gaurav/effGen/issues",
        "Changelog": "https://github.com/ctrl-gaurav/effGen/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Framework :: AsyncIO",
        "Typing :: Typed",
    ],
    keywords=[
        "ai", "agents", "llm", "slm", "language-models", "small-language-models",
        "tool-use", "function-calling", "prompt-engineering", "multi-agent",
        "agent-framework", "transformers", "vllm", "openai", "anthropic", "gemini",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "vllm": [
            "vllm>=0.2.7",
        ],
        "flash-attn": [
            "flash-attn>=2.3.0",
        ],
        "vector-db": [
            "faiss-cpu>=1.7.4",
            "chromadb>=0.4.18",
            "qdrant-client>=1.7.0",
        ],
        "search": [
            "duckduckgo-search>=3.9.0",
            "google-search-results>=2.4.2",
            "google-api-python-client>=2.108.0",
        ],
        "cloud-secrets": [
            "boto3>=1.28.0",
            "hvac>=1.2.0",
            "azure-keyvault-secrets>=4.7.0",
            "azure-identity>=1.14.0",
        ],
        "monitoring": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
        "all": [
            # Include all optional dependencies
            "vllm>=0.2.7",
            "flash-attn>=2.3.0",
            "faiss-cpu>=1.7.4",
            "chromadb>=0.4.18",
            "qdrant-client>=1.7.0",
            "duckduckgo-search>=3.9.0",
            "google-search-results>=2.4.2",
            "google-api-python-client>=2.108.0",
            "boto3>=1.28.0",
            "hvac>=1.2.0",
            "azure-keyvault-secrets>=4.7.0",
            "azure-identity>=1.14.0",
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "effgen=effgen.cli:main",
            "effgen-agent=effgen.cli:main",
            "effgen-web=effgen.cli:web_agent_main",
        ],
    },
    include_package_data=True,
    package_data={
        "effgen": [
            "prompts/templates/*.yaml",
            "prompts/templates/*.json",
            "config/schemas/*.json",
            "config/schemas/*.yaml",
            "config/*.yaml",
            "config/*.json",
            "py.typed",  # PEP 561 marker for type checking
        ],
    },
    zip_safe=False,  # Don't install as a zip file
)
