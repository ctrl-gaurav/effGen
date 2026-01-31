"""
Memory systems for effGen.

This package provides short-term, long-term, and vector-based memory systems
for managing conversation history, persistent data, and semantic search.
"""

from .short_term import (
    ShortTermMemory,
    Message,
    MessageRole,
    ConversationSummary
)

from .long_term import (
    LongTermMemory,
    MemoryEntry,
    Session,
    MemoryType,
    ImportanceLevel,
    StorageBackend,
    JSONStorageBackend,
    SQLiteStorageBackend
)

from .vector_store import (
    VectorMemoryStore,
    VectorMemoryEntry,
    SearchResult,
    EmbeddingProvider,
    SentenceTransformerEmbedding,
    SimpleEmbedding,
    VectorStoreBackend,
    FAISSBackend,
    ChromaBackend,
    EmbeddingModel,
    FAISS_AVAILABLE,
    CHROMA_AVAILABLE
)

__all__ = [
    # Short-term memory
    "ShortTermMemory",
    "Message",
    "MessageRole",
    "ConversationSummary",

    # Long-term memory
    "LongTermMemory",
    "MemoryEntry",
    "Session",
    "MemoryType",
    "ImportanceLevel",
    "StorageBackend",
    "JSONStorageBackend",
    "SQLiteStorageBackend",

    # Vector store
    "VectorMemoryStore",
    "VectorMemoryEntry",
    "SearchResult",
    "EmbeddingProvider",
    "SentenceTransformerEmbedding",
    "SimpleEmbedding",
    "VectorStoreBackend",
    "FAISSBackend",
    "ChromaBackend",
    "EmbeddingModel",
    "FAISS_AVAILABLE",
    "CHROMA_AVAILABLE",
]
