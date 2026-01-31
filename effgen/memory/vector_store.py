"""
Vector-based memory store for semantic search and retrieval.

This module provides vector-based memory storage using FAISS or Chroma,
supporting semantic search, similarity retrieval, and periodic consolidation.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum

# Optional imports for vector backends
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None


class EmbeddingModel(Enum):
    """Supported embedding models."""
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class VectorMemoryEntry:
    """
    Represents a memory entry with vector embedding.

    Attributes:
        id: Unique identifier
        content: Original text content
        embedding: Vector embedding
        metadata: Additional metadata
        timestamp: Creation timestamp
        access_count: Number of times retrieved
        last_accessed: Last access timestamp
    """
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without embedding for serialization)."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed
        }


@dataclass
class SearchResult:
    """
    Represents a search result with similarity score.

    Attributes:
        entry: The memory entry
        similarity: Similarity score (0-1, higher is better)
        rank: Rank in search results
    """
    entry: VectorMemoryEntry
    similarity: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.entry.id,
            "content": self.entry.content,
            "metadata": self.entry.metadata,
            "similarity": self.similarity,
            "rank": self.rank
        }


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize sentence-transformers embedding.

        Args:
            model_name: Name of sentence-transformers model
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self._embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb for emb in embeddings]

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim


class SimpleEmbedding(EmbeddingProvider):
    """
    Simple embedding provider using TF-IDF.
    Fallback when other providers are not available.
    """

    def __init__(self, max_features: int = 384):
        """
        Initialize simple embedding.

        Args:
            max_features: Maximum number of features (embedding dimension)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._embedding_dim = max_features
        self._fitted = False
        self._corpus = []

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if not self._fitted:
            # Fit on first text if not fitted
            self._corpus.append(text)
            self.vectorizer.fit(self._corpus)
            self._fitted = True

        vector = self.vectorizer.transform([text]).toarray()[0]
        # Ensure correct dimension
        if len(vector) < self._embedding_dim:
            vector = np.pad(vector, (0, self._embedding_dim - len(vector)))
        return vector

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self._fitted:
            self._corpus.extend(texts)
            self.vectorizer.fit(self._corpus)
            self._fitted = True

        vectors = self.vectorizer.transform(texts).toarray()
        # Ensure correct dimension
        result = []
        for vector in vectors:
            if len(vector) < self._embedding_dim:
                vector = np.pad(vector, (0, self._embedding_dim - len(vector)))
            result.append(vector)
        return result

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def add(self, entry: VectorMemoryEntry) -> None:
        """Add a vector memory entry."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            List of (entry_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save index to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load index from disk."""
        pass


class FAISSBackend(VectorStoreBackend):
    """FAISS-based vector store backend."""

    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS backend.

        Args:
            embedding_dim: Dimension of embeddings
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

        self.embedding_dim = embedding_dim
        # Use inner product index (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.id_map: Dict[int, str] = {}  # FAISS index -> entry ID
        self.reverse_id_map: Dict[str, int] = {}  # entry ID -> FAISS index
        self.next_idx = 0

    def add(self, entry: VectorMemoryEntry) -> None:
        """Add a vector memory entry."""
        # Normalize vector for cosine similarity
        vector = entry.embedding.astype('float32')
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Add to index
        self.index.add(np.array([vector]))
        self.id_map[self.next_idx] = entry.id
        self.reverse_id_map[entry.id] = self.next_idx
        self.next_idx += 1

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        # Normalize query vector
        query = query_vector.astype('float32')
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # Search
        k = min(k, self.index.ntotal)
        if k == 0:
            return []

        distances, indices = self.index.search(np.array([query]), k)

        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx in self.id_map:
                entry_id = self.id_map[idx]
                # Convert inner product to similarity score (already in 0-1 range)
                similarity = float(dist)
                results.append((entry_id, similarity))

        return results

    def delete(self, entry_id: str) -> bool:
        """
        Delete an entry.
        Note: FAISS doesn't support efficient deletion, so this just updates mappings.
        """
        if entry_id in self.reverse_id_map:
            idx = self.reverse_id_map[entry_id]
            del self.reverse_id_map[entry_id]
            del self.id_map[idx]
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.id_map.clear()
        self.reverse_id_map.clear()
        self.next_idx = 0

    def save(self, path: Path) -> None:
        """Save index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))

        # Save mappings
        with open(path / "mappings.json", 'w') as f:
            json.dump({
                "id_map": {str(k): v for k, v in self.id_map.items()},
                "reverse_id_map": self.reverse_id_map,
                "next_idx": self.next_idx
            }, f)

    def load(self, path: Path) -> None:
        """Load index from disk."""
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))

        # Load mappings
        with open(path / "mappings.json", 'r') as f:
            data = json.load(f)
            self.id_map = {int(k): v for k, v in data["id_map"].items()}
            self.reverse_id_map = data["reverse_id_map"]
            self.next_idx = data["next_idx"]


class ChromaBackend(VectorStoreBackend):
    """Chroma-based vector store backend."""

    def __init__(self, embedding_dim: int, persist_directory: Optional[str] = None):
        """
        Initialize Chroma backend.

        Args:
            embedding_dim: Dimension of embeddings
            persist_directory: Directory to persist data
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma not available. Install with: pip install chromadb")

        self.embedding_dim = embedding_dim

        # Initialize Chroma client
        if persist_directory:
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                chroma_db_impl="duckdb+parquet"
            ))
        else:
            self.client = chromadb.Client()

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="vector_memory",
            metadata={"embedding_dim": embedding_dim}
        )

    def add(self, entry: VectorMemoryEntry) -> None:
        """Add a vector memory entry."""
        self.collection.add(
            ids=[entry.id],
            embeddings=[entry.embedding.tolist()],
            documents=[entry.content],
            metadatas=[entry.metadata]
        )

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k
        )

        # Extract results
        output = []
        if results['ids'] and results['distances']:
            for entry_id, distance in zip(results['ids'][0], results['distances'][0]):
                # Convert distance to similarity (Chroma uses L2 distance)
                similarity = 1.0 / (1.0 + distance)
                output.append((entry_id, similarity))

        return output

    def delete(self, entry_id: str) -> bool:
        """Delete an entry."""
        try:
            self.collection.delete(ids=[entry_id])
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Clear all entries."""
        self.client.delete_collection("vector_memory")
        self.collection = self.client.create_collection(
            name="vector_memory",
            metadata={"embedding_dim": self.embedding_dim}
        )

    def save(self, path: Path) -> None:
        """Save index to disk (Chroma handles persistence automatically)."""
        pass  # Chroma handles this if persist_directory is set

    def load(self, path: Path) -> None:
        """Load index from disk (Chroma handles persistence automatically)."""
        pass  # Chroma handles this if persist_directory is set


class VectorMemoryStore:
    """
    Vector-based memory store with semantic search capabilities.

    Features:
    - Semantic similarity search
    - Multiple backend support (FAISS, Chroma)
    - Automatic embedding generation
    - Periodic consolidation
    - Efficient retrieval
    """

    def __init__(self,
                 backend_type: str = "faiss",
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 persist_directory: Optional[Union[str, Path]] = None,
                 consolidation_threshold: int = 1000,
                 max_entries: int = 10000):
        """
        Initialize vector memory store.

        Args:
            backend_type: Type of backend ("faiss" or "chroma")
            embedding_provider: Custom embedding provider
            persist_directory: Directory for persistence
            consolidation_threshold: Number of entries before consolidation
            max_entries: Maximum number of entries to keep
        """
        # Set up embedding provider
        if embedding_provider is None:
            try:
                embedding_provider = SentenceTransformerEmbedding()
            except ImportError:
                embedding_provider = SimpleEmbedding()

        self.embedding_provider = embedding_provider

        # Set up backend
        persist_path = Path(persist_directory) if persist_directory else None
        if backend_type == "faiss":
            self.backend = FAISSBackend(embedding_provider.embedding_dim)
        elif backend_type == "chroma":
            self.backend = ChromaBackend(
                embedding_provider.embedding_dim,
                persist_directory=str(persist_path) if persist_path else None
            )
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        self.backend_type = backend_type
        self.persist_directory = persist_path

        # Storage for entries
        self.entries: Dict[str, VectorMemoryEntry] = {}

        # Configuration
        self.consolidation_threshold = consolidation_threshold
        self.max_entries = max_entries

        # Statistics
        self.total_entries_added = 0
        self.total_searches = 0
        self.total_consolidations = 0

        # Load existing data if available
        if persist_path and persist_path.exists():
            self.load()

    def add(self,
            content: str,
            entry_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None) -> VectorMemoryEntry:
        """
        Add a new memory entry.

        Args:
            content: Text content to store
            entry_id: Optional custom ID
            metadata: Optional metadata

        Returns:
            Created VectorMemoryEntry
        """
        # Generate ID if not provided
        if entry_id is None:
            entry_id = f"mem_{int(time.time() * 1000)}_{len(self.entries)}"

        # Generate embedding
        embedding = self.embedding_provider.embed(content)

        # Create entry
        entry = VectorMemoryEntry(
            id=entry_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )

        # Store entry
        self.entries[entry_id] = entry
        self.backend.add(entry)
        self.total_entries_added += 1

        # Check for consolidation
        if len(self.entries) >= self.consolidation_threshold:
            self.consolidate()

        return entry

    def add_batch(self,
                 contents: List[str],
                 metadatas: Optional[List[Dict[str, Any]]] = None) -> List[VectorMemoryEntry]:
        """
        Add multiple memory entries efficiently.

        Args:
            contents: List of text contents
            metadatas: Optional list of metadata dicts

        Returns:
            List of created VectorMemoryEntry objects
        """
        if metadatas is None:
            metadatas = [{}] * len(contents)

        # Generate embeddings in batch
        embeddings = self.embedding_provider.embed_batch(contents)

        # Create and store entries
        entries = []
        for content, embedding, metadata in zip(contents, embeddings, metadatas):
            entry_id = f"mem_{int(time.time() * 1000)}_{len(self.entries)}"
            entry = VectorMemoryEntry(
                id=entry_id,
                content=content,
                embedding=embedding,
                metadata=metadata
            )
            self.entries[entry_id] = entry
            self.backend.add(entry)
            entries.append(entry)

        self.total_entries_added += len(entries)

        # Check for consolidation
        if len(self.entries) >= self.consolidation_threshold:
            self.consolidate()

        return entries

    def search(self,
              query: str,
              k: int = 10,
              min_similarity: float = 0.0) -> List[SearchResult]:
        """
        Search for similar memories.

        Args:
            query: Query text
            k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of SearchResult objects
        """
        # Generate query embedding
        query_vector = self.embedding_provider.embed(query)

        # Search backend
        raw_results = self.backend.search(query_vector, k)

        # Build results
        results = []
        for rank, (entry_id, similarity) in enumerate(raw_results):
            if similarity < min_similarity:
                continue

            if entry_id not in self.entries:
                continue

            entry = self.entries[entry_id]

            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()

            results.append(SearchResult(
                entry=entry,
                similarity=similarity,
                rank=rank
            ))

        self.total_searches += 1
        return results

    def get(self, entry_id: str) -> Optional[VectorMemoryEntry]:
        """
        Get entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            VectorMemoryEntry or None
        """
        return self.entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        """
        Delete an entry.

        Args:
            entry_id: Entry ID

        Returns:
            True if deleted, False if not found
        """
        if entry_id in self.entries:
            del self.entries[entry_id]
            self.backend.delete(entry_id)
            return True
        return False

    def consolidate(self) -> int:
        """
        Consolidate memory by removing least useful entries.

        Returns:
            Number of entries removed
        """
        if len(self.entries) <= self.max_entries:
            return 0

        # Score entries based on recency and access
        scored_entries = []
        current_time = time.time()

        for entry in self.entries.values():
            # Calculate score based on:
            # - Recency (newer is better)
            # - Access count (more accessed is better)
            # - Last accessed (recently accessed is better)
            recency_score = 1.0 / (1.0 + (current_time - entry.timestamp) / 86400)  # Days
            access_score = entry.access_count / (1.0 + entry.access_count)

            if entry.last_accessed:
                last_access_score = 1.0 / (1.0 + (current_time - entry.last_accessed) / 3600)  # Hours
            else:
                last_access_score = 0.0

            total_score = recency_score + access_score + last_access_score
            scored_entries.append((entry.id, total_score))

        # Sort by score (ascending) and remove lowest scoring
        scored_entries.sort(key=lambda x: x[1])
        num_to_remove = len(self.entries) - self.max_entries

        removed = 0
        for entry_id, _ in scored_entries[:num_to_remove]:
            if self.delete(entry_id):
                removed += 1

        self.total_consolidations += 1
        return removed

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()
        self.backend.clear()

    def save(self) -> None:
        """Save memory store to disk."""
        if not self.persist_directory:
            raise ValueError("No persist directory configured")

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Save backend
        self.backend.save(self.persist_directory)

        # Save entries
        entries_data = {
            entry_id: {
                "id": entry.id,
                "content": entry.content,
                "embedding": entry.embedding.tolist(),
                "metadata": entry.metadata,
                "timestamp": entry.timestamp,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed
            }
            for entry_id, entry in self.entries.items()
        }

        with open(self.persist_directory / "entries.json", 'w') as f:
            json.dump(entries_data, f, indent=2)

        # Save statistics
        stats = {
            "total_entries_added": self.total_entries_added,
            "total_searches": self.total_searches,
            "total_consolidations": self.total_consolidations
        }

        with open(self.persist_directory / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

    def load(self) -> None:
        """Load memory store from disk."""
        if not self.persist_directory or not self.persist_directory.exists():
            return

        # Load backend
        try:
            self.backend.load(self.persist_directory)
        except Exception:
            pass  # Backend might not have data yet

        # Load entries
        entries_file = self.persist_directory / "entries.json"
        if entries_file.exists():
            with open(entries_file, 'r') as f:
                entries_data = json.load(f)

            for entry_data in entries_data.values():
                entry = VectorMemoryEntry(
                    id=entry_data["id"],
                    content=entry_data["content"],
                    embedding=np.array(entry_data["embedding"]),
                    metadata=entry_data["metadata"],
                    timestamp=entry_data["timestamp"],
                    access_count=entry_data["access_count"],
                    last_accessed=entry_data.get("last_accessed")
                )
                self.entries[entry.id] = entry

        # Load statistics
        stats_file = self.persist_directory / "stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                self.total_entries_added = stats.get("total_entries_added", 0)
                self.total_searches = stats.get("total_searches", 0)
                self.total_consolidations = stats.get("total_consolidations", 0)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get store statistics.

        Returns:
            Dictionary with various statistics
        """
        return {
            "total_entries": len(self.entries),
            "max_entries": self.max_entries,
            "total_entries_added": self.total_entries_added,
            "total_searches": self.total_searches,
            "total_consolidations": self.total_consolidations,
            "backend_type": self.backend_type,
            "embedding_dim": self.embedding_provider.embedding_dim
        }
