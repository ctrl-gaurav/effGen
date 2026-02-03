"""
Standard Retrieval (RAG) tool with embedding-based search.

This module provides a retrieval tool that uses embeddings to search
through a knowledge base and return relevant documents/chunks.
"""

import os
import json
import pickle
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np

from ..base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    ParameterSpec,
    ParameterType,
)


logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document in the knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """A retrieval result with score."""
    document: Document
    score: float
    rank: int


class EmbeddingProvider:
    """Base class for embedding providers."""

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query text."""
        return self.embed([query])[0]


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Sentence Transformers embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings


class SimpleEmbedding(EmbeddingProvider):
    """Simple TF-IDF based embedding for environments without sentence-transformers."""

    def __init__(self):
        """Initialize the simple embedding provider."""
        self._vectorizer = None
        self._fitted = False

    def _get_vectorizer(self):
        """Get or create the vectorizer."""
        if self._vectorizer is None:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._vectorizer = TfidfVectorizer(
                    max_features=512,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            except ImportError:
                raise RuntimeError(
                    "scikit-learn not installed. "
                    "Install with: pip install scikit-learn"
                )
        return self._vectorizer

    def fit(self, texts: List[str]):
        """Fit the vectorizer on texts."""
        vectorizer = self._get_vectorizer()
        vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using TF-IDF."""
        vectorizer = self._get_vectorizer()
        if not self._fitted:
            # Fit on the provided texts if not already fitted
            embeddings = vectorizer.fit_transform(texts).toarray()
            self._fitted = True
        else:
            embeddings = vectorizer.transform(texts).toarray()
        return embeddings


class Retrieval(BaseTool):
    """
    Standard Retrieval (RAG) tool with embedding-based search.

    Features:
    - Embedding-based semantic search
    - Multiple embedding providers (Sentence Transformers, TF-IDF fallback)
    - Document chunking with configurable sizes
    - Persistent index storage
    - Metadata filtering
    - Score thresholds

    Usage:
        retrieval = Retrieval()
        # Index documents
        await retrieval.add_documents([
            {"content": "...", "id": "doc1", "metadata": {...}},
            ...
        ])
        # Search
        result = await retrieval.execute(query="your question", top_k=5)
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        index_path: Optional[str] = None,
    ):
        """
        Initialize the retrieval tool.

        Args:
            embedding_provider: Provider for text embeddings
            chunk_size: Size of document chunks in characters
            chunk_overlap: Overlap between chunks in characters
            index_path: Path to persist the index
        """
        super().__init__(
            metadata=ToolMetadata(
                name="retrieval",
                description=(
                    "Search a knowledge base using semantic similarity. "
                    "Returns the most relevant documents/passages for a given query. "
                    "Use this tool when you need to find information from the indexed knowledge base."
                ),
                category=ToolCategory.INFORMATION_RETRIEVAL,
                parameters=[
                    ParameterSpec(
                        name="query",
                        type=ParameterType.STRING,
                        description="The search query to find relevant documents",
                        required=True,
                        min_length=1,
                        max_length=1000,
                    ),
                    ParameterSpec(
                        name="top_k",
                        type=ParameterType.INTEGER,
                        description="Number of top results to return",
                        required=False,
                        default=5,
                        min_value=1,
                        max_value=50,
                    ),
                    ParameterSpec(
                        name="score_threshold",
                        type=ParameterType.FLOAT,
                        description="Minimum similarity score threshold (0-1)",
                        required=False,
                        default=0.0,
                        min_value=0.0,
                        max_value=1.0,
                    ),
                    ParameterSpec(
                        name="filter_metadata",
                        type=ParameterType.OBJECT,
                        description="Filter results by metadata fields",
                        required=False,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "score": {"type": "number"},
                                    "metadata": {"type": "object"},
                                },
                            },
                        },
                        "total_found": {"type": "integer"},
                    },
                },
                timeout_seconds=30,
                tags=["retrieval", "rag", "search", "knowledge-base", "semantic"],
                examples=[
                    {
                        "query": "What is machine learning?",
                        "top_k": 3,
                        "output": {
                            "results": [
                                {
                                    "content": "Machine learning is a subset of AI...",
                                    "score": 0.92,
                                    "metadata": {"source": "ml_intro.txt"},
                                }
                            ],
                            "total_found": 3,
                        },
                    }
                ],
            )
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path

        # Initialize embedding provider
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            # Try to use sentence transformers, fall back to TF-IDF
            try:
                # Check if sentence_transformers is available before trying to use it
                import importlib.util
                if importlib.util.find_spec("sentence_transformers") is not None:
                    self.embedding_provider = SentenceTransformerEmbedding()
                else:
                    raise ImportError("sentence_transformers not found")
            except (ImportError, Exception) as e:
                logger.warning(
                    f"Sentence Transformers not available ({e}), using TF-IDF fallback"
                )
                self.embedding_provider = SimpleEmbedding()

        # Document storage
        self.documents: Dict[str, Document] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []

        # Load existing index if available
        if index_path and os.path.exists(index_path):
            self._load_index(index_path)

    async def initialize(self) -> None:
        """Initialize the retrieval tool."""
        await super().initialize()
        logger.info("Retrieval tool initialized")

    def _chunk_text(self, text: str, doc_id: str) -> List[Document]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            doc_id: Base document ID

        Returns:
            List of Document chunks
        """
        chunks = []
        start = 0
        chunk_num = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_sep = chunk_text.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        chunk_text = chunk_text[:last_sep + len(sep)]
                        end = start + len(chunk_text)
                        break

            chunk_id = f"{doc_id}_chunk_{chunk_num}"
            chunks.append(Document(
                id=chunk_id,
                content=chunk_text.strip(),
                metadata={
                    "parent_doc_id": doc_id,
                    "chunk_index": chunk_num,
                    "start_char": start,
                    "end_char": end,
                }
            ))

            chunk_num += 1
            start = end - self.chunk_overlap

            # Avoid infinite loop
            if start >= end:
                break

        return chunks

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk: bool = True,
    ) -> int:
        """
        Add documents to the index.

        Args:
            documents: List of documents with 'content', optional 'id' and 'metadata'
            chunk: Whether to chunk documents

        Returns:
            Number of documents/chunks added
        """
        all_chunks = []

        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue

            doc_id = doc.get("id") or hashlib.md5(content.encode()).hexdigest()[:12]
            metadata = doc.get("metadata", {})

            if chunk and len(content) > self.chunk_size:
                # Create chunks
                chunks = self._chunk_text(content, doc_id)
                for c in chunks:
                    c.metadata.update(metadata)
                all_chunks.extend(chunks)
            else:
                # Store as single document
                all_chunks.append(Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                ))

        if not all_chunks:
            return 0

        # Generate embeddings
        texts = [c.content for c in all_chunks]

        # For SimpleEmbedding, we need to fit on all texts
        if isinstance(self.embedding_provider, SimpleEmbedding):
            # Collect all existing texts and new texts
            all_texts = [d.content for d in self.documents.values()] + texts
            self.embedding_provider.fit(all_texts)

        embeddings = self.embedding_provider.embed(texts)

        # Store documents and embeddings
        for i, chunk in enumerate(all_chunks):
            chunk.embedding = embeddings[i]
            self.documents[chunk.id] = chunk
            self.doc_ids.append(chunk.id)

        # Rebuild embedding matrix
        self._rebuild_embedding_matrix()

        logger.info(f"Added {len(all_chunks)} documents/chunks to index")
        return len(all_chunks)

    def add_from_file(
        self,
        file_path: str,
        file_type: str = "auto",
        chunk: bool = True,
    ) -> int:
        """
        Add documents from a file.

        Args:
            file_path: Path to the file
            file_type: File type (auto, txt, json, jsonl)
            chunk: Whether to chunk documents

        Returns:
            Number of documents added
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect file type
        if file_type == "auto":
            suffix = path.suffix.lower()
            if suffix == ".json":
                file_type = "json"
            elif suffix == ".jsonl":
                file_type = "jsonl"
            else:
                file_type = "txt"

        documents = []

        if file_type == "txt":
            content = path.read_text(encoding="utf-8")
            documents.append({
                "content": content,
                "id": path.stem,
                "metadata": {"source": str(path)},
            })

        elif file_type == "json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        doc = {
                            "content": item.get("content") or item.get("text") or str(item),
                            "id": item.get("id") or f"{path.stem}_{i}",
                            "metadata": {k: v for k, v in item.items() if k not in ["content", "text", "id"]},
                        }
                        doc["metadata"]["source"] = str(path)
                        documents.append(doc)
                    else:
                        documents.append({
                            "content": str(item),
                            "id": f"{path.stem}_{i}",
                            "metadata": {"source": str(path)},
                        })
            elif isinstance(data, dict):
                documents.append({
                    "content": data.get("content") or data.get("text") or str(data),
                    "id": data.get("id") or path.stem,
                    "metadata": {"source": str(path)},
                })

        elif file_type == "jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    doc = {
                        "content": item.get("content") or item.get("text") or item.get("question") or str(item),
                        "id": item.get("id") or f"{path.stem}_{i}",
                        "metadata": {k: v for k, v in item.items() if k not in ["content", "text", "id"]},
                    }
                    doc["metadata"]["source"] = str(path)
                    documents.append(doc)

        return self.add_documents(documents, chunk=chunk)

    def _rebuild_embedding_matrix(self):
        """Rebuild the embedding matrix from stored documents."""
        if not self.documents:
            self.embeddings_matrix = None
            self.doc_ids = []
            return

        self.doc_ids = list(self.documents.keys())
        embeddings = [self.documents[doc_id].embedding for doc_id in self.doc_ids]
        self.embeddings_matrix = np.vstack(embeddings)

    def _cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all documents."""
        if self.embeddings_matrix is None:
            return np.array([])

        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.embeddings_matrix / (
            np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True) + 1e-8
        )

        # Compute similarities
        similarities = np.dot(doc_norms, query_norm)
        return similarities

    async def _execute(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute retrieval query.

        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_metadata: Filter by metadata fields

        Returns:
            Dict with results and metadata
        """
        if not self.documents:
            return {
                "results": [],
                "total_found": 0,
                "message": "No documents in index. Add documents first using add_documents() or add_from_file().",
            }

        # Get query embedding
        query_embedding = self.embedding_provider.embed_query(query)

        # Compute similarities
        similarities = self._cosine_similarity(query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1]

        # Build results
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break

            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            score = float(similarities[idx])

            # Apply score threshold
            if score < score_threshold:
                continue

            # Apply metadata filter
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "content": doc.content,
                "score": round(score, 4),
                "id": doc.id,
                "metadata": doc.metadata,
                "rank": len(results) + 1,
            })

        return {
            "results": results,
            "total_found": len(results),
            "query": query,
        }

    def save_index(self, path: Optional[str] = None):
        """
        Save the index to disk.

        Args:
            path: Path to save to (uses self.index_path if not provided)
        """
        path = path or self.index_path
        if not path:
            raise ValueError("No index path specified")

        data = {
            "documents": {
                doc_id: {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": doc.embedding.tolist() if doc.embedding is not None else None,
                }
                for doc_id, doc in self.documents.items()
            },
            "doc_ids": self.doc_ids,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved index to {path}")

    def _load_index(self, path: str):
        """
        Load the index from disk.

        Args:
            path: Path to load from
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.documents = {}
        for doc_id, doc_data in data["documents"].items():
            self.documents[doc_id] = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                embedding=np.array(doc_data["embedding"]) if doc_data["embedding"] else None,
            )

        self.doc_ids = data["doc_ids"]
        self.chunk_size = data.get("chunk_size", self.chunk_size)
        self.chunk_overlap = data.get("chunk_overlap", self.chunk_overlap)

        self._rebuild_embedding_matrix()
        logger.info(f"Loaded index from {path} with {len(self.documents)} documents")

    def clear(self):
        """Clear all documents from the index."""
        self.documents = {}
        self.embeddings_matrix = None
        self.doc_ids = []
        logger.info("Cleared retrieval index")

    @property
    def num_documents(self) -> int:
        """Return the number of documents in the index."""
        return len(self.documents)
