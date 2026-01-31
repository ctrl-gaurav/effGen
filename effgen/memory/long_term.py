"""
Long-term memory system with persistent storage.

This module provides long-term memory with multiple backend support (JSON, SQLite),
session management, memory consolidation, and efficient retrieval of historical data.
"""

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum


class MemoryType(Enum):
    """Type of memory entry."""
    CONVERSATION = "conversation"
    FACT = "fact"
    OBSERVATION = "observation"
    TASK = "task"
    TOOL_RESULT = "tool_result"
    REFLECTION = "reflection"


class ImportanceLevel(Enum):
    """Importance level of memory."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryEntry:
    """
    Represents a single long-term memory entry.

    Attributes:
        id: Unique identifier
        content: Memory content
        memory_type: Type of memory
        importance: Importance level
        timestamp: When memory was created
        session_id: Associated session ID
        metadata: Additional metadata
        access_count: Number of times accessed
        last_accessed: Last access timestamp
        tags: Tags for categorization
    """
    content: str
    memory_type: MemoryType
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            importance=ImportanceLevel(data["importance"]),
            timestamp=data["timestamp"],
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            tags=data.get("tags", [])
        )


@dataclass
class Session:
    """
    Represents a conversation or task session.

    Attributes:
        id: Unique session identifier
        name: Human-readable session name
        start_time: Session start timestamp
        end_time: Session end timestamp (None if active)
        metadata: Session metadata
        memory_count: Number of memories in session
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create from dictionary."""
        return cls(**data)

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.end_time is None

    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_memory(self, entry: MemoryEntry) -> None:
        """Save a memory entry."""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        pass

    @abstractmethod
    def search_memories(self,
                       query: Optional[str] = None,
                       memory_type: Optional[MemoryType] = None,
                       session_id: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       min_importance: Optional[ImportanceLevel] = None,
                       limit: Optional[int] = None) -> List[MemoryEntry]:
        """Search memories with filters."""
        pass

    @abstractmethod
    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """Clear all memories."""
        pass


class JSONStorageBackend(StorageBackend):
    """JSON file-based storage backend."""

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize JSON storage backend.

        Args:
            filepath: Path to JSON file
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._load_or_create()

    def _load_or_create(self) -> None:
        """Load existing data or create new file."""
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {"memories": {}, "sessions": {}}
            self._save()

    def _save(self) -> None:
        """Save data to file."""
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def save_memory(self, entry: MemoryEntry) -> None:
        """Save a memory entry."""
        self.data["memories"][entry.id] = entry.to_dict()
        self._save()

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        mem_data = self.data["memories"].get(memory_id)
        if mem_data:
            return MemoryEntry.from_dict(mem_data)
        return None

    def search_memories(self,
                       query: Optional[str] = None,
                       memory_type: Optional[MemoryType] = None,
                       session_id: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       min_importance: Optional[ImportanceLevel] = None,
                       limit: Optional[int] = None) -> List[MemoryEntry]:
        """Search memories with filters."""
        results = []

        for mem_data in self.data["memories"].values():
            entry = MemoryEntry.from_dict(mem_data)

            # Apply filters
            if memory_type and entry.memory_type != memory_type:
                continue
            if session_id and entry.session_id != session_id:
                continue
            if min_importance and entry.importance.value < min_importance.value:
                continue
            if tags and not any(tag in entry.tags for tag in tags):
                continue
            if query:
                query_lower = query.lower()
                if query_lower not in entry.content.lower():
                    continue

            results.append(entry)

        # Sort by importance and recency
        results.sort(key=lambda x: (x.importance.value, x.timestamp), reverse=True)

        if limit:
            results = results[:limit]

        return results

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        if memory_id not in self.data["memories"]:
            return False

        self.data["memories"][memory_id].update(updates)
        self._save()
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        if memory_id in self.data["memories"]:
            del self.data["memories"][memory_id]
            self._save()
            return True
        return False

    def clear_all(self) -> None:
        """Clear all memories."""
        self.data = {"memories": {}, "sessions": {}}
        self._save()

    def save_session(self, session: Session) -> None:
        """Save a session."""
        self.data["sessions"][session.id] = session.to_dict()
        self._save()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        sess_data = self.data["sessions"].get(session_id)
        if sess_data:
            return Session.from_dict(sess_data)
        return None


class SQLiteStorageBackend(StorageBackend):
    """SQLite database storage backend."""

    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize SQLite storage backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                session_id TEXT,
                metadata TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                tags TEXT
            )
        """)

        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                start_time REAL NOT NULL,
                end_time REAL,
                metadata TEXT,
                memory_count INTEGER DEFAULT 0
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(str(self.db_path))

    def save_memory(self, entry: MemoryEntry) -> None:
        """Save a memory entry."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO memories
            (id, content, memory_type, importance, timestamp, session_id,
             metadata, access_count, last_accessed, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.content,
            entry.memory_type.value,
            entry.importance.value,
            entry.timestamp,
            entry.session_id,
            json.dumps(entry.metadata),
            entry.access_count,
            entry.last_accessed,
            json.dumps(entry.tags)
        ))

        conn.commit()
        conn.close()

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_memory(row)
        return None

    def search_memories(self,
                       query: Optional[str] = None,
                       memory_type: Optional[MemoryType] = None,
                       session_id: Optional[str] = None,
                       tags: Optional[List[str]] = None,
                       min_importance: Optional[ImportanceLevel] = None,
                       limit: Optional[int] = None) -> List[MemoryEntry]:
        """Search memories with filters."""
        conn = self._get_connection()
        cursor = conn.cursor()

        sql = "SELECT * FROM memories WHERE 1=1"
        params = []

        if query:
            sql += " AND content LIKE ?"
            params.append(f"%{query}%")

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type.value)

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        if min_importance:
            sql += " AND importance >= ?"
            params.append(min_importance.value)

        sql += " ORDER BY importance DESC, timestamp DESC"

        if limit:
            sql += " LIMIT ?"
            params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        results = [self._row_to_memory(row) for row in rows]

        # Filter by tags if provided (post-query filter)
        if tags:
            results = [
                entry for entry in results
                if any(tag in entry.tags for tag in tags)
            ]

        return results

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Build update query
        set_clauses = []
        params = []

        for key, value in updates.items():
            if key in ["metadata", "tags"]:
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            params.append(value)

        params.append(memory_id)
        sql = f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = ?"

        cursor.execute(sql, params)
        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def clear_all(self) -> None:
        """Clear all memories."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories")
        cursor.execute("DELETE FROM sessions")

        conn.commit()
        conn.close()

    def _row_to_memory(self, row: Tuple) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row[0],
            content=row[1],
            memory_type=MemoryType(row[2]),
            importance=ImportanceLevel(row[3]),
            timestamp=row[4],
            session_id=row[5],
            metadata=json.loads(row[6]) if row[6] else {},
            access_count=row[7] or 0,
            last_accessed=row[8],
            tags=json.loads(row[9]) if row[9] else []
        )

    def save_session(self, session: Session) -> None:
        """Save a session."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO sessions
            (id, name, start_time, end_time, metadata, memory_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session.id,
            session.name,
            session.start_time,
            session.end_time,
            json.dumps(session.metadata),
            session.memory_count
        ))

        conn.commit()
        conn.close()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Session(
                id=row[0],
                name=row[1],
                start_time=row[2],
                end_time=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
                memory_count=row[5] or 0
            )
        return None


class LongTermMemory:
    """
    Long-term memory manager with persistent storage.

    Features:
    - Multiple storage backends (JSON, SQLite)
    - Session management
    - Memory consolidation
    - Efficient search and retrieval
    - Importance-based retention
    - Automatic cleanup of old/low-importance memories
    """

    def __init__(self,
                 backend: StorageBackend,
                 consolidation_interval: int = 100,
                 max_memories: int = 10000,
                 min_importance_to_keep: ImportanceLevel = ImportanceLevel.LOW):
        """
        Initialize long-term memory.

        Args:
            backend: Storage backend to use
            consolidation_interval: Number of new memories before consolidation
            max_memories: Maximum number of memories to retain
            min_importance_to_keep: Minimum importance level to keep during consolidation
        """
        self.backend = backend
        self.consolidation_interval = consolidation_interval
        self.max_memories = max_memories
        self.min_importance_to_keep = min_importance_to_keep

        # Current session
        self.current_session: Optional[Session] = None

        # Statistics
        self.memories_added_since_consolidation = 0
        self.total_consolidations = 0

    def start_session(self, name: Optional[str] = None) -> Session:
        """
        Start a new session.

        Args:
            name: Optional session name

        Returns:
            Created Session object
        """
        self.current_session = Session(name=name)
        if hasattr(self.backend, 'save_session'):
            self.backend.save_session(self.current_session)
        return self.current_session

    def end_session(self) -> None:
        """End the current session."""
        if self.current_session:
            self.current_session.end_time = time.time()
            if hasattr(self.backend, 'save_session'):
                self.backend.save_session(self.current_session)
            self.current_session = None

    def add_memory(self,
                  content: str,
                  memory_type: MemoryType,
                  importance: ImportanceLevel = ImportanceLevel.MEDIUM,
                  tags: Optional[List[str]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """
        Add a memory entry.

        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance level
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Created MemoryEntry object
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            session_id=self.current_session.id if self.current_session else None,
            tags=tags or [],
            metadata=metadata or {}
        )

        self.backend.save_memory(entry)

        # Update session memory count
        if self.current_session:
            self.current_session.memory_count += 1
            if hasattr(self.backend, 'save_session'):
                self.backend.save_session(self.current_session)

        # Check if consolidation is needed
        self.memories_added_since_consolidation += 1
        if self.memories_added_since_consolidation >= self.consolidation_interval:
            self.consolidate()

        return entry

    def get_memory(self, memory_id: str, update_access: bool = True) -> Optional[MemoryEntry]:
        """
        Get a memory by ID.

        Args:
            memory_id: Memory ID
            update_access: Whether to update access statistics

        Returns:
            MemoryEntry or None if not found
        """
        entry = self.backend.get_memory(memory_id)

        if entry and update_access:
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.backend.update_memory(entry.id, {
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed
            })

        return entry

    def search(self,
              query: Optional[str] = None,
              memory_type: Optional[MemoryType] = None,
              session_id: Optional[str] = None,
              tags: Optional[List[str]] = None,
              min_importance: Optional[ImportanceLevel] = None,
              limit: int = 50) -> List[MemoryEntry]:
        """
        Search memories.

        Args:
            query: Text query
            memory_type: Filter by memory type
            session_id: Filter by session
            tags: Filter by tags
            min_importance: Minimum importance level
            limit: Maximum results

        Returns:
            List of matching MemoryEntry objects
        """
        return self.backend.search_memories(
            query=query,
            memory_type=memory_type,
            session_id=session_id,
            tags=tags,
            min_importance=min_importance,
            limit=limit
        )

    def consolidate(self) -> int:
        """
        Consolidate memories by removing low-importance or old entries.

        Returns:
            Number of memories removed
        """
        # Get all memories
        all_memories = self.backend.search_memories()

        if len(all_memories) <= self.max_memories:
            self.memories_added_since_consolidation = 0
            return 0

        # Sort by importance and access patterns
        def score_memory(entry: MemoryEntry) -> float:
            """Calculate retention score for memory."""
            importance_score = entry.importance.value * 100
            recency_score = (time.time() - entry.timestamp) / (24 * 3600)  # Days old
            access_score = entry.access_count * 10

            # Higher score = more likely to keep
            return importance_score + access_score - recency_score

        all_memories.sort(key=score_memory, reverse=True)

        # Keep top memories
        to_keep = all_memories[:self.max_memories]
        to_remove = all_memories[self.max_memories:]

        # Remove low-scoring memories
        removed_count = 0
        for entry in to_remove:
            if entry.importance.value >= self.min_importance_to_keep.value:
                continue  # Keep high-importance memories regardless
            self.backend.delete_memory(entry.id)
            removed_count += 1

        self.memories_added_since_consolidation = 0
        self.total_consolidations += 1

        return removed_count

    def clear_all(self) -> None:
        """Clear all memories."""
        self.backend.clear_all()
        self.current_session = None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with various statistics
        """
        all_memories = self.backend.search_memories()

        return {
            "total_memories": len(all_memories),
            "max_memories": self.max_memories,
            "current_session_id": self.current_session.id if self.current_session else None,
            "total_consolidations": self.total_consolidations,
            "memories_by_type": self._count_by_type(all_memories),
            "memories_by_importance": self._count_by_importance(all_memories),
        }

    def _count_by_type(self, memories: List[MemoryEntry]) -> Dict[str, int]:
        """Count memories by type."""
        counts = {}
        for entry in memories:
            type_name = entry.memory_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def _count_by_importance(self, memories: List[MemoryEntry]) -> Dict[str, int]:
        """Count memories by importance."""
        counts = {}
        for entry in memories:
            importance_name = entry.importance.name
            counts[importance_name] = counts.get(importance_name, 0) + 1
        return counts
