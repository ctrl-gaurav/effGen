"""
Short-term memory system for conversation management.

This module provides short-term memory for managing recent conversation history
with automatic summarization when approaching context limits, efficient retrieval,
and token counting.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
from datetime import datetime


class MessageRole(Enum):
    """Role of message in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """
    Represents a single message in conversation.

    Attributes:
        role: Role of the message sender
        content: Message content
        timestamp: When the message was created
        metadata: Additional metadata (tool calls, tokens, etc.)
        tokens: Estimated token count for the message
    """
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tokens": self.tokens
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            tokens=data.get("tokens")
        )

    def estimate_tokens(self) -> int:
        """
        Estimate token count for the message.
        Uses rough approximation: 1 token ≈ 4 characters.

        Returns:
            Estimated token count
        """
        if self.tokens is None:
            # Rough estimation: 1 token ≈ 4 characters
            self.tokens = len(self.content) // 4 + len(str(self.metadata)) // 4
        return self.tokens


@dataclass
class ConversationSummary:
    """
    Summary of conversation segment.

    Attributes:
        summary: Condensed summary text
        message_count: Number of messages summarized
        token_count: Total tokens in original messages
        timestamp: When summary was created
        metadata: Additional information about the summary
    """
    summary: str
    message_count: int
    token_count: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create summary from dictionary."""
        return cls(**data)


class ShortTermMemory:
    """
    Short-term memory manager for conversation history.

    Features:
    - Store recent conversation messages
    - Automatic summarization when approaching context limits
    - Efficient retrieval of recent context
    - Token counting and management
    - Message filtering and search
    - Rolling window of recent messages

    This class maintains a balance between keeping recent messages
    and older summaries to stay within context limits while preserving
    important conversation history.
    """

    def __init__(self,
                 max_tokens: int = 4096,
                 max_messages: int = 100,
                 summarization_threshold: float = 0.8,
                 summary_length_ratio: float = 0.3,
                 keep_recent_messages: int = 10):
        """
        Initialize short-term memory.

        Args:
            max_tokens: Maximum total tokens to keep in memory
            max_messages: Maximum number of messages to store
            summarization_threshold: Threshold (0-1) of max_tokens to trigger summarization
            summary_length_ratio: Target ratio of summary length to original (0-1)
            keep_recent_messages: Number of recent messages to always keep unsummarized
        """
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.summarization_threshold = summarization_threshold
        self.summary_length_ratio = summary_length_ratio
        self.keep_recent_messages = keep_recent_messages

        # Storage
        self.messages: deque[Message] = deque(maxlen=max_messages)
        self.summaries: List[ConversationSummary] = []

        # Statistics
        self.total_messages_added = 0
        self.total_summarizations = 0
        self._current_token_count = 0

    def add_message(self,
                   role: MessageRole,
                   content: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   tokens: Optional[int] = None) -> Message:
        """
        Add a message to short-term memory.

        Args:
            role: Role of the message sender
            content: Message content
            metadata: Optional metadata
            tokens: Pre-computed token count (will estimate if not provided)

        Returns:
            The created Message object
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            tokens=tokens
        )

        # Estimate tokens if not provided
        message_tokens = message.estimate_tokens()

        # Add message
        self.messages.append(message)
        self.total_messages_added += 1
        self._current_token_count += message_tokens

        # Check if we need to summarize
        if self._should_summarize():
            self._summarize_old_messages()

        return message

    def add_user_message(self, content: str, **kwargs) -> Message:
        """Convenience method to add user message."""
        return self.add_message(MessageRole.USER, content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs) -> Message:
        """Convenience method to add assistant message."""
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)

    def add_system_message(self, content: str, **kwargs) -> Message:
        """Convenience method to add system message."""
        return self.add_message(MessageRole.SYSTEM, content, **kwargs)

    def add_tool_message(self, content: str, **kwargs) -> Message:
        """Convenience method to add tool message."""
        return self.add_message(MessageRole.TOOL, content, **kwargs)

    def get_recent_messages(self, n: Optional[int] = None) -> List[Message]:
        """
        Get the n most recent messages.

        Args:
            n: Number of recent messages to retrieve (None = all)

        Returns:
            List of recent messages
        """
        if n is None:
            return list(self.messages)
        return list(self.messages)[-n:]

    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """
        Get all messages by a specific role.

        Args:
            role: Message role to filter by

        Returns:
            List of messages with matching role
        """
        return [msg for msg in self.messages if msg.role == role]

    def get_conversation_context(self,
                                max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation context suitable for LLM input.

        Args:
            max_tokens: Maximum tokens to include (uses instance max if None)

        Returns:
            List of message dictionaries in format expected by LLMs
        """
        max_tokens = max_tokens or self.max_tokens
        context = []
        current_tokens = 0

        # Add summaries first if they exist
        for summary in self.summaries:
            summary_tokens = len(summary.summary) // 4
            if current_tokens + summary_tokens > max_tokens:
                break
            context.append({
                "role": "system",
                "content": f"[Summary of previous conversation: {summary.summary}]"
            })
            current_tokens += summary_tokens

        # Add recent messages
        for message in self.messages:
            msg_tokens = message.estimate_tokens()
            if current_tokens + msg_tokens > max_tokens:
                break
            context.append({
                "role": message.role.value,
                "content": message.content
            })
            current_tokens += msg_tokens

        return context

    def search_messages(self, query: str, case_sensitive: bool = False) -> List[Message]:
        """
        Search messages by content.

        Args:
            query: Search query string
            case_sensitive: Whether search should be case-sensitive

        Returns:
            List of messages matching the query
        """
        results = []
        query_str = query if case_sensitive else query.lower()

        for message in self.messages:
            content = message.content if case_sensitive else message.content.lower()
            if query_str in content:
                results.append(message)

        return results

    def get_token_count(self) -> int:
        """
        Get current total token count.

        Returns:
            Total tokens currently stored
        """
        # Recalculate to ensure accuracy
        total = sum(msg.estimate_tokens() for msg in self.messages)
        total += sum(len(s.summary) // 4 for s in self.summaries)
        return total

    def _should_summarize(self) -> bool:
        """
        Check if summarization should be triggered.

        Returns:
            True if summarization is needed
        """
        current_tokens = self.get_token_count()
        threshold = self.max_tokens * self.summarization_threshold

        # Only summarize if we have enough messages to make it worthwhile
        return (current_tokens > threshold and
                len(self.messages) > self.keep_recent_messages)

    def _summarize_old_messages(self) -> None:
        """
        Summarize old messages to save space.

        This method:
        1. Identifies messages to summarize (excluding recent ones)
        2. Generates a summary
        3. Removes original messages
        4. Stores the summary
        """
        if len(self.messages) <= self.keep_recent_messages:
            return

        # Calculate how many messages to summarize
        num_to_summarize = len(self.messages) - self.keep_recent_messages

        # Get messages to summarize
        messages_to_summarize = list(self.messages)[:num_to_summarize]

        # Generate summary
        summary_text = self._generate_summary(messages_to_summarize)

        # Calculate token savings
        original_tokens = sum(msg.estimate_tokens() for msg in messages_to_summarize)
        summary_tokens = len(summary_text) // 4

        # Create summary object
        summary = ConversationSummary(
            summary=summary_text,
            message_count=len(messages_to_summarize),
            token_count=original_tokens,
            metadata={
                "summary_tokens": summary_tokens,
                "compression_ratio": summary_tokens / original_tokens if original_tokens > 0 else 0
            }
        )

        # Store summary
        self.summaries.append(summary)

        # Remove summarized messages
        for _ in range(num_to_summarize):
            self.messages.popleft()

        # Update statistics
        self.total_summarizations += 1
        self._current_token_count = self.get_token_count()

    def _generate_summary(self, messages: List[Message]) -> str:
        """
        Generate a summary of messages.

        For now, this is a simple extractive summary. In production,
        you would call an LLM to generate a better summary.

        Args:
            messages: Messages to summarize

        Returns:
            Summary text
        """
        # Simple extractive summary: key points from each message
        key_points = []

        for msg in messages:
            # Take first sentence or first 100 chars
            content = msg.content.strip()
            if not content:
                continue

            # Extract first sentence
            first_sentence = content.split('.')[0] + '.'
            if len(first_sentence) > 100:
                first_sentence = content[:100] + "..."

            key_points.append(f"{msg.role.value}: {first_sentence}")

        # Limit summary length
        summary = " | ".join(key_points)
        target_length = int(sum(len(m.content) for m in messages) * self.summary_length_ratio)

        if len(summary) > target_length:
            summary = summary[:target_length] + "..."

        return summary

    def clear(self) -> None:
        """Clear all messages and summaries."""
        self.messages.clear()
        self.summaries.clear()
        self._current_token_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with various statistics
        """
        return {
            "current_messages": len(self.messages),
            "current_tokens": self.get_token_count(),
            "max_tokens": self.max_tokens,
            "utilization": self.get_token_count() / self.max_tokens,
            "total_messages_added": self.total_messages_added,
            "total_summarizations": self.total_summarizations,
            "summaries_count": len(self.summaries),
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize memory to dictionary.

        Returns:
            Dictionary representation of memory state
        """
        return {
            "config": {
                "max_tokens": self.max_tokens,
                "max_messages": self.max_messages,
                "summarization_threshold": self.summarization_threshold,
                "summary_length_ratio": self.summary_length_ratio,
                "keep_recent_messages": self.keep_recent_messages,
            },
            "messages": [msg.to_dict() for msg in self.messages],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "statistics": self.get_statistics()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShortTermMemory":
        """
        Deserialize memory from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ShortTermMemory instance
        """
        config = data.get("config", {})
        memory = cls(**config)

        # Restore messages
        for msg_data in data.get("messages", []):
            message = Message.from_dict(msg_data)
            memory.messages.append(message)

        # Restore summaries
        for summary_data in data.get("summaries", []):
            summary = ConversationSummary.from_dict(summary_data)
            memory.summaries.append(summary)

        # Restore statistics
        stats = data.get("statistics", {})
        memory.total_messages_added = stats.get("total_messages_added", 0)
        memory.total_summarizations = stats.get("total_summarizations", 0)
        memory._current_token_count = memory.get_token_count()

        return memory

    def save_to_file(self, filepath: str) -> None:
        """
        Save memory to JSON file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> "ShortTermMemory":
        """
        Load memory from JSON file.

        Args:
            filepath: Path to load file

        Returns:
            ShortTermMemory instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
