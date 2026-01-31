"""
Agent state management for effGen framework.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import pickle


@dataclass
class AgentState:
    """
    Represents the complete state of an agent.
    
    This includes conversation history, tool usage, memory, and configuration.
    Can be saved and loaded for persistence.
    """
    
    agent_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        self.updated_at = datetime.now()
    
    def add_tool_call(self, tool_name: str, args: Dict, result: Any, error: Optional[str] = None):
        """Record a tool call."""
        self.tool_history.append({
            "tool": tool_name,
            "args": args,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def save(self, filepath: str, format: str = "json"):
        """Save state to file."""
        self.updated_at = datetime.now()
        
        if format == "json":
            with open(filepath, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, filepath: str, format: str = "json") -> "AgentState":
        """Load state from file."""
        if format == "json":
            with open(filepath, "r") as f:
                data = json.load(f)
            # Convert ISO strings back to datetime
            if "created_at" in data:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])
            return cls(**data)
        elif format == "pickle":
            with open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)
    
    def clear_history(self):
        """Clear conversation and tool history."""
        self.conversation_history = []
        self.tool_history = []
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent messages."""
        return self.conversation_history[-n:]
    
    def get_token_count_estimate(self) -> int:
        """Estimate total tokens in conversation history."""
        # Rough estimate: 4 characters per token
        total_chars = sum(len(str(msg.get("content", ""))) for msg in self.conversation_history)
        return total_chars // 4
