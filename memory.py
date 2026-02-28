"""
Memory and conversation management for agents
"""
from typing import List, Dict, Optional
from datetime import datetime


class Message:
    """Represents a conversation message"""
    def __init__(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


class Memory:
    """Manages conversation history and context"""
    def __init__(self, max_history: int = 10):
        self.messages: List[Message] = []
        self.max_history = max_history
        self.agent_thoughts: List[Dict] = []

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to memory"""
        msg = Message(role, content, metadata)
        self.messages.append(msg)
        
        # Keep only recent history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def add_thought(self, agent: str, thought: str, action: Optional[str] = None):
        """Record an agent's thought process"""
        self.agent_thoughts.append({
            "agent": agent,
            "thought": thought,
            "action": action,
            "timestamp": datetime.now()
        })

    def get_context(self, last_n: int = 5) -> str:
        """Get recent conversation context"""
        recent = self.messages[-last_n:] if self.messages else []
        return "\n".join([f"{m.role}: {m.content}" for m in recent])

    def get_thoughts_summary(self) -> str:
        """Get summary of agent thoughts"""
        if not self.agent_thoughts:
            return "No thoughts recorded yet."
        return "\n".join([
            f"[{t['agent']}] {t['thought']}" 
            for t in self.agent_thoughts[-5:]
        ])

    def clear(self):
        """Clear all stored messages and agent thoughts"""
        self.messages.clear()
        self.agent_thoughts.clear()
