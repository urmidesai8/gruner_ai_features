from pydantic import BaseModel
from typing import List, Optional, Dict
from fastapi import WebSocket
import uuid

class ChatMessage(BaseModel):
    """Model for chat message storage"""
    sender: str
    message: str
    timestamp: str
    message_id: str
    ai_enabled: bool = True  # Whether AI was enabled when this message was created

class FeatureRequest(BaseModel):
    """Model for AI feature requests"""
    id: str
    sender: str
    message: str


class SmartRepliesRequest(BaseModel):
    """Request model for smart replies generation"""
    messages: List[FeatureRequest]
    tone: str = "auto"  # auto, professional, casual, friendly, formal


class TranslationRequest(BaseModel):
    """Request model for language translation"""
    id: str
    text: str
    target_language: str  # e.g. 'en', 'es', 'fr', 'de', 'hi', 'zh', 'ja'


class SummarizeRequest(BaseModel):
    """Request model for chat summarization"""
    username: Optional[str] = None
    total_messages: Optional[int] = 100  # Number of recent messages to consider (default: 100)


class ReminderSuggestionRequest(BaseModel):
    """Request model for context-based reminder suggestions"""
    username: Optional[str] = None
    context_window: Optional[int] = None  # Number of recent messages to consider


class ReminderCreateRequest(BaseModel):
    """Request model for one-click reminder creation from action items"""
    task_id: str
    title: str
    description: Optional[str] = None
    due_date: Optional[str] = None  # ISO date format (YYYY-MM-DD)
    assignee: Optional[str] = None
    reminder_time: Optional[str] = None  # ISO datetime format for when to remind


class Reminder(BaseModel):
    """Model for a reminder"""
    id: str
    title: str
    description: Optional[str] = None
    due_date: Optional[str] = None
    assignee: Optional[str] = None
    reminder_time: Optional[str] = None
    created_at: str
    source_task_id: Optional[str] = None
    status: str = "pending"  # pending, completed, cancelled


class ChatHistory:
    """Stores chat message history with unread tracking and AI state management."""

    def __init__(self) -> None:
        self.messages: List[ChatMessage] = []
        # username -> last message index read
        self.user_last_read: Dict[str, int] = {}
        # AI state tracking
        self.ai_enabled: bool = True  # Default: AI is enabled
        self.ai_toggle_history: List[Dict] = []  # Track when AI was toggled

    def add_message(self, sender: str, message: str, timestamp: str, ai_enabled: Optional[bool] = None) -> ChatMessage:
        """Add a new message to the history.
        
        Args:
            sender: Message sender
            message: Message text
            timestamp: Message timestamp
            ai_enabled: Whether AI was enabled (defaults to current AI state)
        """
        if ai_enabled is None:
            ai_enabled = self.ai_enabled
        
        msg = ChatMessage(
            sender=sender,
            message=message,
            timestamp=timestamp,
            message_id=str(uuid.uuid4()),
            ai_enabled=ai_enabled,
        )
        self.messages.append(msg)
        return msg
    
    def set_ai_enabled(self, enabled: bool) -> None:
        """Set the AI enabled state."""
        from datetime import datetime
        self.ai_enabled = enabled
        self.ai_toggle_history.append({
            "enabled": enabled,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_ai_enabled(self) -> bool:
        """Get the current AI enabled state."""
        return self.ai_enabled
    
    def get_ai_enabled_messages(self) -> List[dict]:
        """Get only messages that were created when AI was enabled."""
        return [msg.dict() for msg in self.messages if msg.ai_enabled]
    
    def get_all_messages_for_summary(self) -> List[dict]:
        """Get all messages (for chat summary feature which always uses all messages)."""
        return [msg.dict() for msg in self.messages]

    def get_all_messages(self) -> List[dict]:
        """Get all messages as dictionaries (for display purposes)."""
        return [msg.dict() for msg in self.messages]

    def get_messages_since(self, since_index: int = 0) -> List[dict]:
        """Get messages since a specific index."""
        return [msg.dict() for msg in self.messages[since_index:]]

    def get_unread_count(self, username: str) -> int:
        """Get count of unread messages for a user."""
        last_read = self.user_last_read.get(username, 0)
        return len(self.messages) - last_read

    def mark_as_read(self, username: str) -> None:
        """Mark all messages as read for a user."""
        self.user_last_read[username] = len(self.messages)

    def get_unread_messages(self, username: str) -> List[dict]:
        """Get unread messages for a user."""
        last_read = self.user_last_read.get(username, 0)
        return [msg.dict() for msg in self.messages[last_read:]]


class ConnectionManager:
    """Manages WebSocket connections for multiple users."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_names: Dict[str, str] = {}

    async def connect(self, websocket: WebSocket, user_id: str, username: str) -> str:
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_names[user_id] = username
        return user_id

    def disconnect(self, user_id: str) -> None:
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_names:
            del self.user_names[user_id]

    async def send_personal_message(self, message: dict, user_id: str) -> None:
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)

    async def broadcast(self, message: dict, exclude_user_id: str | None = None) -> None:
        """Broadcast message to all connected users except the sender."""
        disconnected: List[str] = []
        for user_id, connection in self.active_connections.items():
            if user_id != exclude_user_id:
                try:
                    await connection.send_json(message)
                except Exception as e:  # pragma: no cover - defensive
                    print(f"Error sending to {user_id}: {e}")
                    disconnected.append(user_id)

        # Clean up disconnected users
        for user_id in disconnected:
            self.disconnect(user_id)

    def get_user_count(self) -> int:
        return len(self.active_connections)

    def get_username(self, user_id: str) -> str:
        return self.user_names.get(user_id, "Unknown")


# Global instances
chat_history = ChatHistory()
manager = ConnectionManager()
