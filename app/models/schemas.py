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

class FeatureRequest(BaseModel):
    """Model for AI feature requests"""
    id: str
    sender: str
    message: str


class TranslationRequest(BaseModel):
    """Request model for language translation"""
    id: str
    text: str
    target_language: str  # e.g. 'en', 'es', 'fr', 'de', 'hi', 'zh', 'ja'


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
    """Stores chat message history with unread tracking."""

    def __init__(self) -> None:
        self.messages: List[ChatMessage] = []
        # username -> last message index read
        self.user_last_read: Dict[str, int] = {}

    def add_message(self, sender: str, message: str, timestamp: str) -> ChatMessage:
        """Add a new message to the history."""
        msg = ChatMessage(
            sender=sender,
            message=message,
            timestamp=timestamp,
            message_id=str(uuid.uuid4()),
        )
        self.messages.append(msg)
        return msg

    def get_all_messages(self) -> List[dict]:
        """Get all messages as dictionaries."""
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
