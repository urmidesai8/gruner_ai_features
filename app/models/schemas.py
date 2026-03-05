from pydantic import BaseModel
from typing import List, Optional, Dict
from fastapi import WebSocket
import uuid
import json
import redis
from app.core.config import settings

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

class AIAnalysisRequest(BaseModel):
    """Generic request model for AI analysis (prioritize, moderate, tasks)"""
    messages: List[FeatureRequest]
    model: Optional[str] = None

class SmartRepliesRequest(BaseModel):
    """Request model for smart replies generation"""
    messages: List[FeatureRequest]
    tone: str = "auto"
    model: Optional[str] = None


class DraftResponseRequest(BaseModel):
    """Request model for draft response rewriting (grammar + tone)."""
    message: str
    tone: str

class TranslationRequest(BaseModel):
    """Request model for language translation"""
    id: str
    text: str
    target_language: str
    model: Optional[str] = None

class TextTranslationRequest(BaseModel):
    """Request model for generic text translation (e.g. transcripts)"""
    text: str
    target_language: str
    model: Optional[str] = None

class SummarizeRequest(BaseModel):
    """Request model for chat summarization"""
    username: Optional[str] = None
    total_messages: Optional[int] = 100
    model: Optional[str] = None

class ReminderSuggestionRequest(BaseModel):
    """Request model for context-based reminder suggestions"""
    username: Optional[str] = None
    context_window: Optional[int] = None
    model: Optional[str] = None


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
    """Stores chat message history with unread tracking and AI state management using Redis."""

    def __init__(self) -> None:
        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                db=settings.REDIS_DB,
                decode_responses=settings.REDIS_DECODE_RESPONSES,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            print("Connected to Redis successfully")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"Warning: Could not connect to Redis: {e}")
            print("Falling back to in-memory storage")
            self.redis_client = None
            self.messages: List[ChatMessage] = []
        
        # Redis keys
        self.MESSAGES_KEY = "chat:messages"
        self.USER_LAST_READ_KEY = "chat:user_last_read"
        self.AI_ENABLED_KEY = "chat:ai_enabled"
        self.AI_TOGGLE_HISTORY_KEY = "chat:ai_toggle_history"
        
        # Initialize AI state in Redis if not exists
        if self.redis_client:
            if not self.redis_client.exists(self.AI_ENABLED_KEY):
                self.redis_client.set(self.AI_ENABLED_KEY, "true")
        else:
            # Fallback to in-memory
            self.user_last_read: Dict[str, int] = {}
            self.ai_enabled: bool = True
            self.ai_toggle_history: List[Dict] = []

    def add_message(self, sender: str, message: str, timestamp: str, ai_enabled: Optional[bool] = None) -> ChatMessage:
        """Add a new message to the history.
        
        Args:
            sender: Message sender
            message: Message text
            timestamp: Message timestamp
            ai_enabled: Whether AI was enabled (defaults to current AI state)
        """
        if ai_enabled is None:
            ai_enabled = self.get_ai_enabled()
        
        msg = ChatMessage(
            sender=sender,
            message=message,
            timestamp=timestamp,
            message_id=str(uuid.uuid4()),
            ai_enabled=ai_enabled,
        )
        
        if self.redis_client:
            # Store in Redis as JSON in a list
            msg_dict = msg.dict()
            self.redis_client.rpush(self.MESSAGES_KEY, json.dumps(msg_dict))
        else:
            # Fallback to in-memory
            self.messages.append(msg)
        
        return msg
    
    def set_ai_enabled(self, enabled: bool) -> None:
        """Set the AI enabled state."""
        from datetime import datetime
        toggle_entry = {
            "enabled": enabled,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if self.redis_client:
            self.redis_client.set(self.AI_ENABLED_KEY, str(enabled).lower())
            self.redis_client.rpush(self.AI_TOGGLE_HISTORY_KEY, json.dumps(toggle_entry))
        else:
            # Fallback to in-memory
            self.ai_enabled = enabled
            self.ai_toggle_history.append(toggle_entry)
    
    def get_ai_enabled(self) -> bool:
        """Get the current AI enabled state."""
        if self.redis_client:
            value = self.redis_client.get(self.AI_ENABLED_KEY)
            return value.lower() == "true" if value else True
        else:
            # Fallback to in-memory
            return self.ai_enabled
    
    def _get_all_messages_from_redis(self) -> List[dict]:
        """Internal method to get all messages from Redis or memory."""
        if self.redis_client:
            messages_json = self.redis_client.lrange(self.MESSAGES_KEY, 0, -1)
            return [json.loads(msg) for msg in messages_json]
        else:
            # Fallback to in-memory
            return [msg.dict() for msg in self.messages]
    
    def get_ai_enabled_messages(self) -> List[dict]:
        """Get only messages that were created when AI was enabled."""
        all_messages = self._get_all_messages_from_redis()
        return [msg for msg in all_messages if msg.get("ai_enabled", True)]
    
    def get_all_messages_for_summary(self) -> List[dict]:
        """Get all messages (for chat summary feature which always uses all messages)."""
        return self._get_all_messages_from_redis()

    def get_all_messages(self) -> List[dict]:
        """Get all messages as dictionaries (for display purposes)."""
        return self._get_all_messages_from_redis()

    def get_messages_since(self, since_index: int = 0) -> List[dict]:
        """Get messages since a specific index."""
        all_messages = self._get_all_messages_from_redis()
        return all_messages[since_index:]

    def get_unread_count(self, username: str) -> int:
        """Get count of unread messages for a user."""
        all_messages = self._get_all_messages_from_redis()
        total_count = len(all_messages)
        
        if self.redis_client:
            last_read_str = self.redis_client.hget(self.USER_LAST_READ_KEY, username)
            last_read = int(last_read_str) if last_read_str else 0
        else:
            # Fallback to in-memory
            last_read = self.user_last_read.get(username, 0)
        
        return total_count - last_read

    def mark_as_read(self, username: str) -> None:
        """Mark all messages as read for a user."""
        all_messages = self._get_all_messages_from_redis()
        total_count = len(all_messages)
        
        if self.redis_client:
            self.redis_client.hset(self.USER_LAST_READ_KEY, username, total_count)
        else:
            # Fallback to in-memory
            self.user_last_read[username] = total_count

    def get_unread_messages(self, username: str) -> List[dict]:
        """Get unread messages for a user."""
        all_messages = self._get_all_messages_from_redis()
        
        if self.redis_client:
            last_read_str = self.redis_client.hget(self.USER_LAST_READ_KEY, username)
            last_read = int(last_read_str) if last_read_str else 0
        else:
            # Fallback to in-memory
            last_read = self.user_last_read.get(username, 0)
        
        return all_messages[last_read:]


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
        # Iterate over a copy to allow modification during iteration
        for user_id, connection in list(self.active_connections.items()):
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
