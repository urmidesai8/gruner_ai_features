from typing import List, Dict
from fastapi import WebSocket
import uuid
from ..models.schemas import ChatMessage

class ChatHistory:
    """Stores chat message history"""
    
    def __init__(self):
        self.messages: List[ChatMessage] = []
    
    def add_message(self, sender: str, message: str, timestamp: str) -> ChatMessage:
        msg = ChatMessage(
            sender=sender,
            message=message,
            timestamp=timestamp,
            message_id=str(uuid.uuid4())
        )
        self.messages.append(msg)
        return msg
    
    def get_all_messages(self) -> List[dict]:
        return [msg.dict() for msg in self.messages]

chat_history = ChatHistory()

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
    
    async def broadcast(self, message: dict, exclude_user_id: str = None):
        """Broadcast message to all connected users except the sender (optional)"""
        for user_id, connection in self.active_connections.items():
            if user_id != exclude_user_id:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

manager = ConnectionManager()
