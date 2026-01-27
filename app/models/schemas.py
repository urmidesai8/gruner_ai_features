from pydantic import BaseModel
from typing import List, Optional

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
