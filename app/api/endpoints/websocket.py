from datetime import datetime
import json
import uuid
from typing import Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.schemas import chat_history, manager

router = APIRouter()

# Static user IDs allowed for group chat (same as frontend dummy)
GROUP_CHAT_USER_IDS = {
    "550e8400-e29b-41d4-a716-446655440001",  # User 1
    "550e8400-e29b-41d4-a716-446655440002",  # User 2
    "550e8400-e29b-41d4-a716-446655440003",  # User 3
}


class GroupConnectionManager:
    """Manages WebSocket connections per group. Each connection is identified by user_id."""

    def __init__(self) -> None:
        # group_id -> { user_id -> WebSocket }
        self._groups: Dict[str, Dict[str, WebSocket]] = {}
        # group_id -> { user_id -> username }
        self._names: Dict[str, Dict[str, str]] = {}

    async def connect(
        self, websocket: WebSocket, group_id: str, user_id: str, username: str
    ) -> None:
        await websocket.accept()
        if group_id not in self._groups:
            self._groups[group_id] = {}
            self._names[group_id] = {}
        self._groups[group_id][user_id] = websocket
        self._names[group_id][user_id] = username

    def disconnect(self, group_id: str, user_id: str) -> None:
        if group_id in self._groups and user_id in self._groups[group_id]:
            del self._groups[group_id][user_id]
        if group_id in self._names and user_id in self._names[group_id]:
            del self._names[group_id][user_id]

    async def broadcast_to_group(
        self, group_id: str, message: dict, exclude_user_id: str | None = None
    ) -> None:
        if group_id not in self._groups:
            return
        disconnected: List[str] = []
        for uid, connection in list(self._groups[group_id].items()):
            if uid != exclude_user_id:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Group broadcast error to {uid}: {e}")
                    disconnected.append(uid)
        for uid in disconnected:
            self.disconnect(group_id, uid)

    def get_username(self, group_id: str, user_id: str) -> str:
        return (self._names.get(group_id) or {}).get(user_id, "Unknown")


group_manager = GroupConnectionManager()
GROUP_ID = "default"


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, username: str = "Anonymous") -> None:
    """WebSocket endpoint for multi-user chat."""
    user_id = str(uuid.uuid4())
    client_address = websocket.client.host if websocket.client else "unknown"

    await manager.connect(websocket, user_id, username)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] User '{username}' ({user_id[:8]}...) connected from {client_address}")
    print(f"[{timestamp}] Total users online: {manager.get_user_count()}")

    await manager.broadcast(
        {
            "type": "system",
            "message": f"{username} joined the chat",
        },
        exclude_user_id=user_id,
    )

    await manager.send_personal_message(
        {
            "type": "system",
            "message": f"Welcome to the chat, {username}!",
        },
        user_id,
    )

    await manager.send_personal_message(
        {
            "type": "user_count",
            "count": manager.get_user_count(),
        },
        user_id,
    )

    await manager.broadcast(
        {
            "type": "user_count",
            "count": manager.get_user_count(),
        }
    )

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)
                message_text = message_data.get("message", data)
            except json.JSONDecodeError:
                message_text = data

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {username} ({user_id[:8]}...): {message_text}")

            # For audio messages ([AUDIO]: URL), keep the full text for broadcasting
            # so the frontend can render the audio player, but store a cleaned
            # placeholder in history so users don't see internal URLs in summaries.
            stored_message = message_text
            if isinstance(message_text, str) and message_text.startswith("[AUDIO]: "):
                stored_message = "[AUDIO]"

            # Add message to history and get the message object with message_id
            # Pass current AI state when adding message
            chat_message = chat_history.add_message(
                username,
                stored_message,
                timestamp,
                ai_enabled=chat_history.get_ai_enabled(),
            )

            await manager.broadcast(
                {
                    "type": "message",
                    "sender": username,
                    "message": message_text,
                    "timestamp": timestamp,
                    "message_id": chat_message.message_id,  # Include message_id for frontend
                    "ai_enabled": chat_message.ai_enabled,  # Include AI state for frontend
                },
                exclude_user_id=user_id,
            )

    except WebSocketDisconnect:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] User '{username}' ({user_id[:8]}...) disconnected")
        manager.disconnect(user_id)
        print(f"[{timestamp}] Total users online: {manager.get_user_count()}")

        await manager.broadcast(
            {
                "type": "system",
                "message": f"{username} left the chat",
            }
        )

        await manager.broadcast(
            {
                "type": "user_count",
                "count": manager.get_user_count(),
            }
        )

    except Exception as e:  # pragma: no cover - defensive
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Error with user {username}: {e}")
        manager.disconnect(user_id)


@router.websocket("/ws/group")
async def group_websocket_endpoint(
    websocket: WebSocket,
    user_id: str = "",
    username: str = "Anonymous",
) -> None:
    """WebSocket endpoint for 3-user group chat. Uses static user_id for User 1, 2, 3."""
    if not user_id or user_id not in GROUP_CHAT_USER_IDS:
        await websocket.close(code=4000, reason="Invalid or missing user_id for group chat")
        return

    client_address = websocket.client.host if websocket.client else "unknown"
    await group_manager.connect(websocket, GROUP_ID, user_id, username)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Group: '{username}' ({user_id}) connected from {client_address}")

    await group_manager.broadcast_to_group(
        GROUP_ID,
        {"type": "system", "message": f"{username} joined the group chat"},
        exclude_user_id=user_id,
    )
    await websocket.send_json(
        {"type": "system", "message": f"Welcome to the group chat, {username}!"}
    )

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                message_text = message_data.get("message", data)
            except json.JSONDecodeError:
                message_text = data

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Group {username} ({user_id}): {message_text}")

            await group_manager.broadcast_to_group(
                GROUP_ID,
                {
                    "type": "message",
                    "sender": username,
                    "sender_id": user_id,
                    "message": message_text,
                    "timestamp": timestamp,
                },
            )
    except WebSocketDisconnect:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Group: '{username}' ({user_id}) disconnected")
        group_manager.disconnect(GROUP_ID, user_id)
        await group_manager.broadcast_to_group(
            GROUP_ID,
            {"type": "system", "message": f"{username} left the group chat"},
        )
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Group error for {username}: {e}")
        group_manager.disconnect(GROUP_ID, user_id)
