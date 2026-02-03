from datetime import datetime
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.schemas import chat_history, manager

router = APIRouter()


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


