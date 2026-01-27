from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import json
import uuid
from ...services.chat_service import manager, chat_history

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, username: str = "Anonymous"):
    user_id = str(uuid.uuid4())
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                msg_data = json.loads(data)
                text = msg_data.get("message", "")
            except:
                text = data
            
            chat_history.add_message(username, text, timestamp)
            
            await manager.broadcast({
                "type": "message",
                "sender": username,
                "message": text,
                "timestamp": timestamp,
                "message_id": str(uuid.uuid4())
            }, exclude_user_id=user_id)
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
