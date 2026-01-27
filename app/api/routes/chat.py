from datetime import datetime
import json
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from app.models.chat import chat_history, manager
from app.services.summarizer import generate_chat_summary


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def get_chat_client() -> HTMLResponse:
    """Serve a simple HTML client for testing."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Chat Client</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
            }
            #messages {
                border: 1px solid #ccc;
                height: 400px;
                overflow-y: auto;
                padding: 10px;
                margin-bottom: 10px;
                background-color: #f9f9f9;
            }
            #usernameInput {
                width: 70%;
                padding: 10px;
                font-size: 16px;
                margin-bottom: 10px;
            }
            #messageInput {
                width: 70%;
                padding: 10px;
                font-size: 16px;
            }
            .message {
                margin: 5px 0;
                padding: 5px;
            }
            .message.own {
                background-color: #e3f2fd;
                text-align: right;
            }
            .message.other {
                background-color: #f5f5f5;
            }
            .message.system {
                background-color: #fff3cd;
                font-style: italic;
                text-align: center;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            button:disabled {
                background-color: #cccccc;
                cursor: not-allowed;
            }
            .status {
                margin: 10px 0;
                padding: 5px;
                font-weight: bold;
            }
            .connected {
                color: green;
            }
            .disconnected {
                color: red;
            }
            #summaryBtn {
                background-color: #2196F3;
                margin-left: 10px;
            }
            #summaryBtn:hover {
                background-color: #1976D2;
            }
            #summaryModal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
            }
            #summaryContent {
                background-color: #fefefe;
                margin: 5% auto;
                padding: 20px;
                border: 1px solid #888;
                width: 80%;
                max-width: 700px;
                max-height: 80vh;
                overflow-y: auto;
                border-radius: 10px;
            }
            .close {
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }
            .close:hover {
                color: black;
            }
            .summary-section {
                margin: 20px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .summary-section h3 {
                margin-top: 0;
                color: #2196F3;
            }
            .summary-section ul {
                margin: 10px 0;
                padding-left: 20px;
            }
            .summary-section li {
                margin: 5px 0;
            }
            #summaryLoading {
                text-align: center;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Multi-User WebSocket Chat</h1>
        <div id="status" class="status disconnected">Disconnected</div>
        <div id="userCount">Users online: 0</div>
        <input type="text" id="usernameInput" placeholder="Enter your username..." maxlength="20">
        <div id="messages"></div>
        <input type="text" id="messageInput" placeholder="Type your message..." disabled>
        <button id="sendBtn" onclick="sendMessage()" disabled>Send</button>
        <button id="connectBtn" onclick="connect()">Connect</button>
        <button id="disconnectBtn" onclick="disconnect()" disabled>Disconnect</button>
        <button id="summaryBtn" onclick="generateSummary()" disabled>📊 Generate Summary</button>
        
        <!-- Summary Modal -->
        <div id="summaryModal">
            <div id="summaryContent">
                <span class="close" onclick="closeSummary()">&times;</span>
                <h2>Chat Summary</h2>
                <div id="summaryLoading" style="display: none;">Generating summary...</div>
                <div id="summaryResults"></div>
            </div>
        </div>

        <script>
            let ws = null;
            let currentUsername = '';

            function connect() {
                const usernameInput = document.getElementById('usernameInput');
                const username = usernameInput.value.trim();
                
                if (!username) {
                    alert('Please enter a username');
                    return;
                }
                
                currentUsername = username;
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws?username=${encodeURIComponent(username)}`);

                ws.onopen = function(event) {
                    updateStatus('Connected', true);
                    document.getElementById('usernameInput').disabled = true;
                    document.getElementById('messageInput').disabled = false;
                    document.getElementById('sendBtn').disabled = false;
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('disconnectBtn').disabled = false;
                    document.getElementById('summaryBtn').disabled = false;
                };

                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'user_count') {
                        document.getElementById('userCount').textContent = `Users online: ${data.count}`;
                    } else if (data.type === 'message') {
                        const isOwn = data.sender === currentUsername;
                        addMessage(data.sender, data.message, isOwn ? 'own' : 'other');
                    } else if (data.type === 'system') {
                        addMessage('System', data.message, 'system');
                    }
                };

                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    addMessage('System', 'Error occurred', 'system');
                };

                ws.onclose = function(event) {
                    updateStatus('Disconnected', false);
                    document.getElementById('usernameInput').disabled = false;
                    document.getElementById('messageInput').disabled = true;
                    document.getElementById('sendBtn').disabled = true;
                    document.getElementById('connectBtn').disabled = false;
                    document.getElementById('disconnectBtn').disabled = true;
                    document.getElementById('summaryBtn').disabled = true;
                    document.getElementById('userCount').textContent = 'Users online: 0';
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }

            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (message && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'message',
                        message: message
                    }));
                    input.value = '';
                }
            }

            function addMessage(sender, message, messageType = 'other') {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${messageType}`;
                
                if (messageType === 'system') {
                    messageDiv.innerHTML = message;
                } else {
                    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
                }
                
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            function updateStatus(status, connected) {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = status;
                statusDiv.className = connected ? 'status connected' : 'status disconnected';
            }

            // Allow Enter key to send message
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Allow Enter key to connect
            document.getElementById('usernameInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    connect();
                }
            });
            
            // Summary functionality
            async function generateSummary() {
                if (!currentUsername) {
                    alert('Please connect first');
                    return;
                }
                
                const modal = document.getElementById('summaryModal');
                const loading = document.getElementById('summaryLoading');
                const results = document.getElementById('summaryResults');
                
                modal.style.display = 'block';
                loading.style.display = 'block';
                results.innerHTML = '';
                
                try {
                    const response = await fetch(`/api/summarize?username=${encodeURIComponent(currentUsername)}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    });
                    const data = await response.json();

                    if (!response.ok) {
                        const detail = data && data.detail ? data.detail : 'Server returned an error';
                        throw new Error(detail);
                    }

                    const summary = data.summary || 'No summary available.';
                    const totalMessages = typeof data.total_messages === 'number' ? data.total_messages : 0;
                    const participants = Array.isArray(data.participants) ? data.participants : [];
                    const bulletPoints = Array.isArray(data.bullet_points) ? data.bullet_points : [];
                    const keyDecisions = Array.isArray(data.key_decisions) ? data.key_decisions : [];
                    const actionItems = Array.isArray(data.action_items) ? data.action_items : [];
                    const unreadSummary = data.unread_summary || 'No unread summary available.';
                    
                    loading.style.display = 'none';
                    
                    let html = `
                        <div class="summary-section">
                            <h3>📋 Overview</h3>
                            <p>${summary}</p>
                            <p><strong>Total Messages:</strong> ${totalMessages}</p>
                            <p><strong>Participants:</strong> ${participants.join(', ')}</p>
                        </div>
                        
                        <div class="summary-section">
                            <h3>💬 What Did I Miss?</h3>
                            <p>${unreadSummary}</p>
                        </div>
                        
                        <div class="summary-section">
                            <h3>📝 Key Points</h3>
                            <ul>
                                ${bulletPoints.map(point => `<li>${point}</li>`).join('')}
                            </ul>
                        </div>
                        
                        <div class="summary-section">
                            <h3>✅ Key Decisions</h3>
                            <ul>
                                ${keyDecisions.map(decision => `<li>${decision}</li>`).join('')}
                            </ul>
                        </div>
                        
                        <div class="summary-section">
                            <h3>🎯 Action Items</h3>
                            <ul>
                                ${actionItems.map(action => `<li>${action}</li>`).join('')}
                            </ul>
                        </div>
                    `;
                    
                    results.innerHTML = html;
                } catch (error) {
                    loading.style.display = 'none';
                    results.innerHTML = `<p style="color: red;">Error generating summary: ${error.message}</p>`;
                }
            }
            
            function closeSummary() {
                document.getElementById('summaryModal').style.display = 'none';
            }
            
            // Close modal when clicking outside
            window.onclick = function(event) {
                const modal = document.getElementById('summaryModal');
                if (event.target == modal) {
                    modal.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


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

            chat_history.add_message(username, message_text, timestamp)

            await manager.broadcast(
                {
                    "type": "message",
                    "sender": username,
                    "message": message_text,
                    "timestamp": timestamp,
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


@router.post("/api/summarize")
async def summarize_chat(username: Optional[str] = None) -> JSONResponse:
    """
    Generate chat summary.

    Query parameters:
    - username: Optional username to get personalized "What did I miss?" summary.
    """
    try:
        if username:
            messages = chat_history.get_unread_messages(username)
            if not messages:
                messages = chat_history.get_all_messages()
        else:
            messages = chat_history.get_all_messages()

        summary = generate_chat_summary(messages, username)

        if username:
            chat_history.mark_as_read(username)

        return JSONResponse(content=summary)

    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}") from e


@router.get("/api/messages")
async def get_messages(username: Optional[str] = None) -> JSONResponse:
    """Get all chat messages or unread messages for a user."""
    if username:
        return JSONResponse(
            content={
                "messages": chat_history.get_unread_messages(username),
                "unread_count": chat_history.get_unread_count(username),
            }
        )

    return JSONResponse(
        content={
            "messages": chat_history.get_all_messages(),
            "total_count": len(chat_history.messages),
        }
    )

