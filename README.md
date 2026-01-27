# FastAPI WebSocket Multi-User Chat Server

A WebSocket server built with FastAPI that enables real-time chat conversations between multiple users. Messages from one user are broadcast to all other connected users, and all messages are displayed on the server console.

## Features

- **Multi-user chat** - Multiple users can connect and chat with each other
- **Real-time messaging** - Messages are instantly broadcast to all connected users
- **User identification** - Each user has a unique username
- **Connection management** - Tracks user connections and disconnections
- **Server logging** - All messages are displayed on the server console with timestamps
- **User count display** - Shows how many users are currently online
- **Simple HTML client** - Built-in web interface for testing
- **Smart Chat Summaries** - AI-powered summarization using Groq's Llama 3.1 8B instant model:
  - Bullet point summaries of long chat threads
  - "What did I miss?" summaries for unread messages
  - Key decisions extraction
  - Action items identification

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Groq API key:
   - Get your API key from [Groq Console](https://console.groq.com/)
   - Create a `.env` file in the project root:
   ```bash
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
   - Or set it as an environment variable:
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```

## Running the Server

Start the server with:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Enter a username in the username field
3. Click "Connect" to establish a WebSocket connection
4. Type messages in the input field and click "Send" (or press Enter)
5. Messages from all users will be displayed in your browser
6. Open multiple browser tabs/windows with different usernames to test multi-user chat
7. All messages are also printed on the server console

### Smart Chat Summaries

To generate a summary of the conversation:

1. Click the **"📊 Generate Summary"** button (available after connecting)
2. A modal will appear with:
   - **Overview** - Total messages and participants
   - **What Did I Miss?** - Summary of unread messages
   - **Key Points** - Bullet point summary of the conversation
   - **Key Decisions** - Extracted decisions from the chat
   - **Action Items** - Identified tasks and action items

The summary is personalized based on your username and will show unread messages if you've been away.

## Server Output

All user connections and messages are displayed on the server console with timestamps:
```
[2026-01-27 10:30:45] User 'Alice' (a1b2c3d4...) connected from 127.0.0.1
[2026-01-27 10:30:45] Total users online: 1
[2026-01-27 10:30:50] User 'Bob' (e5f6g7h8...) connected from 127.0.0.1
[2026-01-27 10:30:50] Total users online: 2
[2026-01-27 10:30:55] Alice (a1b2c3d4...): Hello, Bob!
[2026-01-27 10:31:00] Bob (e5f6g7h8...): Hi Alice, how are you?
[2026-01-27 10:31:05] Alice (a1b2c3d4...) disconnected
[2026-01-27 10:31:05] Total users online: 1
```

## API Endpoints

- `GET /` - Serves the HTML chat client interface
- `WS /ws?username=<username>` - WebSocket endpoint for chat messages (username is required)
- `POST /api/summarize?username=<username>` - Generate chat summary (username is optional)
- `GET /api/messages?username=<username>` - Get all messages or unread messages for a user

## Message Format

Messages sent to the server should be JSON:
```json
{
  "type": "message",
  "message": "Your message text here"
}
```

The server broadcasts messages in this format:
```json
{
  "type": "message",
  "sender": "username",
  "message": "message text",
  "timestamp": "2026-01-27 10:30:55"
}
```

## Testing with Custom Clients

You can also connect to the WebSocket endpoint using any WebSocket client:

```python
import asyncio
import websockets
import json

async def test_client(username):
    uri = f"ws://localhost:8000/ws?username={username}"
    async with websockets.connect(uri) as websocket:
        # Send a message
        message = {
            "type": "message",
            "message": f"Hello from {username}!"
        }
        await websocket.send(json.dumps(message))
        
        # Receive responses
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Received: {data}")

# Run multiple clients to test multi-user chat
asyncio.run(test_client("User1"))
```

## Smart Chat Summary API

The summary endpoint can be called programmatically:

```python
import requests

# Generate summary for a specific user
response = requests.post("http://localhost:8000/api/summarize", params={"username": "Alice"})
summary = response.json()

print("Summary:", summary["summary"])
print("Bullet Points:", summary["bullet_points"])
print("Key Decisions:", summary["key_decisions"])
print("Action Items:", summary["action_items"])
print("Unread Summary:", summary["unread_summary"])
```

The summary response includes:
- `summary` - Overall conversation summary
- `bullet_points` - List of key message points
- `key_decisions` - Extracted decisions from the conversation
- `action_items` - Identified action items and tasks
- `unread_summary` - Personalized "What did I miss?" summary
- `total_messages` - Total number of messages
- `participants` - List of participants in the conversation
