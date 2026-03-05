import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.endpoints import chat, memory, documents, audio, meeting, websocket, assistant


def _connection_aware_exception_handler(loop, context):
    """Suppress noisy connection errors when clients disconnect (e.g. tab close, refresh)."""
    exc = context.get("exception")
    if exc is not None and isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
        return  # Client closed connection; no traceback
    asyncio.default_exception_handler(loop, context)


app = FastAPI()


@app.on_event("startup")
async def set_asyncio_exception_handler():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_connection_aware_exception_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(chat.router, prefix="/api/features", tags=["Chat"])
app.include_router(memory.router, prefix="/api/features", tags=["Memory"])
app.include_router(documents.router, prefix="/api/features", tags=["Documents"])
app.include_router(audio.router, prefix="/api/features", tags=["Audio"])
app.include_router(meeting.router, prefix="/api/features", tags=["Meeting"])
app.include_router(assistant.router, prefix="/api/features", tags=["Assistant"])
app.include_router(websocket.router, tags=["websocket"])

# Mount Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """Serve the dual-user AI chat interface."""
    return FileResponse('static/index.html')


if __name__ == "__main__":
    import uvicorn
    print("Starting WebSocket server on http://localhost:8000")
    print("Open http://localhost:8000 in your browser to test the dual-user chat client")
    print("Open http://localhost:8000/chat in your browser to test the multi-user chat with AI features")
    uvicorn.run(app, host="0.0.0.0", port=8000)
