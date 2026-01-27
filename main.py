from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import api_router


app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes (HTML, WebSocket, REST)
app.include_router(api_router)


if __name__ == "__main__":
    print("Starting WebSocket server on http://localhost:8000")
    print("Open http://localhost:8000 in your browser to test the chat client")
    uvicorn.run(app, host="0.0.0.0", port=8000)