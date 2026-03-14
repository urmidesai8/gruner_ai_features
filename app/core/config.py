import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    AI_MODEL: str = "llama-3.3-70b-versatile"
    # Qdrant vector DB settings
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_INDIVIDUAL_COLLECTION: str = os.getenv(
        "QDRANT_INDIVIDUAL_COLLECTION", "individual_chats"
    )
    QDRANT_GROUP_COLLECTION: str = os.getenv(
        "QDRANT_GROUP_COLLECTION", "group_chats"
    )
    QDRANT_MEETING_TRANSCRIPTION_COLLECTION: str = os.getenv(
        "QDRANT_MEETING_TRANSCRIPTION_COLLECTION", "meeting_transcription"
    )
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", None)
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_DECODE_RESPONSES: bool = True
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8001")

    # Assistant session / memory (production)
    ASSISTANT_SESSION_TTL_SECONDS: int = int(
        os.getenv("ASSISTANT_SESSION_TTL_SECONDS", "86400")
    )  # 24h
    ASSISTANT_MAX_HISTORY_MESSAGES: int = int(
        os.getenv("ASSISTANT_MAX_HISTORY_MESSAGES", "40")
    )  # last 20 user+assistant pairs
    ASSISTANT_MAX_MESSAGE_LENGTH: int = int(
        os.getenv("ASSISTANT_MAX_MESSAGE_LENGTH", "16384")
    )
    ASSISTANT_EXECUTOR_WORKERS: int = int(
        os.getenv("ASSISTANT_EXECUTOR_WORKERS", "4")
    )

    # V2V (voice-to-voice): TTS voice for assistant reply (edge-tts voice id)
    TTS_VOICE: str = os.getenv("TTS_VOICE", "en-US-JennyNeural")

    # Nova Sonic voice backend (optional) — Socket.IO server URL for live voice
    NOVA_VOICE_URL: str = os.getenv("NOVA_VOICE_URL", "")

    # Postgres settings for assistant feedback
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "gruner")

settings = Settings()
