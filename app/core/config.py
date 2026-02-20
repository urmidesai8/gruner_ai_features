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
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", None)
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_DECODE_RESPONSES: bool = True

settings = Settings()
