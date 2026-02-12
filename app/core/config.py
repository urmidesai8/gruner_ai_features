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
    
    # Model Cache Directory (local project folder)
    # This logic assumes app/core/config.py is 2 levels deep from app root, so 3 levels from project root
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_CACHE_DIR: str = os.path.join(PROJECT_ROOT, "models")

settings = Settings()
