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

settings = Settings()
