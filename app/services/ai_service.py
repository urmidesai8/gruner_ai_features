from groq import Groq
from ..core.config import settings

groq_client = Groq(api_key=settings.GROQ_API_KEY)

def call_groq_ai(prompt: str) -> str:
    if not groq_client.api_key:
        return "Error: GROQ_API_KEY not set."
    
    try:
        completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=settings.AI_MODEL,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"
