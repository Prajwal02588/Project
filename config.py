import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── Groq API ──────────────────────────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # Models available on this Groq account:
    #   openai/gpt-oss-120b   ← best quality (default)
    #   openai/gpt-oss-20b    ← faster / lower latency
    #   qwen/qwen3.8-27b      ← alternative mid-size model
    MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # ── Flask ─────────────────────────────────────────────────────────────────
    SECRET_KEY          = os.getenv("SECRET_KEY", "smart-study-secret-2024")
    DEBUG               = os.getenv("DEBUG", "True").lower() == "true"
    MAX_CONTENT_LENGTH  = 16 * 1024 * 1024   # 16 MB

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    DATA_FOLDER   = os.path.join(BASE_DIR, "data")

    ALLOWED_EXTENSIONS = {"pdf", "txt", "png", "jpg", "jpeg", "wav", "mp3", "m4a"}

    # ── Agent behaviour ───────────────────────────────────────────────────────
    MAX_TOKENS  = 4096
    TEMPERATURE = 0.7

    SYSTEM_PROMPT = (
        "You are SmartStudy AI, an expert educational assistant. "
        "You help students learn effectively by summarizing content, "
        "creating flashcards, generating quizzes, building study plans, "
        "and predicting important exam topics. "
        "Always structure your responses clearly with headings and bullet points. "
        "Be encouraging, concise, and pedagogically sound."
    )
