"""
Config via environment variables. No hardcoded secrets.
Copy .env.example -> .env and edit as needed.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Storage
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./storage/chroma")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_corpus")

    # Chunking defaults (word-based sliding window)
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "200"))       # words per chunk
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "40"))  # words of overlap

    # Retrieval
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "4"))
    MIN_SIMILARITY: float = float(os.getenv("MIN_SIMILARITY", "0.15"))  # below -> "no relevant context"

    # LLM (optional). If absent, app runs in extractive fallback mode.
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")  # auto | anthropic | openai | none
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    # Embedding model (chromadb built-in onnx MiniLM, no torch dependency)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = 384

    DATA_DIR: str = os.getenv("DATA_DIR", "./data/corpus")


settings = Settings()
