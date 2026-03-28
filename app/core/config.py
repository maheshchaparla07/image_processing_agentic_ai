import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Allowed image MIME types.
ALLOWED_IMAGE_TYPES: set[str] = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_MB: int = 10

# Accepted media MIME types (image-only pipeline).
ALLOWED_MEDIA_TYPES: set[str] = ALLOWED_IMAGE_TYPES

# OpenAI configuration.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_IMAGE_MODEL: str = os.getenv("OPENAI_IMAGE_MODEL", "gpt-4o")
