import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

# Allowed image MIME types.
ALLOWED_IMAGE_TYPES: set[str] = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_MB: int = 10

# Allowed video MIME types.
ALLOWED_VIDEO_TYPES: set[str] = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
}
MAX_VIDEO_SIZE_MB: int = 100

# Combined set of all accepted media MIME types.
ALLOWED_MEDIA_TYPES: set[str] = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES

# OpenAI configuration.
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_IMAGE_MODEL: str = os.getenv("OPENAI_IMAGE_MODEL", "gpt-4o")
