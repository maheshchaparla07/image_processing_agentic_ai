import os

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

#Image types 
ALLOWED_IMAGE_TYPES: set[str] = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_IMAGE_SIZE_MB: int = 10

#Video types
ALLOWED_VIDEO_TYPES: set[str] = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
}
MAX_VIDEO_SIZE_MB: int = 100

#  Combined 
ALLOWED_MEDIA_TYPES: set[str] = ALLOWED_IMAGE_TYPES | ALLOWED_VIDEO_TYPES

#  OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_IMAGE_MODEL: str = os.getenv("OPENAI_IMAGE_MODEL", "gpt-4.1-mini")
