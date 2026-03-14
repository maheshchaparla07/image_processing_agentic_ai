from typing import Any, Literal, Optional

from pydantic import BaseModel


class MediaResponse(BaseModel):
    """Unified response for the /media/upload endpoint."""

    # ── File identity ─────────────────────────────────────────────────────────
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None

    # ── Pipeline outputs ──────────────────────────────────────────────────────
    media_type: Optional[Literal["image", "video", "unknown"]] = None
    metadata: Optional[dict[str, Any]] = None
    ai_analysis: Optional[str] = None
    ai_detection_result: Optional[Literal["AI_GENERATED", "NOT_AI_GENERATED"]] = None
    decision: Optional[Literal["REAL", "AI_GENERATED"]] = None

    # ── Storage paths ─────────────────────────────────────────────────────────
    stored_file_path: Optional[str] = None        # set when decision == REAL
    watermarked_file_path: Optional[str] = None   # set when decision == AI_GENERATED

    # ── Audit trail ───────────────────────────────────────────────────────────
    processing_log: list[str] = []

    class Config:
        from_attributes = True


# Back-compat alias so any code still importing ImageResponse keeps working
ImageResponse = MediaResponse
