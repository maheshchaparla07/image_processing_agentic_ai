from typing import Any, Literal, Optional

from pydantic import BaseModel


class MediaResponse(BaseModel):
    """Unified response for the /media/upload endpoint."""

    # Basic file identity.
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None

    # Pipeline outputs.
    media_type: Optional[Literal["image", "unknown"]] = None
    metadata: Optional[dict[str, Any]] = None
    ai_analysis: Optional[str] = None
    ai_detection_result: Optional[Literal["AI_GENERATED", "NOT_AI_GENERATED"]] = None
    decision: Optional[
        Literal[
            "DEEP_FAKE",
            "AI_GENERATED",
            "DIGITALLY_EDITED",
            "REAL",
            "OTHER",
            "ABSTAIN",
        ]
    ] = None

    # Output storage paths.
    stored_file_path: Optional[str] = None        # Present when decision == REAL.
    watermarked_file_path: Optional[str] = None   # Present when decision == AI_GENERATED.

    # Processing audit trail.
    processing_log: list[str] = []

    class Config:
        from_attributes = True


# Backward-compatible alias for older imports.
ImageResponse = MediaResponse
