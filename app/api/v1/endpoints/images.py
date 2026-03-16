"""
Media upload endpoint (`/api/v1/media/upload`).

Accepts image and video files, then runs the complete 9-agent LangGraph pipeline:

    Upload Agent → File Type Classifier
             ├── Image Agent → AI Detection → Decision
             │                                   ├── REAL          → Store File
             └── Video Agent → AI Detection →    └── AI_GENERATED → Watermark → Store Result
"""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import (
    ALLOWED_MEDIA_TYPES,
    MAX_IMAGE_SIZE_MB,
    MAX_VIDEO_SIZE_MB,
    OPENAI_API_KEY,
    UPLOAD_DIR,
)
from app.core.database import SessionLocal
from app.models.image import Image
from app.schemas.image import MediaResponse
from app.services.media_pipeline_graph import run_media_pipeline

router = APIRouter(prefix="/media", tags=["Media"])
_executor = ThreadPoolExecutor(max_workers=4)

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


# Upload endpoint

@router.post(
    "/upload",
    response_model=MediaResponse,
    summary="Upload image or video for AI analysis",
)
async def upload_media(file: UploadFile = File(...)) -> MediaResponse:
    """
    Accept an image (JPEG, PNG, GIF, WebP) or video (MP4, MPEG, MOV, AVI, WebM).

    Runs the full pipeline and returns:
    - extracted metadata
    - OpenAI analysis
    - decision (`DEEP_FAKE`, `AI_GENERATED`, `DIGITALLY_EDITED`, `REAL`, or `OTHER`)
    - output path (`stored_file_path` or `watermarked_file_path`)
    - full processing log
    """
    # Validate content type.
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_MEDIA_TYPES))}"
            ),
        )

    content = await file.read()
    size_bytes = len(content)

    # Validate size (images and videos have different limits).
    is_video = content_type.startswith("video/")
    max_mb = MAX_VIDEO_SIZE_MB if is_video else MAX_IMAGE_SIZE_MB
    if size_bytes > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum allowed size is {max_mb} MB.",
        )

    # Run the pipeline in a worker thread to keep the event loop responsive.
    loop = asyncio.get_running_loop()
    pipeline_result = await loop.run_in_executor(
        _executor,
        lambda: run_media_pipeline(
            file_bytes=content,
            original_filename=file.filename or "upload",
            content_type=content_type,
            upload_dir=UPLOAD_DIR,
            openai_api_key=OPENAI_API_KEY or "",
        ),
    )

    # Surface pipeline failures as HTTP 422.
    if pipeline_result.get("error"):
        raise HTTPException(
            status_code=422,
            detail=pipeline_result["error"],
        )

    # Try to persist the result in the database.
    db = None
    try:
        db = SessionLocal()
        record = Image(
            filename=file.filename,
            content_type=content_type,
            size_bytes=size_bytes,
            media_type=pipeline_result.get("media_type"),
            metadata_json=json.dumps(pipeline_result.get("metadata") or {}),
            ai_analysis=pipeline_result.get("ai_analysis"),
            ai_detection_result=pipeline_result.get("ai_detection_result"),
            decision=pipeline_result.get("decision"),
            stored_file_path=pipeline_result.get("stored_file_path"),
            watermarked_file_path=pipeline_result.get("watermarked_file_path"),
        )
        db.add(record)
        db.commit()
    except Exception:
        # Do not hide a successful pipeline result because of a DB write issue.
        pass
    finally:
        if db:
            db.close()

    # Build and return API response.
    return MediaResponse(
        filename=file.filename,
        content_type=content_type,
        size_bytes=size_bytes,
        media_type=pipeline_result.get("media_type"),
        metadata=pipeline_result.get("metadata"),
        ai_analysis=pipeline_result.get("ai_analysis"),
        ai_detection_result=pipeline_result.get("ai_detection_result"),
        decision=pipeline_result.get("decision"),
        stored_file_path=pipeline_result.get("stored_file_path"),
        watermarked_file_path=pipeline_result.get("watermarked_file_path"),
        processing_log=pipeline_result.get("processing_log") or [],
    )