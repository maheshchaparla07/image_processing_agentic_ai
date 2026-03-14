"""
Multi-agent LangGraph pipeline for media AI-detection and processing.

Diagram flow
────────────
User Upload
    │
    ▼
Upload Agent          ← validates file, initialises processing log
    │
    ▼
File Type Classifier  ← routes to Image Agent or Video Agent
    │
  ┌─┴─────────────┐
  ▼               ▼
Image Agent   Video Agent   ← extract rich metadata
  └──────┬────────┘
         ▼
  AI Detection Agent        ← OpenAI vision / text analysis
         │
         ▼
  Decision Agent            ← REAL ──► Store File Agent ──► END
                              AI_GENERATED ──► Watermark Agent
                                                  │
                                                  ▼
                                           Store Result Agent ──► END
"""
from __future__ import annotations

import base64
import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# Media type registries
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
})

VIDEO_CONTENT_TYPES: frozenset[str] = frozenset({
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
})

ALLOWED_MEDIA_TYPES: frozenset[str] = IMAGE_CONTENT_TYPES | VIDEO_CONTENT_TYPES


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state
# ─────────────────────────────────────────────────────────────────────────────

class MediaPipelineState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    file_bytes: bytes
    original_filename: str
    content_type: str
    upload_dir: str
    openai_api_key: str

    # ── Intermediate ──────────────────────────────────────────────────────────
    media_type: Literal["image", "video", "unknown"]
    metadata: dict[str, Any]
    ai_analysis: str
    ai_detection_result: Literal["AI_GENERATED", "NOT_AI_GENERATED"]
    decision: Literal["REAL", "AI_GENERATED"]

    # ── Outputs ───────────────────────────────────────────────────────────────
    stored_file_path: str
    watermarked_file_path: str
    processing_log: list[str]
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_log(state: MediaPipelineState, message: str) -> list[str]:
    log: list[str] = list(state.get("processing_log") or [])
    log.append(message)
    return log


def _unique_path(upload_dir: Path, stem: str, suffix: str, tag: str = "") -> Path:
    tag_part = f"_{tag}" if tag else ""
    return upload_dir / f"{stem}_{uuid.uuid4().hex}{tag_part}{suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# Agent 1 – Upload Agent
# ─────────────────────────────────────────────────────────────────────────────

def upload_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Validates the incoming file and initialises the processing log."""
    if not state.get("file_bytes"):
        return {
            "error": "No file content received.",
            "processing_log": _append_log(state, "upload_agent: ERROR – empty file"),
        }
    if not state.get("original_filename"):
        return {
            "error": "No filename provided.",
            "processing_log": _append_log(state, "upload_agent: ERROR – missing filename"),
        }

    size_kb = round(len(state["file_bytes"]) / 1024, 2)
    msg = (
        f"upload_agent: accepted  filename={state['original_filename']!r}  "
        f"size={size_kb} KB  type={state.get('content_type', 'unknown')}"
    )
    return {"processing_log": _append_log(state, msg)}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 2 – File Type Classifier Agent
# ─────────────────────────────────────────────────────────────────────────────

def file_type_classifier_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Classifies upload as 'image', 'video', or 'unknown' and routes accordingly."""
    ct = (state.get("content_type") or "").lower()
    if ct in IMAGE_CONTENT_TYPES:
        media_type: Literal["image", "video", "unknown"] = "image"
    elif ct in VIDEO_CONTENT_TYPES:
        media_type = "video"
    else:
        media_type = "unknown"

    log = _append_log(state, f"file_type_classifier_agent: classified as '{media_type}'")
    return {"media_type": media_type, "processing_log": log}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 3 – Image Agent
# ─────────────────────────────────────────────────────────────────────────────

def image_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Extracts full image metadata (dimensions, mode, EXIF, colour stats) via Pillow."""
    with Image.open(BytesIO(state["file_bytes"])) as img:
        exif_raw = img.getexif()
        exif: dict[str, str] = (
            {str(k): str(v) for k, v in exif_raw.items()} if exif_raw else {}
        )

        # Basic colour statistics for the first 3 channels
        colour_stats: dict[str, Any] = {}
        try:
            from PIL import ImageStat
            stat = ImageStat.Stat(img)
            colour_stats = {
                "mean": [round(v, 2) for v in stat.mean[:3]],
                "stddev": [round(v, 2) for v in stat.stddev[:3]],
            }
        except Exception:
            pass

        metadata: dict[str, Any] = {
            "format": img.format,
            "mode": img.mode,
            "width": img.width,
            "height": img.height,
            "megapixels": round((img.width * img.height) / 1_000_000, 2),
            "aspect_ratio": f"{img.width}:{img.height}",
            "has_transparency": img.mode in ("RGBA", "LA", "P"),
            "has_exif": bool(exif),
            "exif_fields_count": len(exif),
            "exif_data": exif,
            "colour_stats": colour_stats,
            "sha256": hashlib.sha256(state["file_bytes"]).hexdigest(),
        }

    log = _append_log(
        state,
        f"image_agent: extracted metadata  {img.width}x{img.height}  "
        f"format={img.format}  megapixels={metadata['megapixels']}",
    )
    return {"metadata": metadata, "processing_log": log}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 4 – Video Agent
# ─────────────────────────────────────────────────────────────────────────────

def video_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Extracts video container metadata from binary headers (no ffmpeg required)."""
    raw = state["file_bytes"]

    container = "unknown"
    extra: dict[str, Any] = {}

    # MP4 / MOV: ftyp box at byte 4
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        container = "mp4/mov"
        brand = raw[8:12].decode("latin-1", errors="ignore").strip()
        extra["major_brand"] = brand
    # WebM / MKV: EBML magic
    elif len(raw) >= 4 and raw[:4] == b"\x1a\x45\xdf\xa3":
        container = "webm/mkv"
    # MPEG-1/2 PS: pack start code
    elif len(raw) >= 4 and raw[:4] == b"\x00\x00\x01\xba":
        container = "mpeg-ps"
    # AVI: RIFF....AVI
    elif len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"AVI ":
        container = "avi"

    metadata: dict[str, Any] = {
        "content_type": state.get("content_type"),
        "filename": state.get("original_filename"),
        "size_bytes": len(raw),
        "size_mb": round(len(raw) / (1024 * 1024), 2),
        "container": container,
        "sha256": hashlib.sha256(raw).hexdigest(),
        **extra,
    }

    log = _append_log(
        state,
        f"video_agent: extracted metadata  size={metadata['size_mb']} MB  "
        f"container={container}",
    )
    return {"metadata": metadata, "processing_log": log}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 5 – AI Detection Agent
# ─────────────────────────────────────────────────────────────────────────────

def ai_detection_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Calls OpenAI to determine whether the media was AI-generated.

    • Images  → GPT-4.1-mini vision with base64 data-URL
    • Videos  → GPT-4.1-mini text analysis of extracted metadata
    """
    from openai import OpenAI  # local import; avoids circular deps at module load

    api_key = state.get("openai_api_key") or ""
    if not api_key:
        log = _append_log(state, "ai_detection_agent: OPENAI_API_KEY not set – skipping")
        return {
            "ai_analysis": "OPENAI_API_KEY not configured; AI detection skipped.",
            "ai_detection_result": "NOT_AI_GENERATED",
            "processing_log": log,
        }

    client = OpenAI(api_key=api_key)
    media_type = state.get("media_type", "unknown")

    if media_type == "image":
        img_b64 = base64.b64encode(state["file_bytes"]).decode()
        mime = state.get("content_type") or "image/png"
        data_url = f"data:{mime};base64,{img_b64}"

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image carefully.\n\n"
                            "1. Determine if it appears AI-generated or photographed.\n"
                            "2. Identify specific visual artifacts, lighting inconsistencies, "
                            "unnatural textures, or generative model signatures.\n\n"
                            "Respond EXACTLY in this format:\n"
                            "RESULT: AI_GENERATED or NOT_AI_GENERATED\n"
                            "ANALYSIS: <concise explanation>"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=400,
        )
    else:
        # Video: send metadata as context for text-based analysis
        meta_str = json.dumps(state.get("metadata") or {}, indent=2)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Based on the video file metadata below, assess whether the file "
                    "is likely AI-generated or from a real camera.\n\n"
                    f"```json\n{meta_str}\n```\n\n"
                    "Respond EXACTLY in this format:\n"
                    "RESULT: AI_GENERATED or NOT_AI_GENERATED\n"
                    "ANALYSIS: <concise explanation>"
                ),
            }],
            max_tokens=400,
        )

    raw_reply: str = response.choices[0].message.content.strip()

    # Parse structured result
    detection: Literal["AI_GENERATED", "NOT_AI_GENERATED"] = "NOT_AI_GENERATED"
    for line in raw_reply.splitlines():
        if line.upper().startswith("RESULT:"):
            token = line.split(":", 1)[1].strip().upper()
            if "AI_GENERATED" in token and "NOT" not in token:
                detection = "AI_GENERATED"
            break

    log = _append_log(
        state, f"ai_detection_agent: detection={detection}"
    )
    return {
        "ai_analysis": raw_reply,
        "ai_detection_result": detection,
        "processing_log": log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent 6 – Decision Agent
# ─────────────────────────────────────────────────────────────────────────────

def decision_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Translates AI detection result into a routing decision.
    NOT_AI_GENERATED → REAL (store as-is)
    AI_GENERATED     → AI_GENERATED (watermark, then store)
    """
    result = state.get("ai_detection_result", "NOT_AI_GENERATED")
    decision: Literal["REAL", "AI_GENERATED"] = (
        "AI_GENERATED" if result == "AI_GENERATED" else "REAL"
    )
    log = _append_log(state, f"decision_agent: routing decision='{decision}'")
    return {"decision": decision, "processing_log": log}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 7 – Store File Agent  (REAL branch)
# ─────────────────────────────────────────────────────────────────────────────

def store_file_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Persists a REAL (non-AI) media file to the upload directory."""
    upload_dir = Path(state["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    original = state.get("original_filename") or "file"
    stem = Path(original).stem
    suffix = Path(original).suffix or ".bin"
    dest = _unique_path(upload_dir, stem, suffix, tag="real")
    dest.write_bytes(state["file_bytes"])

    log = _append_log(state, f"store_file_agent: saved REAL media → {dest.name}")
    return {"stored_file_path": str(dest), "processing_log": log}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 8 – Watermark Agent  (AI_GENERATED branch)
# ─────────────────────────────────────────────────────────────────────────────

def watermark_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Stamps a visible 'AI GENERATED' watermark on flagged media.

    • Images  → Pillow alpha-composite red banner at bottom centre
    • Videos  → Original file saved + JSON sidecar flagging AI detection
    """
    upload_dir = Path(state["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    media_type = state.get("media_type", "unknown")
    original = state.get("original_filename") or "file"
    stem = Path(original).stem
    suffix = Path(original).suffix or ".png"

    if media_type == "image":
        with Image.open(BytesIO(state["file_bytes"])) as img:
            rgba = img.convert("RGBA")
            overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            w, h = rgba.size
            text = "⚠  AI GENERATED"
            font_size = max(28, w // 18)

            try:
                # Try common system fonts
                for face in ("arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf"):
                    try:
                        font = ImageFont.truetype(face, size=font_size)
                        break
                    except (OSError, IOError):
                        continue
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (w - tw) // 2
            pad = max(14, h // 40)
            y = h - th - pad * 2

            # Semi-transparent dark-red banner
            draw.rectangle(
                [0, y - pad, w, y + th + pad],
                fill=(160, 0, 0, 190),
            )
            draw.text((x, y), text, fill=(255, 255, 255, 240), font=font)

            watermarked = Image.alpha_composite(rgba, overlay).convert("RGB")
            wm_path = _unique_path(upload_dir, stem, suffix, tag="ai_watermarked")
            watermarked.save(wm_path)

        log = _append_log(
            state, f"watermark_agent: watermarked image saved → {wm_path.name}"
        )
        return {"watermarked_file_path": str(wm_path), "processing_log": log}

    else:
        # Video: save raw bytes + JSON sidecar
        dest = _unique_path(upload_dir, stem, suffix, tag="ai_flagged")
        dest.write_bytes(state["file_bytes"])
        sidecar = dest.with_suffix(".ai_flag.json")
        sidecar.write_text(
            json.dumps(
                {
                    "flagged_as_ai_generated": True,
                    "original_filename": original,
                    "ai_analysis": state.get("ai_analysis"),
                    "metadata": state.get("metadata"),
                },
                indent=2,
            )
        )

        log = _append_log(
            state,
            f"watermark_agent: AI-flagged video saved → {dest.name}  "
            f"(sidecar: {sidecar.name})",
        )
        return {"watermarked_file_path": str(dest), "processing_log": log}


# ─────────────────────────────────────────────────────────────────────────────
# Agent 9 – Store Result Agent
# ─────────────────────────────────────────────────────────────────────────────

def store_result_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Final bookkeeping node – confirms watermarked result is persisted."""
    wm = state.get("watermarked_file_path", "—")
    log = _append_log(
        state,
        f"store_result_agent: pipeline complete. watermarked_path={wm}",
    )
    return {"processing_log": log}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_by_media_type(state: MediaPipelineState) -> str:
    if state.get("error"):
        return END  # type: ignore[return-value]
    mt = state.get("media_type", "unknown")
    if mt == "image":
        return "image_agent"
    if mt == "video":
        return "video_agent"
    return END  # type: ignore[return-value]


def _route_by_decision(state: MediaPipelineState) -> str:
    return (
        "watermark_agent"
        if state.get("decision") == "AI_GENERATED"
        else "store_file_agent"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build & compile graph
# ─────────────────────────────────────────────────────────────────────────────

_builder = StateGraph(MediaPipelineState)

# Register all agents as nodes
_builder.add_node("upload_agent", upload_agent)
_builder.add_node("file_type_classifier_agent", file_type_classifier_agent)
_builder.add_node("image_agent", image_agent)
_builder.add_node("video_agent", video_agent)
_builder.add_node("ai_detection_agent", ai_detection_agent)
_builder.add_node("decision_agent", decision_agent)
_builder.add_node("store_file_agent", store_file_agent)
_builder.add_node("watermark_agent", watermark_agent)
_builder.add_node("store_result_agent", store_result_agent)

# Linear spine
_builder.add_edge(START, "upload_agent")
_builder.add_edge("upload_agent", "file_type_classifier_agent")

# Branch: image or video (or END on unknown/error)
_builder.add_conditional_edges(
    "file_type_classifier_agent",
    _route_by_media_type,
    {
        "image_agent": "image_agent",
        "video_agent": "video_agent",
        END: END,
    },
)

# Both media agents converge to AI detection
_builder.add_edge("image_agent", "ai_detection_agent")
_builder.add_edge("video_agent", "ai_detection_agent")
_builder.add_edge("ai_detection_agent", "decision_agent")

# Branch: real → store, AI-generated → watermark
_builder.add_conditional_edges(
    "decision_agent",
    _route_by_decision,
    {
        "store_file_agent": "store_file_agent",
        "watermark_agent": "watermark_agent",
    },
)

# Terminal edges
_builder.add_edge("store_file_agent", END)
_builder.add_edge("watermark_agent", "store_result_agent")
_builder.add_edge("store_result_agent", END)

media_pipeline_graph = _builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_media_pipeline(
    file_bytes: bytes,
    original_filename: str,
    content_type: str,
    upload_dir: str,
    openai_api_key: str = "",
) -> MediaPipelineState:
    """
    Execute the full multi-agent media-detection pipeline.

    Returns the final :class:`MediaPipelineState` containing metadata,
    AI analysis, decision, stored paths, and the complete processing log.
    """
    return media_pipeline_graph.invoke(
        {
            "file_bytes": file_bytes,
            "original_filename": original_filename,
            "content_type": content_type,
            "upload_dir": upload_dir,
            "openai_api_key": openai_api_key,
            "processing_log": [],
        }
    )
