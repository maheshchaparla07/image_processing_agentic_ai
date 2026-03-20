"""
LangGraph-based media processing pipeline for AI-origin detection.

Flow overview
-------------
User upload
        │
        ▼
Upload Agent            ← validates input and starts a processing log
        │
        ▼
File Type Classifier    ← routes to Image Agent or Video Agent
        │
    ┌─┴─────────────┐
    ▼               ▼
Image Agent   Video Agent   ← extract media metadata
    └──────┬────────┘
                 ▼
AI Detection Agent      ← OpenAI vision/text analysis
                 │
                 ▼
Decision Agent          ← REAL ──► Store File Agent ──► END
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

# Supported media type registries

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


# Pipeline state model

class MediaPipelineState(TypedDict, total=False):
    # Input values
    file_bytes: bytes
    original_filename: str
    content_type: str
    upload_dir: str
    openai_api_key: str

    # Intermediate values
    media_type: Literal["image", "video", "unknown"]
    metadata: dict[str, Any]
    ai_analysis: str
    ai_detection_result: Literal["AI_GENERATED", "NOT_AI_GENERATED"]
    reverification_analysis: str
    reverification_result: Literal["AI_GENERATED", "NOT_AI_GENERATED"]
    reverification_classification: Literal[
        "DEEP_FAKE",
        "AI_GENERATED",
        "DIGITALLY_EDITED",
        "REAL",
        "OTHER",
    ]
    digital_edit_analysis: str
    digital_edit_result: Literal["DIGITALLY_EDITED", "NOT_DIGITALLY_EDITED"]
    ela_score: float
    decision: Literal[
        "DEEP_FAKE",
        "AI_GENERATED",
        "DIGITALLY_EDITED",
        "REAL",
        "OTHER",
    ]

    # Output values
    stored_file_path: str
    watermarked_file_path: str
    processing_log: list[str]
    error: Optional[str]


# Internal helper utilities

def _append_log(state: MediaPipelineState, message: str) -> list[str]:
    log: list[str] = list(state.get("processing_log") or [])
    log.append(message)
    return log


def _unique_path(upload_dir: Path, stem: str, suffix: str, tag: str = "") -> Path:
    tag_part = f"_{tag}" if tag else ""
    return upload_dir / f"{stem}_{uuid.uuid4().hex}{tag_part}{suffix}"


# ELA helpers

def _compute_ela(file_bytes: bytes, quality: int = 75) -> tuple[float, bytes]:
    """
    Error Level Analysis – resave at *quality* and measure pixel differences.

    Regions that were previously saved at a different quality (e.g. pasted
    layers) show higher difference values than uniformly-captured areas.
    Returns (anomaly_score, ela_png_bytes).
    """
    from PIL import ImageChops, ImageEnhance

    with Image.open(BytesIO(file_bytes)) as tmp:
        orig = tmp.convert("RGB")

    buf = BytesIO()
    orig.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(orig, recompressed)
    from PIL import ImageStat as _IS
    stat = _IS.Stat(diff)
    anomaly_score = round(sum(stat.mean) / max(len(stat.mean), 1), 4)

    enhanced = ImageEnhance.Brightness(diff).enhance(10.0)
    ela_buf = BytesIO()
    enhanced.save(ela_buf, format="PNG")
    ela_buf.seek(0)
    return anomaly_score, ela_buf.read()


_EDITING_SOFTWARE_KEYWORDS: frozenset[str] = frozenset({
    "adobe", "photoshop", "lightroom", "gimp", "affinity", "pixelmator",
    "capture one", "darktable", "snapseed", "facetune", "meitu",
    "retouch", "photo editor", "picasa", "luminar",
})


def _detect_editing_software(exif_data: dict[str, str]) -> tuple[bool, str]:
    """Return (found, software_name) by inspecting the EXIF Software tag (305)."""
    software = (exif_data.get("305") or "").lower()
    for kw in _EDITING_SOFTWARE_KEYWORDS:
        if kw in software:
            return True, exif_data.get("305", "")
    return False, ""


AGENT_EXECUTION_PROMPTS: dict[str, str] = {
    "upload_agent": "PROMPT: upload_agent running (validate upload). Next -> file_type_classifier_agent",
    "file_type_classifier_agent": "PROMPT: file_type_classifier_agent running (detect image/video). Next -> image_agent | video_agent | END",
    "image_agent": "PROMPT: image_agent running (extract image metadata). Next -> ai_detection_agent",
    "video_agent": "PROMPT: video_agent running (extract video metadata). Next -> ai_detection_agent",
    "ai_detection_agent": "PROMPT: ai_detection_agent running (primary OpenAI analysis). Next -> reverification_agent",
    "reverification_agent": "PROMPT: reverification_agent running (second-pass OpenAI check). Next -> digital_edit_detection_agent",
    "digital_edit_detection_agent": "PROMPT: digital_edit_detection_agent running (ELA + EXIF + GPT-4o edit forensics). Next -> decision_agent",
    "decision_agent": "PROMPT: decision_agent running (map to DEEP_FAKE/AI_GENERATED/DIGITALLY_EDITED/REAL/OTHER). Next -> store_file_agent | watermark_agent",
    "store_file_agent": "PROMPT: store_file_agent running (persist as-is). Next -> END",
    "watermark_agent": "PROMPT: watermark_agent running (flag/watermark suspicious media). Next -> store_result_agent",
    "store_result_agent": "PROMPT: store_result_agent running (finalize persistence). Next -> END",
}


# Agent 1 – Upload Agent

def upload_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Validate incoming file data and initialize the processing log."""
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["upload_agent"])
    if not state.get("file_bytes"):
        return {
            "error": "No file content received.",
            "processing_log": _append_log({"processing_log": log}, "upload_agent: ERROR – empty file"),
        }
    if not state.get("original_filename"):
        return {
            "error": "No filename provided.",
            "processing_log": _append_log({"processing_log": log}, "upload_agent: ERROR – missing filename"),
        }

    size_kb = round(len(state["file_bytes"]) / 1024, 2)
    msg = (
        f"upload_agent: accepted  filename={state['original_filename']!r}  "
        f"size={size_kb} KB  type={state.get('content_type', 'unknown')}"
    )
    return {"processing_log": _append_log({"processing_log": log}, msg)}


# Agent 2 – File Type Classifier Agent

def file_type_classifier_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Classify the upload as image, video, or unknown for downstream routing."""
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["file_type_classifier_agent"])
    ct = (state.get("content_type") or "").lower()
    if ct in IMAGE_CONTENT_TYPES:
        media_type: Literal["image", "video", "unknown"] = "image"
    elif ct in VIDEO_CONTENT_TYPES:
        media_type = "video"
    else:
        media_type = "unknown"

    log = _append_log({"processing_log": log}, f"file_type_classifier_agent: classified as '{media_type}'")
    return {"media_type": media_type, "processing_log": log}


# Agent 3 – Image Agent

def image_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Extract rich image metadata (size, mode, EXIF, and color statistics)."""
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["image_agent"])
    with Image.open(BytesIO(state["file_bytes"])) as img:
        exif_raw = img.getexif()
        exif: dict[str, str] = (
            {str(k): str(v) for k, v in exif_raw.items()} if exif_raw else {}
        )

        # Basic color statistics for the first three channels
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
        {"processing_log": log},
        f"image_agent: extracted metadata  {img.width}x{img.height}  "
        f"format={img.format}  megapixels={metadata['megapixels']}",
    )
    return {"metadata": metadata, "processing_log": log}


# Agent 4 – Video Agent

def video_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Extract video container metadata from binary headers (FFmpeg not required)."""
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["video_agent"])
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
        {"processing_log": log},
        f"video_agent: extracted metadata  size={metadata['size_mb']} MB  "
        f"container={container}",
    )
    return {"metadata": metadata, "processing_log": log}


# Agent 5 – AI Detection Agent

def ai_detection_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Use OpenAI to estimate whether media appears AI-generated.

    • Images  → GPT-4o vision with a base64 data URL
    • Videos  → GPT-4o text analysis based on extracted metadata
    """
    from openai import OpenAI  # Local import to avoid module-load side effects.

    log = _append_log(state, AGENT_EXECUTION_PROMPTS["ai_detection_agent"])
    api_key = state.get("openai_api_key") or ""
    if not api_key:
        log = _append_log({"processing_log": log}, "ai_detection_agent: OPENAI_API_KEY not set – skipping")
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
            model="gpt-4o",
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
                            "CLASSIFICATION: DEEP_FAKE or AI_GENERATED or DIGITALLY_EDITED or REAL or OTHER\n"
                            "ANALYSIS: <concise explanation>"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=400,
        )
    else:
        # For video, send extracted metadata as text context.
        meta_str = json.dumps(state.get("metadata") or {}, indent=2)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": (
                    "Based on the video file metadata below, assess whether the file "
                    "is likely AI-generated or from a real camera.\n\n"
                    f"```json\n{meta_str}\n```\n\n"
                    "Respond EXACTLY in this format:\n"
                    "RESULT: AI_GENERATED or NOT_AI_GENERATED\n"
                    "CLASSIFICATION: DEEP_FAKE or AI_GENERATED or DIGITALLY_EDITED or REAL or OTHER\n"
                    "ANALYSIS: <concise explanation>"
                ),
            }],
            max_tokens=400,
        )

    raw_reply: str = response.choices[0].message.content.strip()

    # Parse the structured RESULT line from the model response.
    detection: Literal["AI_GENERATED", "NOT_AI_GENERATED"] = "NOT_AI_GENERATED"
    for line in raw_reply.splitlines():
        if line.upper().startswith("RESULT:"):
            token = line.split(":", 1)[1].strip().upper()
            if "AI_GENERATED" in token and "NOT" not in token:
                detection = "AI_GENERATED"
            break

    log = _append_log({"processing_log": log}, f"ai_detection_agent: detection={detection}")
    return {
        "ai_analysis": raw_reply,
        "ai_detection_result": detection,
        "processing_log": log,
    }


# Agent 6 – Reverification Agent

def reverification_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Perform a second-pass verification using a smaller model.

    Uses gpt-4.1-mini ("ChatGPT mini 4") to re-check the first analysis.
    """
    from openai import OpenAI

    log = _append_log(state, AGENT_EXECUTION_PROMPTS["reverification_agent"])
    api_key = state.get("openai_api_key") or ""
    if not api_key:
        log = _append_log({"processing_log": log}, "reverification_agent: OPENAI_API_KEY not set – skipping")
        return {
            "reverification_analysis": state.get("ai_analysis", ""),
            "reverification_result": state.get("ai_detection_result", "NOT_AI_GENERATED"),
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
                            "You are a reverification agent. Re-check this media classification.\n\n"
                            f"Initial analysis:\n{state.get('ai_analysis', '')}\n\n"
                            "Respond EXACTLY in this format:\n"
                            "RESULT: AI_GENERATED or NOT_AI_GENERATED\n"
                            "CLASSIFICATION: DEEP_FAKE or AI_GENERATED or DIGITALLY_EDITED or REAL or OTHER\n"
                            "ANALYSIS: <concise explanation>"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=300,
        )
    else:
        meta_str = json.dumps(state.get("metadata") or {}, indent=2)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": (
                    "You are a reverification agent. Re-check this media classification.\n\n"
                    f"Initial analysis:\n{state.get('ai_analysis', '')}\n\n"
                    f"Metadata:\n```json\n{meta_str}\n```\n\n"
                    "Respond EXACTLY in this format:\n"
                    "RESULT: AI_GENERATED or NOT_AI_GENERATED\n"
                    "CLASSIFICATION: DEEP_FAKE or AI_GENERATED or DIGITALLY_EDITED or REAL or OTHER\n"
                    "ANALYSIS: <concise explanation>"
                ),
            }],
            max_tokens=300,
        )

    raw_reply: str = response.choices[0].message.content.strip()

    detection: Literal["AI_GENERATED", "NOT_AI_GENERATED"] = "NOT_AI_GENERATED"
    for line in raw_reply.splitlines():
        if line.upper().startswith("RESULT:"):
            token = line.split(":", 1)[1].strip().upper()
            if "AI_GENERATED" in token and "NOT" not in token:
                detection = "AI_GENERATED"
            break

    # Parse the CLASSIFICATION line from reverification output.
    allowed: tuple[str, ...] = (
        "DEEP_FAKE",
        "AI_GENERATED",
        "DIGITALLY_EDITED",
        "REAL",
        "OTHER",
    )
    reverif_classification: Literal[
        "DEEP_FAKE", "AI_GENERATED", "DIGITALLY_EDITED", "REAL", "OTHER"
    ] = "OTHER"
    for line in raw_reply.upper().splitlines():
        if line.startswith("CLASSIFICATION:"):
            token = line.split(":", 1)[1].strip().replace("-", "_").replace(" ", "_")
            if token in allowed:
                reverif_classification = token  # type: ignore[assignment]
            break

    log = _append_log({"processing_log": log}, f"reverification_agent: detection={detection}  classification={reverif_classification}")
    return {
        "reverification_analysis": raw_reply,
        "reverification_result": detection,
        "reverification_classification": reverif_classification,
        "processing_log": log,
    }


# Agent 7 – Digital Edit Detection Agent

def digital_edit_detection_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Dedicated digital-manipulation detector for images.

    Combines three independent forensic signals:
    1. ELA (Error Level Analysis) – JPEG compression inconsistencies
    2. EXIF Software tag – Photoshop / GIMP / Lightroom fingerprint
    3. GPT-4o vision – inspects both the original and the ELA map

    Video files are passed through as NOT_DIGITALLY_EDITED.
    """
    from openai import OpenAI

    log = _append_log(state, AGENT_EXECUTION_PROMPTS["digital_edit_detection_agent"])
    media_type = state.get("media_type", "unknown")

    if media_type != "image":
        log = _append_log({"processing_log": log}, "digital_edit_detection_agent: skipped (non-image)")
        return {
            "digital_edit_analysis": "Skipped – not an image.",
            "digital_edit_result": "NOT_DIGITALLY_EDITED",
            "ela_score": 0.0,
            "processing_log": log,
        }

    # ── Signal 1: ELA ─────────────────────────────────────────────────────
    ela_score = 0.0
    ela_bytes = b""
    ela_note = ""
    try:
        ela_score, ela_bytes = _compute_ela(state["file_bytes"])
        ela_note = f"ELA anomaly_score={ela_score}"
    except Exception as exc:
        ela_note = f"ELA failed: {exc}"

    # ── Signal 2: EXIF software fingerprint ───────────────────────────────
    exif_data: dict[str, str] = (state.get("metadata") or {}).get("exif_data") or {}
    edited_by_software, sw_name = _detect_editing_software(exif_data)
    exif_note = (
        f"editing_software_detected={edited_by_software}"
        + (f" ({sw_name})" if sw_name else "")
    )

    pre_signals = f"{ela_note}  |  {exif_note}"

    # ── Signal 3: GPT-4o vision ───────────────────────────────────────────
    api_key = state.get("openai_api_key") or ""
    if not api_key:
        combined_analysis = f"{pre_signals}  |  GPT-4o skipped (no key)"
        result: Literal["DIGITALLY_EDITED", "NOT_DIGITALLY_EDITED"] = (
            "DIGITALLY_EDITED" if (edited_by_software or ela_score > 8.0) else "NOT_DIGITALLY_EDITED"
        )
        log = _append_log(
            {"processing_log": log},
            f"digital_edit_detection_agent: result={result}  ela={ela_score}  exif={edited_by_software}",
        )
        return {
            "digital_edit_analysis": combined_analysis,
            "digital_edit_result": result,
            "ela_score": ela_score,
            "processing_log": log,
        }

    client = OpenAI(api_key=api_key)
    mime = state.get("content_type") or "image/png"
    orig_b64 = base64.b64encode(state["file_bytes"]).decode()
    orig_url = f"data:{mime};base64,{orig_b64}"

    ela_image_block: list[dict] = []
    ela_context_note = ""
    if ela_bytes:
        ela_b64 = base64.b64encode(ela_bytes).decode()
        ela_image_block = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ela_b64}"}}
        ]
        ela_context_note = (
            "The SECOND image is an Error Level Analysis (ELA) map. "
            "Bright/high-contrast regions indicate areas that were inserted, "
            "cloned, or edited after the original JPEG was captured. "
            f"Computed ELA anomaly score: {ela_score:.2f} (0=clean, >5 suspicious).\n\n"
        )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a forensic image analyst specializing in digital manipulation detection.\n\n"
                        f"Pre-computed signals: {pre_signals}\n\n"
                        + ela_context_note
                        + "Examine the image(s) for signs of DIGITAL EDITING:\n"
                        "  • Clone stamping / healing brush artifacts\n"
                        "  • Inconsistent noise grain across regions\n"
                        "  • Spliced backgrounds or composited elements\n"
                        "  • Unnatural edge halos or double-edge artifacts\n"
                        "  • Lighting / shadow direction inconsistencies\n"
                        "  • Color fringing or chromatic aberration mismatch\n"
                        "  • JPEG block boundary artifacts in specific regions\n"
                        "  • EXIF software metadata evidence\n\n"
                        "Respond EXACTLY in this format:\n"
                        "RESULT: DIGITALLY_EDITED or NOT_DIGITALLY_EDITED\n"
                        "CONFIDENCE: HIGH or MEDIUM or LOW\n"
                        "SIGNALS: <comma-separated detected signals>\n"
                        "ANALYSIS: <concise forensic explanation>"
                    ),
                },
                {"type": "image_url", "image_url": {"url": orig_url}},
                *ela_image_block,
            ],
        }],
        max_tokens=400,
    )

    raw_reply: str = response.choices[0].message.content.strip()

    gpt_result: Literal["DIGITALLY_EDITED", "NOT_DIGITALLY_EDITED"] = "NOT_DIGITALLY_EDITED"
    for line in raw_reply.splitlines():
        upper = line.upper()
        if upper.startswith("RESULT:"):
            token = upper.split(":", 1)[1].strip()
            if "DIGITALLY_EDITED" in token and "NOT" not in token:
                gpt_result = "DIGITALLY_EDITED"
            break

    # EXIF software fingerprint overrides GPT "not edited" verdict.
    final_result: Literal["DIGITALLY_EDITED", "NOT_DIGITALLY_EDITED"] = gpt_result
    if edited_by_software and final_result == "NOT_DIGITALLY_EDITED":
        final_result = "DIGITALLY_EDITED"

    combined = f"{pre_signals}\n\nGPT-4o forensic response:\n{raw_reply}"
    log = _append_log(
        {"processing_log": log},
        f"digital_edit_detection_agent: result={final_result}  ela={ela_score}  exif={edited_by_software}",
    )
    return {
        "digital_edit_analysis": combined,
        "digital_edit_result": final_result,
        "ela_score": ela_score,
        "processing_log": log,
    }


# Agent 8 – Decision Agent

def decision_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Classify media into one of:
    DEEP_FAKE, AI_GENERATED, DIGITALLY_EDITED, REAL, OTHER.

    Priority:
    1) explicit CLASSIFICATION line from the model output
    2) fallback from ai_detection_result
    """
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["decision_agent"])
    result = state.get("reverification_result", state.get("ai_detection_result", "NOT_AI_GENERATED"))
    source_analysis = state.get("reverification_analysis") or state.get("ai_analysis") or ""
    raw = source_analysis.upper()

    allowed: tuple[str, ...] = (
        "DEEP_FAKE",
        "AI_GENERATED",
        "DIGITALLY_EDITED",
        "REAL",
        "OTHER",
    )

    decision: Literal[
        "DEEP_FAKE",
        "AI_GENERATED",
        "DIGITALLY_EDITED",
        "REAL",
        "OTHER",
    ] = "OTHER"

    # Priority 1: DEEP_FAKE from reverification takes absolute precedence.
    reverif_cls = state.get("reverification_classification")
    if reverif_cls == "DEEP_FAKE":
        decision = "DEEP_FAKE"

    # Priority 2: dedicated digital-edit detection agent result.
    if decision == "OTHER" and state.get("digital_edit_result") == "DIGITALLY_EDITED":
        decision = "DIGITALLY_EDITED"

    # Priority 3: any other reverification_classification value.
    if decision == "OTHER" and reverif_cls and reverif_cls in allowed:
        decision = reverif_cls  # type: ignore[assignment]

    # Priority 4: explicit CLASSIFICATION line in analysis text.
    if decision == "OTHER":
        for line in raw.splitlines():
            if line.startswith("CLASSIFICATION:"):
                token = line.split(":", 1)[1].strip().replace("-", "_").replace(" ", "_")
                if token in allowed:
                    decision = token  # type: ignore[assignment]
                break

    # Priority 5: fallback keyword heuristics.
    if decision == "OTHER":
        if "DEEP_FAKE" in raw or "DEEPFAKE" in raw:
            decision = "DEEP_FAKE"
        elif "DIGITALLY_EDITED" in raw or "DIGITALLY EDITED" in raw:
            decision = "DIGITALLY_EDITED"
        elif result == "AI_GENERATED":
            decision = "AI_GENERATED"
        elif result == "NOT_AI_GENERATED":
            decision = "REAL"

    log = _append_log({"processing_log": log}, f"decision_agent: routing decision='{decision}'")
    return {"decision": decision, "processing_log": log}


# Agent 9 – Store File Agent (REAL branch)

def store_file_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Save REAL (non-AI) media to the upload directory."""
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["store_file_agent"])
    upload_dir = Path(state["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    original = state.get("original_filename") or "file"
    stem = Path(original).stem
    suffix = Path(original).suffix or ".bin"
    dest = _unique_path(upload_dir, stem, suffix, tag="real")
    dest.write_bytes(state["file_bytes"])

    log = _append_log({"processing_log": log}, f"store_file_agent: saved REAL media → {dest.name}")
    return {"stored_file_path": str(dest), "processing_log": log}


# Agent 10 – Watermark Agent (AI_GENERATED branch)

def watermark_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Add visible AI-generated marking for flagged media.

    • Images  → Pillow alpha-composited banner near the bottom center
    • Videos  → Original file + JSON sidecar with AI flag details
    """
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["watermark_agent"])
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
            text = "  AI GENERATED"
            font_size = max(28, w // 18)

            try:
                # Try a few commonly available system fonts.
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

            # Draw a semi-transparent dark-red banner.
            draw.rectangle(
                [0, y - pad, w, y + th + pad],
                fill=(160, 0, 0, 190),
            )
            draw.text((x, y), text, fill=(255, 255, 255, 240), font=font)

            watermarked = Image.alpha_composite(rgba, overlay).convert("RGB")
            wm_path = _unique_path(upload_dir, stem, suffix, tag="ai_watermarked")
            watermarked.save(wm_path)

        log = _append_log(
            {"processing_log": log}, f"watermark_agent: watermarked image saved → {wm_path.name}"
        )
        return {"watermarked_file_path": str(wm_path), "processing_log": log}

    else:
        # For video, save raw bytes and write a JSON sidecar.
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
            {"processing_log": log},
            f"watermark_agent: AI-flagged video saved → {dest.name}  "
            f"(sidecar: {sidecar.name})",
        )
        return {"watermarked_file_path": str(dest), "processing_log": log}


# Agent 11 – Store Result Agent

def store_result_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Final bookkeeping step that confirms result persistence."""
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["store_result_agent"])
    wm = state.get("watermarked_file_path", "—")
    log = _append_log(
        {"processing_log": log},
        f"store_result_agent: pipeline complete. watermarked_path={wm}",
    )
    return {"processing_log": log}


# Conditional routing helpers

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
        if state.get("decision") in {"DEEP_FAKE", "AI_GENERATED", "DIGITALLY_EDITED"}
        else "store_file_agent"
    )


# Build and compile the graph

_builder = StateGraph(MediaPipelineState)

# Register all agents as graph nodes.
_builder.add_node("upload_agent", upload_agent)
_builder.add_node("file_type_classifier_agent", file_type_classifier_agent)
_builder.add_node("image_agent", image_agent)
_builder.add_node("video_agent", video_agent)
_builder.add_node("ai_detection_agent", ai_detection_agent)
_builder.add_node("reverification_agent", reverification_agent)
_builder.add_node("digital_edit_detection_agent", digital_edit_detection_agent)
_builder.add_node("decision_agent", decision_agent)
_builder.add_node("store_file_agent", store_file_agent)
_builder.add_node("watermark_agent", watermark_agent)
_builder.add_node("store_result_agent", store_result_agent)

# Linear spine
_builder.add_edge(START, "upload_agent")
_builder.add_edge("upload_agent", "file_type_classifier_agent")

# Branch: image/video route, or end on unknown/error.
_builder.add_conditional_edges(
    "file_type_classifier_agent",
    _route_by_media_type,
    {
        "image_agent": "image_agent",
        "video_agent": "video_agent",
        END: END,
    },
)

# Both media branches converge into AI detection.
_builder.add_edge("image_agent", "ai_detection_agent")
_builder.add_edge("digital_edit_detection_agent", "decision_agent")
_builder.add_edge("ai_detection_agent", "reverification_agent")
_builder.add_edge("reverification_agent", "digital_edit_detection_agent")


# Branch: non-REAL suspicious classes go to watermarking.
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


# Public API

def run_media_pipeline(
    file_bytes: bytes,
    original_filename: str,
    content_type: str,
    upload_dir: str,
    openai_api_key: str = "",
) -> MediaPipelineState:
    """
    Execute the full media detection pipeline.

    Returns final :class:`MediaPipelineState` including metadata,
    AI analysis, decision, output paths, and full processing log.
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
