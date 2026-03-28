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
File Type Classifier    ← routes to Image Agent
        │
    ▼
Image Agent   ← extract image metadata
    │
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
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS as EXIF_TAGS

try:
    import cv2
except Exception:
    cv2 = None

# Supported media type registries

IMAGE_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
})

ALLOWED_MEDIA_TYPES: frozenset[str] = IMAGE_CONTENT_TYPES

EXECUTION_ORDER: tuple[str, ...] = (
    "upload_agent",
    "file_type_classifier_agent",
    "image_agent",
    "ai_detection_agent",
    "reverification_agent",
    "digital_edit_detection_agent",
    "decision_agent",
    "watermark_agent",
    "store_file_agent",
    "store_result_agent",
)


# Pipeline state model

class MediaPipelineState(TypedDict, total=False):
    # Input values
    file_bytes: bytes
    original_filename: str
    content_type: str
    upload_dir: str
    openai_api_key: str

    # Intermediate values
    media_type: Literal["image", "unknown"]
    metadata: dict[str, Any]
    ai_analysis: str
    ai_detection_result: Literal["AI_GENERATED", "NOT_AI_GENERATED"]
    ai_confidence: float
    confidence_scores: dict[str, Any]
    abstention_threshold: float
    abstained: bool
    reverification_analysis: str
    reverification_result: Literal["AI_GENERATED", "NOT_AI_GENERATED"]
    reverification_confidence: float
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
        "ABSTAIN",
    ]

    # Output values
    stored_file_path: str
    watermarked_file_path: str
    processing_log: list[str]
    error: Optional[str]


# Internal helper utilities

def _append_log(state: MediaPipelineState, message: str) -> list[str]:
    log: list[str] = list(state.get("processing_log") or [])
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.append(f"[{ts}] {message}")
    return log


def _unique_path(upload_dir: Path, stem: str, suffix: str, tag: str = "") -> Path:
    tag_part = f"_{tag}" if tag else ""
    return upload_dir / f"{stem}_{uuid.uuid4().hex}{tag_part}{suffix}"


def _extract_agent_name_from_log(entry: str) -> Optional[str]:
    """Extract pipeline agent name from a log entry if present."""
    msg = entry
    if "] " in entry:
        msg = entry.split("] ", 1)[1]

    match = re.search(r"\b([a-z_]+_agent)\b", msg)
    if not match:
        return None
    return match.group(1)


def _rearrange_log_by_execution(log: list[str]) -> list[str]:
    """Reorder processing logs by pipeline execution flow."""
    grouped: dict[str, list[str]] = {name: [] for name in EXECUTION_ORDER}
    unmatched: list[str] = []

    for entry in log:
        agent = _extract_agent_name_from_log(entry)
        if agent and agent in grouped:
            grouped[agent].append(entry)
        else:
            unmatched.append(entry)

    ordered: list[str] = []
    for agent in EXECUTION_ORDER:
        ordered.extend(grouped[agent])
    ordered.extend(unmatched)
    return ordered


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


def _compute_ela_script_style(file_bytes: bytes, quality: int = 90) -> tuple[float, bytes]:
    """
    Script-style ELA variant for comparison with ela_analysis.py.

    - Re-save JPEG at quality=90
    - Enhance difference with brightness x15
    - Return ELA map as JPEG bytes
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

    enhanced = ImageEnhance.Brightness(diff).enhance(15.0)
    ela_buf = BytesIO()
    enhanced.save(ela_buf, format="JPEG", quality=95)
    ela_buf.seek(0)
    return anomaly_score, ela_buf.read()


_EDITING_SOFTWARE_KEYWORDS: frozenset[str] = frozenset({
    "adobe", "photoshop", "lightroom", "gimp", "affinity", "pixelmator",
    "capture one", "darktable", "snapseed", "facetune", "meitu",
    "retouch", "photo editor", "picasa", "luminar",
})


def _detect_editing_software(exif_data: dict[str, str]) -> tuple[bool, str]:
    """Return (found, software_name) by inspecting the EXIF Software tag."""
    software = (exif_data.get("Software") or "").lower()
    for kw in _EDITING_SOFTWARE_KEYWORDS:
        if kw in software:
            return True, exif_data.get("Software", "")
    return False, ""


def _extract_human_readable_exif(exif_raw) -> dict[str, str]:
    """Convert numeric EXIF tag IDs to human-readable tag names."""
    if not exif_raw:
        return {}
    
    result = {}
    for tag_id, value in exif_raw.items():
        tag_name = EXIF_TAGS.get(tag_id, f"Tag_{tag_id}")
        result[tag_name] = str(value)
    return result


def _extract_exiftool_metadata(file_bytes: bytes, original_filename: str) -> dict[str, Any]:
    """
    Extract metadata using exiftool if available on PATH.

    Returns a dict with status + parsed metadata payload.
    """
    exiftool_bin = shutil.which("exiftool")
    if not exiftool_bin:
        return {
            "available": False,
            "error": "exiftool executable not found on PATH",
            "data": {},
        }

    suffix = Path(original_filename or "upload.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(file_bytes)

    try:
        proc = subprocess.run(
            [exiftool_bin, "-j", "-n", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        if proc.returncode != 0:
            return {
                "available": True,
                "error": (proc.stderr or "exiftool failed").strip(),
                "data": {},
            }

        payload = json.loads(proc.stdout or "[]")
        data = payload[0] if isinstance(payload, list) and payload else {}
        if isinstance(data, dict):
            data.pop("SourceFile", None)

        return {
            "available": True,
            "error": "",
            "data": data if isinstance(data, dict) else {},
        }
    except Exception as exc:
        return {
            "available": True,
            "error": str(exc),
            "data": {},
        }
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _extract_opencv_metadata(file_bytes: bytes) -> dict[str, Any]:
    """Extract basic image stats with OpenCV, if OpenCV is installed."""
    if cv2 is None:
        return {
            "available": False,
            "error": "opencv-python not installed",
            "data": {},
        }

    try:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            return {
                "available": True,
                "error": "cv2.imdecode returned None",
                "data": {},
            }

        height, width = decoded.shape[:2]
        channels = 1 if len(decoded.shape) == 2 else int(decoded.shape[2])
        mean_bgr = cv2.mean(decoded)[: min(channels, 3)]

        return {
            "available": True,
            "error": "",
            "data": {
                "width": int(width),
                "height": int(height),
                "channels": channels,
                "dtype": str(decoded.dtype),
                "mean_bgr": [round(float(v), 2) for v in mean_bgr],
            },
        }
    except Exception as exc:
        return {
            "available": True,
            "error": str(exc),
            "data": {},
        }


def _build_pillow_opencv_difference(
    pillow_meta: dict[str, Any],
    opencv_meta: dict[str, Any],
    exif_data: dict[str, str],
    exiftool_meta: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact consistency/difference summary for model prompts."""
    cv_data = (opencv_meta or {}).get("data") or {}
    exiftool_data = (exiftool_meta or {}).get("data") or {}

    diffs: list[str] = []

    pw = pillow_meta.get("width")
    ph = pillow_meta.get("height")
    cw = cv_data.get("width")
    ch = cv_data.get("height")
    if cw is not None and ch is not None and (pw != cw or ph != ch):
        diffs.append(f"dimension_mismatch: pillow={pw}x{ph} opencv={cw}x{ch}")

    pillow_mode = str(pillow_meta.get("mode") or "")
    expected_channels = {
        "L": 1,
        "LA": 2,
        "RGB": 3,
        "RGBA": 4,
        "P": 1,
    }.get(pillow_mode)
    cv_channels = cv_data.get("channels")
    if expected_channels is not None and cv_channels is not None and expected_channels != cv_channels:
        diffs.append(
            f"channel_mismatch: pillow_mode={pillow_mode}({expected_channels}) opencv_channels={cv_channels}"
        )

    exif_software = exif_data.get("Software") or ""
    exiftool_software = str(exiftool_data.get("Software") or "")
    if exif_software and exiftool_software and exif_software != exiftool_software:
        diffs.append(
            f"software_tag_mismatch: pillow_software={exif_software!r} exiftool_software={exiftool_software!r}"
        )

    return {
        "consistency": "match" if not diffs else "mismatch",
        "differences": diffs,
        "opencv_available": bool((opencv_meta or {}).get("available")),
        "exiftool_available": bool((exiftool_meta or {}).get("available")),
    }


def _build_authenticity_markers(metadata: dict[str, Any]) -> dict[str, Any]:
    exif = metadata.get("exif_data") or {}
    exiftool_data = (metadata.get("exiftool_metadata") or {}).get("data") or {}

    return {
        "camera_make": exif.get("Make") or exiftool_data.get("Make"),
        "camera_model": exif.get("Model") or exiftool_data.get("Model"),
        "iso": exif.get("ISOSpeedRatings") or exiftool_data.get("ISO"),
        "gps": {
            "lat": exif.get("GPSLatitude") or exiftool_data.get("GPSLatitude"),
            "lon": exif.get("GPSLongitude") or exiftool_data.get("GPSLongitude"),
        },
    }


def _build_synthetic_traces(metadata: dict[str, Any]) -> dict[str, Any]:
    exif = metadata.get("exif_data") or {}
    software = str(exif.get("Software") or "")
    user_comment = str(exif.get("UserComment") or "")
    markers = (
        "midjourney",
        "stable diffusion",
        "dall",
        "prompt",
        "generated",
        "adobe firefly",
    )
    combined = f"{software} {user_comment}".lower()
    hits = [m for m in markers if m in combined]
    return {
        "software": software,
        "user_comment": user_comment,
        "markers_detected": hits,
        "has_synthetic_markers": bool(hits),
    }


def _build_provenance_signals_placeholder() -> dict[str, Any]:
    return {
        "reverse_image_search_available": False,
        "provider": "not_configured",
        "matches_found": None,
        "note": "placeholder only; external reverse image API not configured",
    }


def _build_error_recovery_rate_placeholder() -> dict[str, Any]:
    return {
        "dynamic_replanning_enabled": False,
        "state_mismatch_detected": None,
        "recovery_success_rate": None,
        "note": "placeholder only; no dynamic re-planning module integrated",
    }


def _build_compression_analysis(metadata: dict[str, Any]) -> dict[str, Any]:
    """Heuristic compression/recompression analysis from metadata + ELA signals."""
    image_format = str(metadata.get("format") or "").upper()
    megapixels = float(metadata.get("megapixels") or 0.0)
    exif = metadata.get("exif_data") or {}
    exiftool_data = (metadata.get("exiftool_metadata") or {}).get("data") or {}

    software = str(exif.get("Software") or exiftool_data.get("Software") or "")
    jpeg_quality_est = exiftool_data.get("JPEGQualityEstimate")
    compressed_tag = exif.get("Compression") or exiftool_data.get("Compression")

    size_bytes = int(metadata.get("size_bytes") or 0)
    bytes_per_mp = round(size_bytes / megapixels, 2) if size_bytes > 0 and megapixels > 0 else None

    reasons: list[str] = []
    score = 0.0

    if image_format == "JPEG":
        score += 0.35
        reasons.append("jpeg_container_detected")
    if compressed_tag:
        score += 0.2
        reasons.append(f"compression_tag_present:{compressed_tag}")
    if software:
        score += 0.15
        reasons.append("software_field_present")
    if isinstance(jpeg_quality_est, (int, float)) and jpeg_quality_est < 95:
        score += 0.2
        reasons.append(f"jpeg_quality_estimate:{jpeg_quality_est}")
    if bytes_per_mp is not None and bytes_per_mp < 900_000:
        score += 0.15
        reasons.append(f"low_bytes_per_mp:{bytes_per_mp}")

    score = max(0.0, min(score, 1.0))

    verdict = "likely_not_compressed"
    if score >= 0.6:
        verdict = "likely_compressed"

    return {
        "verdict": verdict,
        "confidence": round(score, 2),
        "bytes_per_megapixel": bytes_per_mp,
        "jpeg_quality_estimate": jpeg_quality_est,
        "software": software,
        "signals": reasons,
        "note": "heuristic estimate; not absolute proof of original compression state",
    }


def _parse_confidence_from_reply(raw_reply: str) -> float:
    for line in raw_reply.splitlines():
        if line.upper().startswith("CONFIDENCE:"):
            token = line.split(":", 1)[1].strip().upper()
            if token == "HIGH":
                return 0.85
            if token == "MEDIUM":
                return 0.65
            if token == "LOW":
                return 0.45
            try:
                value = float(token)
                if 0.0 <= value <= 1.0:
                    return value
            except Exception:
                pass
    return 0.5


AGENT_EXECUTION_PROMPTS: dict[str, str] = {
    "upload_agent": "PROMPT: upload_agent running (validate upload). Next -> file_type_classifier_agent",
    "file_type_classifier_agent": "PROMPT: file_type_classifier_agent running (detect image). Next -> image_agent | END",
    "image_agent": "PROMPT: image_agent running (extract image metadata). Next -> ai_detection_agent",
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
    """Classify the upload as image or unknown for downstream routing."""
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["file_type_classifier_agent"])
    ct = (state.get("content_type") or "").lower()
    if ct in IMAGE_CONTENT_TYPES:
        media_type: Literal["image", "unknown"] = "image"
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
        exif: dict[str, str] = _extract_human_readable_exif(exif_raw)

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
            "size_bytes": len(state["file_bytes"]),
            "megapixels": round((img.width * img.height) / 1_000_000, 2),
            "aspect_ratio": f"{img.width}:{img.height}",
            "has_transparency": img.mode in ("RGBA", "LA", "P"),
            "has_exif": bool(exif),
            "exif_fields_count": len(exif),
            "exif_data": exif,
            "colour_stats": colour_stats,
            "sha256": hashlib.sha256(state["file_bytes"]).hexdigest(),
        }

    opencv_meta = _extract_opencv_metadata(state["file_bytes"])
    exiftool_meta = _extract_exiftool_metadata(
        state["file_bytes"],
        state.get("original_filename", "upload.bin"),
    )
    metadata["opencv_metadata"] = opencv_meta
    metadata["exiftool_metadata"] = exiftool_meta
    metadata["pillow_opencv_difference"] = _build_pillow_opencv_difference(
        metadata,
        opencv_meta,
        metadata.get("exif_data") or {},
        exiftool_meta,
    )
    metadata["authenticity_markers"] = _build_authenticity_markers(metadata)
    metadata["synthetic_traces"] = _build_synthetic_traces(metadata)
    metadata["provenance_signals"] = _build_provenance_signals_placeholder()
    metadata["error_recovery_rate"] = _build_error_recovery_rate_placeholder()
    metadata["compression_analysis"] = _build_compression_analysis(metadata)

    log = _append_log(
        {"processing_log": log},
        f"image_agent: extracted metadata  {img.width}x{img.height}  "
        f"format={img.format}  megapixels={metadata['megapixels']}",
    )
    return {"metadata": metadata, "processing_log": log}


# Agent 4 – AI Detection Agent

def ai_detection_agent(state: MediaPipelineState) -> MediaPipelineState:
    
    from openai import OpenAI  # Local import to avoid module-load side effects.

    log = _append_log(state, AGENT_EXECUTION_PROMPTS["ai_detection_agent"])
    api_key = state.get("openai_api_key") or ""
    if not api_key:
        log = _append_log({"processing_log": log}, "ai_detection_agent: OPENAI_API_KEY not set – skipping")
        return {
            "ai_analysis": "OPENAI_API_KEY not configured; AI detection skipped.",
            "ai_detection_result": "NOT_AI_GENERATED",
            "ai_confidence": 0.0,
            "confidence_scores": {
                "primary_model": 0.0,
                "note": "model call skipped",
            },
            "abstention_threshold": 0.6,
            "abstained": True,
            "processing_log": log,
        }

    client = OpenAI(api_key=api_key)
    media_type = state.get("media_type", "unknown")

    if media_type == "image":
        img_b64 = base64.b64encode(state["file_bytes"]).decode()
        mime = state.get("content_type") or "image/png"
        data_url = f"data:{mime};base64,{img_b64}"
        metadata = state.get("metadata") or {}
        diff_summary = metadata.get("pillow_opencv_difference") or {}
        exiftool_summary = metadata.get("exiftool_metadata") or {}
        authenticity_markers = metadata.get("authenticity_markers") or {}
        synthetic_traces = metadata.get("synthetic_traces") or {}
        provenance_signals = metadata.get("provenance_signals") or {}
        error_recovery_rate = metadata.get("error_recovery_rate") or {}
        compression_analysis = metadata.get("compression_analysis") or {}
        abstention_threshold = float(state.get("abstention_threshold", 0.6))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image carefully.\n\n"
                            f"Structured metadata context:\n{json.dumps({'authenticity_markers': authenticity_markers, 'synthetic_traces': synthetic_traces, 'compression_analysis': compression_analysis, 'visual_pixel_artifacts': {'pillow_opencv_difference': diff_summary}, 'semantic_abnormalities': {'text_rendering_abnormalities': 'inspect image text for gibberish/warping'}, 'provenance_signals': provenance_signals, 'error_recovery_rate': error_recovery_rate, 'exiftool_status': {'available': exiftool_summary.get('available'), 'error': exiftool_summary.get('error')}, 'abstention_threshold': abstention_threshold}, indent=2)}\n\n"
                            "1. Determine if it appears AI-generated or photographed.\n"
                            "2. Identify specific visual artifacts, lighting inconsistencies, unnatural textures, text rendering abnormalities, or generative model signatures.\n"
                            "3. If evidence is weak and confidence is below the abstention threshold, state LOW confidence.\n\n"
                            "Respond EXACTLY in this format:\n"
                            "RESULT: AI_GENERATED or NOT_AI_GENERATED\n"
                            "CONFIDENCE: HIGH or MEDIUM or LOW\n"
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
        log = _append_log({"processing_log": log}, "ai_detection_agent: skipped (non-image media_type)")
        return {
            "ai_analysis": "Skipped – not an image.",
            "ai_detection_result": "NOT_AI_GENERATED",
            "ai_confidence": 0.0,
            "confidence_scores": {
                "primary_model": 0.0,
                "note": "model call skipped for non-image",
            },
            "abstention_threshold": float(state.get("abstention_threshold", 0.6)),
            "abstained": True,
            "processing_log": log,
        }

    raw_reply: str = response.choices[0].message.content.strip()
    confidence = _parse_confidence_from_reply(raw_reply)
    if media_type == "image":
        raw_reply = (
            "DIFF_SUMMARY:\n"
            f"{json.dumps(diff_summary, indent=2)}\n\n"
            + raw_reply
        )

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
        "ai_confidence": confidence,
        "confidence_scores": {
            "primary_model": confidence,
            "source": "parsed_from_confidence_line_with_heuristic_fallback",
        },
        "abstention_threshold": float(state.get("abstention_threshold", 0.6)),
        "processing_log": log,
    }


# Agent 5 – Reverification Agent

def reverification_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Perform a second-pass verification using a smaller model.

    Uses gpt-4.1-mini ("ChatGPT mini 4") to re-check the first analysis.
    """
    from openai import OpenAI

    log = _append_log(state, AGENT_EXECUTION_PROMPTS["reverification_agent"])
    api_key = state.get("openai_api_key") or ""
    if not api_key:
        log = _append_log({"processing_log": log}, "reverification_agent: OPENAI_API_KEY not set - skipping")
        return {
            "reverification_analysis": state.get("ai_analysis", ""),
            "reverification_result": state.get("ai_detection_result", "NOT_AI_GENERATED"),
            "reverification_confidence": float(state.get("ai_confidence", 0.0)),
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
    reverification_confidence = _parse_confidence_from_reply(raw_reply)

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
        "reverification_confidence": reverification_confidence,
        "reverification_classification": reverif_classification,
        "processing_log": log,
    }


# Agent 6 – Digital Edit Detection Agent

def digital_edit_detection_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Dedicated digital-manipulation detector for images.

    Combines three independent forensic signals:
    1. ELA (Error Level Analysis) – JPEG compression inconsistencies
    2. EXIF Software tag – Photoshop / GIMP / Lightroom fingerprint
    3. GPT-4o vision – inspects both the original and the ELA map
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

    # Signal 1: ELA (existing + script-style comparison)
    ela_score = 0.0
    ela_bytes = b""
    script_ela_score = 0.0
    script_ela_bytes = b""
    ela_note = ""
    ela_comparison: dict[str, Any] = {}
    try:
        ela_score, ela_bytes = _compute_ela(state["file_bytes"])
        script_ela_score, script_ela_bytes = _compute_ela_script_style(state["file_bytes"])
        delta = round(script_ela_score - ela_score, 4)
        ela_comparison = {
            "existing_pipeline": {
                "quality": 75,
                "brightness_enhance": 10.0,
                "output_format": "PNG",
                "anomaly_score": ela_score,
            },
            "script_style": {
                "quality": 90,
                "brightness_enhance": 15.0,
                "output_format": "JPEG",
                "anomaly_score": script_ela_score,
            },
            "score_delta_script_minus_existing": delta,
            "difference_summary": (
                "script_style_higher" if delta > 0 else "script_style_lower" if delta < 0 else "equal_scores"
            ),
        }
        ela_note = (
            f"ELA existing={ela_score} (q75,b10,png) | "
            f"script_style={script_ela_score} (q90,b15,jpg)"
        )
    except Exception as exc:
        ela_note = f"ELA failed: {exc}"
        ela_comparison = {
            "error": str(exc),
            "existing_pipeline": {"anomaly_score": ela_score},
            "script_style": {"anomaly_score": script_ela_score},
        }

    #  Signal 2: EXIF software fingerprint 
    exif_data: dict[str, str] = (state.get("metadata") or {}).get("exif_data") or {}
    edited_by_software, sw_name = _detect_editing_software(exif_data)
    exif_note = (
        f"editing_software_detected={edited_by_software}"
        + (f" ({sw_name})" if sw_name else "")
    )
    metadata = state.get("metadata") or {}
    diff_summary = metadata.get("pillow_opencv_difference") or {}
    compression_analysis = metadata.get("compression_analysis") or {}
    exiftool_meta = metadata.get("exiftool_metadata") or {}

    pre_signals = f"{ela_note}  |  {exif_note}"

    #  Signal 3: GPT-4o vision
    api_key = state.get("openai_api_key") or ""
    if not api_key:
        combined_analysis = (
            f"{pre_signals}\n\n"
            "ELA_COMPARISON:\n"
            f"{json.dumps(ela_comparison, indent=2)}\n\n"
            "GPT-4o skipped (no key)"
        )
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

    if script_ela_bytes:
        script_ela_b64 = base64.b64encode(script_ela_bytes).decode()
        ela_image_block.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{script_ela_b64}"}}
        )
        ela_context_note += (
            "The THIRD image is script-style ELA (q90, brightness x15, JPEG output), "
            "used to compare sensitivity against the pipeline ELA map.\n\n"
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
                        "ELA comparison (existing pipeline vs standalone script style):\n"
                        f"{json.dumps(ela_comparison, indent=2)}\n\n"
                        "Compression analysis context:\n"
                        f"{json.dumps(compression_analysis, indent=2)}\n\n"
                        "Cross-tool metadata consistency context (Pillow/OpenCV/ExifTool):\n"
                        f"{json.dumps({'pillow_opencv_difference': diff_summary, 'exiftool_status': {'available': exiftool_meta.get('available'), 'error': exiftool_meta.get('error')}}, indent=2)}\n\n"
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

    combined = (
        f"{pre_signals}\n\n"
        "ELA_COMPARISON:\n"
        f"{json.dumps(ela_comparison, indent=2)}\n\n"
        "COMPRESSION_ANALYSIS:\n"
        f"{json.dumps(compression_analysis, indent=2)}\n\n"
        "DIFF_SUMMARY:\n"
        f"{json.dumps(diff_summary, indent=2)}\n\n"
        "GPT-4o forensic response:\n"
        f"{raw_reply}"
    )
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


# Agent 7 – Decision Agent

def decision_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Classify media into one of:
    DEEP_FAKE, AI_GENERATED, DIGITALLY_EDITED, REAL, OTHER, ABSTAIN.

    Priority:
    1) explicit CLASSIFICATION line from the model output
    2) fallback from ai_detection_result
    """
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["decision_agent"])
    result = state.get("reverification_result", state.get("ai_detection_result", "NOT_AI_GENERATED"))
    source_analysis = state.get("reverification_analysis") or state.get("ai_analysis") or ""
    raw = source_analysis.upper()
    threshold = float(state.get("abstention_threshold", 0.6))
    confidence = float(state.get("reverification_confidence", state.get("ai_confidence", 0.5)))

    allowed: tuple[str, ...] = (
        "DEEP_FAKE",
        "AI_GENERATED",
        "DIGITALLY_EDITED",
        "REAL",
        "OTHER",
        "ABSTAIN",
    )

    decision: Literal[
        "DEEP_FAKE",
        "AI_GENERATED",
        "DIGITALLY_EDITED",
        "REAL",
        "OTHER",
        "ABSTAIN",
    ] = "OTHER"

    if confidence < threshold:
        decision = "ABSTAIN"
        log = _append_log(
            {"processing_log": log},
            f"decision_agent: abstained (confidence={confidence:.2f} < threshold={threshold:.2f})",
        )
        return {"decision": decision, "abstained": True, "processing_log": log}

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
    return {"decision": decision, "abstained": False, "processing_log": log}


# Agent 8 – Store File Agent (REAL branch)

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


# Agent 9 – Watermark Agent (AI_GENERATED branch)

def watermark_agent(state: MediaPipelineState) -> MediaPipelineState:
    """
    Add visible AI-generated marking for flagged media.

    • Images  → Pillow alpha-composited banner near the bottom center
    """
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["watermark_agent"])
    upload_dir = Path(state["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    media_type = state.get("media_type", "unknown")
    original = state.get("original_filename") or "file"
    stem = Path(original).stem
    suffix = Path(original).suffix or ".png"

    if media_type != "image":
        log = _append_log({"processing_log": log}, "watermark_agent: skipped (non-image)")
        return {"processing_log": log}

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


# Agent 10 – Store Result Agent

def store_result_agent(state: MediaPipelineState) -> MediaPipelineState:
    """Final bookkeeping step that confirms result persistence."""
    log = _append_log(state, AGENT_EXECUTION_PROMPTS["store_result_agent"])
    wm = state.get("watermarked_file_path", "—")
    log = _append_log(
        {"processing_log": log},
        f"store_result_agent: pipeline complete. watermarked_path={wm}",
    )
    ordered_log = _rearrange_log_by_execution(log)
    return {"processing_log": ordered_log}


# Conditional routing helpers

def _route_by_media_type(state: MediaPipelineState) -> str:
    if state.get("error"):
        return END  # type: ignore[return-value]
    mt = state.get("media_type", "unknown")
    if mt == "image":
        return "image_agent"
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

# Branch: image route, or end on unknown/error.
_builder.add_conditional_edges(
    "file_type_classifier_agent",
    _route_by_media_type,
    {
        "image_agent": "image_agent",
        END: END,
    },
)

# Image branch converges into AI detection.
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
