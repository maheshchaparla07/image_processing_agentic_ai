from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, TypedDict
import uuid

from langgraph.graph import END, START, StateGraph
from PIL import Image


class ImageProcessingState(TypedDict, total=False):
    image_bytes: bytes
    original_filename: str
    upload_dir: str
    metadata: dict[str, Any]
    reversed_file_path: str


def _extract_metadata(state: ImageProcessingState) -> ImageProcessingState:
    with Image.open(BytesIO(state["image_bytes"])) as image:
        exif_data = image.getexif()
        metadata = {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "has_exif": bool(exif_data),
            "exif_tags_count": len(exif_data) if exif_data else 0,
        }
    return {"metadata": metadata}


def _reverse_image(state: ImageProcessingState) -> ImageProcessingState:
    upload_dir = Path(state["upload_dir"])
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_filename = state.get("original_filename") or "image"
    original_suffix = Path(original_filename).suffix or ".png"
    reversed_name = f"{Path(original_filename).stem}_{uuid.uuid4().hex}_reversed{original_suffix}"
    reversed_path = upload_dir / reversed_name

    with Image.open(BytesIO(state["image_bytes"])) as image:
        reversed_image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        reversed_image.save(reversed_path)

    return {"reversed_file_path": str(reversed_path)}


_graph_builder = StateGraph(ImageProcessingState)
_graph_builder.add_node("extract_metadata", _extract_metadata)
_graph_builder.add_node("reverse_image", _reverse_image)
_graph_builder.add_edge(START, "extract_metadata")
_graph_builder.add_edge("extract_metadata", "reverse_image")
_graph_builder.add_edge("reverse_image", END)
_image_processing_graph = _graph_builder.compile()


def run_image_processing_graph(
    image_bytes: bytes,
    original_filename: str,
    upload_dir: str,
) -> ImageProcessingState:
    return _image_processing_graph.invoke(
        {
            "image_bytes": image_bytes,
            "original_filename": original_filename,
            "upload_dir": upload_dir,
        }
    )
