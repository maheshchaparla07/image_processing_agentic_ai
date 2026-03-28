from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageChops, ImageEnhance


def run_ela(image_path: str = "image.jpg", output_path: str = "ela_result.jpg", quality: int = 90) -> None:
    with Image.open(image_path) as original_image:
        original_rgb = original_image.convert("RGB")

    buffer = BytesIO()
    original_rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    recompressed = Image.open(buffer).convert("RGB")
    difference = ImageChops.difference(original_rgb, recompressed)

    enhanced = ImageEnhance.Brightness(difference).enhance(15.0)
    enhanced.save(output_path, format="JPEG", quality=95)
    enhanced.show()


def main() -> None:
    run_ela("image.jpg", "ela_result.jpg", 90)
    print("ELA analysis complete")


if __name__ == "__main__":
    main()
