import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_generation.composite_image_generator import generate_composite_scene_images
from image_generation.image_generation_client import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_OUTPUT_DIR,
)


def generate_scene_images(
    json_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scene_output_dir: str | Path = DEFAULT_SCENE_OUTPUT_DIR,
    size: str = "512x512",
) -> list[dict[str, Any]]:
    """
    Generate one image for every scene in the story JSON.

    The active pipeline always uses the composite scene generator.
    """
    return generate_composite_scene_images(
        json_path,
        output_dir=output_dir,
        scene_output_dir=scene_output_dir,
        size=size,
    )


if __name__ == "__main__":
    images = generate_scene_images()
    print(json.dumps(images, indent=2))
