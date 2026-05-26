import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_generation.composite_image_generator import generate_composite_scene_images
from image_generation.prompt_config import CompositeGenerationOptions
from image_generation.image_generation_client import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCENE_OUTPUT_DIR,
    PROJECT_ROOT,
)


def clear_directory_contents(directory_path: str | Path) -> None:
    """Remove all existing files and folders inside a project output directory."""
    resolved_path = Path(directory_path).resolve()
    project_root = PROJECT_ROOT.resolve()

    if not resolved_path.exists():
        resolved_path.mkdir(parents=True, exist_ok=True)
        return

    if not resolved_path.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {resolved_path}")

    if project_root not in [resolved_path, *resolved_path.parents]:
        raise ValueError(f"Refusing to clear a directory outside the project: {resolved_path}")

    for child_path in resolved_path.iterdir():
        if child_path.is_dir():
            shutil.rmtree(child_path)
        else:
            child_path.unlink()


def clear_scene_generation_outputs(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scene_output_dir: str | Path = DEFAULT_SCENE_OUTPUT_DIR,
) -> None:
    """Clear stale generated image outputs before starting a fresh script run."""
    clear_directory_contents(output_dir)
    clear_directory_contents(scene_output_dir)


def generate_scene_images(
    json_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scene_output_dir: str | Path = DEFAULT_SCENE_OUTPUT_DIR,
    size: str = "512x512",
    options: CompositeGenerationOptions | None = None,
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
        options=options,
    )


if __name__ == "__main__":
    clear_scene_generation_outputs()
    images = generate_scene_images()
    print(json.dumps(images, indent=2))
