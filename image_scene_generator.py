import base64
import json
import os
from pathlib import Path
from typing import Any, Final

import requests


DEFAULT_INPUT_PATH = "example_input.json"
DEFAULT_OUTPUT_DIR = "tmp_image_generated"
IMAGE_GENERATION_URL = "https://saia.gwdg.de/v1/images/generations"
USE_LLM_PROMPT: Final[bool] = True
VERBOSE: Final[bool] = True


def load_env_file_if_available() -> None:
    """Load a local .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def load_story_spec(json_path: str | Path = DEFAULT_INPUT_PATH) -> dict[str, Any]:
    """Load the story specification containing characters and scenes."""
    input_path = Path(json_path)
    with input_path.open("r", encoding="utf-8") as file:
        story_spec = json.load(file)

    if not isinstance(story_spec, dict):
        raise ValueError("The input JSON must contain a JSON object.")

    if not isinstance(story_spec.get("characters"), list):
        raise ValueError("The input JSON must contain a 'characters' array.")

    if not isinstance(story_spec.get("scenes"), list):
        raise ValueError("The input JSON must contain a 'scenes' array.")

    return story_spec


def build_scene_prompt(
    characters: list[dict[str, Any]],
    scene: dict[str, Any],
) -> str:
    """Create a consistent image prompt for one scene."""
    if USE_LLM_PROMPT:
        from llm_prompt_builder import build_image_prompt_with_llm

        return build_image_prompt_with_llm(characters, scene, verbose=VERBOSE)

    character_lines = [
        f"- {character.get('name', 'Unnamed character')}, "
        f"age {character.get('age', 'unknown')}: "
        f"{character.get('description', 'No description provided.')}"
        for character in characters
    ]

    return "\n".join(
        [
            "Create a charming storybook illustration for a children's story.",
            "Keep character appearance consistent across scenes.",
            "",
            "Characters:",
            *character_lines,
            "",
            "Scene:",
            str(scene.get("summary", "")),
            "",
            "Opening sentence:",
            str(scene.get("first_sentence", "")),
            "",
            "Closing sentence:",
            str(scene.get("last_sentence", "")),
        ]
    )


def b64_to_image(b64_string: str, output_path: str | Path) -> None:
    """Decode a base64 image string and write it to disk."""
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]

    image_bytes = base64.b64decode("".join(b64_string.split()), validate=True)
    Path(output_path).write_bytes(image_bytes)


def generate_image_b64(
    prompt: str,
    *,
    size: str = "512x512",
    api_key: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Generate one image and return the first API data item."""
    load_env_file_if_available()

    resolved_api_key = api_key or os.getenv("ACADEMIC_CLOUD_CHATAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Missing ACADEMIC_CLOUD_CHATAI_API_KEY environment variable."
        )

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "size": size,
    }

    if VERBOSE:
        print("API call: image generation")
        print("role: image_generation")
        print(f"prompt: {prompt}")

    response = requests.post(
        IMAGE_GENERATION_URL,
        headers=headers,
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    response_json = response.json()
    data = response_json.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("The image generation response did not contain image data.")

    first_image = data[0]
    if not isinstance(first_image, dict) or "b64_json" not in first_image:
        raise ValueError("The image generation response did not contain 'b64_json'.")

    return first_image


def generate_scene_images(
    json_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    size: str = "512x512",
) -> list[dict[str, Any]]:
    """
    Generate one image for every scene in the story JSON.

    Images are saved into output_dir and the returned list contains one entry
    per generated scene image.
    """
    story_spec = load_story_spec(json_path)
    characters = story_spec["characters"]
    scenes = story_spec["scenes"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_images: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene {scene_index} must be a JSON object.")

        prompt = build_scene_prompt(characters, scene)
        image_data = generate_image_b64(prompt, size=size)

        image_path = output_path / f"scene_{scene_index:03d}.png"
        b64_to_image(image_data["b64_json"], image_path)

        generated_images.append(
            {
                "scene_index": scene_index,
                "output_path": str(image_path),
                "prompt": prompt,
                "api_result": image_data,
            }
        )

    return generated_images


if __name__ == "__main__":
    images = generate_scene_images()
    print(json.dumps(images, indent=2))
