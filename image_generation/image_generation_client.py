import base64
import json
import os
from pathlib import Path
from typing import Any, Final

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "tmp" / "story.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp/tmp_image_generated"
DEFAULT_SCENE_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "scene_images"
IMAGE_GENERATION_URL = "https://saia.gwdg.de/v1/images/generations"
IMAGE_EDIT_URL = "https://saia.gwdg.de/v1/images/edits/"
FLUX_MODEL = "flux"
QWEN_IMAGE_EDIT_INFERENCE_SERVICE = "image-edit-2511"
LAST_RUN_PROMPTS_FILENAME = "last_run_prompts.json"
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


def save_prompt_log(prompt_log: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """Persist all prompts sent to APIs during the latest run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prompt_log_path = output_path / LAST_RUN_PROMPTS_FILENAME
    prompt_log_path.write_text(
        json.dumps(prompt_log, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return prompt_log_path


def sanitize_filename(value: str) -> str:
    """Create a small filesystem-safe stem from a display name."""
    safe_characters = [
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    ]
    safe_name = "".join(safe_characters).strip("_")
    return safe_name or "unnamed"


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
    model: str = FLUX_MODEL,
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
        "model": model,
        "size": size,
        "n": 1,
        "response_format": "b64_json",
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


def get_scene_text(scene: dict[str, Any]) -> str:
    """Collect searchable scene text from the JSON scene fields."""
    return " ".join(str(value) for value in scene.values()).lower()


def select_character_images_for_scene(
    scene: dict[str, Any],
    character_images: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select only character references mentioned in the scene text."""
    scene_text = get_scene_text(scene)
    selected_images = [
        character_image
        for character_image in character_images
        if str(character_image["name"]).lower() in scene_text
    ]

    return selected_images or character_images


def save_image_edit_response(response: requests.Response, output_path: str | Path) -> None:
    """Save a Qwen image-edit response that may be binary or JSON/base64."""
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        Path(output_path).write_bytes(response.content)
        return

    response_json = response.json()
    data = response_json.get("data")
    if isinstance(data, list) and data:
        first_image = data[0]
        if isinstance(first_image, dict) and "b64_json" in first_image:
            b64_to_image(first_image["b64_json"], output_path)
            return

    b64_json = response_json.get("b64_json")
    if isinstance(b64_json, str):
        b64_to_image(b64_json, output_path)
        return

    raise ValueError("The image edit response did not contain image bytes or base64.")


def generate_scene_image_with_qwen(
    prompt: str,
    input_image_path: str | Path,
    output_path: str | Path,
    *,
    api_key: str | None = None,
    timeout_seconds: int = 180,
) -> None:
    """Generate one scene image with Qwen image edit."""
    load_env_file_if_available()

    resolved_api_key = api_key or os.getenv("ACADEMIC_CLOUD_CHATAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Missing ACADEMIC_CLOUD_CHATAI_API_KEY environment variable."
        )

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "inference-service": QWEN_IMAGE_EDIT_INFERENCE_SERVICE,
    }

    if VERBOSE:
        print("API call: Qwen scene generation")
        print("role: image_generation")
        print(f"prompt: {prompt}")
        print(f"input_image: {input_image_path}")

    files: dict[str, Any] = {
        "prompt": (None, prompt),
    }
    with Path(input_image_path).open("rb") as image_file:
        files["image"] = (
            Path(input_image_path).name,
            image_file,
            "image/png",
        )
        response = requests.post(
            IMAGE_EDIT_URL,
            headers=headers,
            files=files,
            timeout=timeout_seconds,
        )

    response.raise_for_status()
    save_image_edit_response(response, output_path)
