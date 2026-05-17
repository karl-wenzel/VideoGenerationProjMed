import base64
import json
import os
from pathlib import Path
from typing import Any, Final

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "tmp" / "story.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp_image_generated"
DEFAULT_SCENE_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "scene_images"
IMAGE_GENERATION_URL = "https://saia.gwdg.de/v1/images/generations"
IMAGE_EDIT_URL = "https://saia.gwdg.de/v1/images/edits/"
FLUX_MODEL = "flux"
QWEN_IMAGE_EDIT_INFERENCE_SERVICE = "image-edit-2511"
CHARACTER_OUTPUT_DIR_NAME = "characters"
SCENE_OUTPUT_DIR_NAME = "scenes"
SCENE_REFERENCE_OUTPUT_DIR_NAME = "scene_references"
CHARACTER_REFERENCE_SHEET_FILENAME = "character_reference_sheet.png"
LAST_RUN_PROMPTS_FILENAME = "last_run_prompts.json"
USE_LLM_PROMPT: Final[bool] = True
USE_CHARACTER_REFERENCE_SHEET: Final[bool] = True
COMPOSITE_IMAGE: Final[bool] = True
USE_QWEN_FOR_COMPOSITE: Final[bool] = True
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


def build_scene_prompt(
    characters: list[dict[str, Any]],
    scene: dict[str, Any],
    prompt_log: list[dict[str, Any]] | None = None,
) -> str:
    """Create a consistent image prompt for one scene."""
    reference_instruction = ""
    if USE_CHARACTER_REFERENCE_SHEET:
        reference_instruction = (
            "Use the uploaded character reference sheet only to preserve character "
            "appearance, proportions, clothing, and identity. Do not copy the "
            "reference sheet layout, do not draw panels, and do not add any text "
            "or labels. Generate a new scene matching the description below."
        )

    if USE_LLM_PROMPT:
        try:
            from .llm_prompt_builder import build_image_prompt_with_llm
        except ImportError:
            from llm_prompt_builder import build_image_prompt_with_llm

        scene_prompt = build_image_prompt_with_llm(
            characters,
            scene,
            verbose=VERBOSE,
            prompt_log=prompt_log,
        )
        if reference_instruction:
            return f"{reference_instruction}\n\n{scene_prompt}"

        return scene_prompt

    character_lines = []
    for character in characters:
        name = character.get("name", "Unnamed character")
        details = "; ".join(format_character_details(character))
        character_lines.append(f"- {name}: {details}")

    try:
        from .llm_prompt_builder import STYLE_PROMPT
    except ImportError:
        from llm_prompt_builder import STYLE_PROMPT

    visual_description = "\n".join(
        [
            "Keep character appearance consistent across scenes.",
            "Prioritize the exact character traits below over extra scene detail.",
            "",
            "Fixed character traits:",
            *character_lines,
            "",
            "Brief scene action:",
            str(scene.get("summary", "")),
        ]
    )
    if reference_instruction:
        return f"{reference_instruction}\n\n{visual_description}\n\n{STYLE_PROMPT}"

    return f"{visual_description}\n\n{STYLE_PROMPT}"


def sanitize_filename(value: str) -> str:
    """Create a small filesystem-safe stem from a display name."""
    safe_characters = [
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    ]
    safe_name = "".join(safe_characters).strip("_")
    return safe_name or "unnamed"


def format_character_details(character: dict[str, Any]) -> list[str]:
    """Format all character metadata as prompt-ready consistency details."""
    detail_lines: list[str] = []
    for key, value in character.items():
        if value is None or value == "":
            continue

        label = key.replace("_", " ").title()
        detail_lines.append(f"{label}: {value}")

    return detail_lines


def build_character_reference_prompt(character: dict[str, Any]) -> str:
    """Create a FLUX prompt for one reusable character reference image."""
    try:
        from .llm_prompt_builder import STYLE_PROMPT
    except ImportError:
        from llm_prompt_builder import STYLE_PROMPT

    character_details = format_character_details(character)
    return "\n".join(
        [
            "Create a single-character reference image.",
            "Show the character clearly in a neutral full-body pose.",
            "Use a simple light background with no story scene action.",
            "The character should be easy to reuse as a visual reference.",
            "Do not include text, letters, labels, captions, signs, or watermarks.",
            "Treat every character detail below as a fixed visual identity constraint.",
            "",
            *character_details,
            "",
            STYLE_PROMPT,
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


def generate_character_reference_images(
    characters: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    size: str = "512x512",
) -> list[dict[str, Any]]:
    """Generate and save one FLUX reference image per character."""
    character_output_dir = Path(output_dir)
    character_output_dir.mkdir(parents=True, exist_ok=True)

    generated_characters: list[dict[str, Any]] = []
    for character_index, character in enumerate(characters, start=1):
        if not isinstance(character, dict):
            raise ValueError(f"Character {character_index} must be a JSON object.")

        character_name = str(character.get("name", f"character_{character_index}"))
        prompt = build_character_reference_prompt(character)
        image_data = generate_image_b64(prompt, size=size, model=FLUX_MODEL)

        image_path = (
            character_output_dir
            / f"character_{character_index:03d}_{sanitize_filename(character_name)}.png"
        )
        b64_to_image(image_data["b64_json"], image_path)

        generated_characters.append(
            {
                "character_index": character_index,
                "name": character_name,
                "output_path": str(image_path),
                "prompt": prompt,
                "api_result": image_data,
            }
        )

    return generated_characters


def create_character_reference_sheet(
    character_images: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Combine generated character references into one image for Qwen edits."""
    if not character_images:
        raise ValueError("At least one character reference image is required.")

    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "The 'Pillow' package is required to build the character reference sheet."
        ) from error

    cell_width = 384
    cell_height = 440
    padding = 24
    sheet_width = padding + len(character_images) * (cell_width + padding)
    sheet_height = cell_height + padding * 2

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")

    for image_index, character_image in enumerate(character_images):
        source_path = Path(character_image["output_path"])
        with Image.open(source_path) as source_image:
            image = ImageOps.contain(
                source_image.convert("RGB"),
                (cell_width, cell_height),
            )

        x = padding + image_index * (cell_width + padding)
        image_x = x + (cell_width - image.width) // 2
        image_y = padding + (cell_height - image.height) // 2
        sheet.paste(image, (image_x, image_y))

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(resolved_output_path)
    return resolved_output_path


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


def generate_scene_images(
    json_path: str | Path = DEFAULT_INPUT_PATH,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scene_output_dir: str | Path = DEFAULT_SCENE_OUTPUT_DIR,
    size: str = "512x512",
) -> list[dict[str, Any]]:
    """
    Generate one image for every scene in the story JSON.

    Images are saved into output_dir and the returned list contains one entry
    per generated scene image.
    """
    if COMPOSITE_IMAGE:
        try:
            from .composite_image_generator import generate_composite_scene_images
        except ImportError:
            from composite_image_generator import generate_composite_scene_images

        return generate_composite_scene_images(
            json_path,
            output_dir=output_dir,
            scene_output_dir=scene_output_dir,
            size=size,
        )

    story_spec = load_story_spec(json_path)
    characters = story_spec["characters"]
    scenes = story_spec["scenes"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    character_output_path = output_path / CHARACTER_OUTPUT_DIR_NAME
    scene_output_path = Path(scene_output_dir)
    scene_reference_output_path = output_path / SCENE_REFERENCE_OUTPUT_DIR_NAME
    scene_output_path.mkdir(parents=True, exist_ok=True)
    prompt_log: list[dict[str, Any]] = []

    generated_images: list[dict[str, Any]] = []
    try:
        character_images: list[dict[str, Any]] = []
        if USE_CHARACTER_REFERENCE_SHEET:
            character_images = generate_character_reference_images(
                characters,
                character_output_path,
                size=size,
            )
            prompt_log.extend(
                {
                    "api_call": "flux_character_reference_generation",
                    "character_index": character_image["character_index"],
                    "character_name": character_image["name"],
                    "prompt": character_image["prompt"],
                }
                for character_image in character_images
            )

        for scene_index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise ValueError(f"Scene {scene_index} must be a JSON object.")

            prompt = build_scene_prompt(characters, scene, prompt_log)
            image_path = scene_output_path / f"scene_{scene_index:03d}.png"
            scene_reference_sheet_path: Path | None = None
            if USE_CHARACTER_REFERENCE_SHEET:
                scene_character_images = select_character_images_for_scene(
                    scene,
                    character_images,
                )
                scene_reference_sheet_path = create_character_reference_sheet(
                    scene_character_images,
                    scene_reference_output_path
                    / f"scene_{scene_index:03d}_reference.png",
                )
                prompt_log.append(
                    {
                        "api_call": "qwen_scene_image_edit",
                        "scene_index": scene_index,
                        "reference_image_path": str(scene_reference_sheet_path),
                        "reference_character_names": [
                            str(character_image["name"])
                            for character_image in scene_character_images
                        ],
                        "prompt": prompt,
                    }
                )
                generate_scene_image_with_qwen(
                    prompt,
                    scene_reference_sheet_path,
                    image_path,
                )
            else:
                prompt_log.append(
                    {
                        "api_call": "flux_scene_image_generation",
                        "scene_index": scene_index,
                        "prompt": prompt,
                    }
                )
                image_data = generate_image_b64(prompt, size=size, model=FLUX_MODEL)
                b64_to_image(image_data["b64_json"], image_path)

            generated_images.append(
                {
                    "scene_index": scene_index,
                    "output_path": str(image_path),
                    "prompt": prompt,
                    "character_reference_sheet_path": (
                        str(scene_reference_sheet_path)
                        if scene_reference_sheet_path is not None
                        else None
                    ),
                    "character_reference_images": character_images,
                }
            )
    finally:
        prompt_log_path = save_prompt_log(prompt_log, output_path)

    for generated_image in generated_images:
        generated_image["prompt_log_path"] = str(prompt_log_path)

    return generated_images


if __name__ == "__main__":
    images = generate_scene_images()
    print(json.dumps(images, indent=2))
