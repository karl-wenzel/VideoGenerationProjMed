import re
import statistics
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps


CHARACTER_REFERENCE_OUTPUT_DIR_NAME = "character_references"
SCENE_BACKGROUND_OUTPUT_DIR_NAME = "scene_backgrounds"
SCENE_CHARACTER_POSE_OUTPUT_DIR_NAME = "scene_character_poses"
CHARACTER_CUTOUT_OUTPUT_DIR_NAME = "character_cutouts"
COMPOSITE_SCENE_SIZE = "1344x768"
COMPOSITE_STYLE_PROMPT = (
    "Style: warm children's storybook illustration, gentle natural light, "
    "soft painterly finish, expressive friendly subject, coherent composition, "
    "suitable for a picture book."
)
FORBIDDEN_IMAGE_PROMPT_WORDS = (
    "text",
    "texts",
    "letter",
    "letters",
    "label",
    "labels",
    "caption",
    "captions",
    "sign",
    "signs",
    "watermark",
    "watermarks",
)
SOLID_BACKGROUND_RGB = (255, 0, 255)
KEY_COLOR_TOLERANCE = 55
MASK_FEATHER_RADIUS = 1.2


def generate_composite_scene_images(
    json_path: str | Path,
    *,
    output_dir: str | Path,
    size: str = "512x512",
) -> list[dict[str, Any]]:
    """Generate scenes by compositing posed character cutouts over backgrounds."""
    from image_scene_generator import (
        FLUX_MODEL,
        VERBOSE,
        b64_to_image,
        generate_image_b64,
        generate_scene_image_with_qwen,
        load_story_spec,
        sanitize_filename,
        save_prompt_log,
        select_character_images_for_scene,
    )

    story_spec = load_story_spec(json_path)
    characters = story_spec["characters"]
    scenes = story_spec["scenes"]

    output_path = Path(output_dir)
    character_reference_dir = output_path / CHARACTER_REFERENCE_OUTPUT_DIR_NAME
    background_dir = output_path / SCENE_BACKGROUND_OUTPUT_DIR_NAME
    pose_dir = output_path / SCENE_CHARACTER_POSE_OUTPUT_DIR_NAME
    cutout_dir = output_path / CHARACTER_CUTOUT_OUTPUT_DIR_NAME
    scene_dir = output_path / "scenes"

    for directory_path in [
        character_reference_dir,
        background_dir,
        pose_dir,
        cutout_dir,
        scene_dir,
    ]:
        directory_path.mkdir(parents=True, exist_ok=True)

    prompt_log: list[dict[str, Any]] = []
    generated_images: list[dict[str, Any]] = []

    try:
        character_references = generate_composite_character_references(
            characters,
            character_reference_dir,
            size=size,
            prompt_log=prompt_log,
        )

        for scene_index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict):
                raise ValueError(f"Scene {scene_index} must be a JSON object.")

            scene_character_references = select_character_images_for_scene(
                scene,
                character_references,
            )
            layout = build_scene_layout(
                characters,
                scene,
                scene_character_references,
                prompt_log,
                verbose=VERBOSE,
            )

            background_prompt = ensure_style_prompt(
                sanitize_image_prompt(str(layout["background_prompt"]), characters)
            )
            background_path = background_dir / f"scene_{scene_index:03d}_background.png"
            prompt_log.append(
                {
                    "api_call": "flux_composite_background_generation",
                    "scene_index": scene_index,
                    "prompt": background_prompt,
                }
            )
            background_data = generate_image_b64(
                background_prompt,
                size=COMPOSITE_SCENE_SIZE,
                model=FLUX_MODEL,
            )
            b64_to_image(background_data["b64_json"], background_path)

            character_layers: list[dict[str, Any]] = []
            layout_characters = map_layout_characters_by_name(layout, characters)
            fallback_boxes = build_fallback_placement_boxes(
                len(scene_character_references)
            )

            for character_position, character_reference in enumerate(
                scene_character_references
            ):
                character_name = str(character_reference["name"])
                layout_character = layout_characters.get(
                    character_name.lower(),
                    {},
                )
                prompt = build_pose_prompt(
                    character_reference["character"],
                    scene,
                    layout_character,
                    characters,
                )
                pose_path = (
                    pose_dir
                    / f"scene_{scene_index:03d}_character_"
                    f"{character_reference['character_index']:03d}_"
                    f"{sanitize_filename(character_name)}.png"
                )
                cutout_path = (
                    cutout_dir
                    / f"scene_{scene_index:03d}_character_"
                    f"{character_reference['character_index']:03d}_"
                    f"{sanitize_filename(character_name)}.png"
                )
                prompt_log.append(
                    {
                        "api_call": "qwen_composite_character_pose_generation",
                        "scene_index": scene_index,
                        "character_index": character_reference["character_index"],
                        "character_name": character_name,
                        "reference_image_path": character_reference["output_path"],
                        "prompt": prompt,
                    }
                )
                generate_scene_image_with_qwen(
                    prompt,
                    character_reference["output_path"],
                    pose_path,
                )

                cutout_character_image(pose_path, cutout_path)
                character_layers.append(
                    {
                        "character_index": character_reference["character_index"],
                        "name": character_name,
                        "pose_path": str(pose_path),
                        "cutout_path": str(cutout_path),
                        "placement_box": normalize_placement_box(
                            layout_character.get("placement_box"),
                            fallback_boxes[character_position],
                        ),
                        "layer_order": to_int(
                            layout_character.get("layer_order"),
                            character_position,
                        ),
                        "prompt": prompt,
                    }
                )

            final_path = scene_dir / f"scene_{scene_index:03d}.png"
            composite_layers(background_path, character_layers, final_path)

            generated_images.append(
                {
                    "scene_index": scene_index,
                    "output_path": str(final_path),
                    "background_path": str(background_path),
                    "background_prompt": background_prompt,
                    "character_layers": character_layers,
                    "layout": layout,
                    "character_references": character_references,
                }
            )
    finally:
        prompt_log_path = save_prompt_log(prompt_log, output_path)

    for generated_image in generated_images:
        generated_image["prompt_log_path"] = str(prompt_log_path)

    return generated_images


def generate_composite_character_references(
    characters: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    size: str,
    prompt_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate one unnamed, single-subject reference image per character."""
    from image_scene_generator import (
        FLUX_MODEL,
        b64_to_image,
        generate_image_b64,
        sanitize_filename,
    )

    generated_references: list[dict[str, Any]] = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for character_index, character in enumerate(characters, start=1):
        if not isinstance(character, dict):
            raise ValueError(f"Character {character_index} must be a JSON object.")

        character_name = str(character.get("name", f"character_{character_index}"))
        prompt = build_character_reference_prompt(character, characters)
        prompt_log.append(
            {
                "api_call": "flux_composite_character_reference_generation",
                "character_index": character_index,
                "character_name": character_name,
                "prompt": prompt,
            }
        )

        image_data = generate_image_b64(prompt, size=size, model=FLUX_MODEL)
        image_path = (
            output_path
            / f"character_{character_index:03d}_{sanitize_filename(character_name)}.png"
        )
        b64_to_image(image_data["b64_json"], image_path)
        generated_references.append(
            {
                "character_index": character_index,
                "name": character_name,
                "character": character,
                "output_path": str(image_path),
                "prompt": prompt,
                "api_result": image_data,
            }
        )

    return generated_references


def build_character_reference_prompt(
    character: dict[str, Any],
    all_characters: list[dict[str, Any]],
) -> str:
    """Build a positive prompt for a single unnamed character reference."""
    details = format_visual_character_details(character, all_characters)
    prompt_parts = [
        "Single full-body subject.",
        "Neutral standing pose.",
        "Centered composition.",
        "Plain solid light studio backdrop.",
        "Clear silhouette and complete outfit.",
        "Consistent storybook illustration.",
        *details,
        COMPOSITE_STYLE_PROMPT,
    ]
    return sanitize_image_prompt("\n".join(prompt_parts), all_characters)


def build_scene_layout(
    characters: list[dict[str, Any]],
    scene: dict[str, Any],
    scene_character_references: list[dict[str, Any]],
    prompt_log: list[dict[str, Any]],
    *,
    verbose: bool,
) -> dict[str, Any]:
    """Return an LLM layout, falling back to deterministic prompts and boxes."""
    from llm_prompt_builder import build_composite_scene_layout_with_llm

    scene_characters = [
        character_reference["character"]
        for character_reference in scene_character_references
    ]
    try:
        layout = build_composite_scene_layout_with_llm(
            characters,
            scene,
            scene_characters,
            verbose=verbose,
            prompt_log=prompt_log,
        )
        return validate_scene_layout(layout, scene_character_references)
    except Exception as error:
        fallback_layout = build_fallback_scene_layout(scene, scene_character_references)
        prompt_log.append(
            {
                "api_call": "composite_scene_layout_fallback",
                "error": str(error),
                "layout": fallback_layout,
            }
        )
        return fallback_layout


def validate_scene_layout(
    layout: dict[str, Any],
    scene_character_references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate required layout shape while allowing later prompt sanitization."""
    if not isinstance(layout.get("background_prompt"), str):
        raise ValueError("Composite layout requires a string background_prompt.")

    layout_characters = layout.get("characters")
    if not isinstance(layout_characters, list):
        raise ValueError("Composite layout requires a characters array.")

    known_names = {
        str(character_reference["name"]).lower()
        for character_reference in scene_character_references
    }
    valid_characters = []
    for item in layout_characters:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).lower()
        if name not in known_names:
            continue

        valid_characters.append(item)

    if not valid_characters:
        raise ValueError("Composite layout did not include any scene characters.")

    return {
        "background_prompt": layout["background_prompt"],
        "characters": valid_characters,
    }


def build_fallback_scene_layout(
    scene: dict[str, Any],
    scene_character_references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a deterministic layout if LLM layout planning is unavailable."""
    fallback_boxes = build_fallback_placement_boxes(len(scene_character_references))
    return {
        "background_prompt": build_fallback_background_prompt(scene),
        "characters": [
            {
                "name": character_reference["name"],
                "pose_prompt": build_fallback_pose_action(scene),
                "placement_box": fallback_boxes[index],
                "layer_order": index,
            }
            for index, character_reference in enumerate(scene_character_references)
        ],
    }


def build_fallback_background_prompt(scene: dict[str, Any]) -> str:
    """Build a simple setting-only prompt from scene text."""
    scene_text = collect_scene_description(scene)
    return ensure_style_prompt(
        "\n".join(
            [
                "Empty storybook setting.",
                "Open foreground space for later character compositing.",
                scene_text,
            ]
        )
    )


def ensure_style_prompt(prompt: str) -> str:
    """Append the composite style prompt once."""
    if COMPOSITE_STYLE_PROMPT in prompt:
        return prompt

    return f"{prompt.strip()}\n{COMPOSITE_STYLE_PROMPT}"


def collect_full_scene_description(scene: dict[str, Any]) -> str:
    """Collect all available scene prose for pose planning."""
    scene_lines = []
    for key in ["summary", "first_sentence", "last_sentence"]:
        value = scene.get(key)
        if value:
            scene_lines.append(str(value))

    return " ".join(scene_lines) or "Gentle story moment in a picture book scene."


def build_pose_alignment_hint(scene: dict[str, Any]) -> str:
    """Create a compact scene-aware posture hint for the pose prompt."""
    scene_text = collect_full_scene_description(scene)
    lower_scene_text = scene_text.lower()
    posture_hint = ""
    if "picnic" in lower_scene_text:
        posture_hint = "Picnic body language: seated or kneeling near ground level."
    elif any(word in lower_scene_text for word in ["gather", "collect", "moss"]):
        posture_hint = (
            "Gathering body language: crouching, bending, reaching, or carrying."
        )
    elif any(word in lower_scene_text for word in ["look up", "tree", "nest", "birds"]):
        posture_hint = (
            "Discovery body language: looking upward and reacting to something above."
        )
    elif any(word in lower_scene_text for word in ["meet", "stream", "friend"]):
        posture_hint = "Meeting body language: friendly greeting and open posture."

    if posture_hint:
        return f"{scene_text} {posture_hint}"

    return scene_text


def append_pose_clarity(prompt: str, scene: dict[str, Any]) -> str:
    """Add scene context and posture constraints to a pose prompt."""
    return "\n".join(
        [
            "Scene context for matching the final background:",
            build_pose_alignment_hint(scene),
            "Use a body posture that matches this scene activity.",
            prompt,
        ]
    )


def build_fallback_pose_action(scene: dict[str, Any]) -> str:
    """Build a compact action phrase from scene text."""
    return collect_scene_description(scene)


def collect_scene_description(scene: dict[str, Any]) -> str:
    """Collect scene prose in a stable priority order."""
    for key in ["summary", "first_sentence", "last_sentence"]:
        value = scene.get(key)
        if value:
            return str(value)

    return "Gentle story moment in a picture book scene."


def build_pose_prompt(
    character: dict[str, Any],
    scene: dict[str, Any],
    layout_character: dict[str, Any],
    all_characters: list[dict[str, Any]],
) -> str:
    """Build a positive Qwen edit prompt for an isolated posed character."""
    pose_action = str(
        layout_character.get("pose_prompt") or build_fallback_pose_action(scene)
    )
    details = format_visual_character_details(character, all_characters)
    prompt_parts = [
        "Use the provided reference image as the identity guide.",
        "Single full-body subject.",
        "Plain solid pure magenta #ff00ff studio backdrop.",
        "Centered isolated subject.",
        "Clear silhouette.",
        "Scene-specific body pose and expression:",
        append_pose_clarity(pose_action, scene),
        *details,
        COMPOSITE_STYLE_PROMPT,
    ]
    return sanitize_image_prompt("\n".join(prompt_parts), all_characters)


def format_visual_character_details(
    character: dict[str, Any],
    all_characters: list[dict[str, Any]],
) -> list[str]:
    """Format visual character metadata while omitting internal identifiers."""
    omitted_keys = {"name", "id", "character_name"}
    detail_lines: list[str] = []
    for key, value in character.items():
        if key in omitted_keys or value is None or value == "":
            continue

        label = key.replace("_", " ").title()
        detail_lines.append(f"{label}: {value}")

    return [
        sanitize_image_prompt(detail_line, all_characters)
        for detail_line in detail_lines
    ]


def sanitize_image_prompt(
    prompt: str,
    characters: list[dict[str, Any]],
) -> str:
    """Remove image-model trigger words from prompts sent to image endpoints."""
    sanitized_prompt = prompt
    for character in characters:
        name = character.get("name")
        if not name:
            continue

        sanitized_prompt = re.sub(
            rf"\b{re.escape(str(name))}\b",
            "the subject",
            sanitized_prompt,
            flags=re.IGNORECASE,
        )

    forbidden_pattern = "|".join(
        re.escape(forbidden_word)
        for forbidden_word in FORBIDDEN_IMAGE_PROMPT_WORDS
    )
    sanitized_prompt = re.sub(
        rf"\b({forbidden_pattern})\b",
        "",
        sanitized_prompt,
        flags=re.IGNORECASE,
    )
    sanitized_prompt = re.sub(r"[ \t]{2,}", " ", sanitized_prompt)
    sanitized_prompt = re.sub(r" +([.,:;])", r"\1", sanitized_prompt)
    cleaned_lines = []
    for line in sanitized_prompt.splitlines():
        cleaned_line = line.strip()
        for _ in range(3):
            cleaned_line = re.sub(
                r"\b(with|beside|near|holding|wearing|showing|including|and|a|an|the)\s*$",
                "",
                cleaned_line,
                flags=re.IGNORECASE,
            ).strip()

        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines).strip()


def map_layout_characters_by_name(
    layout: dict[str, Any],
    characters: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Map valid layout character entries by case-insensitive name."""
    known_names = {
        str(character.get("name", "")).lower()
        for character in characters
        if character.get("name")
    }
    mapped_characters: dict[str, dict[str, Any]] = {}
    for layout_character in layout.get("characters", []):
        if not isinstance(layout_character, dict):
            continue

        name = str(layout_character.get("name", "")).lower()
        if name in known_names:
            mapped_characters[name] = layout_character

    return mapped_characters


def build_fallback_placement_boxes(character_count: int) -> list[dict[str, float]]:
    """Place characters evenly along the lower scene area."""
    if character_count <= 0:
        return []

    width = min(0.34, 0.82 / character_count)
    height = 0.72
    gap = (1.0 - character_count * width) / (character_count + 1)
    return [
        {
            "x": gap + index * (width + gap),
            "y": 0.24,
            "width": width,
            "height": height,
        }
        for index in range(character_count)
    ]


def normalize_placement_box(
    candidate_box: Any,
    fallback_box: dict[str, float],
) -> dict[str, float]:
    """Clamp an LLM placement box to usable normalized bounds."""
    if not isinstance(candidate_box, dict):
        return fallback_box

    normalized_box: dict[str, float] = {}
    for key in ["x", "y", "width", "height"]:
        value = candidate_box.get(key)
        if not isinstance(value, int | float):
            return fallback_box

        normalized_box[key] = float(value)

    normalized_box["width"] = clamp(normalized_box["width"], 0.08, 1.0)
    normalized_box["height"] = clamp(normalized_box["height"], 0.08, 1.0)
    normalized_box["x"] = clamp(
        normalized_box["x"],
        0.0,
        1.0 - normalized_box["width"],
    )
    normalized_box["y"] = clamp(
        normalized_box["y"],
        0.0,
        1.0 - normalized_box["height"],
    )
    return normalized_box


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a float into the provided range."""
    return max(minimum, min(maximum, value))


def to_int(value: Any, fallback: int) -> int:
    """Convert arbitrary layout order values into an integer."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def cutout_character_image(
    input_path: str | Path,
    output_path: str | Path,
    *,
    key_color: tuple[int, int, int] | None = None,
    tolerance: int = KEY_COLOR_TOLERANCE,
) -> Path:
    """Remove border-connected solid-color background from a character image."""
    with Image.open(input_path) as source_image:
        rgba_image = source_image.convert("RGBA")

    resolved_key_color = key_color or infer_border_background_color(rgba_image)
    background_mask = find_border_connected_key_pixels(
        rgba_image,
        resolved_key_color,
        tolerance,
    )
    alpha = ImageChops.invert(background_mask).filter(
        ImageFilter.GaussianBlur(MASK_FEATHER_RADIUS)
    )
    cutout_image = rgba_image.copy()
    cutout_image.putalpha(alpha)

    visible_bbox = alpha.getbbox()
    if visible_bbox:
        cutout_image = cutout_image.crop(visible_bbox)

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    cutout_image.save(resolved_output_path)
    return resolved_output_path


def find_border_connected_key_pixels(
    image: Image.Image,
    key_color: tuple[int, int, int],
    tolerance: int,
) -> Image.Image:
    """Return an L-mode mask for key-color pixels connected to the image border."""
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    pixels = rgb_image.load()
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    mask = Image.new("L", (width, height), 0)
    mask_pixels = mask.load()

    for x in range(width):
        queue.append((x, 0))
        queue.append((x, height - 1))
    for y in range(height):
        queue.append((0, y))
        queue.append((width - 1, y))

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue

        visited.add((x, y))
        if not pixel_matches_background(pixels[x, y], key_color, tolerance):
            continue

        mask_pixels[x, y] = 255
        for next_x, next_y in [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]:
            if 0 <= next_x < width and 0 <= next_y < height:
                queue.append((next_x, next_y))

    return mask


def infer_border_background_color(image: Image.Image) -> tuple[int, int, int]:
    """Infer the solid backdrop color from the image border."""
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    pixels = rgb_image.load()
    border_pixels: list[tuple[int, int, int]] = []

    for x in range(width):
        border_pixels.append(pixels[x, 0])
        border_pixels.append(pixels[x, height - 1])
    for y in range(height):
        border_pixels.append(pixels[0, y])
        border_pixels.append(pixels[width - 1, y])

    return tuple(
        int(statistics.median(pixel[channel] for pixel in border_pixels))
        for channel in range(3)
    )


def pixel_matches_background(
    pixel: tuple[int, int, int],
    key_color: tuple[int, int, int],
    tolerance: int,
) -> bool:
    """Check if a pixel is close enough to the inferred solid backdrop."""
    return pixel_matches_key(pixel, key_color, tolerance) or pixel_is_magenta_backdrop(
        pixel
    )


def pixel_matches_key(
    pixel: tuple[int, int, int],
    key_color: tuple[int, int, int],
    tolerance: int,
) -> bool:
    """Check if a pixel is close enough to a key color."""
    return all(
        abs(pixel[channel] - key_color[channel]) <= tolerance for channel in range(3)
    )


def pixel_is_magenta_backdrop(pixel: tuple[int, int, int]) -> bool:
    """Recognize generated magenta-family backdrop pixels."""
    red, green, blue = pixel
    return (
        green <= 70
        and red >= 90
        and blue >= 80
        and abs(red - blue) <= 85
        and red > green * 2
        and blue > green * 2
    )


def composite_layers(
    background_path: str | Path,
    character_layers: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Composite cutout layers over a background image."""
    with Image.open(background_path) as background_image:
        canvas = background_image.convert("RGBA")

    for layer in sorted(character_layers, key=lambda item: item["layer_order"]):
        with Image.open(layer["cutout_path"]) as cutout_image:
            cutout = cutout_image.convert("RGBA")

        placed_cutout, paste_position = fit_cutout_to_box(
            cutout,
            layer["placement_box"],
            canvas.size,
        )
        canvas.alpha_composite(placed_cutout, paste_position)

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(resolved_output_path)
    return resolved_output_path


def fit_cutout_to_box(
    cutout: Image.Image,
    placement_box: dict[str, float],
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, tuple[int, int]]:
    """Resize a cutout into a normalized placement box."""
    canvas_width, canvas_height = canvas_size
    box_x = int(placement_box["x"] * canvas_width)
    box_y = int(placement_box["y"] * canvas_height)
    box_width = max(1, int(placement_box["width"] * canvas_width))
    box_height = max(1, int(placement_box["height"] * canvas_height))

    fitted_cutout = ImageOps.contain(cutout, (box_width, box_height))
    paste_x = box_x + (box_width - fitted_cutout.width) // 2
    paste_y = box_y + box_height - fitted_cutout.height
    return fitted_cutout, (paste_x, paste_y)
