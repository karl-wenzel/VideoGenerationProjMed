import json
import os
from typing import Any


CHAT_COMPLETIONS_BASE_URL = "https://saia.gwdg.de/v1"
PROMPT_BUILDER_MODEL = "openai-gpt-oss-120b"
STYLE_PROMPT = (
    "Style: warm children's storybook illustration, gentle natural light, "
    "soft painterly textures, expressive but friendly characters, coherent "
    "composition, suitable for a picture book."
)


def _load_env_file_if_available() -> None:
    """Load a local .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def build_image_prompt_with_llm(
    characters: list[dict[str, Any]],
    scene: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str = PROMPT_BUILDER_MODEL,
    verbose: bool = False,
    prompt_log: list[dict[str, Any]] | None = None,
) -> str:
    """Ask a chat model for scene content, then append the fixed style."""
    _load_env_file_if_available()

    resolved_api_key = api_key or os.getenv("ACADEMIC_CLOUD_CHATAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Missing ACADEMIC_CLOUD_CHATAI_API_KEY environment variable."
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The 'openai' package is required for LLM prompt generation."
        ) from error

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=CHAT_COMPLETIONS_BASE_URL,
    )

    story_context = {
        "characters": characters,
        "scene": scene,
    }
    output_contract = (
        "Return a compact image prompt with should include: "
        "character traits from the JSON, including clothing, "
        "hair, accessories, and any other visual identity fields; "
        "a brief description of the scene action and setting. "
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You write compact image prompts for image generation. "
                "Your main job is enforcing character consistency. Include "
                "all fixed visual character attributes provided by the user. "
                "Keep the scene description brief and avoid unnecessary "
                "background, composition, mood, prop, or lighting details. "
                "Do not choose or mention an art style, medium, rendering "
                "technique, camera model, artist, genre label, or visual "
                "finish. Return only the prompt text, with no "
                "markdown, labels, or extra explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{output_contract}\n\n"
                "Use this JSON. Do not add style information.\n\n"
                f"{json.dumps(story_context, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

    if verbose:
        print("API call: LLM prompt builder")
        for message in messages:
            print(f"role: {message['role']}")
            print(f"prompt: {message['content']}")

    if prompt_log is not None:
        prompt_log.append(
            {
                "api_call": "llm_prompt_builder",
                "model": model,
                "messages": [
                    {
                        "role": str(message["role"]),
                        "prompt": str(message["content"]),
                    }
                    for message in messages
                ],
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    visual_description = response.choices[0].message.content
    if not visual_description:
        raise ValueError("The LLM did not return a visual scene description.")

    return f"{visual_description.strip()}\n\n{STYLE_PROMPT}"


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response."""
    stripped_text = raw_text.strip()
    try:
        parsed_json = json.loads(stripped_text)
    except json.JSONDecodeError:
        start_index = stripped_text.find("{")
        end_index = stripped_text.rfind("}")
        if start_index == -1 or end_index == -1 or end_index <= start_index:
            raise

        parsed_json = json.loads(stripped_text[start_index : end_index + 1])

    if not isinstance(parsed_json, dict):
        raise ValueError("The LLM response must be a JSON object.")

    return parsed_json


def build_composite_scene_layout_with_llm(
    characters: list[dict[str, Any]],
    scene: dict[str, Any],
    scene_characters: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    model: str = PROMPT_BUILDER_MODEL,
    verbose: bool = False,
    prompt_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ask a chat model for background, pose prompts, and layout boxes."""
    _load_env_file_if_available()

    resolved_api_key = api_key or os.getenv("ACADEMIC_CLOUD_CHATAI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Missing ACADEMIC_CLOUD_CHATAI_API_KEY environment variable."
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The 'openai' package is required for LLM prompt generation."
        ) from error

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=CHAT_COMPLETIONS_BASE_URL,
    )

    story_context = {
        "all_characters": characters,
        "scene_characters": scene_characters,
        "scene": scene,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You plan layered children's book image generation. Return only "
                "valid JSON. The JSON must contain background_prompt and "
                "characters. characters must be an array where each item has "
                "name, pose_prompt, placement_box, and layer_order. "
                "placement_box must contain x, y, width, and height as numbers "
                "from 0 to 1. Names are only for internal matching. "
                "background_prompt must describe only the setting. pose_prompt "
                "must describe a full-body isolated character pose that can be "
                "composited into the background. Infer concrete posture from "
                "the scene activity, including whether the body is standing, "
                "sitting, kneeling, crouching, reaching, looking, pointing, "
                "carrying, or resting. Include facing direction, gesture, "
                "expression, limb positions, and held objects. Keep "
                "pose_prompt limited to the isolated character body, clothing, "
                "held objects, expression, and action. Put all location, "
                "surface, scenery, lighting, weather, props that are not held, "
                "and environment details in background_prompt instead. Use "
                "visual traits, not proper names. Image prompts should use "
                "positive visual phrasing only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create a composite layout plan for this scene. "
                "Use compact image prompts. Make each character box large "
                "enough for a clear full-body subject. The character pose "
                "prompts will be rendered separately on a solid backdrop, so "
                "they must be meaningful isolated body poses that visibly fit "
                "the scene action after compositing. Keep scene objects and "
                "setting details in background_prompt, and keep pose_prompt as "
                "a character-only layer prompt.\n\n"
                f"{json.dumps(story_context, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

    if verbose:
        print("API call: composite scene layout prompt builder")
        for message in messages:
            print(f"role: {message['role']}")
            print(f"prompt: {message['content']}")

    if prompt_log is not None:
        prompt_log.append(
            {
                "api_call": "composite_scene_layout_prompt_builder",
                "model": model,
                "messages": [
                    {
                        "role": str(message["role"]),
                        "prompt": str(message["content"]),
                    }
                    for message in messages
                ],
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    layout_text = response.choices[0].message.content
    if not layout_text:
        raise ValueError("The LLM did not return a composite scene layout.")

    try:
        return _extract_json_object(layout_text)
    except (json.JSONDecodeError, ValueError):
        repair_messages = [
            *messages,
            {
                "role": "assistant",
                "content": layout_text,
            },
            {
                "role": "user",
                "content": (
                    "Repair the previous response into only one valid JSON "
                    "object with background_prompt and characters. Keep the "
                    "same scene intent and preserve clear scene-matching body "
                    "poses for every character."
                ),
            },
        ]
        if prompt_log is not None:
            prompt_log.append(
                {
                    "api_call": "composite_scene_layout_repair_prompt_builder",
                    "model": model,
                    "messages": [
                        {
                            "role": str(message["role"]),
                            "prompt": str(message["content"]),
                        }
                        for message in repair_messages
                    ],
                }
            )

        repair_response = client.chat.completions.create(
            model=model,
            messages=repair_messages,
        )
        repaired_layout_text = repair_response.choices[0].message.content
        if not repaired_layout_text:
            raise ValueError("The LLM did not return a repaired scene layout.")

        return _extract_json_object(repaired_layout_text)
