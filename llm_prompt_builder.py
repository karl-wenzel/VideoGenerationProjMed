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
        "Return a compact image prompt with exactly two short parts: "
        "1. fixed character traits copied from the JSON, including clothing, "
        "hair, accessories, and any other visual identity fields; "
        "2. one brief sentence for the scene action and setting. "
        "Prioritize character consistency over scenic detail."
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
