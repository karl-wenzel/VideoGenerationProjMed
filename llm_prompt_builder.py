import json
import os
from typing import Any


CHAT_COMPLETIONS_BASE_URL = "https://saia.gwdg.de/v1"
PROMPT_BUILDER_MODEL = "openai-gpt-oss-120b"


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
) -> str:
    """Ask a chat model to write a polished image-generation prompt."""
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
    messages = [
        {
            "role": "system",
            "content": (
                "You write concise, vivid prompts for an image generation "
                "model. Return only the final image prompt, with no "
                "markdown, labels, or extra explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create one children's storybook illustration prompt from "
                "this JSON. Preserve the character identities and include "
                "the visible action, setting, mood, and style.\n\n"
                f"{json.dumps(story_context, ensure_ascii=False, indent=2)}"
            ),
        },
    ]

    if verbose:
        print("API call: LLM prompt builder")
        for message in messages:
            print(f"role: {message['role']}")
            print(f"prompt: {message['content']}")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    prompt = response.choices[0].message.content
    if not prompt:
        raise ValueError("The LLM did not return an image prompt.")

    return prompt.strip()
