import json
import os
from typing import Any

from pipeline_timing import track_api_call

from .prompt_config import (
    LLM_COMPOSITE_LAYOUT_REPAIR_USER_MESSAGE,
    LLM_COMPOSITE_LAYOUT_SYSTEM_MESSAGE,
    LLM_COMPOSITE_LAYOUT_USER_PREFIX,
)


CHAT_COMPLETIONS_BASE_URL = "https://saia.gwdg.de/v1"
PROMPT_BUILDER_MODEL = "openai-gpt-oss-120b"


def _load_env_file_if_available() -> None:
    """Load a local .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


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
            "content": LLM_COMPOSITE_LAYOUT_SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": (
                f"{LLM_COMPOSITE_LAYOUT_USER_PREFIX}\n\n"
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

    with track_api_call(
        "text",
        "composite_scene_layout_prompt_builder",
        {"model": model},
    ):
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
                "content": LLM_COMPOSITE_LAYOUT_REPAIR_USER_MESSAGE,
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

        with track_api_call(
            "text",
            "composite_scene_layout_repair_prompt_builder",
            {"model": model},
        ):
            repair_response = client.chat.completions.create(
                model=model,
                messages=repair_messages,
            )
        repaired_layout_text = repair_response.choices[0].message.content
        if not repaired_layout_text:
            raise ValueError("The LLM did not return a repaired scene layout.")

        return _extract_json_object(repaired_layout_text)
