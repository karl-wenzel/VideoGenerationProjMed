"""
main.py — Story-to-video pipeline orchestrator.

Pipeline:
  1. prompt  →  story JSON          (story_generation/prompt2storyjson.py)
  2. story JSON  →  scene images    (image_generation/image_scene_generator.py)
  3. story JSON  →  seamless audio  (audio_generation/audio_generation.py)
  4. images + audio  →  final video (video_generation/video_generation.py)
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

# IMPORTANT:
# Load the .env file before importing project modules.
# Some modules may read API keys immediately when imported.
ENV_PATH = ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Per-module imports
# ---------------------------------------------------------------------------

from story_generation.prompt2storyjson import generate_story

from image_generation.image_scene_generator import (
    generate_scene_images,
    clear_scene_generation_outputs,
)

from audio_generation.audio_generation import generate_story_audio

from video_generation.video_generation import create_story_video

from pipeline_timing import (
    reset_api_timeline,
    save_api_timeline,
)

from image_generation.image_generation_client import (
    DEFAULT_OUTPUT_DIR as IMAGE_OUTPUT_DIR,
    DEFAULT_SCENE_OUTPUT_DIR as SCENE_OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

TMP_DIR = ROOT / "tmp"
STORY_JSON_PATH = TMP_DIR / "story.json"
VIDEO_OUTPUT = TMP_DIR / "final_story.mp4"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def generate_story_json(user_prompt: str) -> dict:
    """Convert a free-text prompt into a structured story JSON."""

    print("\n=== Step 1: Generating story JSON ===")

    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f".env file not found: {ENV_PATH}"
        )

    story = generate_story(user_prompt)

    print(f"  Characters : {len(story['characters'])}")
    print(f"  Scenes     : {len(story['scenes'])}")

    STORY_JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with STORY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            story,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"  Story JSON saved to: {STORY_JSON_PATH}")

    return story


def generate_images() -> list[dict]:
    """Generate one scene image per scene."""

    print("\n=== Step 2: Generating scene images ===")

    clear_scene_generation_outputs()

    images = generate_scene_images(
        json_path=STORY_JSON_PATH,
        output_dir=IMAGE_OUTPUT_DIR,
        scene_output_dir=SCENE_OUTPUT_DIR,
    )

    print(f"  Images generated: {len(images)}")

    return images


async def generate_audio(story: dict) -> dict:
    """Generate one seamless audio track for the complete story."""

    print("\n=== Step 3: Generating scene audio ===")

    audio_result = await generate_story_audio(story)

    print(
        "  Scene narrations generated: "
        f"{len(audio_result['scene_durations'])}"
    )

    print(
        "  Full audio duration: "
        f"{audio_result['total_duration_seconds']:.2f}s"
    )

    print(
        "  Full audio saved to: "
        f"{audio_result['audio_path']}"
    )

    return audio_result


def generate_video(audio_result: dict) -> None:
    """Build the final video using one continuous audio track."""

    print("\n=== Step 4: Assembling final video ===")

    create_story_video(
        image_folder=str(SCENE_OUTPUT_DIR),
        audio_path=audio_result["audio_path"],
        scene_durations=audio_result["scene_durations"],
        output_name=str(VIDEO_OUTPUT),
    )

    print(f"  Video saved to: {VIDEO_OUTPUT}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(user_prompt: str) -> None:
    reset_api_timeline()

    try:
        story = generate_story_json(user_prompt)
        generate_images()

        audio_result = await generate_audio(story)
        generate_video(audio_result)

    except Exception as exc:
        print("\n❌ Pipeline failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")

        # Re-raise so gui.py receives the failure instead of appearing stuck.
        raise

    finally:
        TMP_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        timeline_json_path, timeline_svg_path = save_api_timeline(
            TMP_DIR / "api_timeline.json",
            TMP_DIR / "api_timeline.svg",
        )

        print(
            "\nAPI timeline JSON saved to: "
            f"{timeline_json_path}"
        )

        print(
            "API timeline diagram saved to: "
            f"{timeline_svg_path}"
        )

    print("\n✅ Pipeline complete!")


DEFAULT_PROMPT = (
    "A bedtime story about a shy fox named Mika with brown hair "
    "and a brave firefly who help the moon find its glow again."
)


def main(user_prompt: str = DEFAULT_PROMPT) -> None:
    """Synchronous entry point used by gui.py."""

    asyncio.run(
        run_pipeline(user_prompt)
    )


if __name__ == "__main__":
    main(
        "two girls exploring a hidden village "
        "and meeting animals which can speak?"
    )
