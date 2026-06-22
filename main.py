"""
main.py — Story-to-video pipeline orchestrator.

Pipeline:
  1. prompt  →  story JSON          (story_generation/prompt2storyjson.py)
  2. story JSON  →  scene images    (image_generation/image_scene_generator.py)
  3. story JSON  →  scene audio     (audio_generation/audio_generation.py)
  4. images + audio  →  final video (video_generation/video_generation.py)
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make every sub-package importable when running from root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Per-module imports
# ---------------------------------------------------------------------------
from story_generation.prompt2storyjson import generate_story          # step 1

from image_generation.image_scene_generator import (                  # step 2
    generate_scene_images,
    clear_scene_generation_outputs,
)

from audio_generation.audio_generation import generate_scene_audio    # step 3

from video_generation.video_generation import create_story_video      # step 4
from pipeline_timing import reset_api_timeline, save_api_timeline

# ---------------------------------------------------------------------------
# Shared paths (mirrors the defaults used in each sub-module)
# ---------------------------------------------------------------------------
TMP_DIR          = ROOT / "tmp"
STORY_JSON_PATH  = TMP_DIR / "story.json"

# image_generation defaults (re-used here so we can pass them to step 4)
from image_generation.image_generation_client import (
    DEFAULT_OUTPUT_DIR    as IMAGE_OUTPUT_DIR,
    DEFAULT_SCENE_OUTPUT_DIR as SCENE_OUTPUT_DIR,
)

AUDIO_OUTPUT_DIR = TMP_DIR / "tmp_audio_generation"    / "final"   # audio_generation writes here already
VIDEO_OUTPUT     = ROOT / "tmp" / "final_story.mp4"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def generate_story_json(user_prompt: str) -> dict:
    """Convert a free-text prompt into a structured story JSON."""
    print("\n=== Step 1: Generating story JSON ===")
    story = generate_story(user_prompt)
    print(f"  Characters : {len(story['characters'])}")
    print(f"  Scenes     : {len(story['scenes'])}")

    tmp_copy = TMP_DIR / "story.json"
    tmp_copy.parent.mkdir(parents=True, exist_ok=True)
    with tmp_copy.open("w", encoding="utf-8") as fh:
        json.dump(story, fh, indent=2, ensure_ascii=False)

    return story


def generate_images() -> list[dict]:
    """Generate one scene image per scene from the saved story JSON."""
    print("\n=== Step 2: Generating scene images ===")
    clear_scene_generation_outputs()
    images = generate_scene_images(
        json_path=STORY_JSON_PATH,
        output_dir=IMAGE_OUTPUT_DIR,
        scene_output_dir=SCENE_OUTPUT_DIR,
    )
    print(f"  Images generated: {len(images)}")
    return images


async def generate_audio(story: dict) -> None:
    """Generate narration + background-music audio for every scene."""
    print("\n=== Step 3: Generating scene audio ===")
    for index, scene in enumerate(story["scenes"], start=1):
        await generate_scene_audio(scene, index)
    print(f"  Audio files generated: {len(story['scenes'])}")


def generate_video() -> None:
    """Stitch scene images and audio tracks into the final MP4."""
    print("\n=== Step 4: Assembling final video ===")
    create_story_video(
        image_folder=str(SCENE_OUTPUT_DIR),
        audio_folder=str(AUDIO_OUTPUT_DIR),
        output_name=str(VIDEO_OUTPUT),
    )
    print(f"  Video saved to: {VIDEO_OUTPUT}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_pipeline(user_prompt: str) -> None:
    reset_api_timeline()
    try:
        story  = generate_story_json(user_prompt)
        generate_images()
        await generate_audio(story)
        generate_video()
    finally:
        timeline_json_path, timeline_svg_path = save_api_timeline(
            TMP_DIR / "api_timeline.json",
            TMP_DIR / "api_timeline.svg",
        )
        print(f"\nAPI timeline JSON saved to: {timeline_json_path}")
        print(f"API timeline diagram saved to: {timeline_svg_path}")

    print("\n✅  Pipeline complete!")


DEFAULT_PROMPT = (
    "A bedtime story about a shy fox named Mika with brown hair "
    "and a brave firefly who help the moon find its glow again."
)


def main(user_prompt: str = DEFAULT_PROMPT) -> None:
    asyncio.run(run_pipeline(user_prompt))


if __name__ == "__main__":
    main("two girls exploring a hidden village and meeting animals which can speak?")
