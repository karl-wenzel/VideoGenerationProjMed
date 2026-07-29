import json
import os
import re
import textwrap
from pathlib import Path

from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)


# ---------------------------------------------------------------------------
# Subtitle helpers
# ---------------------------------------------------------------------------

def natural_sort_key(filename: str) -> list:
    """Sort scene_2 before scene_10."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", filename)
    ]


def get_scene_subtitle_text(scene: dict) -> str:
    parts = [
        scene.get("first_sentence", ""),
        scene.get("narration", ""),
        scene.get("last_sentence", ""),
    ]

    return " ".join(
        part.strip()
        for part in parts
        if isinstance(part, str) and part.strip()
    )


def split_subtitle_sentences(text: str) -> list[str]:
    text = " ".join(text.split())

    parts = re.split(
        r"(?<=[.!?。！？])\s+",
        text,
    )

    parts = [
        part.strip()
        for part in parts
        if part.strip()
    ]

    final_parts: list[str] = []

    for part in parts:
        if len(part) > 120:
            smaller_parts = re.split(
                r"(?<=[,，;；:：])\s+",
                part,
            )

            final_parts.extend(
                smaller_part.strip()
                for smaller_part in smaller_parts
                if smaller_part.strip()
            )
        else:
            final_parts.append(part)

    return final_parts or ([text] if text else [])


def create_precise_subtitle_timings(
    subtitle_data: dict,
    scene_duration: float,
) -> list[tuple[str, float, float]]:
    """Read TTS boundary timing and keep every subtitle inside its scene."""

    timings: list[tuple[str, float, float]] = []

    for item in subtitle_data.get("words", []):
        text = str(item.get("text", "")).strip()
        start = max(0.0, float(item.get("start", 0.0)))
        duration = max(0.0, float(item.get("duration", 0.0)))

        if not text or start >= scene_duration:
            continue

        duration = min(
            duration,
            max(0.0, scene_duration - start),
        )

        if duration > 0:
            timings.append((text, start, duration))

    return timings


def create_fallback_subtitle_timings(
    subtitle_text: str,
    scene_duration: float,
) -> list[tuple[str, float, float]]:
    """Use evenly distributed sentence timing when no TTS timing exists."""

    subtitle_parts = split_subtitle_sentences(subtitle_text)

    if not subtitle_parts:
        return []

    part_duration = scene_duration / len(subtitle_parts)

    return [
        (
            part,
            index * part_duration,
            part_duration,
        )
        for index, part in enumerate(subtitle_parts)
    ]


def create_subtitle_clips(
    subtitle_timings: list[tuple[str, float, float]],
    image_width: int,
    image_height: int,
    font_path: Path,
) -> list[TextClip]:
    subtitle_clips: list[TextClip] = []

    for subtitle_text, start_time, duration in subtitle_timings:
        wrapped_text = "\n".join(
            textwrap.wrap(
                subtitle_text,
                width=50,
            )
        )

        subtitle_clip = (
            TextClip(
                text=wrapped_text,
                font=str(font_path),
                font_size=30,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(image_width * 0.82), 140),
            )
            .with_start(start_time)
            .with_duration(duration)
            .with_position(("center", image_height - 180))
        )

        subtitle_clips.append(subtitle_clip)

    return subtitle_clips


# ---------------------------------------------------------------------------
# Video creation
# ---------------------------------------------------------------------------

def create_story_video(
    image_folder: str,
    audio_path: str,
    scene_durations: list[float],
    output_name: str,
) -> None:
    """
    Build all visual scenes first, then attach one continuous audio file.

    This removes the audio boundaries that existed when every scene used its
    own separately encoded MP3 file.
    """

    output_folder = os.path.dirname(output_name)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(
            f"Scene-image folder not found: {image_folder}"
        )

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"Complete-story audio file not found: {audio_path}"
        )

    images = sorted(
        [
            filename
            for filename in os.listdir(image_folder)
            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ],
        key=natural_sort_key,
    )

    if not images:
        raise FileNotFoundError(
            f"No scene images found in: {image_folder}"
        )

    scene_count = min(
        len(images),
        len(scene_durations),
    )

    if scene_count == 0:
        raise ValueError(
            "No matching scene images and scene durations were provided."
        )

    if len(images) != len(scene_durations):
        print(
            "Warning: image count and scene-duration count differ. "
            f"Images={len(images)}, durations={len(scene_durations)}. "
            f"Using {scene_count} scenes."
        )

    project_root = Path(__file__).resolve().parent.parent
    font_path = project_root / "fonts" / "DejaVuSans.ttf"
    story_path = project_root / "tmp" / "story.json"

    subtitle_folder = (
        project_root
        / "tmp"
        / "tmp_audio_generation"
        / "subtitles"
    )

    if not font_path.exists():
        raise FileNotFoundError(
            f"Subtitle font not found: {font_path}"
        )

    if not story_path.exists():
        raise FileNotFoundError(
            f"Story JSON not found: {story_path}"
        )

    story = json.loads(
        story_path.read_text(encoding="utf-8")
    )

    scenes = story.get("scenes", [])
    full_audio_clip = AudioFileClip(audio_path)

    adjusted_durations = [
        float(duration)
        for duration in scene_durations[:scene_count]
    ]

    expected_duration = sum(adjusted_durations)

    # WAV and MoviePy can still differ by a tiny fraction because of rounding.
    # The final scene absorbs that fraction so the visual and audio timelines
    # finish at exactly the same time.
    duration_difference = (
        full_audio_clip.duration
        - expected_duration
    )

    adjusted_durations[-1] = max(
        0.1,
        adjusted_durations[-1] + duration_difference,
    )

    scene_clips: list[CompositeVideoClip] = []

    try:
        for scene_index in range(scene_count):
            image_path = os.path.join(
                image_folder,
                images[scene_index],
            )

            scene_duration = adjusted_durations[scene_index]

            image_clip = (
                ImageClip(image_path)
                .with_duration(scene_duration)
            )

            scene = (
                scenes[scene_index]
                if scene_index < len(scenes)
                else {}
            )

            subtitle_text = get_scene_subtitle_text(scene)

            subtitle_json_path = (
                subtitle_folder
                / f"scene_{scene_index + 1}_words.json"
            )

            subtitle_timings: list[tuple[str, float, float]] = []

            if subtitle_json_path.exists():
                subtitle_data = json.loads(
                    subtitle_json_path.read_text(encoding="utf-8")
                )

                subtitle_timings = create_precise_subtitle_timings(
                    subtitle_data,
                    scene_duration,
                )

            if not subtitle_timings:
                subtitle_timings = create_fallback_subtitle_timings(
                    subtitle_text,
                    scene_duration,
                )

            subtitle_clips = create_subtitle_clips(
                subtitle_timings=subtitle_timings,
                image_width=image_clip.w,
                image_height=image_clip.h,
                font_path=font_path,
            )

            scene_clip = CompositeVideoClip(
                [image_clip, *subtitle_clips]
            ).with_duration(scene_duration)

            scene_clips.append(scene_clip)

        print(
            f"Processing {len(scene_clips)} scenes "
            "with one continuous audio track..."
        )

        visual_timeline = concatenate_videoclips(
            scene_clips,
            method="compose",
        )

        final_video = visual_timeline.with_audio(
            full_audio_clip
        )

        try:
            final_video.write_videofile(
                output_name,
                fps=24,
                codec="libx264",
                audio_codec="aac",
            )
        finally:
            final_video.close()
            visual_timeline.close()

        print(f"Success! Video created: {output_name}")

    finally:
        full_audio_clip.close()

        for clip in scene_clips:
            clip.close()


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    test_image_folder = project_root / "tmp" / "scene_images"

    test_audio_path = (
        project_root
        / "tmp"
        / "tmp_audio_generation"
        / "final"
        / "full_story_audio.wav"
    )

    timeline_path = (
        project_root
        / "tmp"
        / "tmp_audio_generation"
        / "final"
        / "audio_timeline.json"
    )

    test_output_name = (
        project_root
        / "tmp"
        / "final_story_standalone.mp4"
    )

    if (
        test_image_folder.exists()
        and test_audio_path.exists()
        and timeline_path.exists()
    ):
        timeline = json.loads(
            timeline_path.read_text(encoding="utf-8")
        )

        create_story_video(
            image_folder=str(test_image_folder),
            audio_path=str(test_audio_path),
            scene_durations=timeline["scene_durations"],
            output_name=str(test_output_name),
        )
    else:
        print(
            "Required test files were not found.\n"
            f"Images: {test_image_folder}\n"
            f"Audio: {test_audio_path}\n"
            f"Timeline: {timeline_path}"
        )
