import asyncio
import json
import os
import shutil
from pathlib import Path

import edge_tts
from pydub import AudioSegment

from pipeline_timing import track_api_call_async


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_MODULE_DIR = Path(__file__).resolve().parent

AUDIO_ROOT_DIR = BASE_DIR / "tmp" / "tmp_audio_generation"
VOICE_OUTPUT_DIR = AUDIO_ROOT_DIR / "voice"
SUBTITLE_OUTPUT_DIR = AUDIO_ROOT_DIR / "subtitles"
FINAL_OUTPUT_DIR = AUDIO_ROOT_DIR / "final"

FULL_STORY_AUDIO_PATH = FINAL_OUTPUT_DIR / "full_story_audio.wav"
AUDIO_TIMELINE_PATH = FINAL_OUTPUT_DIR / "audio_timeline.json"

NARRATOR_VOICE = "en-GB-RyanNeural"

SCENE_BUFFER_MS = 400
BGM_FADE_IN_MS = 700
BGM_FADE_OUT_MS = 1000
BGM_VOLUME_REDUCTION_DB = 20
MAX_AUDIO_WORKERS = 4


MOOD_STYLE = {
    "happy": {"rate": "-2%", "pitch": "+5Hz"},
    "calm": {"rate": "-3%", "pitch": "+0Hz"},
    "warm": {"rate": "-4%", "pitch": "+3Hz"},
    "worried": {"rate": "-3%", "pitch": "-3Hz"},
    "mysterious": {"rate": "-4%", "pitch": "-5Hz"},
    "sad": {"rate": "-4%", "pitch": "-5Hz"},
    "scary": {"rate": "-3%", "pitch": "-10Hz"},
    "adventurous": {"rate": "-2%", "pitch": "+2Hz"},
}


BGM_MAP = {
    "happy": "bgm/happy.wav",
    "sad": "bgm/sad.wav",
    "mysterious": "bgm/mysterious.wav",
    "scary": "bgm/scary.wav",
    "calm": "bgm/calm.wav",
    "adventurous": "bgm/adventurous.wav",
    "worried": "bgm/worried.wav",
    "warm": "bgm/warm.wav",
}

# Tail duration to remove before looping each BGM, in milliseconds.
BGM_END_TRIM_MS = {
    "calm": 2000,
    "happy": 2000,
    "warm": 2000,
    "sad": 5000,
    "worried": 8000,
    "scary": 7000,
    "mysterious": 3000,
    "adventurous": 0,
}

BGM_LOOP_CROSSFADE_MS = 300


# ---------------------------------------------------------------------------
# Mood detection
# ---------------------------------------------------------------------------

def detect_mood(scene: dict) -> str:
    text = (
        scene.get("summary", "")
        + " "
        + scene.get("first_sentence", "")
        + " "
        + scene.get("last_sentence", "")
    ).lower()

    mood_keywords = {
        "happy": [
            "happy", "joyful", "cheerful", "delighted", "excited", "thrilled",
            "pleased", "content", "satisfied", "optimistic",
            "bright", "sunny", "lively", "playful", "energetic", "vibrant",
            "uplifting", "smiling", "laughing", "celebrating", "dancing",
            "enjoying", "euphoric", "blissful", "radiant", "carefree",
            "ecstatic", "celebrate", "picnic", "magical", "proud",
        ],
        "worried": [
            "worried", "anxious", "nervous", "stressed", "uneasy",
            "concerned", "afraid", "fearful", "tense", "dark", "uncertain",
            "uncomfortable", "restless", "troubled", "shaking", "hesitating",
            "panicking", "overthinking", "insecure", "paranoid",
            "overwhelmed", "distressed", "apprehensive", "storm", "lost",
            "scared", "danger", "fear",
        ],
        "calm": [
            "calm", "peaceful", "relaxed", "quiet", "still", "gentle",
            "serene", "soft", "smooth", "tranquil", "balanced", "stable",
            "meditative", "ocean breeze", "silence", "slow", "flowing",
            "harmony", "soothing", "composed", "restful", "mindful",
            "untroubled", "fireflies", "evening", "stars",
        ],
        "warm": [
            "warm", "cozy", "comforting", "affectionate", "friendly",
            "loving", "tender", "soft light", "fireplace", "homey",
            "intimate", "golden", "caring", "heartfelt", "welcoming",
            "supportive", "blanket", "candlelight", "hug", "family",
            "sunset", "nostalgic", "wholesome", "sincere",
            "gentle-hearted", "help", "rebuild", "smile",
        ],
        "mysterious": [
            "mysterious", "strange", "unknown", "hidden", "secretive",
            "cryptic", "enigmatic", "shadowy", "foggy", "eerie", "haunting",
            "silent", "moonlight", "whisper", "forest", "abandoned", "mist",
            "surreal", "uncanny", "obscure", "puzzling", "supernatural",
            "shadows", "secret",
        ],
        "adventurous": [
            "adventurous", "daring", "bold", "fearless", "curious", "wild",
            "brave", "exciting", "dynamic", "unpredictable", "thrilling",
            "energetic", "journey", "exploration", "mountain", "ocean",
            "travel", "discovery", "climbing", "wandering", "discovering",
            "risking", "chasing", "pioneering", "ambitious", "rebellious",
            "untamed", "courage",
        ],
        "sad": [
            "sad", "sorrow", "sorrowful", "unhappy", "lonely", "crying",
            "tears", "heartbroken", "grief", "gloomy", "melancholy",
            "miserable", "loss", "lost friend", "missed", "missing",
        ],
        "scary": [
            "scary", "terrifying", "frightening", "horror", "monster",
            "ghost", "scream", "nightmare", "creepy", "spooky",
        ],
    }

    scores = {
        mood: sum(1 for keyword in keywords if keyword in text)
        for mood, keywords in mood_keywords.items()
    }

    best_mood = max(scores, key=scores.get)

    if scores[best_mood] == 0:
        return "calm"

    return best_mood


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clear_audio_outputs() -> None:
    """Remove outputs from the previous run and recreate the audio folders."""

    if AUDIO_ROOT_DIR.exists():
        shutil.rmtree(AUDIO_ROOT_DIR)

    VOICE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBTITLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_scene_text(scene: dict) -> str:
    """Combine the three story fields into the spoken narration."""

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


def repeat_audio_to_length(
    audio: AudioSegment,
    target_length_ms: int,
    mood: str,
) -> AudioSegment:
    """
    Trim the configured tail and repeat the BGM with a short crossfade.

    If the BGM is already long enough for the mood group, it is only shortened
    to the requested duration and no loop is created.
    """

    if target_length_ms <= 0:
        return AudioSegment.empty()

    if len(audio) <= 0:
        raise ValueError("The selected background-music file is empty.")

    trim_end_ms = BGM_END_TRIM_MS.get(mood, 0)

    # Remove the long tail before creating a loop.
    if trim_end_ms > 0:
        if trim_end_ms >= len(audio):
            raise ValueError(
                f"The configured tail trim for mood '{mood}' "
                f"({trim_end_ms} ms) is longer than the BGM itself "
                f"({len(audio)} ms)."
            )

        loop_source = audio[:-trim_end_ms]
    else:
        loop_source = audio

    if len(loop_source) <= 0:
        raise ValueError(
            f"No usable BGM remains after trimming mood '{mood}'."
        )

    # No loop is needed when the usable BGM is already long enough.
    if len(loop_source) >= target_length_ms:
        return loop_source[:target_length_ms]

    crossfade_ms = min(
        BGM_LOOP_CROSSFADE_MS,
        len(loop_source) // 4,
    )

    repeated = loop_source

    while len(repeated) < target_length_ms:
        repeated = repeated.append(
            loop_source,
            crossfade=crossfade_ms,
        )

    return repeated[:target_length_ms]


# ---------------------------------------------------------------------------
# Narration generation
# ---------------------------------------------------------------------------

async def generate_scene_voice(scene: dict, scene_id: int) -> dict:
    """
    Generate narration for one scene.

    No background music is added here. Background music is built later as one
    continuous timeline for the complete story.
    """

    text = get_scene_text(scene)

    if not text:
        raise ValueError(f"Scene {scene_id} contains no narration text.")

    mood = scene.get("mood") or detect_mood(scene)
    style = MOOD_STYLE.get(mood, MOOD_STYLE["calm"])

    voice_output = VOICE_OUTPUT_DIR / f"scene_{scene_id}_voice.mp3"
    subtitle_output = SUBTITLE_OUTPUT_DIR / f"scene_{scene_id}_words.json"

    communicate = edge_tts.Communicate(
        text=text,
        voice=NARRATOR_VOICE,
        rate=style["rate"],
        pitch=style["pitch"],
        boundary="SentenceBoundary",
    )

    audio_bytes = bytearray()
    subtitle_boundaries: list[dict] = []

    async with track_api_call_async(
        "audio",
        "edge_tts_scene_narration",
        {
            "scene_id": scene_id,
            "voice": NARRATOR_VOICE,
            "mood": mood,
            "rate": style["rate"],
            "pitch": style["pitch"],
        },
    ):
        async for chunk in communicate.stream():
            chunk_type = chunk.get("type")

            if chunk_type == "audio":
                audio_bytes.extend(chunk["data"])

            elif chunk_type in {"SentenceBoundary", "WordBoundary"}:
                subtitle_boundaries.append(
                    {
                        "text": chunk.get("text", ""),
                        "start": (
                            chunk.get("offset", 0) / 10_000_000
                            + SCENE_BUFFER_MS / 1000
                        ),
                        "duration": chunk.get("duration", 0) / 10_000_000,
                    }
                )

    if not audio_bytes:
        raise RuntimeError(
            f"Edge TTS returned no audio data for scene {scene_id}."
        )

    voice_output.write_bytes(audio_bytes)

    raw_voice = AudioSegment.from_file(voice_output)

    buffered_duration_ms = (
        SCENE_BUFFER_MS
        + len(raw_voice)
        + SCENE_BUFFER_MS
    )

    subtitle_payload = {
        "scene_id": scene_id,
        "full_text": text,
        "mood": mood,
        "scene_duration_ms": buffered_duration_ms,
        "words": subtitle_boundaries,
    }

    subtitle_output.write_text(
        json.dumps(subtitle_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"Generated narration for scene {scene_id} | "
        f"mood={mood} | "
        f"duration={buffered_duration_ms / 1000:.2f}s"
    )

    return {
        "scene_id": scene_id,
        "mood": mood,
        "voice_path": str(voice_output),
        "subtitle_path": str(subtitle_output),
        "duration_ms": buffered_duration_ms,
    }


async def generate_all_scene_voices(story: dict) -> list[dict]:
    """Generate all TTS narrations concurrently, with a worker limit."""

    scenes = story.get("scenes", [])

    if not scenes:
        raise ValueError("The story contains no scenes.")

    semaphore = asyncio.Semaphore(MAX_AUDIO_WORKERS)

    async def limited_generate(scene: dict, scene_id: int) -> dict:
        async with semaphore:
            return await generate_scene_voice(scene, scene_id)

    tasks = [
        limited_generate(scene, scene_id)
        for scene_id, scene in enumerate(scenes, start=1)
    ]

    results = await asyncio.gather(*tasks)

    return sorted(results, key=lambda item: item["scene_id"])


# ---------------------------------------------------------------------------
# Complete-story audio timeline
# ---------------------------------------------------------------------------

def build_narration_timeline(
    scene_results: list[dict],
) -> AudioSegment:
    """Concatenate all buffered narrations into one complete timeline."""

    narration_timeline = AudioSegment.empty()

    for result in scene_results:
        voice = AudioSegment.from_file(result["voice_path"])

        buffered_voice = (
            AudioSegment.silent(duration=SCENE_BUFFER_MS)
            + voice
            + AudioSegment.silent(duration=SCENE_BUFFER_MS)
        )

        narration_timeline += buffered_voice

    return narration_timeline


def find_mood_groups(scene_results: list[dict]) -> list[dict]:
    """
    Combine consecutive scenes with the same mood into one music group.

    Example:
        calm, calm, mysterious, mysterious, warm

    becomes:
        calm: scenes 1-2
        mysterious: scenes 3-4
        warm: scene 5
    """

    groups: list[dict] = []
    current_start_ms = 0

    for result in scene_results:
        mood = result["mood"]
        duration_ms = int(result["duration_ms"])

        if groups and groups[-1]["mood"] == mood:
            groups[-1]["duration_ms"] += duration_ms
            groups[-1]["scene_ids"].append(result["scene_id"])
        else:
            groups.append(
                {
                    "mood": mood,
                    "start_ms": current_start_ms,
                    "duration_ms": duration_ms,
                    "scene_ids": [result["scene_id"]],
                }
            )

        current_start_ms += duration_ms

    return groups


def build_background_music_timeline(
    scene_results: list[dict],
    total_duration_ms: int,
) -> AudioSegment:
    """
    Build one continuous background-music timeline.

    Consecutive scenes with the same mood share one uninterrupted BGM segment.
    The BGM starts over only when the detected mood changes.
    """

    background_timeline = AudioSegment.silent(
        duration=total_duration_ms
    )

    mood_groups = find_mood_groups(scene_results)

    for group in mood_groups:
        mood = group["mood"]
        start_ms = int(group["start_ms"])
        duration_ms = int(group["duration_ms"])

        relative_bgm_path = BGM_MAP.get(mood, BGM_MAP["calm"])
        bgm_path = AUDIO_MODULE_DIR / relative_bgm_path

        if not bgm_path.exists():
            raise FileNotFoundError(
                f"Background music not found: {bgm_path}"
            )

        source_bgm = AudioSegment.from_file(bgm_path)

        group_bgm = repeat_audio_to_length(
            source_bgm,
            duration_ms,
            mood,
        )

        group_bgm = group_bgm - BGM_VOLUME_REDUCTION_DB

        fade_in_duration = min(BGM_FADE_IN_MS, duration_ms)
        fade_out_duration = min(BGM_FADE_OUT_MS, duration_ms)

        group_bgm = group_bgm.fade_in(fade_in_duration)
        group_bgm = group_bgm.fade_out(fade_out_duration)

        background_timeline = background_timeline.overlay(
            group_bgm,
            position=start_ms,
        )

        print(
            f"BGM group | mood={mood} | "
            f"scenes={group['scene_ids']} | "
            f"start={start_ms / 1000:.2f}s | "
            f"duration={duration_ms / 1000:.2f}s"
        )

    return background_timeline


def export_full_story_audio(
    scene_results: list[dict],
) -> dict:
    """
    Mix narration and BGM once, then export one lossless complete-story WAV.
    """

    narration_timeline = build_narration_timeline(scene_results)
    total_duration_ms = len(narration_timeline)

    background_timeline = build_background_music_timeline(
        scene_results,
        total_duration_ms,
    )

    full_story_audio = background_timeline.overlay(
        narration_timeline
    )

    if total_duration_ms > 0:
        full_story_audio = full_story_audio.fade_out(
            min(500, total_duration_ms)
        )

    # WAV is used intentionally. It avoids MP3 encoder delay at file boundaries.
    full_story_audio.export(
        FULL_STORY_AUDIO_PATH,
        format="wav",
    )

    scene_durations = [
        result["duration_ms"] / 1000
        for result in scene_results
    ]

    manifest = {
        "audio_path": str(FULL_STORY_AUDIO_PATH),
        "total_duration_seconds": total_duration_ms / 1000,
        "scene_durations": scene_durations,
        "scenes": [
            {
                "scene_id": result["scene_id"],
                "mood": result["mood"],
                "duration_seconds": result["duration_ms"] / 1000,
                "voice_path": result["voice_path"],
                "subtitle_path": result["subtitle_path"],
            }
            for result in scene_results
        ],
    }

    AUDIO_TIMELINE_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Full story audio saved: {FULL_STORY_AUDIO_PATH}")
    print(f"Audio timeline saved: {AUDIO_TIMELINE_PATH}")

    return manifest


async def generate_story_audio(story: dict) -> dict:
    """Main audio entry point used by main.py."""

    clear_audio_outputs()
    scene_results = await generate_all_scene_voices(story)
    return export_full_story_audio(scene_results)


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

async def main() -> None:
    story_path = BASE_DIR / "tmp" / "story.json"

    if not story_path.exists():
        raise FileNotFoundError(f"Story JSON not found: {story_path}")

    story = json.loads(story_path.read_text(encoding="utf-8"))
    await generate_story_audio(story)


if __name__ == "__main__":
    asyncio.run(main())


