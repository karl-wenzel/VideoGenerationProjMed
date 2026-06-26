import asyncio
import json
import edge_tts
import re
from pydub import AudioSegment
import os

from pipeline_timing import track_api_call_async

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BGM_DIR = os.path.dirname(os.path.abspath(__file__))  # ./audio_generation/

NARRATOR_VOICE = "en-GB-RyanNeural"

MOOD_STYLE = {
    "happy": {"rate": "-2%", "pitch": "+5Hz"},
    "calm": {"rate": "-3%", "pitch": "+0Hz"},
    "warm": {"rate": "-4%", "pitch": "+3Hz"},
    "worried": {"rate": "-3%", "pitch": "-3Hz"},
    "mysterious": {"rate": "-4%", "pitch": "-5Hz"},
    "sad": {"rate": "-4%", "pitch": "-5Hz"},
    "scary": {"rate": "-3%", "pitch": "-10Hz"},
    "adventurous": {"rate": "-2%", "pitch": "+2Hz"}
}

def detect_mood(scene):
    text = (
        scene.get("summary", "") + " " +
        scene.get("first_sentence", "") + " " +
        scene.get("last_sentence", "")
    ).lower()

    mood_keywords = {
        "happy": [
            "happy", "joyful", "cheerful", "delighted", "excited", "thrilled",
            "pleased", "content", "satisfied", "optimistic",
            "bright", "sunny", "lively", "playful", "energetic", "vibrant", "uplifting",
            "smiling", "laughing", "celebrating", "dancing", "enjoying",
            "euphoric", "blissful", "radiant", "carefree", "ecstatic",
            "celebrate", "picnic", "magical", "proud"
        ],

        "worried": [
            "worried", "anxious", "nervous", "stressed", "uneasy", "concerned",
            "afraid", "fearful", "tense",
            "dark", "uncertain", "uncomfortable", "restless", "troubled",
            "shaking", "hesitating", "panicking", "overthinking",
            "insecure", "paranoid", "overwhelmed", "distressed", "apprehensive",
            "storm", "lost", "scared", "danger", "fear"
        ],

        "calm": [
            "calm", "peaceful", "relaxed", "quiet", "still", "gentle", "serene",
            "soft", "smooth", "tranquil", "balanced", "stable", "meditative",
            "ocean breeze", "silence", "slow", "flowing", "harmony",
            "soothing", "composed", "restful", "mindful", "untroubled",
            "fireflies", "evening", "stars"
        ],

        "warm": [
            "warm", "cozy", "comforting", "affectionate", "friendly", "loving", "tender",
            "soft light", "fireplace", "homey", "intimate", "golden",
            "caring", "heartfelt", "welcoming", "supportive",
            "blanket", "candlelight", "hug", "family", "sunset",
            "nostalgic", "wholesome", "sincere", "gentle-hearted",
            "help", "rebuild", "smile"
        ],

        "mysterious": [
            "mysterious", "strange", "unknown", "hidden", "secretive", "cryptic", "enigmatic",
            "shadowy", "foggy", "eerie", "haunting", "silent",
            "moonlight", "whisper", "forest", "abandoned", "mist",
            "surreal", "uncanny", "obscure", "puzzling", "supernatural",
            "shadows", "secret"
        ],

        "adventurous": [
            "adventurous", "daring", "bold", "fearless", "curious", "wild", "brave",
            "exciting", "dynamic", "unpredictable", "thrilling", "energetic",
            "journey", "exploration", "mountain", "ocean", "travel", "discovery",
            "climbing", "wandering", "discovering", "risking", "chasing",
            "pioneering", "ambitious", "rebellious", "untamed",
            "courage"
        ]
    }

    scores = {}

    for mood, keywords in mood_keywords.items():
        score = 0
        for word in keywords:
            if word in text:
                score += 1
        scores[mood] = score

    best_mood = max(scores, key=scores.get)

    if scores[best_mood] == 0:
        return "calm"

    return best_mood


BGM_MAP = {
    "happy": "bgm/happy.wav",
    "sad": "bgm/sad.wav",
    "mysterious": "bgm/mysterious.wav",
    "scary": "bgm/scary.wav",
    "calm": "bgm/calm.wav",
    "adventurous": "bgm/adventurous.wav",
    "worried": "bgm/worried.wav",
    "warm": "bgm/warm.wav"
}


async def generate_scene_audio(scene, scene_id):
    tmp_voice_dir = os.path.join(BASE_DIR, "tmp", "tmp_audio_generation", "voice")
    os.makedirs(tmp_voice_dir, exist_ok=True)

    tmp_final_dir = os.path.join(BASE_DIR, "tmp", "tmp_audio_generation", "final")
    os.makedirs(tmp_final_dir, exist_ok=True)

    #combine scene text
    text = scene["first_sentence"] + " " + scene["summary"] + " " + scene["last_sentence"]

    #get mood from JSON or auto_detect
    mood = scene.get("mood", detect_mood(scene))
    style = MOOD_STYLE.get(mood, MOOD_STYLE["calm"])

    #output file names
    voice_output = os.path.join(tmp_voice_dir, f"scene_{scene_id}_voice.mp3")
    final_output = os.path.join(tmp_final_dir, f"scene_{scene_id}_final.mp3")

    #create TTS narrator
    communicate = edge_tts.Communicate(
        text=text,
        voice=NARRATOR_VOICE,
        rate=style["rate"],
        pitch=style["pitch"]
    )

    #generate narration audio
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
        subtitle_dir = os.path.join(BASE_DIR, "tmp", "tmp_audio_generation", "subtitles")
        os.makedirs(subtitle_dir, exist_ok=True)

        subtitle_output = os.path.join(subtitle_dir, f"scene_{scene_id}_words.json")

        word_boundaries = []
        audio_bytes = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes.extend(chunk["data"])

            elif chunk["type"] == "SentenceBoundary":
                word_boundaries.append({
                    "text": chunk["text"],
                    "start": chunk["offset"] / 10_000_000,
                    "duration": chunk["duration"] / 10_000_000,
                })

        with open(voice_output, "wb") as f:
            f.write(audio_bytes)

        with open(subtitle_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene_id": scene_id,
                    "full_text": text,
                    "words": word_boundaries,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    
    #add background music
    add_background_music(
        voice_output,
        mood,
        final_output
    )

    print(f"Generated {final_output} | mood={mood} | rate={style['rate']} | pitch={style['pitch']}")



async def generate_all_scene_audio(story: dict) -> None:

    semaphore = asyncio.Semaphore(MAX_AUDIO_WORKERS)

    async def limited_generate(scene, scene_id):

        async with semaphore:

            await generate_scene_audio(scene, scene_id)

    tasks = [

        limited_generate(scene, index)

        for index, scene in enumerate(story["scenes"], start=1)

    ]

    await asyncio.gather(*tasks)    


SCENE_BUFFER_MS = 400
BGM_FADE_IN_MS = 500
BGM_FADE_OUT_MS = 1000
MAX_AUDIO_WORKERS = 4

def add_background_music(voice_file, mood, output_file):

    # load narration audio
    voice = AudioSegment.from_file(voice_file)

    # small buffer
    silence = AudioSegment.silent(duration=SCENE_BUFFER_MS)

    voice = silence + voice + silence

    # select BGM
    bgm_path = os.path.join(BGM_DIR, BGM_MAP.get(mood, "bgm/calm.wav"))

    # load BGM
    bgm = AudioSegment.from_file(bgm_path)

    while len(bgm) < len(voice):
        bgm += bgm

    bgm = bgm[:len(voice)]
    bgm = bgm - 20

    # fade_in,fade_out
    bgm = bgm.fade_in(BGM_FADE_IN_MS)
    bgm = bgm.fade_out(BGM_FADE_OUT_MS)

    # mix the narration with bgm
    final_audio = bgm.overlay(voice)

    
    final_audio = final_audio.fade_out(500)

    #save the file in tmp
    final_audio.export(output_file, format="mp3")

    print(f"Final audio with BGM saved: {output_file}")






async def generate_all_scene_audio(story: dict) -> None:
    semaphore = asyncio.Semaphore(MAX_AUDIO_WORKERS)

    async def limited_generate(scene, scene_id):
        async with semaphore:
            await generate_scene_audio(scene, scene_id)

    tasks = [
        limited_generate(scene, index)
        for index, scene in enumerate(story["scenes"], start=1)
    ]

    await asyncio.gather(*tasks)


async def main():
    with open(os.path.join(BASE_DIR, "tmp", "story.json"), "r", encoding="utf-8") as file:
        story = json.load(file)

    await generate_all_scene_audio(story)







if __name__ == "__main__":
    asyncio.run(main())

