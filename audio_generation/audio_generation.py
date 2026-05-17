import asyncio
import json
import edge_tts
from pydub import AudioSegment

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
    text = (scene["summary"] + " " + scene["first_sentence"] + " " + scene["last_sentence"]).lower()

    if "storm" in text or "lost" in text:
        return "worried"
    if "celebrate" in text or "picnic" in text or "magical" in text or "proud" in text:
        return "happy"
    if "fireflies" in text or "evening" in text or "stars" in text or "quiet" in text:
        return "calm"
    if "help" in text or "rebuild" in text or "warm" in text:
        return "warm"
    if "shadows" in text:
        return "mysterious"
    if "brave" in text:
        return "adventure"
    return "calm"

BGM_MAP = {
    "happy": "bgm/happy.wav",
    "sad": "bgm/sad.wav",
    "mysterious": "bgm/mysterious.wav",
    "scary": "bgm/scary.wav",
    "calm": "bgm/calm.wav",
    "adventure": "bgm/adventure.wav",
    "worried": "bgm/worried.wav",
    "warm": "bgm/warm.wav"
}


async def generate_scene_audio(scene, scene_id):
    #combine scene text
    text = scene["first_sentence"] + " " + scene["summary"] + " " + scene["last_sentence"]

    #get mood from JSON or auto_detect
    mood = scene.get("mood", detect_mood(scene))
    style = MOOD_STYLE.get(mood, MOOD_STYLE["calm"])

    #output file names
    voice_output = f"scene(2)_{scene_id}_voice.mp3"
    final_output = f"scene(2)_{scene_id}_final.mp3"

    #create TTS narrator
    communicate = edge_tts.Communicate(
        text=text,
        voice=NARRATOR_VOICE,
        rate=style["rate"],
        pitch=style["pitch"]
    )

    #generate narration audio
    await communicate.save(voice_output)

    
    #add background music
    add_background_music(
        voice_output,
        mood,
        final_output
    )

    print(f"Generated {final_output} | mood={mood} | rate={style['rate']} | pitch={style['pitch']}")

    


def add_background_music(voice_file, mood, output_file):
    # load narration audio
    voice = AudioSegment.from_file(voice_file)

    #select BGM based on mood
    bgm_path = BGM_MAP.get(mood, "bgm/calm.wav")

    # load BGM music
    bgm = AudioSegment.from_file(bgm_path)

    # reduce BGM volume
    bgm = bgm - 20

    # trim BGN to narration lenth
    bgm = bgm[:len(voice)]

    # mix narration and BGM
    final_audio = bgm.overlay(voice)

    # export final audio
    final_audio.export(output_file, format="mp3")

    print(f"Final audio with BGM saved: {output_file}")



async def main():
    #load JSON input
    with open("example_input2.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    #process each scene
    for index, scene in enumerate(data["scenes"], start=1):
        await generate_scene_audio(scene, index)

asyncio.run(main())
