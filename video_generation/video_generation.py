import os
import json
import re
import textwrap
from pathlib import Path
from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips


def get_scene_subtitle_text(scene):
    return scene.get("narration") or (
        scene.get("first_sentence", "") + " " +
        scene.get("summary", "") + " " +
        scene.get("last_sentence", "")
    )


def split_subtitle_sentences(text):
    text = " ".join(text.split())
    parts = re.split(r'(?<=[.!?。！？])\s+', text)
    parts = [part.strip() for part in parts if part.strip()]

    final_parts = []
    for part in parts:
        if len(part) > 120:
            smaller = re.split(r'(?<=[,，;；:：])\s+', part)
            final_parts.extend([p.strip() for p in smaller if p.strip()])
        else:
            final_parts.append(part)

    return final_parts or [text]


def normalize_word(word):
    return re.sub(r"[^a-zA-Z0-9]", "", word).lower()

def create_precise_subtitle_timings(word_data):
    subtitle_timings = []

    for item in word_data["words"]:
        subtitle_timings.append(
            (
                item["text"],
                item["start"],
                item["duration"]
            )
        )

    return subtitle_timings

def create_story_video(image_folder, audio_folder, output_name):
    
    # 1. Wir holen uns den Ordnernamen 
    target_folder = os.path.dirname(output_name)
    if target_folder:
        os.makedirs(target_folder, exist_ok=True)
    
    # Listen für alle Bilder und Audios holen 
    images = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])
    audios = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])
    
    project_root = Path(__file__).resolve().parent.parent
    font_path = project_root / "fonts" / "DejaVuSans.ttf"
    story_path = project_root / "tmp" / "story.json"

    with open(story_path, "r", encoding="utf-8") as f:
        story = json.load(f)

    scenes = story.get("scenes", [])
    subtitle_folder = project_root / "tmp" / "tmp_audio_generation" / "subtitles"

    clips = []

    # Wir gehen Szene für Szene durch
    for i in range(min(len(images), len(audios))):
        img_path = os.path.join(image_folder, images[i])
        audio_path = os.path.join(audio_folder, audios[i])

        audio_clip = AudioFileClip(audio_path)

        img_clip = ImageClip(img_path).with_duration(audio_clip.duration)

        scene = scenes[i] if i < len(scenes) else {}
        subtitle_text = get_scene_subtitle_text(scene)

        subtitle_json_path = subtitle_folder / f"scene_{i + 1}_words.json"

        if subtitle_json_path.exists():
            with open(subtitle_json_path, "r", encoding="utf-8") as f:
                word_data = json.load(f)

            subtitle_timings = create_precise_subtitle_timings(word_data)
        else:
            subtitle_parts = split_subtitle_sentences(subtitle_text)
            part_duration = audio_clip.duration / max(1, len(subtitle_parts))
            subtitle_timings = [
                (part, index * part_duration, part_duration)
                for index, part in enumerate(subtitle_parts)
            ]

        subtitle_clips = []

        for part, start_time, duration in subtitle_timings:
            wrapped_text = "\n".join(textwrap.wrap(part, width=50))

            subtitle_clip = (
                TextClip(
                    text=wrapped_text,
                    font=str(font_path),
                    font_size=30,
                    color="white",
                    stroke_color="black",
                    stroke_width=2,
                    method="caption",
                    size=(int(img_clip.w * 0.82), 140),
                )
                .with_start(start_time)
                .with_duration(duration)
                .with_position(("center", img_clip.h-180))
            )

            subtitle_clips.append(subtitle_clip)

        scene_clip = CompositeVideoClip([img_clip, *subtitle_clips])
        scene_clip = scene_clip.with_audio(audio_clip)

        clips.append(scene_clip)

    # Alle Szenen hintereinander hängen
    if clips:
        print(f"Verarbeite {len(clips)} Szenen...")
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Direkt an den Pfad speichern
        final_video.write_videofile(output_name, fps=24, codec="libx264", audio_codec="aac")
        print(f"Erfolg! Video erstellt: {output_name}")
    else:
        print(f"Keine passenden Dateien gefunden in:\nBilder: {image_folder}\nAudios: {audio_folder}")


# Der Test-Aufruf unten ist nur aktiv, wenn man das Skript einzeln startet.
# Wenn main.py das Skript startet, wird das hier ignoriert.
if __name__ == "__main__":
    
    # 1. Wir holen uns den exakten Pfad, wo wir uns gerade befinden, und gehen einen Ordner nach oben
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 2. Wir bauen die Pfade zusammen
    test_image_folder = os.path.join(project_root, "tmp", "scene_images")
    test_audio_folder = os.path.join(project_root, "tmp", "tmp_audio_generation", "final")
    test_output_name = os.path.join(project_root, "tmp", "final_story_einzeltest.mp4")
    
    # Prüfen und starten
    if os.path.exists(test_image_folder) and os.path.exists(test_audio_folder):
        print("Ordner gefunden! Baue das Video zusammen...")
        create_story_video(test_image_folder, test_audio_folder, test_output_name)
    else:
        print(f"Fehler: Ordner nicht gefunden.\nSuche Bilder in: {test_image_folder}\nSuche Audios in: {test_audio_folder}")
