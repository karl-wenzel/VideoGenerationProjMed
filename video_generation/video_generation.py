import os
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

def create_story_video(image_folder, audio_folder, output_name="final_story.mp4"):
    # 1. Zielordner definieren und sicherstellen, dass er existiert
    target_folder = "tmp"
    os.makedirs(target_folder, exist_ok=True)
    
    # Der Pfad ist "tmp/final_story.mp4"
    final_output_path = os.path.join(target_folder, output_name)

    # Listen für alle Bilder und Audios holen (sortiert, damit die Reihenfolge stimmt)
    images = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])
    audios = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])

    clips = []

    # Wir gehen Szene für Szene durch
    for i in range(min(len(images), len(audios))):
        img_path = os.path.join(image_folder, images[i])
        audio_path = os.path.join(audio_folder, audios[i])

        # Audio laden, um die Länge zu bestimmen
        audio_clip = AudioFileClip(audio_path)
        
        # Audio-Länge
        gesamtdauer = audio_clip.duration
        img_clip = ImageClip(img_path).with_duration(gesamtdauer)
        
        # Audio dem Bild hinzufügen
        img_clip = img_clip.with_audio(audio_clip)
        
        clips.append(img_clip)

    # Alle Szenen hintereinander hängen
    if clips:
        print(f"Verarbeite {len(clips)} Szenen (inklusive jeweils 1 Sek. Pause)...")
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Rendern und im tmp-Ordner speichern
        final_video.write_videofile(final_output_path, fps=24, codec="libx264", audio_codec="aac")
        print(f"Erfolg! Video erstellt: {final_output_path}")
    else:
        print("Keine passenden Dateien gefunden!")

# START
if __name__ == "__main__":
    create_story_video("generated_images", "generated_audios")
