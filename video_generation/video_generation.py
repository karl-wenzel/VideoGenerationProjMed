import os
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

def create_story_video(image_folder, audio_folder, output_name):
    
    # 1. Wir holen uns den Ordnernamen 
    target_folder = os.path.dirname(output_name)
    if target_folder:
        os.makedirs(target_folder, exist_ok=True)
    
    # Listen für alle Bilder und Audios holen 
    images = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])
    audios = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])

    clips = []

    # Wir gehen Szene für Szene durch
    for i in range(min(len(images), len(audios))):
        img_path = os.path.join(image_folder, images[i])
        audio_path = os.path.join(audio_folder, audios[i])

        # Audio laden
        audio_clip = AudioFileClip(audio_path)
        
        # Audio hinzufügen
        img_clip = img_clip.with_audio(audio_clip)
        
        clips.append(img_clip)

    # Alle Szenen hintereinander hängen
    if clips:
        print(f"Verarbeite {len(clips)} Szenen...")
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Direkt an den Pfad speichern
        final_video.write_videofile(output_name, fps=24, codec="libx264", audio_codec="aac")
        print(f"Erfolg! Video erstellt: {output_name}")
    else:
        print(f"Keine passenden Dateien gefunden in:\nBilder: {image_folder}\nAudios: {audio_folder}")

# Der Test-Aufruf ganz unten ist nur aktiv, wenn man das Skript einzeln startet.
# Wenn main.py das Skript startet, wird das hier ignoriert.
if __name__ == "__main__":
    create_story_video("tmp", "tmp", "tmp/final_story.mp4")
