import os
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

def create_story_video(image_folder, audio_folder, output_name="final_story.mp4"):
    # Listen für alle Bilder und Audios holen (sortiert, damit die Reihenfolge stimmt)
    images = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])
    audios = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])

    clips = []

    # Wir gehen Szene für Szene durch
    for i in range(min(len(images), len(audios))):
        img_path = os.path.join(image_folder, images[i])
        audio_path = os.path.join(audio_folder, audios[i])

        # 1. Audio laden, um die Länge zu bestimmen
        audio_clip = AudioFileClip(audio_path)
        
        # 2. Bild laden und die Dauer exakt auf die Audio-Länge setzen
        img_clip = ImageClip(img_path).with_duration(audio_clip.duration)
        
        # 3. Audio dem Bild hinzufügen
        img_clip = img_clip.with_audio(audio_clip)
        
        clips.append(img_clip)

    # Alle Szenen hintereinander hängen
    if clips:
        print(f"Verarbeite {len(clips)} Szenen...")
        final_video = concatenate_videoclips(clips, method="compose")
        # fps=24 ist Standard für flüssige Diashows
        final_video.write_videofile(output_name, fps=24, codec="libx264", audio_codec="aac")
        print(f"Erfolg! Video erstellt: {output_name}")
    else:
        print("Keine passenden Dateien gefunden!")

# TEST-LAUF
create_story_video("generated_images", "generated_audios") 
