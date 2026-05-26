import os
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

def create_story_video(image_folder, audio_folder, output_name="final_story.mp4"):
    # Zielordner für das fertige Video definieren
    target_folder = "tmp"
    os.makedirs(target_folder, exist_ok=True)
    
    final_output_path = os.path.join(target_folder, output_name)

    # Prüfen, ob der Ordner existiert
    if not os.path.exists(image_folder):
        print(f"Fehler: Der Ordner '{image_folder}' existiert nicht!")
        return

    # Listen für alle Bilder und Audios aus dem selben Ordner holen
    # Dank sorted() werden sie alphabetisch/numerisch sortiert
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
        
        # Im tmp-Ordner speichern
        final_video.write_videofile(final_output_path, fps=24, codec="libx264", audio_codec="aac")
        print(f"Erfolg! Video erstellt: {final_output_path}")
    else:
        print("Keine passenden Bild- und Audiodateien im Ordner gefunden!")

# START
if __name__ == "__main__":
    # Da alles direkt im tmp-Ordner liegt, gilt für beide Ordner "tmp"
    create_story_video(
        image_folder="tmp", 
        audio_folder="tmp"
    )
