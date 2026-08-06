# 🦊 Children's Story Video Generator

A fully automated pipeline that turns a single text prompt into a narrated, illustrated children's story video. Just describe a story idea and the pipeline generates the script, scene images, voiceover audio, and a final MP4 — end to end.

---

## How It Works

```
Text Prompt
    │
    ▼
Story JSON          story_generation/prompt2storyjson.py
    │
    ▼
Scene Images        image_generation/image_scene_generator.py
    │
    ▼
Scene Audio         audio_generation/audio_generation.py
    │
    ▼
Final Video         video_generation/video_generation.py
```

---

## Project Structure

```
root/
├── main.py                          # Pipeline entry point
├── story_generation/
│   └── prompt2storyjson.py          # Prompt → structured story JSON via LLM
├── image_generation/
│   ├── image_scene_generator.py     # Story JSON → scene images
│   └── image_generation_client.py   # Image generation client + path defaults
├── audio_generation/
│   └── audio_generation.py          # Story JSON → narrated MP3s with background music
├── video_generation/
│   └── video_generation.py          # Images + audio → final MP4
├── bgm/                             # Background music files per mood
│   ├── happy.wav
│   ├── calm.wav
│   ├── warm.wav
│   ├── mysterious.wav
│   ├── adventurous.wav
│   ├── worried.wav
│   ├── sad.wav
│   └── scary.wav
└── tmp/                             # All generated outputs (auto-created)
    ├── story.json
    ├── tmp_story.json
    ├── scene(2)_*_voice.mp3
    ├── scene(2)_*_final.mp3
    └── final_story.mp4
```

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Core dependencies:

| Package | Purpose |
|---|---|
| `openai` | LLM API client for story generation |
| `python-dotenv` | Loading API keys from `.env` |
| `edge-tts` | Neural text-to-speech narration |
| `pydub` | Audio mixing (voice + background music) |
| `moviepy` | Stitching images and audio into video |

---

## Setup

1. **Clone the repository** and navigate to the project root.

2. **Create a `.env` file** in the root directory:

    ```env
    ACADEMIC_CLOUD_CHATAI_API_KEY=your_api_key_here
    ```

## Git LFS Setup

The background music `.wav` files in `audio_generation/bgm/` are stored with
[Git Large File Storage (Git LFS)](https://git-lfs.com/). Install Git LFS before
cloning the repository so Git downloads the actual audio files instead of only
their small pointer files.

1. Install Git LFS:

   - Windows: `winget install GitHub.GitLFS`
   - macOS: `brew install git-lfs`
   - Debian/Ubuntu: `sudo apt install git-lfs`

2. Enable Git LFS for your user account:

   ```bash
   git lfs install
   ```

3. Clone the repository normally:

   ```bash
   git clone <repository-url>
   cd TextToImgProMed
   ```

If the repository was cloned before Git LFS was installed, download the audio
files afterward with:

```bash
git lfs pull
```

You can verify that the background audio is managed by Git LFS with:

```bash
git lfs ls-files
```


---

## Usage

Run the pipeline from the project root:

```python
from main import main

# Use the default prompt
main()

# Pass a custom prompt
main("A story about a curious penguin who discovers a hidden library")
```

Or call the pipeline function directly for more control:

```python
import asyncio
from main import run_pipeline

asyncio.run(run_pipeline("Two sisters who find a door hidden behind a waterfall"))
```

---

## Pipeline Details

### 1. Story Generation
The user prompt is sent to an LLM which returns a structured JSON object containing 2–3 characters (with visual descriptions) and 4–8 scenes (each with a summary, opening sentence, and closing sentence). The output is saved to `tmp/story.json`.

### 2. Image Generation
One image is generated per scene using the composite scene generator. Images are saved to the scene output directory defined in `image_generation_client.py`.

### 3. Audio Generation
Each scene is narrated using the `en-GB-RyanNeural` voice via `edge-tts`. The mood of each scene is auto-detected from its text (happy, calm, mysterious, etc.) and matched to a background music track. Voice and music are mixed and saved as MP3s in `tmp/`.

### 4. Video Generation
Scene images and their corresponding audio tracks are combined in order using `moviepy`. Each scene lasts for the duration of its audio plus a 1-second pause. The final video is exported as `tmp/final_story.mp4`.

### 5. GUI
If you want to start the GUI you may use the 'start_pipeline.command'. This script acts as an automated shortcut that navigates to the project folder, activates the necessary Conda environment, and launches the application. You need this code because it allows anyone to start the software with a single click, completely removing the need to manually type complex commands into the terminal.

---

## Notes

- All sub-modules (`audio_generation.py`, `video_generation.py`) must have their standalone execution wrapped in `if __name__ == "__main__":` to prevent them from running on import.
- The `tmp/` directory is created automatically and will be cleared at the start of each image generation run.
- The default story prompt (Mika the fox and the firefly) is used if no prompt is passed to `main()`.