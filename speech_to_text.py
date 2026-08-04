"""
Speech-to-text helper.

Uses the `speech_recognition` library with Google's free recognizer.

Install dependencies:
    pip install SpeechRecognition pyaudio

If pyaudio fails to install on Windows:
    pip install pipwin
    pipwin install pyaudio

For fully offline recognition, swap `recognize_google` for a Vosk model
(see the note at the bottom of this file).
"""

from PyQt6.QtCore import QThread, pyqtSignal

import speech_recognition as sr


# Set to e.g. "de-DE" for German, "en-US" for English, or None for auto.
RECOGNITION_LANGUAGE = "en-US"

# How long to wait for the user to start speaking (seconds).
LISTEN_TIMEOUT = 10

# Maximum length of a single spoken phrase (seconds).
PHRASE_TIME_LIMIT = 60

# How long a silence is allowed before recording is considered finished (seconds).
# Higher = you can pause longer mid-sentence without it cutting you off.
PAUSE_THRESHOLD = 2.0


class SpeechWorker(QThread):
    """Records from the default microphone and transcribes it in the background."""

    listening_signal = pyqtSignal()      # emitted once the mic is actually listening
    result_signal = pyqtSignal(str)      # emitted with the recognised text
    error_signal = pyqtSignal(str)       # emitted with a human-readable error message

    def run(self):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = PAUSE_THRESHOLD
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.listening_signal.emit()
                audio = recognizer.listen(
                    source,
                    timeout=LISTEN_TIMEOUT,
                    phrase_time_limit=PHRASE_TIME_LIMIT,
                )

            if RECOGNITION_LANGUAGE:
                text = recognizer.recognize_google(audio, language=RECOGNITION_LANGUAGE)
            else:
                text = recognizer.recognize_google(audio)

            self.result_signal.emit(text)

        except sr.WaitTimeoutError:
            self.error_signal.emit("No speech detected.")
        except sr.UnknownValueError:
            self.error_signal.emit("Could not understand audio.")
        except sr.RequestError:
            self.error_signal.emit("Recognition service unavailable (check internet).")
        except OSError:
            self.error_signal.emit("No microphone found.")
        except Exception as e:
            self.error_signal.emit(str(e))


# ---------------------------------------------------------------------------
# OFFLINE OPTION (Vosk) — no internet required.
#
#   pip install vosk
#   Download a model from https://alphacephei.com/vosk/models and unzip it,
#   e.g. to ./model
#
# Then replace the recognize_google call above with:
#
#   text = recognizer.recognize_vosk(audio)   # requires a "model" folder
#
# ---------------------------------------------------------------------------