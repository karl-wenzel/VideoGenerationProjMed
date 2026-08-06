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

import audioop
import math
import threading
from collections import deque

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = threading.Event()

    def stop_recording(self):
        """Ask the capture loop to finish with the audio recorded so far."""

        self._stop_requested.set()

    def _listen_until_stopped(self, recognizer, source):
        """Capture one phrase while allowing the GUI to stop it early.

        SpeechRecognition's normal ``listen`` call blocks until its silence or
        time limit. Reading one microphone chunk at a time is required so the
        Stop Recording button can be acted on promptly.
        """

        seconds_per_buffer = source.CHUNK / float(source.SAMPLE_RATE)
        pause_buffers = math.ceil(PAUSE_THRESHOLD / seconds_per_buffer)
        retained_silence_buffers = math.ceil(
            recognizer.non_speaking_duration / seconds_per_buffer
        )
        frames = deque()
        elapsed = 0.0
        speech_started = False
        quiet_buffers = 0

        while elapsed < PHRASE_TIME_LIMIT:
            if self._stop_requested.is_set():
                break

            buffer = source.stream.read(source.CHUNK)
            if not buffer:
                break

            elapsed += seconds_per_buffer
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)

            if not speech_started:
                frames.append(buffer)
                if len(frames) > retained_silence_buffers:
                    frames.popleft()

                if energy > recognizer.energy_threshold:
                    speech_started = True
                elif elapsed >= LISTEN_TIMEOUT:
                    raise sr.WaitTimeoutError()
                continue

            frames.append(buffer)
            quiet_buffers = (
                0
                if energy > recognizer.energy_threshold
                else quiet_buffers + 1
            )
            if quiet_buffers > pause_buffers:
                break

        if not speech_started:
            raise sr.WaitTimeoutError()

        # Keep only the configured amount of trailing silence.
        for _ in range(max(0, quiet_buffers - retained_silence_buffers)):
            frames.pop()

        return sr.AudioData(
            b"".join(frames),
            source.SAMPLE_RATE,
            source.SAMPLE_WIDTH,
        )

    def run(self):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = PAUSE_THRESHOLD
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                self.listening_signal.emit()
                audio = self._listen_until_stopped(recognizer, source)

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
