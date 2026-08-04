import sys
import os
import shutil
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTextEdit, QPushButton, QProgressBar, QListWidget,
                             QListWidgetItem, QMessageBox, QStyle)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QPainter, QColor

from main import main as run_ai_pipeline
from main import VIDEO_OUTPUT

# Speech-to-text worker lives in its own module.
from speech_to_text import SpeechWorker

# === ORDNER UND DATEI FÜR DEN VERLAUF ===
HISTORY_DIR = Path("video_history")
HISTORY_FILE = HISTORY_DIR / "history.json"
HISTORY_DIR.mkdir(exist_ok=True)
if not HISTORY_FILE.exists():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


def open_video_file(video_path):
    path = str(video_path)

    # Windows has no "open" command; use the file association shell hook so
    # generated MP4s open in the user's default video player.
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])


# ==========================================
# 1. Custom Widgets
# ==========================================

class MovingTextProgressBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setTextVisible(False)

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = self.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#11111b"))

        text = f"{self.value()}%"
        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(text)

        fraction = self.value() / (self.maximum() or 100)
        chunk_width = int(self.width() * fraction)

        x_pos = (chunk_width // 2) - (text_width // 2)

        if x_pos < 10:
            x_pos = 10

        y_pos = (self.height() + fm.ascent() - fm.descent()) // 2

        painter.drawText(x_pos, y_pos, text)


class HistoryItemWidget(QWidget):
    def __init__(self, prompt, timestamp, duration, video_path, list_item, parent_gui):
        super().__init__()
        self.list_item = list_item
        self.video_path = video_path
        self.parent_gui = parent_gui

        layout = QVBoxLayout()
        # --- FIX: Rechter Abstand verkleinert (von 5 auf 2), damit der Button weiter nach rechts geht ---
        layout.setContentsMargins(10, 8, 2, 8)
        layout.setSpacing(5)

        # Obere Reihe: Zeit, Dauer und Löschen-Button
        top_layout = QHBoxLayout()

        self.time_label = QLabel(f"🕒 {timestamp}   |   ⏳ Taken: {duration}")
        self.time_label.setStyleSheet("color: #a6adc8; font-size: 11px;")
        top_layout.addWidget(self.time_label)

        top_layout.addStretch()

        self.delete_btn = QPushButton()
        trash_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        self.delete_btn.setIcon(trash_icon)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setFixedSize(30, 30)
        self.delete_btn.setToolTip("Delete Session")
        self.delete_btn.setStyleSheet("""
            QPushButton { background-color: #45475a; border-radius: 15px; }
            QPushButton:hover { background-color: #f38ba8; }
        """)
        self.delete_btn.clicked.connect(self.delete_self)
        top_layout.addWidget(self.delete_btn)

        layout.addLayout(top_layout)

        # --- FIX: Text abschneiden und "..." hinzufügen ---
        display_prompt = prompt
        if len(display_prompt) > 120:
            display_prompt = display_prompt[:117] + "..."

        self.prompt_label = QLabel(f"🎬 {display_prompt}")
        self.prompt_label.setWordWrap(True)

        # PROFI-FEATURE: Wenn der Text abgeschnitten ist, zeigt Hovering den kompletten Text!
        self.prompt_label.setToolTip(prompt)

        self.prompt_label.setStyleSheet("color: #cdd6f4; font-size: 13px; font-weight: bold;")
        layout.addWidget(self.prompt_label)

        self.setLayout(layout)

    def delete_self(self):
        self.parent_gui.delete_single_history_item(self.list_item, self.video_path)


# ==========================================
# 2. Worker Threads
# ==========================================
class PipelineWorker(QThread):
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    target_progress_signal = pyqtSignal(int)
    step_signal = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        class PrintInterceptor:
            def __init__(self, worker):
                self.worker = worker
                self.terminal = sys.__stdout__

            def write(self, text):
                self.terminal.write(text)

                if "Step 1" in text:
                    self.worker.target_progress_signal.emit(15)
                    self.worker.step_signal.emit("Status: ✍️ Writing the story (Step 1/4)...")
                elif "Step 2" in text:
                    self.worker.target_progress_signal.emit(35)
                    self.worker.step_signal.emit("Status: 🎨 Painting scene images (Step 2/4)...")
                elif "Step 3" in text:
                    self.worker.target_progress_signal.emit(80)
                    self.worker.step_signal.emit("Status: 🎙️ Recording voice & music (Step 3/4)...")
                elif "Step 4" in text:
                    self.worker.target_progress_signal.emit(95)
                    self.worker.step_signal.emit("Status: 🎬 Assembling final video (Step 4/4)...")

            def flush(self):
                self.terminal.flush()

        old_stdout = sys.stdout
        sys.stdout = PrintInterceptor(self)

        try:
            run_ai_pipeline(self.prompt)
            self.target_progress_signal.emit(100)
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            sys.stdout = old_stdout


class DemoWorker(QThread):
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    target_progress_signal = pyqtSignal(int)
    step_signal = pyqtSignal(str)

    def run(self):
        self.step_signal.emit("Status: ✍️ Writing the story (Step 1/4)...")
        self.target_progress_signal.emit(20)
        time.sleep(2.5)

        self.step_signal.emit("Status: 🎨 Painting scene images (Step 2/4)...")
        self.target_progress_signal.emit(45)
        time.sleep(3.0)

        self.step_signal.emit("Status: 🎙️ Recording voice & music (Step 3/4)...")
        self.target_progress_signal.emit(85)
        time.sleep(3.0)

        self.step_signal.emit("Status: 🎬 Assembling final video (Step 4/4)...")
        self.target_progress_signal.emit(98)
        time.sleep(1.5)

        self.target_progress_signal.emit(100)
        self.finished_signal.emit()


# ==========================================
# 3. Hauptfenster
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 AI Video Generator (Messe Edition)")

        self.setMinimumSize(1000, 600)
        self.showMaximized()

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QLabel { color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }

            QTextEdit {
                background-color: #313244; color: #cdd6f4;
                border: 2px solid #45475a; border-radius: 10px;
                padding: 12px; font-size: 15px;
            }
            QTextEdit:focus { border: 2px solid #89b4fa; }

            QPushButton {
                background-color: #89b4fa; color: #11111b;
                border-radius: 10px; padding: 15px;
                font-size: 16px; font-weight: bold;
            }
            QPushButton:hover { background-color: #b4befe; }
            QPushButton:disabled { background-color: #45475a; color: #a6adc8; }

            QProgressBar {
                border: 2px solid #45475a; border-radius: 10px;
                background-color: #313244; height: 25px;
                font-weight: bold; font-size: 13px;
            }
            QProgressBar::chunk { background-color: #a6e3a1; border-radius: 8px; }

            QListWidget {
                background-color: #181825; color: #cdd6f4;
                border-radius: 10px; padding: 5px; border: none;
            }
            QListWidget::item { border-bottom: 1px solid #313244; }
            QListWidget::item:hover { background-color: #313244; border-radius: 5px; }
            QListWidget::item:selected { background-color: #45475a; border-radius: 5px; }
        """)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        # --- LINKE SEITE ---
        sidebar_layout = QVBoxLayout()
        self.history_title = QLabel("📚 Session History")
        self.history_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        sidebar_layout.addWidget(self.history_title)

        self.history_list = QListWidget()
        self.history_list.setFixedWidth(350)
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_list.itemDoubleClicked.connect(self.play_history_video)
        sidebar_layout.addWidget(self.history_list)

        main_layout.addLayout(sidebar_layout)

        # --- RECHTE SEITE ---
        content_layout = QVBoxLayout()
        content_layout.setSpacing(15)

        self.title_label = QLabel("Describe your story:")
        self.title_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        content_layout.addWidget(self.title_label)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Once upon a time...")
        content_layout.addWidget(self.prompt_input)

        # --- GENERATE-BUTTON MIT MIKROFON-BUTTON IN EINER REIHE ---
        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.start_button = QPushButton("🚀 Generate Video")
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.clicked.connect(self.start_generation)
        button_row.addWidget(self.start_button)

        self.mic_button = QPushButton("🎤")
        self.mic_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_button.setFixedWidth(70)
        self.mic_button.setToolTip("Click and speak to fill the prompt")
        self.mic_button.setStyleSheet("""
            QPushButton { background-color: #f9e2af; color: #11111b; font-size: 22px; }
            QPushButton:hover { background-color: #f5c97a; }
            QPushButton:disabled { background-color: #45475a; color: #a6adc8; }
        """)
        self.mic_button.clicked.connect(self.start_listening)
        button_row.addWidget(self.mic_button)

        content_layout.addLayout(button_row)

        self.progress_bar = MovingTextProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.hide()
        content_layout.addWidget(self.progress_bar)

        # --- ZEIT & STATUS ---
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        self.time_label = QLabel("")
        self.time_label.setStyleSheet("color: #f9e2af; font-size: 14px; font-weight: bold;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_layout.addWidget(self.time_label)

        content_layout.addLayout(status_layout)
        main_layout.addLayout(content_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.ui_tick)
        self.elapsed_ticks = 0
        self.target_progress = 0
        self.current_eta = "~08:00"

        self.speech_worker = None

        self.load_history()

    # ==========================================
    # 4. Timer & Progress Logik
    # ==========================================
    def ui_tick(self):
        self.elapsed_ticks += 1

        if self.elapsed_ticks % 10 == 0:
            seconds = self.elapsed_ticks // 10
            mins, secs = divmod(seconds, 60)
            self.time_label.setText(f"⏱️ Time: {mins:02d}:{secs:02d} | ETA: {self.current_eta}")

        if self.current_prompt.upper() != "DEMO":
            if self.elapsed_ticks % 40 == 0 and self.target_progress < 95:
                self.target_progress += 1

        current = self.progress_bar.value()
        if current < self.target_progress:
            self.progress_bar.setValue(current + 1)

    def set_target_progress(self, target):
        if target > self.target_progress:
            self.target_progress = target

    # ==========================================
    # 5. History Logik
    # ==========================================
    def load_history(self):
        self.history_list.clear()
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                for entry in reversed(history):
                    t_stamp = entry.get("timestamp", "Unknown time")
                    duration = entry.get("duration", "??:?? min")
                    self.add_to_sidebar(entry["prompt"], t_stamp, duration, entry["video_path"])
        except Exception:
            pass

    def add_to_sidebar(self, prompt, timestamp, duration, video_path):
        item = QListWidgetItem(self.history_list)
        widget = HistoryItemWidget(prompt, timestamp, duration, video_path, item, self)

        # --- FIX: Das Element ist jetzt genau so breit, dass der Button an den Rand gedrückt wird ---
        widget.setFixedWidth(335)

        item.setSizeHint(widget.sizeHint())
        item.setData(Qt.ItemDataRole.UserRole, video_path)

        self.history_list.addItem(item)
        self.history_list.setItemWidget(item, widget)

    def save_to_history(self, prompt, duration_str):
        now = datetime.now()
        timestamp_file = now.strftime("%Y%m%d_%H%M%S")
        timestamp_ui = now.strftime("%d.%m.%Y - %H:%M")

        new_video_path = HISTORY_DIR / f"video_{timestamp_file}.mp4"
        try:
            shutil.copy2(VIDEO_OUTPUT, new_video_path)
        except Exception:
            pass

        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []

        history.append({
            "prompt": prompt,
            "timestamp": timestamp_ui,
            "duration": duration_str,
            "video_path": str(new_video_path)
        })
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        self.load_history()

    def delete_single_history_item(self, list_item, video_path):
        row = self.history_list.row(list_item)
        self.history_list.takeItem(row)

        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            history = [h for h in history if h.get("video_path") != video_path]
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
        except Exception:
            pass

    def play_history_video(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if os.path.exists(path): open_video_file(path)

    # ==========================================
    # 6. Speech-to-Text Logik
    # ==========================================
    def start_listening(self):
        # Don't start a second recording while one is running.
        if self.speech_worker is not None and self.speech_worker.isRunning():
            return

        self.mic_button.setEnabled(False)
        self.mic_button.setText("…")

        self.speech_worker = SpeechWorker()
        self.speech_worker.listening_signal.connect(self.on_speech_listening)
        self.speech_worker.result_signal.connect(self.on_speech_result)
        self.speech_worker.error_signal.connect(self.on_speech_error)
        self.speech_worker.start()

    def on_speech_listening(self):
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.status_label.setText("Status: 🎤 Listening... speak now")

    def on_speech_result(self, text):
        existing = self.prompt_input.toPlainText().strip()
        self.prompt_input.setPlainText(f"{existing} {text}".strip() if existing else text)
        self.status_label.setText("Status: Ready")
        self._reset_mic_button()

    def on_speech_error(self, message):
        self.status_label.setText(f"Status: 🎤 {message}")
        self._reset_mic_button()

    def _reset_mic_button(self):
        self.mic_button.setEnabled(True)
        self.mic_button.setText("🎤")

    # ==========================================
    # 7. Generator Logik
    # ==========================================
    def start_generation(self):
        self.current_prompt = self.prompt_input.toPlainText().strip()
        if not self.current_prompt: return

        self.start_button.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.target_progress = 0
        self.elapsed_ticks = 0

        if self.current_prompt.upper() == "DEMO":
            self.current_eta = "~00:10"
            self.worker = DemoWorker()
        else:
            self.current_eta = "~08:00"
            self.worker = PipelineWorker(self.current_prompt)

        self.ui_timer.start(100)

        self.worker.target_progress_signal.connect(self.set_target_progress)
        self.worker.step_signal.connect(self.status_label.setText)
        self.worker.finished_signal.connect(self.on_generation_finished)
        self.worker.error_signal.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_finished(self):
        self.ui_timer.stop()
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Status: ✅ Done! Video saved.")

        seconds = self.elapsed_ticks // 10
        mins, secs = divmod(seconds, 60)
        final_duration_str = f"{mins:02d}:{secs:02d} min"

        if self.current_prompt.upper() != "DEMO":
            self.save_to_history(self.current_prompt, final_duration_str)
        else:
            self.save_to_history("🎭 Demo Simulation run", final_duration_str)

        QTimer.singleShot(4000, self.hide_progress_bar)

        try:
            open_video_file(VIDEO_OUTPUT)
        except:
            pass

    def on_generation_error(self, error_message):
        self.ui_timer.stop()
        self.start_button.setEnabled(True)
        self.progress_bar.hide()
        self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        self.status_label.setText(f"Status: ❌ ERROR:\n{error_message}")

    def hide_progress_bar(self):
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        self.time_label.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())