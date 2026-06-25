import sys
import os
import shutil
import json
import subprocess
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QTextEdit, QPushButton, QProgressBar, QListWidget, QListWidgetItem,
                             QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont

from main import main as run_ai_pipeline
from main import VIDEO_OUTPUT

# Ordner und Datei für den Verlauf ===
HISTORY_DIR = Path("video_history")
HISTORY_FILE = HISTORY_DIR / "history.json"

# Stellt sicher, dass der Ordner existiert
HISTORY_DIR.mkdir(exist_ok=True)
if not HISTORY_FILE.exists():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# ==========================================
# 1. Background
# ==========================================
class PipelineWorker(QThread):
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
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
                    self.worker.progress_signal.emit(10)
                    self.worker.step_signal.emit("Status: ✍️ Writing the story (Step 1/4)...")
                elif "Step 2" in text:
                    self.worker.progress_signal.emit(30)
                    self.worker.step_signal.emit("Status: 🎨 Painting scene images (Step 2/4)...")
                elif "Step 3" in text:
                    self.worker.progress_signal.emit(65)
                    self.worker.step_signal.emit("Status: 🎙️ Recording voice & music (Step 3/4)...")
                elif "Step 4" in text:
                    self.worker.progress_signal.emit(90)
                    self.worker.step_signal.emit("Status: 🎬 Assembling final video (Step 4/4)...")

            def flush(self):
                self.terminal.flush()

        old_stdout = sys.stdout
        sys.stdout = PrintInterceptor(self)

        try:
            run_ai_pipeline(self.prompt)
            self.progress_signal.emit(100)
            self.finished_signal.emit() 
        except Exception as e:
            self.error_signal.emit(str(e)) 
        finally:
            sys.stdout = old_stdout


# ==========================================
# 2. Hauptfenster
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 AI Video Generator")
        self.setFixedSize(900, 520)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #3494fa; 
            } 
            QLabel {
                color: #FFFDF7; 
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            }
            QTextEdit {
                background-color: #055bb5; 
                color: #FFFDF7;
                border: 2px solid #FFFDF7;
                border-radius: 10px; 
                padding: 12px;
                font-size: 14px;
            }
            QTextEdit:focus {
                border: 2px solid #FFFDF7; 
            }
            QPushButton {
                background-color: #FFFDF7; 
                color: #3494fa; 
                border-radius: 10px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe; 
            }
            QPushButton:disabled {
                background-color: #055bb5; 
                color: #FFFDF7;
            }
            QProgressBar {
                border: 2px solid #FFFDF7;
                border-radius: 10px;
                text-align: center; 
                color: #022a52; 
                background-color: #89b4fa; 
                height: 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #FFFDF7; 
                border-radius: 8px;
            }
            
            /* --- DESIGN FÜR SEITENLEISTE --- */
            QListWidget {
                background-color: #022a52; 
                color: #FFFDF7;
                border-radius: 10px;
                padding: 5px;
                font-size: 13px;
                border: none;
            }
            QListWidget::item {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 5px;
            }
            QListWidget::item:selected {
                background-color: #89b4fa; 
                color: #022a52;
                font-weight: bold;
            }
            
            /* --- CLEAN DESIGN FÜR WARNFENSTER --- */
            QMessageBox {
                background-color: #022a52;
            }
            QMessageBox QLabel {
                color: #FFFDF7;
                font-size: 13px;
                min-width: 300px;
            }
            QMessageBox QPushButton {
                background-color: #FFFDF7; 
                color: #022a52; 
                border-radius: 5px;
                padding: 6px 15px;
                font-weight: bold;
            }
        """)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # LINKE SEITE
        sidebar_layout = QVBoxLayout()
        self.history_title = QLabel("📚 History")
        self.history_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        sidebar_layout.addWidget(self.history_title)

        self.history_list = QListWidget()
        self.history_list.setFixedWidth(200) 
        self.history_list.itemDoubleClicked.connect(self.play_history_video) 
        sidebar_layout.addWidget(self.history_list)

        self.clear_history_button = QPushButton("🗑️ Clear History")
        self.clear_history_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_history_button.setStyleSheet("background-color: #b52222; color: white;")
        self.clear_history_button.clicked.connect(self.clear_history)
        sidebar_layout.addWidget(self.clear_history_button)

        main_layout.addLayout(sidebar_layout)

        # RECHTE SEITE
        content_layout = QVBoxLayout()
        content_layout.setSpacing(15) 
        self.title_label = QLabel("Describe your story:")
        self.title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        content_layout.addWidget(self.title_label)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Once upon a time...")
        content_layout.addWidget(self.prompt_input)

        self.start_button = QPushButton("🚀 Generate video")
        self.start_button.clicked.connect(self.start_generation)
        content_layout.addWidget(self.start_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100) 
        self.progress_bar.hide() 
        content_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Ready")
        content_layout.addWidget(self.status_label)

        main_layout.addLayout(content_layout)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.load_history()

    # --- FUNKTIONEN ---
    def load_history(self):
        self.history_list.clear()
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                for entry in reversed(history):
                    self.add_to_sidebar(entry["prompt"], entry["video_path"])
        except Exception: pass

    def add_to_sidebar(self, prompt, video_path):
        item = QListWidgetItem(f"🎬 {prompt[:20]}...")
        item.setData(Qt.ItemDataRole.UserRole, video_path)
        self.history_list.addItem(item)

    def save_to_history(self, prompt):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_video_path = HISTORY_DIR / f"video_{timestamp}.mp4"
        shutil.copy2(VIDEO_OUTPUT, new_video_path)
        
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: history = json.load(f)
        except: history = []
        history.append({"prompt": prompt, "video_path": str(new_video_path)})
        with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(history, f, indent=4)
        self.load_history()

    def play_history_video(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if os.path.exists(path): subprocess.call(['open', path])

    def clear_history(self):
        msg = QMessageBox(self)
        msg.setWindowTitle('🗑️ Clear History')
        msg.setText('Are you sure you want to delete all saved videos?')
        
        # Wir zwingen die Box dazu, breit genug für den Text zu sein
        msg.setStyleSheet("""
            QMessageBox { background-color: #022a52; }
            QLabel { color: #FFFDF7; font-size: 14px; min-width: 400px; }
            QPushButton { 
                background-color: #FFFDF7; color: #022a52; 
                border-radius: 5px; padding: 6px 15px; font-weight: bold; 
            }
        """)

        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if msg.exec() == QMessageBox.StandardButton.Yes:
            for file in HISTORY_DIR.glob("*.mp4"): file.unlink()
            with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump([], f)
            self.history_list.clear()
            self.status_label.setText("Status: 🗑️ History cleared!")

    def start_generation(self):
        self.current_prompt = self.prompt_input.toPlainText().strip()
        if not self.current_prompt: return
        
        if self.current_prompt.upper() == "DEMO":
            self.progress_bar.show()
            self.progress_bar.setValue(100)
            QTimer.singleShot(4000, self.hide_progress_bar)
            self.save_to_history("🎭 Demo Video")
            try: subprocess.call(['open', str(VIDEO_OUTPUT)])
            except: pass
            return

        self.start_button.setEnabled(False)
        self.progress_bar.show()
        self.worker = PipelineWorker(self.current_prompt)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.step_signal.connect(self.status_label.setText)
        self.worker.finished_signal.connect(self.on_generation_finished)
        self.worker.start()

    def on_generation_finished(self):
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.save_to_history(self.current_prompt)
        QTimer.singleShot(4000, self.hide_progress_bar)
        try: subprocess.call(['open', str(VIDEO_OUTPUT)])
        except: pass

    def hide_progress_bar(self):
        self.progress_bar.hide()
        self.progress_bar.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
