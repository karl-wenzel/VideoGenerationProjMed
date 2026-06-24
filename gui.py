import sys
import subprocess
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QTextEdit, QPushButton)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont

# Wir importieren eure geniale Pipeline!
from main import main as run_ai_pipeline
from main import VIDEO_OUTPUT

# ==========================================
# 1. Der Hintergrund-Arbeiter (Worker)
# ==========================================
class PipelineWorker(QThread):
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            run_ai_pipeline(self.prompt)
            self.finished_signal.emit() 
        except Exception as e:
            self.error_signal.emit(str(e)) 

# ==========================================
# 2. Das Hauptfenster (Mit modernem UI-Design)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 AI Video Generator")
        self.setFixedSize(520, 450) # Etwas mehr Platz gegeben

        # --- MODERNES STYLESHEET (QSS - wie CSS) ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #3494fa; /* Eleganter, dunkler Hintergrund */
            } 
            QLabel {
                color: #FFFDF7; /* Weiches Weiß für Text */
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            }
            QTextEdit {
                background-color: #055bb5; /* Etwas helleres Dunkel für Eingabefelder */
                color: #FFFDF7;
                border: 2px solid #FFFDF7;
                border-radius: 10px; /* Abgerundete Ecken */
                padding: 12px;
                font-size: 14px;
            }
            QTextEdit:focus {
                border: 2px solid #FFFDF7; /* Blauer Rahmen, wenn man reinklickt */
            }
            QPushButton {
                background-color: #FFFDF7; /* Moderner blauer Button */
                color: #3494fa; /* Dunkler Text auf hellem Button */
                border-radius: 10px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b4befe; /* Helleres Blau beim Darüberfahren (Hover) */
            }
            QPushButton:disabled {
                background-color: #45475a; /* Grauer Button, wenn deaktiviert */
                color: #7f849c;
            }
        """)

        # Haupt-Layout mit großzügigen Abständen
        layout = QVBoxLayout()
        layout.setSpacing(15) # Abstand ZWISCHEN den Elementen
        layout.setContentsMargins(25, 25, 25, 25) # Abstand zum Fensterrand

        # --- Die Elemente (Widgets) ---
        
        # 1. Titel
        self.title_label = QLabel("Describe your story:")
        # Wir nutzen eine saubere, größere Schriftart
        title_font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        # 2. Text-Eingabefeld
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Once upon a time...")
        layout.addWidget(self.prompt_input)

        # 3. Der Start-Button
        self.start_button = QPushButton("🚀 Generate video")
        # Cursor ändert sich zu einer Hand beim Drüberfahren
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.clicked.connect(self.start_generation)
        layout.addWidget(self.start_button)

        # 4. Status-Text
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #FFFDF7; font-size: 13px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # ==========================================
    # 3. Die Logik
    # ==========================================
    def start_generation(self):
        user_text = self.prompt_input.toPlainText().strip()
        if not user_text:
            self.status_label.setStyleSheet("color: #d10808;") # Pastell-Rot bei Fehler
            self.status_label.setText("Status: ❌ Please enter a text!")
            return

        # DEMO MODUS
        if user_text.upper() == "DEMO":
            self.status_label.setStyleSheet("color: #FFFDF7;") # Weiß
            self.status_label.setText("Status: 🎭 Demo Mode active! Loading pre-generated video...")
            try:
                subprocess.call(['open', str(VIDEO_OUTPUT)])
            except Exception:
                pass
            return

        self.start_button.setEnabled(False)
        self.start_button.setText("⏳ Generating... (This takes a moment)")
        self.status_label.setStyleSheet("color: #FFFDF7;") # Info-Blau
        self.status_label.setText("Status: Pipeline is running in the background...")

        self.worker = PipelineWorker(user_text)
        self.worker.finished_signal.connect(self.on_generation_finished)
        self.worker.error_signal.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_finished(self):
        self.start_button.setEnabled(True)
        self.start_button.setText("🚀 Generate new video")
        self.status_label.setStyleSheet("color: #FFFDF7;") # Erfolg-Grün
        self.status_label.setText(f"Status: ✅ Done! Video saved at:\n{VIDEO_OUTPUT}")
        
        try:
            subprocess.call(['open', str(VIDEO_OUTPUT)])
        except Exception:
            pass

    def on_generation_error(self, error_message):
        self.start_button.setEnabled(True)
        self.start_button.setText("🚀 Generate video")
        self.status_label.setStyleSheet("color: #d10808; font-weight: bold;") # Fehler-Rot
        self.status_label.setText(f"Status: ❌ ERROR:\n{error_message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
