#!/bin/bash
# Wechsle in den aktuellen Ordner
cd "$(dirname "$0")"

# Aktiviere Conda und den Raum (Pfade können je nach Mac leicht abweichen)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate video_gui

# Starte die App
python gui.py
