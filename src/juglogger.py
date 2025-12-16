import json
import os
from datetime import datetime

LOG_FILE = "Jugo_Training_Data_2.jsonl"
MAX_FILE_SIZE_MB = 50  # auto-rotate at 50MB


def rotate_logs():
    """If the log file gets too big, rotate it."""
    if os.path.exists(LOG_FILE):
        size_mb = os.path.getsize(LOG_FILE) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(LOG_FILE, f"{LOG_FILE}.{timestamp}.bak")
            print(f"[LOG] Rotated log file -> {LOG_FILE}.{timestamp}.bak")


def log_interaction(user_message: str, bot_message: str):
    """Save a single training example in JSONL format."""
    rotate_logs()

    entry = {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": bot_message}
        ]
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[LOG] Logged new conversation entry.")
