import json
import os
from datetime import datetime

MEMORY_FILE = "Jugo_user_memory.json"


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def get_user_memory(user_id: int):
    memory = load_memory()
    return memory.get(str(user_id), {})


def update_user_memory(user_id: int, key: str, value):
    memory = load_memory()
    uid = str(user_id)

    if uid not in memory:
        memory[uid] = {
            "first_seen": datetime.utcnow().isoformat() + "Z",
            "message_count": 0,
            "notes": []
        }

    memory[uid][key] = value
    memory[uid]["last_seen"] = datetime.utcnow().isoformat() + "Z"

    save_memory(memory)


def increment_message_count(user_id: int):
    memory = load_memory()
    uid = str(user_id)

    if uid not in memory:
        memory[uid] = {
            "first_seen": datetime.utcnow().isoformat() + "Z",
            "message_count": 0,
            "notes": []
        }

    memory[uid]["message_count"] += 1
    memory[uid]["last_seen"] = datetime.utcnow().isoformat() + "Z"

    save_memory(memory)


def add_user_note(user_id: int, note: str):
    memory = load_memory()
    uid = str(user_id)

    if uid not in memory:
        memory[uid] = {
            "first_seen": datetime.utcnow().isoformat() + "Z",
            "message_count": 0,
            "notes": []
        }

    if note not in memory[uid]["notes"]:
        memory[uid]["notes"].append(note)

    save_memory(memory)
