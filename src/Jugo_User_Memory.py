import json
import os
from datetime import datetime

MEMORY_FILE = "Jugo_user_memory_v2.json"


# -----------------------------
# Load / Save Utilities
# -----------------------------
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("[WARN] user_memory.json was invalid. Resetting.")
        return {}


def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=4, ensure_ascii=False)


# -----------------------------
# Core: Update user memory
# -----------------------------
def update_user_memory(user_id, guild_id, display_name, global_name=None):
    """
    Store per-user memory:
      - user_id (stable identity)
      - aliases (name per guild)
      - global_name (account name)
      - message_count
      - last_seen timestamp
    """
    memory = load_memory()

    uid = str(user_id)
    gid = str(guild_id)

    if uid not in memory:
        memory[uid] = {
            "global_name": global_name or display_name,
            "aliases": {},
            "message_count": 0,
            "last_seen": None,
        }

    # Always update the alias for the specific guild
    memory[uid]["aliases"][gid] = display_name

    # Only update global_name if missing (avoid overwriting)
    if global_name and not memory[uid].get("global_name"):
        memory[uid]["global_name"] = global_name

    # Update last seen timestamp
    memory[uid]["last_seen"] = datetime.utcnow().isoformat()

    save_memory(memory)


# -----------------------------
# Message Count
# -----------------------------
def increment_message_count(user_id):
    memory = load_memory()
    uid = str(user_id)

    if uid not in memory:
        memory[uid] = {
            "global_name": None,
            "aliases": {},
            "message_count": 0,
            "last_seen": None,
        }

    memory[uid]["message_count"] += 1
    memory[uid]["last_seen"] = datetime.utcnow().isoformat()

    save_memory(memory)


# -----------------------------
# Fetch preferred name
# -----------------------------
def get_preferred_name(user_id, guild_id):
    """
    Choose the best possible name:
      1. Alias specific to this guild
      2. Global username
      3. None (if not in memory yet)
    """
    memory = load_memory()
    uid = str(user_id)
    gid = str(guild_id)

    if uid not in memory:
        return None

    user = memory[uid]

    # Priority 1: Name used in this server
    if "aliases" in user and gid in user["aliases"]:
        return user["aliases"][gid]

    # Priority 2: Global name (Discord username)
    return user.get("global_name")


# -----------------------------
# Debug helper (optional)
# -----------------------------
def get_user_memory(user_id):
    """Returns the full memory entry for debugging/logging."""
    memory = load_memory()
    return memory.get(str(user_id), None)