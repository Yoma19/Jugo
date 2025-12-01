import json
import re

INPUT_FILE = "shitpost.txt"
OUTPUT_FILE = "shitpost_dataset.jsonl"

# Regex to detect Discord message header lines:
# Example: [2/22/2022 1:15 PM] username
message_header = re.compile(r"^\[(.*?)\]\s+(.*)$")

def parse_discord_log(lines):
    conversations = []
    current_conv = []
    current_user = None
    current_text = []

    def finalize_message():
        """Save the most recent message into the current conversation list."""
        nonlocal current_user, current_text, current_conv

        if current_user and current_text:
            msg = {
                "role": "user",   # We can remap roles later if needed
                "name": current_user,
                "content": " ".join(current_text).strip()
            }
            current_conv.append(msg)
        
        current_user = None
        current_text = []

    for line in lines:
        line = line.rstrip("\n")

        # Skip separators
        if line.startswith("===") or line.startswith("{Stickers}") or line.startswith("{Reactions}"):
            continue

        # Skip sticker URLs
        if "cdn.discordapp.com/stickers/" in line:
            continue

        # Match message header
        header_match = message_header.match(line)
        if header_match:
            # New message begins -> finalize previous
            finalize_message()

            timestamp, user = header_match.groups()
            current_user = user
            continue

        # If blank line → end of a message
        if line.strip() == "":
            finalize_message()
            continue

        # Otherwise → it is part of a message body
        current_text.append(line)

    # finalize last message
    finalize_message()

    # convert to JSONL "conversations" (chunk messages into pairs)
    jsonl_entries = []

    # We create training entries of alternating messages
    # Example:
    #   user1 -> user2
    #   user2 -> user1   (next pair)
    # This is simplistic but good for training.
    for i in range(0, len(current_conv) - 1, 2):
        jsonl_entries.append({
            "messages": [
                {"role": "user", "content": current_conv[i]["content"]},
                {"role": "assistant", "content": current_conv[i+1]["content"]}
            ]
        })
    
    return jsonl_entries


# Read logs
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parse
entries = parse_discord_log(lines)

# Write JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for item in entries:
        out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done. Wrote {len(entries)} conversation pairs to {OUTPUT_FILE}")
