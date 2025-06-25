import os
import shutil

# Paths
SOURCE_DIR = "C:/Users/vagha/Music/24_actors"
TARGET_DIR = "data"

# Emotion mappings
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

# Only use selected emotions
USED_EMOTIONS = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '08': 'surprise'
}

# 🔥 DELETE existing 'data/' directory first
if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

# Create base target directory
os.makedirs(TARGET_DIR, exist_ok=True)

# Process actor folders
for actor_folder in os.listdir(SOURCE_DIR):
    actor_path = os.path.join(SOURCE_DIR, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    for file_name in os.listdir(actor_path):
        if file_name.endswith(".wav"):
            parts = file_name.split("-")
            emotion_code = parts[2]  # 3rd segment

            if emotion_code in USED_EMOTIONS:
                emotion = USED_EMOTIONS[emotion_code]
                dest_dir = os.path.join(TARGET_DIR, emotion)
                os.makedirs(dest_dir, exist_ok=True)

                src_path = os.path.join(actor_path, file_name)
                dest_path = os.path.join(dest_dir, file_name)

                shutil.copyfile(src_path, dest_path)

print("✅ Dataset cleaned and re-organized.")
