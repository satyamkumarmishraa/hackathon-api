import base64
import os
import json

INPUT_FOLDER = "../audio_samples"
OUTPUT_FILE = "batch_requests.json"

payloads = []

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(".mp3"):
        file_path = os.path.join(INPUT_FOLDER, filename)

        with open(file_path, "rb") as audio:
            encoded_audio = base64.b64encode(audio.read()).decode("utf-8")

        payloads.append({
            "fileName": filename,
            "language": "English",   # change only if instructed
            "audioFormat": "mp3",
            "audioBase64": encoded_audio
        })

with open(OUTPUT_FILE, "w") as f:
    json.dump(payloads, f, indent=2)

print(f"✅ Converted {len(payloads)} audio files to Base64 JSON")
