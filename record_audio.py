import sounddevice as sd
from scipy.io.wavfile import write

DURATION = 5  # seconds
SAMPLE_RATE = 22050  # CD quality audio
FILENAME = "user_audio.wav"

def record_audio():
    print("🎙️ Speak now... Recording for", DURATION, "seconds.")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(FILENAME, SAMPLE_RATE, recording)
    print(f"✅ Audio recorded and saved as '{FILENAME}'.")

if __name__ == "__main__":
    record_audio()
