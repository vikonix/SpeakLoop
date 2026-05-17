import os
import warnings
import numpy as np
import pyaudio
import soundfile as sf
import sounddevice as sd
import tempfile
import time
import threading
from collections import deque

# AI Libraries
from faster_whisper import WhisperModel
from openai import OpenAI
from kokoro import KModel, KPipeline

# Keyboard library for global hotkeys
import keyboard

# === CONFIGURATION ===
sd.default.latency = "low"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

# 1. LM Studio API Settings
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

# 2. Model Settings
WHISPER_MODEL = "base"
#DEVICE = "cuda"
#COMPUTE_TYPE = "float16" # Use float16 for RTX 3060
DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # faster on CPU

# 3. Kokoro TTS Settings
KOKORO_VOICE = "af_heart"

# 4. System Prompt
SYSTEM_PROMPT = """
You are a helpful language tutor. 
Your goal is to help the user practice English.
Be encouraging, correct mistakes politely, and keep the conversation engaging.
Always keep responses concise (max 4 sentences).
"""

class VoiceTutor:
    def __init__(self):
        print("Initializing Whisper (this may take a moment)...")
        # Initialize Whisper STT
        self.whisper_model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

        print("Initializing Kokoro TTS...")
        # Initialize Kokoro TTS
        #self.kokoro_model = KModel().to("cpu")
        self.kokoro_model = KModel(repo_id="hexgrad/Kokoro-82M").to("cpu")
        self.pipeline = KPipeline(lang_code="a")

        print("Connecting to LM Studio...")
        # Initialize LLM Client
        self.llm_client = OpenAI(
            base_url=LM_STUDIO_URL,
            api_key=LM_STUDIO_API_KEY
        )

        # Conversation History
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "System ready. Hold SPACEBAR to speak."}
        ]

        # Audio Recorder Setup
        self.audio = pyaudio.PyAudio()
        # State
        self.is_recording = False
        self.frames = []

        # Start Keyboard Listener
        print("Listening for 'SPACE' key... (Hold to record, Release to send)")
        keyboard.on_press_key('space', self.on_space_press)
        keyboard.on_release_key('space', self.on_space_release)

        # Keep thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    def on_space_press(self, event):
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            print("\n[Recording...] Release SPACE to send.")

            threading.Thread(
                target=self.record_loop,
                daemon=True
            ).start()

    def on_space_release(self, event):
        if self.is_recording:
            self.is_recording = False
            print("\nProcessing...")

            threading.Thread(
                target=self.process_audio,
                daemon=True
            ).start()

    def record_loop(self):
        stream = None
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )

            while self.is_recording:
                data = stream.read(1024, exception_on_overflow=False)
                self.frames.append(data)

        except OSError as e:
            print(f"❌ Microphone error: {e}")
            self.is_recording = False

        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()

    def process_audio(self):
        """Transcribe -> LLM -> Speak"""
        try:
            # 1. Transcribe
            text = self.transcribe_audio()
            if not text:
                return

            print(f"🎤 You: {text}")

            # 2. Get LLM Response
            response_text = self.get_llm_response(text)
            print(f"🤖 Tutor: {response_text}")

            # 3. Speak
            self.speak(response_text)

        except Exception as e:
            print(f"❌ Error processing: {e}")

    def transcribe_audio(self):
        if not self.frames:
            return ""

        audio_np = np.frombuffer(b''.join(self.frames), dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = self.whisper_model.transcribe(
            audio_np,
            beam_size=1,
            best_of=1,
            language="en",
            vad_filter=True,
            condition_on_previous_text=False,
            without_timestamps=True,
        )

        return "".join(segment.text for segment in segments).strip()

    def get_llm_response(self, user_text):
        """Send to LM Studio and get reply"""
        self.messages.append({"role": "user", "content": user_text})

        try:
            response = self.llm_client.chat.completions.create(
                model="local-model", # Name doesn't matter for the API call
                messages=self.messages,
                temperature=0.7,
                max_tokens=128
            )

            reply = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"Error connecting to LM Studio: {e}"

    def speak(self, text):
        """Generate and play audio using Kokoro"""
        generator = self.pipeline(text, voice=KOKORO_VOICE, model=self.kokoro_model)

        for _, _, audio in generator:
            audio = np.asarray(audio, dtype=np.float32)

            # 200 ms silence before speech
            silence = np.zeros(int(0.2 * 24000), dtype=np.float32)
            audio = np.concatenate([silence, audio])
            sd.play(audio, 24000, blocking=True)

if __name__ == "__main__":
    tutor = VoiceTutor()
