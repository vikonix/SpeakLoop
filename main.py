import os
import time
import queue
import threading
import warnings
from typing import Optional

import numpy as np
import sounddevice as sd
import keyboard
import torch

from faster_whisper import WhisperModel
from kokoro import KModel, KPipeline
from openai import OpenAI


# =========================
# Configuration
# =========================

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "local-model"

WHISPER_MODEL = "base.en"
WHISPER_SAMPLE_RATE = 16_000

KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24_000

MAX_RECORD_SECONDS = 20

SYSTEM_PROMPT = (
    "You are an English tutor. Keep responses very short, 1-2 sentences. "
    "Be encouraging and correct only the most important mistakes."
)


if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"


class VoiceTutor:
    def __init__(self):
        self.shutdown_event = threading.Event()

        self.is_recording = False
        self.record_lock = threading.Lock()
        self.recorded_chunks: list[np.ndarray] = []
        self.record_thread: Optional[threading.Thread] = None

        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.tts_stop_event = threading.Event()

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        self.whisper_model: Optional[WhisperModel] = None
        self.kokoro_model: Optional[KModel] = None
        self.pipeline: Optional[KPipeline] = None
        self.llm_client: Optional[OpenAI] = None

    def start(self):
        print("\n" + "=" * 50)
        print("Voice Tutor MVP")
        print("=" * 50)

        print(f"Audio backend: sounddevice")
        print(f"Device: {DEVICE}, compute type: {COMPUTE_TYPE}")

        self.load_models()
        self.check_lm_studio()
        self.warm_up()

        self.tts_thread = threading.Thread(target=self.process_tts_queue, daemon=True)
        self.tts_thread.start()

        keyboard.on_press_key("space", self.on_space_press)
        keyboard.on_release_key("space", self.on_space_release)
        keyboard.on_press_key("esc", lambda _: self.shutdown())

        print("\nReady. Hold SPACE to speak. Press ESC to quit.\n")

        try:
            while not self.shutdown_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.shutdown()

    def load_models(self):
        print("Loading Whisper...", end=" ", flush=True)
        start = time.time()

        self.whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=4,
            num_workers=1,
        )

        print(f"done in {time.time() - start:.1f}s")

        print("Loading Kokoro TTS...", end=" ", flush=True)
        start = time.time()

        # Keep Kokoro on CPU for MVP stability. Move to CUDA later only after testing.
        self.kokoro_model = KModel(repo_id="hexgrad/Kokoro-82M").to("cpu")
        self.pipeline = KPipeline(lang_code="a")

        print(f"done in {time.time() - start:.1f}s")

        self.llm_client = OpenAI(
            base_url=LM_STUDIO_URL,
            api_key=LM_STUDIO_API_KEY,
            timeout=5.0,
        )

    def check_lm_studio(self):
        print("Checking LM Studio...", end=" ", flush=True)

        try:
            assert self.llm_client is not None
            self.llm_client.models.list()
            print("ready")
        except Exception as error:
            print(f"not available: {error}")
            print("Start LM Studio before using the tutor.")

    def warm_up(self):
        print("Warming up models...", end=" ", flush=True)

        assert self.whisper_model is not None
        assert self.pipeline is not None
        assert self.kokoro_model is not None

        dummy_audio = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)

        list(
            self.whisper_model.transcribe(
                dummy_audio,
                language="en",
                beam_size=1,
                vad_filter=True,
            )[0]
        )

        list(
            self.pipeline(
                "Hi.",
                voice=KOKORO_VOICE,
                model=self.kokoro_model,
            )
        )

        print("done")

    def on_space_press(self, _event):
        with self.record_lock:
            if self.is_recording:
                return

            self.stop_current_tts()
            self.is_recording = True
            self.recorded_chunks = []

        print("\nRecording...", flush=True)

        self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
        self.record_thread.start()

    def on_space_release(self, _event):
        with self.record_lock:
            if not self.is_recording:
                return

            self.is_recording = False

        if self.record_thread:
            self.record_thread.join(timeout=1.0)

        print("Processing...", flush=True)

        threading.Thread(target=self.process_audio, daemon=True).start()

    def record_loop(self):
        start_time = time.time()

        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio input warning: {status}")

            with self.record_lock:
                if self.is_recording:
                    self.recorded_chunks.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=WHISPER_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=1024,
                latency="low",
                callback=callback,
            ):
                while True:
                    with self.record_lock:
                        still_recording = self.is_recording

                    if not still_recording:
                        break

                    if time.time() - start_time >= MAX_RECORD_SECONDS:
                        print("Maximum recording time reached.")
                        with self.record_lock:
                            self.is_recording = False
                        break

                    time.sleep(0.01)

        except Exception as error:
            print(f"Recording error: {error}")
            with self.record_lock:
                self.is_recording = False

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        return audio.astype(np.float32)

    def process_audio(self):
        try:
            audio = self.get_recorded_audio()
            audio = self.normalize_audio(audio)

            if audio is None or len(audio) < WHISPER_SAMPLE_RATE * 0.2:
                print("No useful audio captured.")
                return

            stt_start = time.perf_counter()
            user_text = self.transcribe_audio(audio)
            stt_ms = (time.perf_counter() - stt_start) * 1000

            if not user_text:
                print("Could not understand. Please try again.")
                return

            print(f"\nYou: {user_text}")

            llm_start = time.perf_counter()
            response = self.get_llm_response(user_text)
            llm_ms = (time.perf_counter() - llm_start) * 1000

            print(f"Tutor: {response}")

            tts_start = time.perf_counter()
            self.speak(response)
            tts_ms = (time.perf_counter() - tts_start) * 1000

            total_ms = stt_ms + llm_ms + tts_ms

            print(
                f"\nSTT: {stt_ms:.0f}ms | "
                f"LLM: {llm_ms:.0f}ms | "
                f"TTS queue: {tts_ms:.0f}ms | "
                f"Total before playback: {total_ms:.0f}ms"
            )
            print("-" * 50)

        except Exception as error:
            print(f"Processing error: {error}")

    def get_recorded_audio(self) -> Optional[np.ndarray]:
        with self.record_lock:
            if not self.recorded_chunks:
                return None

            chunks = list(self.recorded_chunks)
            self.recorded_chunks = []

        audio = np.concatenate(chunks, axis=0).flatten()
        return audio.astype(np.float32, copy=False)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        assert self.whisper_model is not None

        segments, _info = self.whisper_model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=3,
            best_of=3,
            vad_filter=True,
            vad_parameters={
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
            },
            no_speech_threshold=0.45,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
            initial_prompt=(
                "This is a conversation with an English tutor. "
                "The speaker is practicing simple English phrases."
            ),
        )
        
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return self.clean_transcript(text)

    def clean_transcript(self, text: str) -> str:
        if not text:
            return ""

        # Keep post-processing minimal. Heavy corrections belong to the tutor model.
        text = " ".join(text.split())

        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        return text

    def get_llm_response(self, user_text: str) -> str:
        assert self.llm_client is not None

        self.messages.append({"role": "user", "content": user_text})
        self.trim_history(max_pairs=4)

        try:
            response = self.llm_client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=self.messages,
                temperature=0.3,
                max_tokens=50,
                top_p=0.9,
                stream=False,
                timeout=4.0,
            )

            reply = (response.choices[0].message.content or "").strip()

            if not reply:
                reply = "Sorry, I did not get a response."

            self.messages.append({"role": "assistant", "content": reply})
            self.trim_history(max_pairs=4)

            return reply

        except Exception as error:
            print(f"LLM error: {error}")
            return "Sorry, please try again."

    def trim_history(self, max_pairs: int):
        system_message = self.messages[0]
        conversation = self.messages[1:]

        # Keep the latest complete user/assistant turns as much as possible.
        max_messages = max_pairs * 2
        conversation = conversation[-max_messages:]

        self.messages = [system_message] + conversation

    def speak(self, text: str):
        if not text:
            return

        self.clear_tts_queue()
        self.tts_stop_event.clear()
        self.tts_queue.put(text)

    def stop_current_tts(self):
        self.tts_stop_event.set()
        self.clear_tts_queue()
        sd.stop()

    def clear_tts_queue(self):
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break

    def process_tts_queue(self):
        while not self.shutdown_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.play_tts(text)
            finally:
                self.tts_queue.task_done()

    def play_tts(self, text: str):
        assert self.pipeline is not None
        assert self.kokoro_model is not None

        try:
            generator = self.pipeline(
                text,
                voice=KOKORO_VOICE,
                model=self.kokoro_model,
            )

            chunks = []

            for _, _, audio in generator:
                if self.tts_stop_event.is_set():
                    return

                chunks.append(np.asarray(audio, dtype=np.float32))

            if not chunks or self.tts_stop_event.is_set():
                return

            audio_np = np.concatenate(chunks)
            sd.play(audio_np, samplerate=KOKORO_SAMPLE_RATE)
            sd.wait()

        except Exception as error:
            print(f"TTS error: {error}")

    def shutdown(self):
        if self.shutdown_event.is_set():
            return

        print("\nShutting down...")

        self.shutdown_event.set()

        with self.record_lock:
            self.is_recording = False

        self.stop_current_tts()
        keyboard.unhook_all()

        print("Goodbye.")


if __name__ == "__main__":
    tutor = VoiceTutor()
    tutor.start()