import time
import queue
import threading
from typing import Optional
import numpy as np
import sounddevice as sd
import keyboard

import config
from stt import STTManager
from llm import LLMManager
from tts import TTSManager

class VoiceTutorApp:
    def __init__(self):
        # Thread management events
        self.shutdown_event = threading.Event()
        self.tts_stop_event = threading.Event()

        # Recording state management
        self.is_recording = False
        self.record_lock = threading.Lock()
        self.recorded_chunks: list[np.ndarray] = []
        self.record_thread: Optional[threading.Thread] = None

        # Text-to-Speech background queue and thread
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.tts_thread: Optional[threading.Thread] = None

        # Initialize core modular sub-managers
        self.stt_mgr = STTManager()
        self.llm_mgr = LLMManager()
        self.tts_mgr = TTSManager()

    def start(self):
        print("\n" + "=" * 50)
        print(f"Voice Tutor MVP ({config.NATIVE_LANGUAGE} -> {config.TARGET_LANGUAGE})")
        print("=" * 50)
        print(f"Audio backend: sounddevice")
        print(f"Device: {config.DEVICE}, compute type: {config.COMPUTE_TYPE}")

        # Core initialization blocks
        print("Loading components...")
        self.stt_mgr.load_model()
        self.tts_mgr.load_model()
        self.llm_mgr.init_client()
        
        if not self.llm_mgr.check_connection():
            print("Warning: Please ensure LM Studio is running before initiating voice chat loops.")

        print("Warming up models...", end=" ", flush=True)
        self.stt_mgr.warm_up()
        self.tts_mgr.warm_up()
        print("done")

        # Spawn background worker thread to process text-to-speech queue
        self.tts_thread = threading.Thread(target=self.process_tts_queue, daemon=True)
        self.tts_thread.start()

        # Register global hotkeys
        keyboard.on_press_key("space", self.on_space_press)
        keyboard.on_release_key("space", self.on_space_release)
        keyboard.on_press_key("esc", lambda _: self.shutdown_event.set())

        print("\nReady. Hold SPACE to speak. Press ESC to quit.\n")

        try:
            while not self.shutdown_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def on_space_press(self, _event):
        with self.record_lock:
            if self.is_recording:
                return  # Key-repeat guard

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
            self.record_thread.join(timeout=1.5)

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
                    samplerate=config.WHISPER_SAMPLE_RATE,
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

                    if time.time() - start_time >= config.MAX_RECORD_SECONDS:
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
        if peak < 0.01:
            return audio.astype(np.float32)
        audio = audio / peak * 0.9
        return np.nan_to_num(audio).astype(np.float32)

    def process_audio(self):
        try:
            audio = self.get_recorded_audio()
            if audio is None or len(audio) < config.WHISPER_SAMPLE_RATE * 0.2:
                print("No useful audio captured.")
                return

            audio = self.normalize_audio(audio)

            # Step 1: Speech-to-Text
            stt_start = time.perf_counter()
            user_text = self.stt_mgr.transcribe(audio)
            stt_ms = (time.perf_counter() - stt_start) * 1000

            if not user_text:
                print("Could not understand. Please try again.")
                return

            print(f"\nYou: {user_text}")
            print("Tutor: ", end="", flush=True)

            # Step 2 & 3: Run LLM streaming and feeding TTS queue concurrently
            llm_start = time.perf_counter()
            self.clear_tts_queue()
            self.tts_stop_event.clear()
            
            _full_response = self.llm_mgr.stream_and_queue_tts(user_text, self.tts_queue, self.tts_stop_event)
            llm_ms = (time.perf_counter() - llm_start) * 1000

            print(f"\nSTT: {stt_ms:.0f}ms | LLM (Total Stream Time): {llm_ms:.0f}ms")
            print("-" * 50)

        except Exception as error:
            print(f"\nProcessing error: {error}")

    def get_recorded_audio(self) -> Optional[np.ndarray]:
        with self.record_lock:
            if not self.recorded_chunks:
                return None
            chunks = list(self.recorded_chunks)
            self.recorded_chunks = []
        return np.concatenate(chunks, axis=0).flatten().astype(np.float32, copy=False)

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
                if not self.tts_stop_event.is_set():
                    self.tts_mgr.play_stream(text, self.tts_stop_event, self.shutdown_event)
            except Exception as e:
                print(f"Error in TTS queue thread: {e}")
            finally:
                self.tts_queue.task_done()

    def cleanup(self):
        print("\nShutting down and cleaning up resources...")
        with self.record_lock:
            self.is_recording = False
        self.stop_current_tts()
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        print("Goodbye.")

if __name__ == "__main__":
    app = VoiceTutorApp()
    app.start()
    