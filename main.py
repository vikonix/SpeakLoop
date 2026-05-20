import os
import time
import queue
import threading
import warnings
import re
from typing import Optional

import numpy as np
import sounddevice as sd
import keyboard
import torch

from faster_whisper import WhisperModel
from kokoro import KModel, KPipeline
from openai import OpenAI

# =====================================================================
# Configuration & Environment Setup
# =====================================================================

# Disable Hugging Face hub symlinks warning for a cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ignore specific deprecation and model warnings from underlying libraries
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

# LM Studio local server configuration API parameters
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "local-model"

# Speech-to-Text (STT) model settings
WHISPER_MODEL = "base.en"
WHISPER_SAMPLE_RATE = 16_000  # Whisper expects 16kHz audio input

# Text-to-Speech (TTS) model settings
KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24_000  # Kokoro outputs 24kHz audio

# Safety threshold to prevent infinite recording if spacebar gets stuck
MAX_RECORD_SECONDS = 20

# System prompt shaping the LLM behavior into a specific persona
SYSTEM_PROMPT = (
    "You are an English tutor named Emma. "
    "Keep responses very short. "
    "Use simple spoken English. "
    "Avoid idioms, abbreviations, complex punctuation, and compressed phrases. "
    "Prefer short clear sentences."
)

# Hardware acceleration setup: Use CUDA (GPU) if available, otherwise fallback to CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"  # Fast FP16 inference for execution on GPU
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"  # Quantized INT8 execution for CPU efficiency


class VoiceTutor:
    def __init__(self):
        # Thread management events
        self.shutdown_event = threading.Event()  # Signals the main loop to terminate
        self.tts_stop_event = threading.Event()  # Interrupted to stop speech playback immediately

        # Recording state management
        self.is_recording = False
        self.record_lock = threading.Lock()  # Protects shared state between recording and main threads
        self.recorded_chunks: list[np.ndarray] = []  # Stores raw incoming audio buffers
        self.record_thread: Optional[threading.Thread] = None

        # Text-to-Speech background queue and thread
        self.tts_queue: queue.Queue[str] = queue.Queue()  # Holds text sentences waiting to be spoken
        self.tts_thread: Optional[threading.Thread] = None

        # Chat history buffer starting with the system instructions
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Model containers placeholder definitions
        self.whisper_model: Optional[WhisperModel] = None
        self.kokoro_model: Optional[KModel] = None
        self.pipeline: Optional[KPipeline] = None
        self.llm_client: Optional[OpenAI] = None

    def start(self):
        """
        Main entry point. Initializes resources, hooks keystrokes, and starts the runtime loop.
        """
        print("\n" + "=" * 50)
        print("Voice Tutor MVP")
        print("=" * 50)

        print(f"Audio backend: sounddevice")
        print(f"Device: {DEVICE}, compute type: {COMPUTE_TYPE}")

        # Step 1: Heavy initialization blocks (I/O & Model loading)
        self.load_models()
        self.check_lm_studio()
        self.warm_up()

        # Step 2: Spawn background worker thread to process sentences ready for synthesis
        self.tts_thread = threading.Thread(target=self.process_tts_queue, daemon=True)
        self.tts_thread.start()

        # Step 3: Register global keyboard hotkeys using the keyboard listener module
        keyboard.on_press_key("space", self.on_space_press)
        keyboard.on_release_key("space", self.on_space_release)
        keyboard.on_press_key("esc", lambda _: self.shutdown_event.set())  # Esc triggers exit event

        print("\nReady. Hold SPACE to speak. Press ESC to quit.\n")

        # Step 4: Keep main execution block active until shutdown event is flagged
        try:
            while not self.shutdown_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass  # Handle structural Ctrl+C exit safely
        finally:
            self.cleanup()  # Ensure resources are freed on termination

    def load_models(self):
        """
        Instantiates AI engines into memory.
        """
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
        # Automatically push Kokoro to GPU/CPU based on discovered system environment
        self.kokoro_model = KModel(repo_id="hexgrad/Kokoro-82M").to(DEVICE)
        self.pipeline = KPipeline(lang_code="a")  # "a" stands for American English voice codes
        print(f"done in {time.time() - start:.1f}s")

        # Configure OpenAI compatible client mapped to look at local LM Studio server instance
        self.llm_client = OpenAI(
            base_url=LM_STUDIO_URL,
            api_key=LM_STUDIO_API_KEY,
            timeout=5.0,
        )

    def check_lm_studio(self):
        """
        Validates connectivity to the local LLM server prior to running conversational workflows.
        """
        print("Checking LM Studio...", end=" ", flush=True)
        try:
            assert self.llm_client is not None
            self.llm_client.models.list()  # Fetch available model listing to test active HTTP endpoint
            print("ready")
        except Exception as error:
            print(f"not available: {error}")
            print("Warning: Please ensure LM Studio is running before initiating voice chat loops.")

    def warm_up(self):
        """
        Runs a mock inference pass through models to compile/cache layers and eliminate initial latency.
        """
        print("Warming up models...", end=" ", flush=True)
        assert self.whisper_model is not None
        assert self.pipeline is not None
        assert self.kokoro_model is not None

        # Warm up Whisper with 1 second of structural silence array
        dummy_audio = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)
        list(self.whisper_model.transcribe(dummy_audio, language="en", beam_size=1, vad_filter=True)[0])

        # Warm up Kokoro synthesis network with a simple token phrase
        list(self.pipeline("Hi.", voice=KOKORO_VOICE, model=self.kokoro_model))
        print("done")

    def on_space_press(self, _event):
        """
        Callback handler executed when the spacebar hotkey is pressed down.
        """
        with self.record_lock:
            if self.is_recording:
                return  # Guard against system keyboard key-repeat firing duplicate calls

            # Disrupt current active playback immediately when starting a new turn
            self.stop_current_tts()
            self.is_recording = True
            self.recorded_chunks = []

        print("\nRecording...", flush=True)

        # Launch non-blocking background thread to capture raw microphone hardware output buffers
        self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
        self.record_thread.start()

    def on_space_release(self, _event):
        """
        Callback handler executed when the spacebar hotkey is released.
        """
        with self.record_lock:
            if not self.is_recording:
                return
            self.is_recording = False  # Tells the record loop worker thread to wind down execution

        # Gracefully await the structural recording thread block closure
        if self.record_thread:
            self.record_thread.join(timeout=1.5)

        print("Processing...", flush=True)
        # Handle heavy transcription, generation and audio streaming sequentially in a separate background context
        threading.Thread(target=self.process_audio, daemon=True).start()

    def record_loop(self):
        """
        Opens input audio stream and aggregates continuous floating-point sample windows.
        """
        start_time = time.time()

        def callback(indata, frames, time_info, status):
            """Internal sounddevice stream processing context callback."""
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
                    blocksize=1024,  # Small block sizes maintain responsive frame intervals
                    latency="low",  # Optimizes underlying sound card capture configuration
                    callback=callback,
            ):
                while True:
                    with self.record_lock:
                        still_recording = self.is_recording

                    if not still_recording:
                        break

                    # Enforce cutoff limit window constraints
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
        """
        Normalizes global maximum audio volume to a stable target gain level.
        Prevents downstream Whisper degradation from over-amplification or structural low-volume input signals.
        """
        peak = np.max(np.abs(audio))
        if peak < 0.01:
            return audio.astype(np.float32)  # Skip near-silent signals to avoid boosting pure noise floor
        audio = audio / peak * 0.9  # Normalize target ceiling output directly to 90%
        return np.nan_to_num(audio).astype(np.float32)

    def process_audio(self):
        """
        Orchestrates pipeline execution sequence: STT -> LLM Stream Generation -> TTS Stream Playback.
        """
        try:
            audio = self.get_recorded_audio()

            # Prevent processing if total captured frame volume duration is less than 0.2 seconds
            if audio is None or len(audio) < WHISPER_SAMPLE_RATE * 0.2:
                print("No useful audio captured.")
                return

            audio = self.normalize_audio(audio)

            # Step 1: Run Speech-to-Text transcription via Whisper
            stt_start = time.perf_counter()
            user_text = self.transcribe_audio(audio)
            stt_ms = (time.perf_counter() - stt_start) * 1000

            if not user_text:
                print("Could not understand. Please try again.")
                return

            print(f"\nYou: {user_text}")
            print("Tutor: ", end="", flush=True)

            # Step 2 & 3: Stream text from LLM and send completed sentences to the TTS queue concurrently
            llm_start = time.perf_counter()
            full_response = self.stream_llm_and_queue_tts(user_text)
            llm_ms = (time.perf_counter() - llm_start) * 1000

            print(f"\nSTT: {stt_ms:.0f}ms | LLM (Total Stream Time): {llm_ms:.0f}ms")
            print("-" * 50)

        except Exception as error:
            print(f"\nProcessing error: {error}")

    def get_recorded_audio(self) -> Optional[np.ndarray]:
        """
        Safely extracts and joins captured chunk segments from the list.
        """
        with self.record_lock:
            if not self.recorded_chunks:
                return None
            chunks = list(self.recorded_chunks)
            self.recorded_chunks = []

        # Merge multidimensional window chunks into one uniform contiguous array track
        audio = np.concatenate(chunks, axis=0).flatten()
        return audio.astype(np.float32, copy=False)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """
        Passes the audio waveform data into Whisper for text extraction.
        """
        assert self.whisper_model is not None

        # Note: beam_size=1 is faster than higher values and perfectly suited for temperature 0.0
        segments, _info = self.whisper_model.transcribe(
            audio,
            language="en",
            task="transcribe",
            beam_size=1,
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
        """Removes trailing structural whitespace and capitalizes the starting character."""
        if not text:
            return ""
        text = " ".join(text.split())
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    def stream_llm_and_queue_tts(self, user_text: str) -> str:
        """
        Streams generated text tokens from the LLM, parses sentences using regex on-the-fly,
        and pushes completed text strings to the TTS worker pipeline for near-zero latency.
        """
        assert self.llm_client is not None

        self.messages.append({"role": "user", "content": user_text})
        self.trim_history(max_pairs=4)

        self.clear_tts_queue()
        self.tts_stop_event.clear()

        try:
            # Call the chat endpoint with streaming enabled
            stream_response = self.llm_client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=self.messages,
                temperature=0.3,
                max_tokens=50,
                top_p=0.9,
                stream=True,  # Crucial flag enabling sequential chunk delivery
                timeout=5.0,
            )

            full_reply = ""
            sentence_buffer = ""

            # Regex tracking common punctuation (. ! ?) followed by spaces to detect completed sentences
            sentence_end = re.compile(r'(?<=[.!?])\s+')

            for chunk in stream_response:
                if self.tts_stop_event.is_set():
                    break

                token = chunk.choices[0].delta.content or ""
                if not token:
                    continue

                print(token, end="", flush=True)  # Print token fragments real-time to standard output console
                full_reply += token
                sentence_buffer += token

                # Parse the working token sequence accumulator to check if a sentence has concluded
                parts = sentence_end.split(sentence_buffer)
                if len(parts) > 1:
                    # Enqueue completed sentences, retaining the last element in case it is cut off mid-word
                    for i in range(len(parts) - 1):
                        text_to_speak = parts[i].strip()
                        if text_to_speak:
                            self.tts_queue.put(text_to_speak)
                    sentence_buffer = parts[-1]

            # Flush any residual text remaining inside the buffer stream loop context
            remaining_text = sentence_buffer.strip()
            if remaining_text and not self.tts_stop_event.is_set():
                self.tts_queue.put(remaining_text)

            final_reply = full_reply.strip() if full_reply.strip() else "Sorry, I did not get a response."

            self.messages.append({"role": "assistant", "content": final_reply})
            self.trim_history(max_pairs=4)

            return final_reply

        except Exception as error:
            print(f"\nLLM Stream error: {error}")
            return "Sorry, please try again."

    def trim_history(self, max_pairs: int):
        """Prunes conversation state structure list sizes to conserve system context window bounds."""
        system_message = self.messages[0]
        conversation = self.messages[1:]
        max_messages = max_pairs * 2
        conversation = conversation[-max_messages:]
        self.messages = [system_message] + conversation

    def stop_current_tts(self):
        """Flags stop notifications, clears text queues, and interrupts active sound card play states."""
        self.tts_stop_event.set()
        self.clear_tts_queue()
        sd.stop()

    def clear_tts_queue(self):
        """Empties all pending sentences inside the synchronized thread queue container."""
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break

    def process_tts_queue(self):
        """
        Background execution worker loop reading items from the string queue and calling synthesis methods.
        """
        while not self.shutdown_event.is_set():
            try:
                text = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if not self.tts_stop_event.is_set():
                    self.play_tts_stream(text)
            except Exception as e:
                print(f"Error in TTS queue thread: {e}")
            finally:
                self.tts_queue.task_done()

    def play_tts_stream(self, text: str):
        """
        Streams audio frames generated by Kokoro directly to sounddevice in chunks.
        Audio starts playing as soon as the first sub-phrase chunk finishes synthesizing.
        """
        assert self.pipeline is not None
        assert self.kokoro_model is not None

        try:
            text = " ".join(text.split())
            if not text:
                return

            # Generator yielding raw audio frames sentence-by-sentence or phrase-by-phrase
            generator = self.pipeline(
                text,
                voice=KOKORO_VOICE,
                model=self.kokoro_model,
            )

            # Open a hardware audio output stream with ultra-low latency configurations
            with sd.OutputStream(
                    samplerate=KOKORO_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=1024,
                    latency="low",
            ) as stream:

                for _, _, audio in generator:
                    # Break loop execution instantly if user initiates a new recording or exits
                    if self.tts_stop_event.is_set() or self.shutdown_event.is_set():
                        return

                    audio_chunk = np.asarray(audio, dtype=np.float32)
                    if audio_chunk.ndim == 1:
                        audio_chunk = audio_chunk.reshape(-1, 1)  # Reshape to expected (samples, channels) structure

                    # Write raw synthesized float array blocks to audio hardware frame channels
                    stream.write(audio_chunk)

        except Exception as error:
            print(f"\nTTS Stream Play error: {error}")

    def cleanup(self):
        """
        Releases global system hooks and resources safely during programmatic shutdowns.
        """
        print("\nShutting down and cleaning up resources...")
        with self.record_lock:
            self.is_recording = False

        self.stop_current_tts()

        try:
            keyboard.unhook_all()  # Clean keyboard listener hooks to prevent lingering driver blocks
        except Exception:
            pass

        print("Goodbye.")


if __name__ == "__main__":
    tutor = VoiceTutor()
    tutor.start()