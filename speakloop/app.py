# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""The application: the Tkinter window and the voice loop, plus run().

Started only through speakloop/cli.py, which calls bootstrap.early_init()
before this module is imported (UTF-8 console, warning filters) - that is why
nothing of the kind is done here. Logging is configured in run(), after the
heavy imports below, for the reason given in bootstrap.setup_logging.
"""

import time
import queue
import threading
from typing import Optional
import os
import logging
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import sounddevice as sd

# config first: it sets HF_HOME and the offline switch, which huggingface_hub
# reads when stt and tts import faster_whisper and kokoro below.
from speakloop import config
from speakloop import bootstrap, detect_hardware, lifecycle
from speakloop.stt import STTManager, WHISPER_SAMPLE_RATE
from speakloop.llm import LLMManager, error_message
from speakloop.llm_server_ctl import LLMServerController
from speakloop.tts import TTSManager

# Sentinel object pushed to the TTS queue after LLM finishes streaming.
# The TTS thread buffers sentences and only starts playback when it sees this object,
# ensuring the LLM has released the GPU before Kokoro synthesis begins.
_TTS_START_SENTINEL = object()

# Technical recording & signal processing parameters
RECORDING_BLOCKSIZE = 1024  # Small block sizes maintain responsive streaming frame intervals

# Signal gain normalization parameters
AUDIO_MIN_PEAK_THRESHOLD = 0.01      # Prevents boosting pure background noise floor during silence
AUDIO_NORMALIZATION_CEILING = 0.9    # Scales the peak target output level directly to 90%

# How long to wait for the recording thread to finish after stopping.
# Covers the last InputStream callback flush; should be well under 1 second in normal use.
RECORD_THREAD_JOIN_TIMEOUT_SEC = 1.5


class VoiceTutorGUI:
    def __init__(self):
        logging.info("Starting Voice Tutor GUI Application...")

        # Core Tkinter setup
        self.root = tk.Tk()
        self.root.title("Emma - Voice Tutor")
        self.root.geometry("500x700")
        self.root.configure(bg="#121214")

        # Thread management events
        self.shutdown_event = threading.Event()
        self.tts_stop_event = threading.Event()

        # Recording state management
        self.is_recording = False
        self.space_is_held = False
        self.record_lock = threading.Lock()
        self.recorded_chunks: list[np.ndarray] = []
        self.record_thread: Optional[threading.Thread] = None

        # Audio processing guard - prevents concurrent process_audio() calls
        self.is_processing_audio = False
        self.processing_lock = threading.Lock()

        # TTS state tracking - avoids reading Tkinter widget from background thread
        self._tts_is_speaking = False
        self.tts_state_lock = threading.Lock()

        # Text-to-Speech background queue and thread
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.tts_thread: Optional[threading.Thread] = None

        # Initialize core modular sub-managers
        self.stt_mgr = STTManager()
        self.tts_mgr = TTSManager()

        # LLM backend. config validates the name and falls back to the default,
        # so only the two known values reach this point.
        self.llm_backend = config.LLM_BACKEND
        logging.info(f"Using the {self.llm_backend} LLM backend.")
        # One client for both backends: they speak the same OpenAI API and only
        # the address differs, which init_client is told at connection time.
        self.llm_mgr = LLMManager()
        # Owns the llama-server subprocess. Built for every backend and left
        # untouched by "lm-studio": all of its methods are no-ops until start().
        self._llm_server = LLMServerController()

        # Setup custom dark styles for UI elements
        self.setup_styles()
        # Build UI layout
        self.build_ui()
        # Bind keyboard events locally
        self.bind_events()

        # Start loading models in a background thread to prevent UI freezing
        threading.Thread(target=self.load_components, daemon=True).start()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Configure scrollbar styling
        self.style.configure("Vertical.TScrollbar",
                             gripcount=0,
                             background="#1a1a1e",
                             troughcolor="#121214",
                             bordercolor="#121214",
                             arrowcolor="#8a2be2")

    def build_ui(self):
        # 1. Header Area (Top)
        header_frame = tk.Frame(self.root, bg="#121214", height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        title_label = tk.Label(header_frame, text="EMMA • Voice Tutor", font=("Segoe UI", 16, "bold"), fg="#8a2be2", bg="#121214")
        title_label.pack(side=tk.LEFT)

        lang_label = tk.Label(header_frame,
                             text=f"{config.NATIVE_LANGUAGE} ➔ {config.TARGET_LANGUAGE}",
                             font=("Segoe UI", 9, "bold"),
                             fg="#a0a0a5",
                             bg="#1a1a1e",
                             padx=10,
                             pady=4,
                             bd=0)
        lang_label.pack(side=tk.RIGHT)

        # 2. Status & Stats Bar (Absolute Bottom)
        self.status_bar = tk.Frame(self.root, bg="#1a1a1e", height=30)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(
            self.status_bar,
            text="Status: Starting...",
            font=("Segoe UI", 9),
            fg="#00e676",
            bg="#1a1a1e"
        )
        self.status_label.pack(side=tk.LEFT, padx=15, pady=4)

        self.stats_label = tk.Label(
            self.status_bar,
            text="STT: --ms | LLM: --ms",
            font=("Segoe UI", 9),
            fg="#a0a0a5",
            bg="#1a1a1e"
        )
        self.stats_label.pack(side=tk.RIGHT, padx=15, pady=4)

        # 3. Bottom Control Panel (Above Status Bar)
        control_frame = tk.Frame(self.root, bg="#121214")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

        # Interactive Canvas Button (Pulsing Mic)
        self.btn_canvas = tk.Canvas(control_frame, width=100, height=100, bg="#121214", highlightthickness=0, cursor="hand2")
        self.btn_canvas.pack(pady=5)
        self.btn_canvas.bind("<ButtonPress-1>", lambda e: self.on_gui_btn_press())
        self.btn_canvas.bind("<ButtonRelease-1>", lambda e: self.on_gui_btn_release())

        self.draw_mic_button("loading")

        # Instruction Text
        self.instruction_label = tk.Label(
            control_frame,
            text="Loading components...",
            font=("Segoe UI", 10),
            fg="#a0a0a5",
            bg="#121214"
        )
        self.instruction_label.pack(pady=5)

        # 4. Chat Transcript Area (Middle - takes up all remaining space)
        chat_frame = tk.Frame(self.root, bg="#121214")
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=5)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            bg="#1a1a1e",
            fg="#f8f8f2",
            insertbackground="#ffffff",
            font=("Segoe UI", 11),
            wrap=tk.WORD,
            bd=0,
            highlightthickness=1,
            highlightbackground="#25252a",
            highlightcolor="#8a2be2",
            padx=15,
            pady=15,
            spacing2=6,
            spacing3=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.configure(state=tk.DISABLED)

        # Define text styles/tags for the chat window
        self.chat_display.tag_configure("user", foreground="#8be9fd", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_configure("emma", foreground="#ff79c6", font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_configure("system", foreground="#6272a4", font=("Segoe UI", 10, "italic"))
        self.chat_display.tag_configure("text_user", foreground="#ffffff", font=("Segoe UI", 11))
        self.chat_display.tag_configure("text_emma", foreground="#f1f1f6", font=("Segoe UI", 11))

    def draw_mic_button(self, state):
        self.btn_canvas.delete("all")

        # Center coordinates
        cx, cy = 50, 50
        r_outer, r_inner = 42, 34

        if state == "loading":
            bg_color = "#1e1e24"
            outline_color = "#44475a"
            emoji = "⌛"
        elif state == "idle":
            bg_color = "#1f1430"
            outline_color = "#8a2be2"
            emoji = "🎤"
        elif state == "recording":
            bg_color = "#3a0c10"
            outline_color = "#ff5555"
            emoji = "🔴"
        elif state == "processing":
            bg_color = "#36220f"
            outline_color = "#ffb86c"
            emoji = "⚡"
        elif state == "speaking":
            bg_color = "#0f2c1d"
            outline_color = "#50fa7b"
            emoji = "🔊"
        else:
            bg_color = "#1e1e24"
            outline_color = "#44475a"
            emoji = "🎤"

        # Draw outer glow circle
        self.btn_canvas.create_oval(cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer, fill="", outline=outline_color, width=3)
        # Draw solid inner circle
        self.btn_canvas.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner, fill=bg_color, outline="")
        # Render Emoji inside
        self.btn_canvas.create_text(cx, cy, text=emoji, font=("Segoe UI", 20), fill="#ffffff")

    def bind_events(self):
        # Keyboard Push-to-Talk bindings
        self.root.bind("<KeyPress-space>", self.on_keyboard_press)
        self.root.bind("<KeyRelease-space>", self.on_keyboard_release)

        # Escape bindings to shut down gracefully
        self.root.bind("<Escape>", lambda _: self.quit_app())
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def append_system_msg(self, text: str):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"[System] {text}\n", "system")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_user_msg(self, text: str):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "You: ", "user")
        self.chat_display.insert(tk.END, f"{text}\n", "text_user")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_emma_start(self):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "Emma: ", "emma")
        # Keep track of where Emma's streamed response starts
        self.emma_start_index = self.chat_display.index(tk.INSERT)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_emma_token(self, token: str):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, token, "text_emma")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_emma_end(self):
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def update_status(self, text: str, color: str = "#a0a0a5"):
        self.status_label.configure(text=f"Status: {text}", fg=color)

    def update_instruction(self, text: str):
        self.instruction_label.configure(text=text)

    def update_stats(self, stt_ms: float, llm_ms: float):
        self.stats_label.configure(text=f"STT: {stt_ms:.0f}ms | LLM: {llm_ms:.0f}ms")

    def load_components(self):
        logging.info("Starting model loading thread...")
        self.root.after(0, self.update_status, "Loading models...", "#ffb86c")
        self.root.after(0, self.append_system_msg, "Loading Whisper (STT) and Kokoro (TTS) models...")

        try:
            self.stt_mgr.load_model()
            logging.info("STT Model loaded successfully.")

            self.tts_mgr.load_model()
            logging.info("TTS Model loaded successfully.")

            if self.llm_backend == "llama-server":
                model_name = os.path.basename(config.EXTERNAL_MODEL_PATH)
                self.root.after(0, self.append_system_msg, f"Starting llama-server with {model_name}...")
                self.root.after(0, self.update_status, "Starting LLM server...", "#ffb86c")
                ready = self._llm_server.start(self.llm_mgr)
                if not ready:
                    self.root.after(0, self.append_system_msg, "Error: LLM server failed to start. Check logs/llm_server.log, the model path and GPU memory.")
                    self.root.after(0, self.update_status, "LLM Server Error", "#ff5555")
                    self.root.after(0, self.update_instruction, "LLM server failed to start. Check the log and restart.")
                    # Do not call make_app_ready - keep the button in loading/disabled state
                    return
                self.root.after(0, self.append_system_msg, "LLM server is ready.")
            else:
                self.llm_mgr.init_client()
                if not self.llm_mgr.check_connection():
                    self.root.after(0, self.append_system_msg, "Warning: LM Studio is offline. Start it to use voice tutor!")
                    logging.warning("LM Studio is offline during initialization.")

            self.root.after(0, self.update_status, "Warming up models...", "#ffb86c")
            self.stt_mgr.warm_up()
            self.tts_mgr.warm_up()
            logging.info("Models warmed up successfully.")

            # Start TTS background thread
            self.tts_thread = threading.Thread(target=self.process_tts_queue, daemon=True)
            self.tts_thread.start()
            logging.info("TTS Queue processor thread started.")

            # Make App Ready
            self.root.after(0, self.make_app_ready)
            logging.info("Voice Tutor initialization fully completed.")

        except Exception as e:
            logging.exception("Error during initialization thread:")
            self.root.after(0, self.append_system_msg, f"Initialization Error: {e}")
            self.root.after(0, self.update_status, "Initialization Failed", "#ff5555")

    def make_app_ready(self):
        with self.tts_state_lock:
            self._tts_is_speaking = False
        self.draw_mic_button("idle")
        self.update_status("Ready", "#00e676")
        self.update_instruction("Hold SPACE or click Button to speak. Press ESC to quit.")
        self.append_system_msg(f"Voice Tutor ready. Practice learning {config.TARGET_LANGUAGE}!")

    def on_gui_btn_press(self):
        # Click behavior (simulates holding space)
        if not self.space_is_held:
            logging.info("GUI microphone button clicked.")
            self.trigger_recording_start()

    def on_gui_btn_release(self):
        with self.record_lock:
            currently_recording = self.is_recording
        if currently_recording and not self.space_is_held:
            logging.info("GUI microphone button released.")
            self.trigger_recording_stop()

    def on_keyboard_press(self, event):
        if event.keysym == "space" and not self.space_is_held:
            self.space_is_held = True
            logging.info("Spacebar keyboard press event.")
            self.trigger_recording_start()

    def on_keyboard_release(self, event):
        if event.keysym == "space" and self.space_is_held:
            self.space_is_held = False
            logging.info("Spacebar keyboard release event.")
            self.trigger_recording_stop()

    def trigger_recording_start(self):
        with self.record_lock:
            if self.is_recording:
                return  # Safety guard

            logging.info("Starting audio recording...")
            self.stop_current_tts()
            self.is_recording = True
            self.recorded_chunks = []

            # All GUI updates scheduled on main thread via root.after
            self.root.after(0, self.draw_mic_button, "recording")
            self.root.after(0, self.update_status, "Recording...", "#ff5555")
            self.root.after(0, self.update_instruction, "Release key or click button when finished speaking.")

            self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
            self.record_thread.start()

    def trigger_recording_stop(self):
        with self.record_lock:
            if not self.is_recording:
                return
            logging.info("Stopping audio recording...")
            self.is_recording = False

        self.root.after(0, self.draw_mic_button, "processing")
        self.root.after(0, self.update_status, "Processing Speech (STT)...", "#ffb86c")

        # Join and audio processing happen off the main thread to prevent UI freeze.
        # record_thread.join() can block up to RECORD_THREAD_JOIN_TIMEOUT_SEC -
        # running it on the main thread would make the window unresponsive.
        threading.Thread(target=self._finalize_recording, daemon=True).start()

    def _finalize_recording(self):
        """Joins the record thread, then starts audio processing - runs off the main thread."""
        if self.record_thread:
            self.record_thread.join(timeout=RECORD_THREAD_JOIN_TIMEOUT_SEC)

        with self.processing_lock:
            if self.is_processing_audio:
                logging.warning("process_audio already running, skipping duplicate.")
                return
            self.is_processing_audio = True

        self._process_audio_safe()

    def _process_audio_safe(self):
        """Wrapper that ensures process_audio runs exactly once and releases the guard."""
        try:
            self.process_audio()
        finally:
            with self.processing_lock:
                self.is_processing_audio = False

    def record_loop(self):
        start_time = time.time()
        logging.info("sd.InputStream thread started.")

        # Warnings from the realtime callback are buffered here and logged from
        # the main record loop - calling logging directly inside a sounddevice
        # callback can block on I/O and cause audio dropouts.
        callback_warnings: list[str] = []

        def callback(indata, frames, time_info, status):
            if status:
                callback_warnings.append(str(status))
            with self.record_lock:
                if self.is_recording:
                    self.recorded_chunks.append(indata.copy())

        try:
            with config.AUDIO_LOCK:
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception as init_err:
                    logging.debug(f"PortAudio reinitialization error: {init_err}")

                stream = sd.InputStream(
                        samplerate=WHISPER_SAMPLE_RATE,
                        channels=config.AUDIO_CHANNELS,
                        dtype="float32",
                        blocksize=RECORDING_BLOCKSIZE,
                        latency=config.AUDIO_LATENCY,
                        device=config.AUDIO_INPUT_DEVICE,
                        callback=callback,
                )
                stream.start()

            try:
                while True:
                    # Flush warnings accumulated by the realtime callback
                    while callback_warnings:
                        logging.warning(f"Audio input warning: {callback_warnings.pop(0)}")

                    with self.record_lock:
                        still_recording = self.is_recording

                    if not still_recording:
                        break

                    if time.time() - start_time >= config.MAX_RECORD_SECONDS:
                        logging.info("Maximum recording duration reached.")
                        self.root.after(0, self.append_system_msg, "Reached maximum record limit.")
                        with self.record_lock:
                            self.is_recording = False
                        break

                    time.sleep(0.01)
            finally:
                with config.AUDIO_LOCK:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception as close_error:
                        logging.debug(f"Error during sound input stream close: {close_error}")

        except Exception:
            logging.exception("Recording InputStream error:")
            with self.record_lock:
                self.is_recording = False
            self.root.after(0, self.update_status, "Recording Error", "#ff5555")

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        logging.info(f"Normalizing audio. Peak signal level: {peak:.4f}")
        if peak < AUDIO_MIN_PEAK_THRESHOLD:
            logging.info("Peak signal is too low (silence). Skipping gain adjustment.")
            return audio.astype(np.float32)
        audio = audio / peak * AUDIO_NORMALIZATION_CEILING
        return np.nan_to_num(audio).astype(np.float32)

    def process_audio(self):
        try:
            audio = self.get_recorded_audio()
            if audio is None or len(audio) < WHISPER_SAMPLE_RATE * 0.2:
                logging.warning("Captured audio too short or empty.")
                self.root.after(0, self.append_system_msg, "Audio is too short. Try holding space longer.")
                self.root.after(0, self.draw_mic_button, "idle")
                self.root.after(0, self.update_status, "Ready", "#00e676")
                self.root.after(0, self.update_instruction, "Hold SPACE or click Button to speak.")
                return

            audio = self.normalize_audio(audio)

            # Speech-to-Text (STT)
            stt_start = time.perf_counter()
            user_text = self.stt_mgr.transcribe(audio)
            stt_ms = (time.perf_counter() - stt_start) * 1000
            logging.info(f"STT transcribed speech: {user_text!r} | Latency: {stt_ms:.0f}ms")

            if not user_text:
                logging.info("STT returned empty transcription.")
                self.root.after(0, self.append_system_msg, "Could not hear you clearly. Please try again.")
                self.root.after(0, self.draw_mic_button, "idle")
                self.root.after(0, self.update_status, "Ready", "#00e676")
                self.root.after(0, self.update_instruction, "Hold SPACE or click Button to speak.")
                return

            # Update User Speech to GUI
            self.root.after(0, self.append_user_msg, user_text)
            self.root.after(0, self.update_status, "Thinking (LLM)...", "#8be9fd")

            # Start LLM stream feeding the TTS queue
            llm_start = time.perf_counter()
            self.clear_tts_queue()
            self.tts_stop_event.clear()

            self.root.after(0, self.append_emma_start)

            # Streaming callback to append tokens live
            def token_cb(token):
                self.root.after(0, self.append_emma_token, token)

            try:
                self.llm_mgr.stream_and_queue_tts(
                    user_text,
                    self.tts_queue,
                    self.tts_stop_event,
                    token_callback=token_cb
                )
            except Exception as llm_error:
                # Handled here and not by the outer handler, which cannot know
                # that a reply line is already open in the chat. Without this
                # the window keeps an empty "Emma:" line and a status bar that
                # still says "Thinking", and the failure is only in the log.
                # llm.py has logged the traceback already.
                self.root.after(0, self.append_emma_end)
                self.root.after(0, self.append_system_msg,
                                f"LLM error: {error_message(llm_error)}")
                self._abort_tts()
                self.root.after(0, self.draw_mic_button, "idle")
                self.root.after(0, self.update_status, "LLM Error", "#ff5555")
                self.root.after(0, self.update_instruction, "Hold SPACE or click Button to speak.")
                return

            llm_ms = (time.perf_counter() - llm_start) * 1000
            logging.info(f"LLM complete streaming and queuing. Duration: {llm_ms:.0f}ms")

            self.root.after(0, self.append_emma_end)
            self.root.after(0, self.update_stats, stt_ms, llm_ms)

            # Signal TTS thread that LLM has finished and GPU is free.
            # The TTS thread buffers sentences until it receives this sentinel,
            # preventing GPU contention between the model server and Kokoro.
            if not self.tts_stop_event.is_set():
                self.tts_queue.put(_TTS_START_SENTINEL)
                with self.tts_state_lock:
                    self._tts_is_speaking = True
                self.root.after(0, self.draw_mic_button, "speaking")
                self.root.after(0, self.update_status, "Emma is speaking...", "#ff79c6")

        except Exception:
            logging.exception("Error in process_audio:")
            self.root.after(0, self.append_system_msg, "Processing Error. Please try again.")
            self.root.after(0, self.draw_mic_button, "idle")
            self.root.after(0, self.update_status, "Error", "#ff5555")

    def get_recorded_audio(self) -> Optional[np.ndarray]:
        with self.record_lock:
            if not self.recorded_chunks:
                return None
            chunks = list(self.recorded_chunks)
            self.recorded_chunks = []
        return np.concatenate(chunks, axis=0).flatten().astype(np.float32, copy=False)

    def stop_current_tts(self):
        logging.info("Stopping active text-to-speech output...")
        with self.tts_state_lock:
            self._tts_is_speaking = False
        self.tts_stop_event.set()
        self.clear_tts_queue()
        self.tts_mgr.stop_playback()

    def _abort_tts(self):
        """Drop everything queued for speech after a failed exchange.

        stop_current_tts() empties the queue, but sentences the TTS thread has
        already taken from it are buffered inside that thread and are dropped
        only when it sees another item. The sentinel is that item: with the
        stop event set it clears the buffer and plays nothing. Without it those
        sentences would be spoken after the NEXT reply.
        """
        self.stop_current_tts()
        self.tts_queue.put(_TTS_START_SENTINEL)

    def clear_tts_queue(self):
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break

    def process_tts_queue(self):
        # Sentences are buffered here while LLM is still running on the GPU.
        # Playback starts only after _TTS_START_SENTINEL arrives (LLM done, GPU free).
        pending_sentences: list[str] = []

        while not self.shutdown_event.is_set():
            try:
                item = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                # Transition to idle only when truly done: sentinel was received (pending
                # is empty) and there are no more items waiting in the queue.
                if not pending_sentences:
                    with self.record_lock:
                        currently_recording = self.is_recording
                    if not currently_recording and not self.tts_stop_event.is_set() and self.tts_queue.empty():
                        with self.tts_state_lock:
                            tts_speaking = self._tts_is_speaking
                        if tts_speaking:
                            self.root.after(0, self.draw_mic_button, "idle")
                            self.root.after(0, self.update_status, "Ready", "#00e676")
                            self.root.after(0, self.update_instruction, "Hold SPACE or click Button to speak.")
                            with self.tts_state_lock:
                                self._tts_is_speaking = False
                continue

            try:
                if item is _TTS_START_SENTINEL:
                    # LLM has finished - GPU is now free. Play all buffered sentences.
                    logging.info(f"TTS sentinel received. Playing {len(pending_sentences)} buffered sentence(s).")
                    remaining = list(pending_sentences)
                    pending_sentences.clear()
                    for sentence in remaining:
                        if self.tts_stop_event.is_set() or self.shutdown_event.is_set():
                            break
                        try:
                            logging.info(f"TTS playing synthesized block: {sentence!r}")
                            self.tts_mgr.play_stream(sentence, self.tts_stop_event, self.shutdown_event)
                        except Exception as play_err:
                            # Skip the failed sentence and continue with the rest
                            logging.exception(f"TTS playback error for {sentence!r}:")
                elif self.tts_stop_event.is_set():
                    # Stop was requested - discard buffered sentences and this one
                    pending_sentences.clear()
                else:
                    # LLM still running - buffer the sentence, do not synthesize yet
                    logging.info(f"TTS buffering sentence (waiting for LLM): {item!r}")
                    pending_sentences.append(item)
            except Exception:
                logging.exception("Error in TTS queue thread:")
            finally:
                self.tts_queue.task_done()

    def quit_app(self):
        logging.info("Shutting down VoiceTutor App...")
        self.shutdown_event.set()
        self.stop_current_tts()

        # Terminate the llama-server subprocess if this app started one. A
        # no-op for the "lm-studio" backend and safe to reach while the loader
        # thread is still in start(): see LLMServerController.shutdown.
        self._llm_server.shutdown()

        self.root.destroy()
        # Not the interpreter's normal exit: with CUDA torch loaded, tearing the
        # CUDA context down during finalization can crash the process on
        # Windows (0xC0000409). hard_exit flushes the logs and ends the process
        # without that teardown. Everything that needs a clean release (the
        # server subprocess, its log file) is released above.
        lifecycle.hard_exit()

    def run(self):
        self.root.mainloop()


def run(append_log: bool = False) -> None:
    """Start the application: logging, the startup checks, then the window.

    The only thing speakloop/cli.py calls. ``append_log`` continues
    logs/main.log instead of truncating it (see bootstrap.setup_logging).
    """
    bootstrap.setup_logging(config.LOG_FILE, append=append_log)
    # One log line when an NVIDIA GPU is present but torch runs on the CPU:
    # the app works then, only several times slower, and nothing else says so.
    detect_hardware.warn_if_gpu_unused(config.DEVICE)
    app = VoiceTutorGUI()
    app.run()
