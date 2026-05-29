import time
import queue
import threading
from typing import Optional
import os
import sys
import warnings
import logging
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import sounddevice as sd

# Disable Hugging Face hub symlinks warning for a cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ignore specific deprecation and model warnings from underlying libraries
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

import config
from stt import STTManager, COMPUTE_TYPE, WHISPER_SAMPLE_RATE
from llm import LLMManager
from tts import TTSManager

# Configure comprehensive events logging (console + file)
log_format = "%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Technical recording & signal processing parameters
AUDIO_BLOCKSIZE = 1024  # Small block sizes maintain responsive streaming frame intervals
AUDIO_LATENCY = "low"   # Optimizes underlying sound card capture profiles
AUDIO_CHANNELS = 1      # Mono recording/playback mode
AUDIO_INPUT_DEVICE = None   # None defaults to OS system default microphone

# Signal gain normalization parameters
AUDIO_MIN_PEAK_THRESHOLD = 0.01      # Prevents boosting pure background noise floor during silence
AUDIO_NORMALIZATION_CEILING = 0.9    # Scales the peak target output level directly to 90%

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

        # Text-to-Speech background queue and thread
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.tts_thread: Optional[threading.Thread] = None

        # Initialize core modular sub-managers
        self.stt_mgr = STTManager()
        self.llm_mgr = LLMManager()
        self.tts_mgr = TTSManager()

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
            logging.error(f"Error during initialization thread: {e}")
            self.root.after(0, self.append_system_msg, f"Initialization Error: {e}")
            self.root.after(0, self.update_status, "Initialization Failed", "#ff5555")

    def make_app_ready(self):
        self.draw_mic_button("idle")
        self.update_status("Ready", "#00e676")
        self.update_instruction(f"Hold SPACE or click Button to speak. Press ESC to quit.")
        self.append_system_msg(f"Voice Tutor ready. Practice learning {config.TARGET_LANGUAGE}!")

    def on_gui_btn_press(self):
        # Click behavior (simulates holding space)
        if not self.space_is_held:
            logging.info("GUI microphone button clicked.")
            self.trigger_recording_start()

    def on_gui_btn_release(self):
        if self.is_recording and not self.space_is_held:
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
            
            self.draw_mic_button("recording")
            self.update_status("Recording...", "#ff5555")
            self.update_instruction("Release key or click button when finished speaking.")
            
            self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
            self.record_thread.start()

    def trigger_recording_stop(self):
        with self.record_lock:
            if not self.is_recording:
                return
            logging.info("Stopping audio recording...")
            self.is_recording = False

        self.draw_mic_button("processing")
        self.update_status("Processing Speech (STT)...", "#ffb86c")
        
        if self.record_thread:
            self.record_thread.join(timeout=1.5)

        threading.Thread(target=self.process_audio, daemon=True).start()

    def record_loop(self):
        start_time = time.time()
        logging.info("sd.InputStream thread started.")

        def callback(indata, frames, time_info, status):
            if status:
                logging.warning(f"Audio input warning: {status}")
            with self.record_lock:
                if self.is_recording:
                    self.recorded_chunks.append(indata.copy())

        try:
            with sd.InputStream(
                    samplerate=WHISPER_SAMPLE_RATE,
                    channels=AUDIO_CHANNELS,
                    dtype="float32",
                    blocksize=AUDIO_BLOCKSIZE,
                    latency=AUDIO_LATENCY,
                    device=AUDIO_INPUT_DEVICE,
                    callback=callback,
            ):
                while True:
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

        except Exception as error:
            logging.error(f"Recording InputStream error: {error}")
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
            
            _full_response = self.llm_mgr.stream_and_queue_tts(
                user_text, 
                self.tts_queue, 
                self.tts_stop_event, 
                token_callback=token_cb
            )
            
            llm_ms = (time.perf_counter() - llm_start) * 1000
            logging.info(f"LLM complete streaming and queuing. Duration: {llm_ms:.0f}ms")

            self.root.after(0, self.append_emma_end)
            self.root.after(0, self.update_stats, stt_ms, llm_ms)
            
            self.root.after(0, self.draw_mic_button, "speaking")
            self.root.after(0, self.update_status, "Emma is speaking...", "#ff79c6")

        except Exception as error:
            logging.error(f"Error in process_audio: {error}")
            self.root.after(0, self.append_system_msg, f"Processing Error: {error}")
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
                # If we are done speaking and no new items in queue, transition status back to ready
                if not self.is_recording and not self.tts_stop_event.is_set() and self.tts_queue.empty():
                    # Only change status if current state was 'speaking'
                    if self.status_label.cget("text") == "Status: Emma is speaking...":
                        self.root.after(0, self.draw_mic_button, "idle")
                        self.root.after(0, self.update_status, "Ready", "#00e676")
                        self.root.after(0, self.update_instruction, "Hold SPACE or click Button to speak.")
                continue

            try:
                if not self.tts_stop_event.is_set():
                    logging.info(f"TTS playing synthesized block: {text!r}")
                    self.tts_mgr.play_stream(text, self.tts_stop_event, self.shutdown_event)
            except Exception as e:
                logging.error(f"Error in TTS queue thread: {e}")
            finally:
                self.tts_queue.task_done()

    def quit_app(self):
        logging.info("Shutting down VoiceTutor App...")
        self.shutdown_event.set()
        self.stop_current_tts()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = VoiceTutorGUI()
    app.run()