import os
import warnings
import numpy as np
import pyaudio
import sounddevice as sd
import time
import threading
import queue
from collections import deque
from functools import lru_cache

# AI Libraries
from faster_whisper import WhisperModel
from kokoro import KModel, KPipeline
from openai import OpenAI
from typing import cast, Iterable
from openai.types.chat import ChatCompletionChunk

# Keyboard library for global hotkeys
import keyboard

# Voice Activity Detection
import webrtcvad

# GPU detection
import torch

# === CONFIGURATION ===
sd.default.latency = "low"

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

# 1. LM Studio API Settings
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

# 2. Model Settings - OPTIMIZED for speed
WHISPER_MODEL = "tiny"

# Auto-detect best device
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    print("🚀 Using GPU (CUDA) for faster processing")
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    print("💻 Using CPU (consider GPU for better performance)")

# 3. Kokoro TTS Settings
KOKORO_VOICE = "af_heart"

# 4. System Prompt
SYSTEM_PROMPT = """
You are a helpful language tutor. 
Your goal is to help the user practice English.
Be encouraging, correct mistakes politely, and keep the conversation engaging.
Always keep responses concise (max 3-4 sentences).
"""

class VoiceTutor:
    def __init__(self):
        print("\n" + "="*50)
        print("🎙️  VOICE TUTOR - Optimized Version")
        print("="*50 + "\n")
        
        # Initialize components
        self.load_whisper()
        self.load_kokoro()
        self.init_llm_client()
        
        # Conversation History
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "System ready. Hold SPACEBAR to speak."}
        ]
        
        # Audio Recorder Setup
        self.audio = pyaudio.PyAudio()
        
        # VAD Setup
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        
        # Audio settings for VAD (must be 10, 20 or 30ms frames)
        self.sample_rate = 16000
        self.frame_duration_ms = 30  # 30ms frames (works best)
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)  # 480 samples at 16kHz
        self.frame_bytes = self.frame_size * 2  # 2 bytes per sample (int16)
        
        # State
        self.is_recording = False
        self.frames = []
        
        # Pre-buffer for capturing beginning of speech
        self.pre_buffer_seconds = 1.0
        self.pre_buffer_max_frames = int(self.pre_buffer_seconds * self.sample_rate / self.frame_size)
        self.pre_buffer = deque(maxlen=self.pre_buffer_max_frames)
        
        # TTS Queue for async playback
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self.process_tts_queue, daemon=True)
        self.tts_thread.start()
        
        # Statistics for profiling
        self.stats = {
            "stt_times": deque(maxlen=10),
            "llm_times": deque(maxlen=10),
            "tts_times": deque(maxlen=10),
        }
        
        # Warmup models
        self.warmup_models()
        
        # Start Keyboard Listener
        print("\n🎤 Listening for 'SPACE' key...")
        print("   Hold SPACE to record, Release to send")
        print("   Press 'ESC' to quit\n")
        
        keyboard.on_press_key('space', self.on_space_press)
        keyboard.on_release_key('space', self.on_space_release)
        keyboard.on_press_key('esc', lambda x: self.shutdown())
        
        # Keep thread alive
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.shutdown()
    
    def load_whisper(self):
        """Load Whisper STT model with progress indicator"""
        print("📥 Loading Whisper STT model...", end=" ", flush=True)
        start = time.time()
        self.whisper_model = WhisperModel(
            WHISPER_MODEL, 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE,
            cpu_threads=4,
            num_workers=1
        )
        print(f"✓ ({time.time()-start:.1f}s)")
    
    def load_kokoro(self):
        """Load Kokoro TTS model"""
        print("📥 Loading Kokoro TTS model...", end=" ", flush=True)
        start = time.time()
        self.kokoro_model = KModel(repo_id="hexgrad/Kokoro-82M").to("cpu")
        self.pipeline = KPipeline(lang_code="a")
        print(f"✓ ({time.time()-start:.1f}s)")
    
    def init_llm_client(self):
        """Initialize LM Studio client"""
        print("🔗 Connecting to LM Studio...", end=" ", flush=True)
        self.llm_client = OpenAI(
            base_url=LM_STUDIO_URL,
            api_key=LM_STUDIO_API_KEY,
            timeout=30.0
        )
        print("✓")
    
    def warmup_models(self):
        """Warm up models to avoid cold-start delays"""
        print("🔥 Warming up models...", end=" ", flush=True)
        
        # Warmup Whisper with silence
        dummy_audio = np.zeros(16000, dtype=np.float32)
        _ = self.whisper_model.transcribe(dummy_audio, language="en")
        
        # Warmup Kokoro
        _ = list(self.pipeline("Hello", voice=KOKORO_VOICE, model=self.kokoro_model))
        
        print("✓\n")
    
    def on_space_press(self, event):
        """Handle spacebar press - start recording with pre-buffer"""
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.pre_buffer.clear()
            print("\n[🎤 Listening...] ", end="", flush=True)
            
            # Start recording in separate thread
            threading.Thread(target=self.record_loop, daemon=True).start()
    
    def on_space_release(self, event):
        """Handle spacebar release - process audio"""
        if self.is_recording:
            self.is_recording = False
            print("\n⏳ Processing...", flush=True)
            
            threading.Thread(target=self.process_audio, daemon=True).start()
    
    def record_loop(self):
        """Record audio with pre-buffering to capture beginning of speech"""
        stream = None
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size
            )
            
            voice_active = False
            silence_frames = 0
            max_silence_frames = int(0.5 / (self.frame_duration_ms / 1000))  # 0.5 seconds of silence
            
            # Buffer for accumulating audio for VAD
            audio_buffer = b''
            
            while self.is_recording:
                # Read exactly one frame
                data = stream.read(self.frame_size, exception_on_overflow=False)
                
                # Always keep pre-buffer
                self.pre_buffer.append(data)
                
                # Accumulate for VAD (VAD needs exact frame sizes)
                audio_buffer += data
                
                # Process complete VAD frames
                while len(audio_buffer) >= self.frame_bytes:
                    # Extract one VAD frame
                    vad_frame = audio_buffer[:self.frame_bytes]
                    audio_buffer = audio_buffer[self.frame_bytes:]
                    
                    # Check for speech
                    is_speech = self.vad.is_speech(vad_frame, self.sample_rate)
                    
                    if is_speech:
                        if not voice_active:
                            # Speech started - include pre-buffer
                            voice_active = True
                            print("🔴 Recording...", end="", flush=True)
                            # Add all pre-buffer frames to recording
                            for pre_frame in self.pre_buffer:
                                self.frames.append(pre_frame)
                            self.frames.append(vad_frame)
                        else:
                            self.frames.append(vad_frame)
                        silence_frames = 0
                        
                    elif voice_active:
                        # Voice ended but keep recording a bit more
                        self.frames.append(vad_frame)
                        silence_frames += 1
                        
                        if silence_frames >= max_silence_frames:
                            print(" ⏸️", end="", flush=True)
                            self.is_recording = False
                            break
                            
        except OSError as e:
            print(f"\n❌ Microphone error: {e}")
            self.is_recording = False
        except Exception as e:
            print(f"\n❌ VAD Error: {e}")
            self.is_recording = False
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
    
    def process_audio(self):
        """Transcribe -> LLM -> Speak with profiling"""
        try:
            if not self.frames:
                print("No audio captured")
                return
            
            start_total = time.perf_counter()
            
            # Debug info
            total_samples = len(self.frames) * self.frame_size
            duration = total_samples / self.sample_rate
            print(f"📊 Audio: {duration:.1f} seconds, {len(self.frames)} frames")
            
            # 1. Transcribe
            stt_start = time.perf_counter()
            text = self.transcribe_audio()
            stt_time = (time.perf_counter() - stt_start) * 1000
            self.stats["stt_times"].append(stt_time)
            
            if not text:
                print("❓ Could not understand. Please try again.")
                return
            
            print(f"\n🎤 You: {text}")
            
            # 2. Get LLM Response
            llm_start = time.perf_counter()
            response_text = self.get_llm_response(text)
            llm_time = (time.perf_counter() - llm_start) * 1000
            self.stats["llm_times"].append(llm_time)
            
            if not response_text:
                return
            
            # 3. Speak (async via queue)
            tts_start = time.perf_counter()
            self.speak(response_text)
            tts_time = (time.perf_counter() - tts_start) * 1000
            self.stats["tts_times"].append(tts_time)
            
            # Print statistics
            total_time = (time.perf_counter() - start_total) * 1000
            self.print_stats(stt_time, llm_time, tts_time, total_time)
            
        except Exception as e:
            print(f"❌ Error processing: {e}")
            import traceback
            traceback.print_exc()
    
    def transcribe_audio(self):
        """Transcribe with optimized parameters and error correction"""
        if not self.frames:
            return ""
        
        # Convert audio data from list of bytes to numpy array
        audio_bytes = b''.join(self.frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Optimized transcription parameters
        segments, info = self.whisper_model.transcribe(
            audio_np,
            beam_size=1,
            best_of=1,
            language="en",
            vad_filter=False,  # Disable VAD filter since we already have VAD
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
        )
        
        text = "".join(segment.text for segment in segments).strip()
        
        # Post-process for common errors
        text = self.correct_common_errors(text)
        
        return text
    
    def correct_common_errors(self, text: str) -> str:
        """Fast correction of common speech recognition errors"""
        corrections = {
            "i is": "I am",
            "you is": "you are",
            "we is": "we are",
            "they is": "they are",
            "a apple": "an apple",
            "a hour": "an hour",
            "an banana": "a banana",
            "an car": "a car",
            "the a": "the",
            "he want": "he wants",
            "she want": "she wants",
            "it want": "it wants",
            "could you": "Could you",
            "can you": "Can you",
            "would you": "Would you",
        }
        
        result = text
        for wrong, correct in corrections.items():
            result = result.replace(wrong, correct)
        
        # Capitalize first letter if needed
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        
        return result
    
    def get_llm_response(self, user_text: str) -> str:
        """Get response from LLM with caching for common queries"""
        
        # Fast cache for common phrases
        cache = {
            "hello": "Hello! How can I help you practice English today?",
            "hi": "Hi there! Ready to practice some English?",
            "hey": "Hey! What would you like to learn today?",
            "thank you": "You're welcome! Keep up the great work!",
            "thanks": "You're welcome! Practice makes perfect!",
            "goodbye": "Goodbye! Come back anytime to practice more!",
            "bye": "Bye! Keep studying and you'll improve quickly!",
        }
        
        user_lower = user_text.lower().strip()
        if user_lower in cache:
            response = cache[user_lower]
            print(f"🤖 Tutor: {response}")
            self.messages.append({"role": "user", "content": user_text})
            self.messages.append({"role": "assistant", "content": response})
            return response
        
        # Dynamic response for longer text
        self.messages.append({"role": "user", "content": user_text})
        
        # Keep only last 10 messages for context
        self.messages = [self.messages[0]] + self.messages[-10:]
        
        # Adaptive max_tokens based on query length
        word_count = len(user_text.split())
        max_tokens = 40 if word_count < 10 else 60
        
        try:
            stream = cast(
                Iterable[ChatCompletionChunk],
                self.llm_client.chat.completions.create(
                    model="local-model",
                    messages=self.messages,
                    temperature=0.5,
                    max_tokens=max_tokens,
                    stream=True,
                    timeout=15.0,
                )
            )
            
            chunks: list[str] = []
            print("🤖 Tutor: ", end="", flush=True)
            
            for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    chunks.append(delta)
            
            print()
            
            reply = "".join(chunks).strip()
            if reply:
                self.messages.append({"role": "assistant", "content": reply})
            return reply
            
        except Exception as e:
            print(f"\n❌ LLM Error: {e}")
            return "Sorry, I'm having trouble responding. Please try again."
    
    def speak(self, text: str):
        """Queue TTS for non-blocking playback"""
        if text:
            self.tts_queue.put(text)
    
    def process_tts_queue(self):
        """Process TTS requests in background"""
        while True:
            text = self.tts_queue.get()
            self._speak_immediate(text)
            self.tts_queue.task_done()
    
    @lru_cache(maxsize=50)
    def _get_cached_tts(self, text: str):
        """Cache TTS audio for frequent phrases"""
        generator = self.pipeline(text, voice=KOKORO_VOICE, model=self.kokoro_model)
        chunks = []
        for _, _, audio in generator:
            chunks.append(np.asarray(audio, dtype=np.float32))
        return np.concatenate(chunks) if chunks else None
    
    def _speak_immediate(self, text: str):
        """Actually play TTS audio"""
        try:
            # Try to get from cache
            audio_np = self._get_cached_tts(text)
            
            if audio_np is None:
                return
            
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=24000,
                output=True,
                frames_per_buffer=1024
            )
            
            try:
                stream.write(audio_np.tobytes())
            finally:
                stream.stop_stream()
                stream.close()
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")
    
    def print_stats(self, stt_time: float, llm_time: float, tts_time: float, total_time: float):
        """Print performance statistics"""
        avg_stt = np.mean(self.stats["stt_times"]) if self.stats["stt_times"] else stt_time
        avg_llm = np.mean(self.stats["llm_times"]) if self.stats["llm_times"] else llm_time
        avg_tts = np.mean(self.stats["tts_times"]) if self.stats["tts_times"] else tts_time
        
        print(f"\n📊 Latency: STT={stt_time:.0f}ms | LLM={llm_time:.0f}ms | TTS={tts_time:.0f}ms | TOTAL={total_time:.0f}ms")
        print(f"   Avg: STT={avg_stt:.0f}ms | LLM={avg_llm:.0f}ms | TTS={avg_tts:.0f}ms")
        print("-" * 50)
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n\n🛑 Shutting down Voice Tutor...")
        keyboard.unhook_all()
        self.audio.terminate()
        print("✓ Goodbye! 👋")
        os._exit(0)

def check_dependencies():
    """Check if all required libraries are installed"""
    missing = []
    
    try:
        import faster_whisper
    except ImportError:
        missing.append("faster-whisper")
    
    try:
        import kokoro
    except ImportError:
        missing.append("kokoro")
    
    try:
        import webrtcvad
    except ImportError:
        missing.append("webrtcvad")
    
    try:
        import keyboard
    except ImportError:
        missing.append("keyboard")
    
    if missing:
        print("❌ Missing dependencies. Install with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════╗
    ║     🎙️  VOICE TUTOR - MVP 2.0        ║
    ║    Optimized for Speed & Accuracy     ║
    ╚═══════════════════════════════════════╝
    """)
    
    if not check_dependencies():
        exit(1)
    
    # Check LM Studio connection
    print("📡 Checking LM Studio connection...")
    try:
        test_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY, timeout=5.0)
        test_client.models.list()
        print("✓ LM Studio is running and accessible\n")
    except Exception as e:
        print(f"⚠️  Warning: Cannot connect to LM Studio at {LM_STUDIO_URL}")
        print("   Please ensure LM Studio is running with the API server enabled\n")
    
    tutor = VoiceTutor()