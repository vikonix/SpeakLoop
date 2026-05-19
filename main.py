import os
import warnings
import numpy as np
import pyaudio
import sounddevice as sd
import time
import threading
import queue
from collections import deque

# AI Libraries
from faster_whisper import WhisperModel
from kokoro import KModel, KPipeline
from openai import OpenAI
from typing import cast, Iterable
from openai.types.chat import ChatCompletionChunk

# Keyboard library for global hotkeys
import keyboard

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

# 2. Model Settings
WHISPER_MODEL = "tiny"

# Auto-detect best device
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    print("🚀 Using GPU (CUDA)")
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    print("💻 Using CPU")

# 3. Kokoro TTS Settings
KOKORO_VOICE = "af_heart"

# 4. System Prompt
SYSTEM_PROMPT = """You are an English tutor. Keep responses very short (1-2 sentences). Be encouraging."""

class VoiceTutor:
    def __init__(self):
        print("\n" + "="*50)
        print("🎙️  VOICE TUTOR - MVP")
        print("="*50 + "\n")
        
        # Load Whisper STT
        print("📥 Loading Whisper...", end=" ", flush=True)
        start = time.time()
        self.whisper_model = WhisperModel(
            WHISPER_MODEL, 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE,
            cpu_threads=4,
            num_workers=1
        )
        print(f"✓ ({time.time()-start:.1f}s)")
        
        # Load Kokoro TTS
        print("📥 Loading Kokoro TTS...", end=" ", flush=True)
        start = time.time()
        self.kokoro_model = KModel(repo_id="hexgrad/Kokoro-82M").to("cpu")
        self.pipeline = KPipeline(lang_code="a")
        print(f"✓ ({time.time()-start:.1f}s)")
        
        # Connect to LM Studio
        print("🔗 Connecting to LM Studio...", end=" ", flush=True)
        self.llm_client = OpenAI(
            base_url=LM_STUDIO_URL,
            api_key=LM_STUDIO_API_KEY,
            timeout=5.0
        )
        print("✓")
        
        # Conversation history
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 16000
        
        # Recording state
        self.is_recording = False
        self.frames = []
        self.record_thread = None
        
        # TTS Queue for async playback
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self.process_tts_queue, daemon=True)
        self.tts_thread.start()
        
        # Warm up models
        print("🔥 Warming up models...", end=" ", flush=True)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        _ = self.whisper_model.transcribe(dummy_audio, language="en")
        _ = list(self.pipeline("Hi", voice=KOKORO_VOICE, model=self.kokoro_model))
        print("✓\n")
        
        # Start keyboard listener
        print("🎤 READY! Hold SPACE to speak, ESC to quit\n")
        keyboard.on_press_key('space', self.on_space_press)
        keyboard.on_release_key('space', self.on_space_release)
        keyboard.on_press_key('esc', lambda x: self.shutdown())
        
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.shutdown()
    
    def on_space_press(self, event):
        """Start recording when spacebar is pressed"""
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            print("\n[🎤 Recording...] ", end="", flush=True)
            self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
            self.record_thread.start()
    
    def on_space_release(self, event):
        """Stop recording and process when spacebar is released"""
        if self.is_recording:
            self.is_recording = False
            print("\n⏳ Processing...", flush=True)
            threading.Thread(target=self.process_audio, daemon=True).start()
    
    def record_loop(self):
        """Simple recording without VAD - just record everything"""
        stream = None
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
            
            while self.is_recording:
                data = stream.read(1024, exception_on_overflow=False)
                self.frames.append(data)
                
        except Exception as e:
            print(f"\n❌ Recording error: {e}")
            self.is_recording = False
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
    
    def process_audio(self):
        """Main pipeline: STT -> LLM -> TTS"""
        try:
            if not self.frames:
                print("No audio captured")
                return
            
            # 1. Speech to Text
            stt_start = time.perf_counter()
            text = self.transcribe_audio()
            stt_time = (time.perf_counter() - stt_start) * 1000
            
            if not text:
                print("❓ Could not understand. Please try again.")
                return
            
            print(f"\n🎤 You: {text}")
            
            # 2. LLM Response
            llm_start = time.perf_counter()
            response = self.get_llm_response(text)
            llm_time = (time.perf_counter() - llm_start) * 1000
            
            print(f"🤖 Tutor: {response}")
            
            # 3. Text to Speech
            tts_start = time.perf_counter()
            self.speak(response)
            tts_time = (time.perf_counter() - tts_start) * 1000
            
            # Statistics
            total_time = stt_time + llm_time + tts_time
            print(f"\n📊 STT: {stt_time:.0f}ms | LLM: {llm_time:.0f}ms | TTS: {tts_time:.0f}ms | TOTAL: {total_time:.0f}ms")
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    def transcribe_audio(self):
        """Transcribe using Whisper"""
        if not self.frames:
            return ""
        
        # Convert bytes to numpy array
        audio_bytes = b''.join(self.frames)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Let Whisper handle VAD internally (better quality)
        segments, info = self.whisper_model.transcribe(
            audio_np,
            beam_size=1,
            best_of=1,
            language="en",
            vad_filter=True,              # Let Whisper do VAD
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
        )
        
        # Combine text
        text = "".join(segment.text for segment in segments).strip()
        
        # Post-processing
        text = self.fix_common_errors(text)
        
        return text
    
    def fix_common_errors(self, text: str) -> str:
        """Fast correction of common errors"""
        corrections = {
            "i is": "I am",
            "you is": "you are", 
            "he like": "he likes",
            "she like": "she likes",
            "i'll leave": "I'll leave",
            "could you": "Could you",
            "can you": "Can you",
        }
        
        result = text
        for wrong, correct in corrections.items():
            result = result.replace(wrong, correct)
        
        # Capitalize first letter
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        
        return result
    
    def get_llm_response(self, user_text: str) -> str:
        """Get response from LLM"""
        
        self.messages.append({"role": "user", "content": user_text})
        
        # Keep last 5 exchanges for context
        if len(self.messages) > 6:
            self.messages = [self.messages[0]] + self.messages[-5:]
        
        try:
            response = self.llm_client.chat.completions.create(
                model="local-model",
                messages=self.messages,
                temperature=0.3,
                max_tokens=40,
                top_p=0.9,
                stream=False,
                timeout=3.0,
            )
            
            reply = response.choices[0].message.content.strip()
            self.messages.append({"role": "assistant", "content": reply})
            
            return reply
            
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            return "Sorry, please try again."
    
    def speak(self, text: str):
        """Queue TTS for playback"""
        if text:
            self.tts_queue.put(text)
    
    def process_tts_queue(self):
        """Background TTS playback"""
        while True:
            text = self.tts_queue.get()
            self._speak_immediate(text)
            self.tts_queue.task_done()
    
    def _speak_immediate(self, text: str):
        """Play TTS audio"""
        try:
            generator = self.pipeline(text, voice=KOKORO_VOICE, model=self.kokoro_model)
            
            chunks = []
            for _, _, audio in generator:
                chunks.append(np.asarray(audio, dtype=np.float32))
            
            if not chunks:
                return
            
            audio_np = np.concatenate(chunks)
            
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
            print(f"❌ TTS error: {e}")
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n\n🛑 Shutting down...")
        keyboard.unhook_all()
        self.audio.terminate()
        print("✓ Goodbye! 👋")
        os._exit(0)

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════╗
    ║     🎙️  VOICE TUTOR - SIMPLE MVP     ║
    ║    Removed VAD, using Whisper VAD     ║
    ╚═══════════════════════════════════════╝
    """)
    
    # Check LM Studio
    print("📡 Checking LM Studio...")
    try:
        test_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LM_STUDIO_API_KEY, timeout=3.0)
        test_client.models.list()
        print("✓ LM Studio ready\n")
    except:
        print("⚠️  Make sure LM Studio is running\n")
    
    tutor = VoiceTutor()