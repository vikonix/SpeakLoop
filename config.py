import torch

# =====================================================================
# Language Pair & Persona Configuration
# =====================================================================
# Define your language learning pair configuration
NATIVE_LANGUAGE = "Russian"
TARGET_LANGUAGE = "English"
TARGET_LANG_CODE = "en"  # ISO code used for Whisper transcription routing

# System prompt shaping the LLM behavior into a specific educational persona
SYSTEM_PROMPT = (
    f"You are a friendly {TARGET_LANGUAGE} tutor named Emma. "
    f"The user's native language is {NATIVE_LANGUAGE}, but you should talk to them in simple {TARGET_LANGUAGE}. "
    "Keep responses very short. Use simple spoken sentences. "
    "Avoid idioms, abbreviations, complex punctuation, and compressed phrases."
)

# =====================================================================
# Controls & Hotkeys
# =====================================================================
# Global keyboard triggers mapped via the keyboard listener module
HOTKEY_RECORD = "space"
HOTKEY_QUIT = "esc"

# Safety threshold to prevent infinite recording loops if a key gets physically stuck
MAX_RECORD_SECONDS = 20

# Hardware Acceleration setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# LLM Backend Settings
# =====================================================================
# Backend selection:
#   "lm-studio"    — external LM Studio app (must be running separately)
#   "local_server" — llm_server.py started automatically as a subprocess
#   "local_gguf"   — llama_cpp loaded directly in-process (legacy, GPU contention risk)
LLM_BACKEND = "local_server"

# =====================================================================
# LM Studio backend (for "lm-studio" backend)
# =====================================================================
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "local-model"

# =====================================================================
# Local LLM Server (for "local_server" backend)
# =====================================================================
LOCAL_SERVER_HOST = "127.0.0.1"
LOCAL_SERVER_PORT = 8765
LOCAL_SERVER_URL = f"http://{LOCAL_SERVER_HOST}:{LOCAL_SERVER_PORT}/v1"
LOCAL_SERVER_API_KEY = "local"
LOCAL_SERVER_MODEL = "local-model"

# How long (seconds) to wait for the server to become ready after launching
LOCAL_SERVER_STARTUP_TIMEOUT = 60

# =====================================================================
# GGUF Model Settings (shared by "local_server" and "local_gguf" backends)
# =====================================================================
EXTERNAL_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"  # Path to your GGUF model file
EXTERNAL_N_GPU_LAYERS = 20  # Number of layers to offload to GPU (10, 20, 25...)
EXTERNAL_N_CTX = 2048       # Context window size

# Generation tuning variables controlling response creativity and lengths (used by both backends)
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 50
LLM_TOP_P = 0.9

# Context buffer constraints (used by both backends)
LLM_HISTORY_MAX_PAIRS = 4  # Number of full conversation turns kept inside short-term memory

# =====================================================================
# Speech-to-Text (Whisper) Settings
# =====================================================================
# Speech transcription configuration parameters
WHISPER_MODEL = "small"
WHISPER_BEAM_SIZE = 1         # Beam size 1 provides optimal inference speed at 0.0 temperature
WHISPER_NO_SPEECH_THRESHOLD = 0.45

# Context conditioning instruction guiding spelling logic styles
WHISPER_INITIAL_PROMPT = (
    f"This is a conversation with a {TARGET_LANGUAGE} tutor. "
    f"The speaker is practicing simple phrases."
)

# =====================================================================
# Text-to-Speech (Kokoro) Settings
# =====================================================================
# Voice synthesis operational parameters
KOKORO_LANG_CODE = "a"        # 'a' stands for American English voice pipelines, 'b' for British
KOKORO_VOICE = "af_heart"     # Chosen voice model matrix file

# =====================================================================
# Logging Settings
# =====================================================================
LOG_FILE = "voice_tutor.log"

