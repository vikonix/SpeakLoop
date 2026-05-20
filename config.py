import os
import warnings
import torch

# =====================================================================
# Environment & Warning Adjustments
# =====================================================================
# Disable Hugging Face hub symlinks warning for a cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ignore specific deprecation and model warnings from underlying libraries
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

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

# =====================================================================
# Audio Hardware & Signal Processing
# =====================================================================
# Hardware interface and buffer processing parameters
AUDIO_BLOCKSIZE = 1024  # Small block sizes maintain responsive streaming frame intervals
AUDIO_LATENCY = "low"   # Optimizes underlying sound card capture profiles
AUDIO_CHANNELS = 1      # Mono recording/playback mode

# Device index overrides. Set to an integer ID if you want to bypass system defaults
AUDIO_INPUT_DEVICE = None   # None defaults to OS system default microphone
AUDIO_OUTPUT_DEVICE = None  # None defaults to OS system default speakers/headphones

# Signal gain normalization parameters
AUDIO_MIN_PEAK_THRESHOLD = 0.01      # Prevents boosting pure background noise floor during silence
AUDIO_NORMALIZATION_CEILING = 0.9    # Scales the peak target output level directly to 90%

# =====================================================================
# LLM (LM Studio) Settings
# =====================================================================
# Local server backend endpoint connections
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "local-model"

# Generation tuning variables controlling response creativity and lengths
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 50
LLM_TOP_P = 0.9
LLM_TIMEOUT = 5.0

# Context buffer constraints
LLM_HISTORY_MAX_PAIRS = 4  # Number of full conversation turns kept inside short-term memory

# =====================================================================
# Speech-to-Text (Whisper) Settings
# =====================================================================
# Speech transcription configuration parameters
WHISPER_MODEL = "small"
WHISPER_SAMPLE_RATE = 16_000   # Whisper architecture requires strict 16kHz audio layouts
WHISPER_BEAM_SIZE = 1         # Beam size 1 provides optimal inference speed at 0.0 temperature
WHISPER_NO_SPEECH_THRESHOLD = 0.45

# Voice Activity Detection (VAD) timeline definitions (in milliseconds)
WHISPER_VAD_MIN_SPEECH_MS = 250   # Shortest duration considered as valid spoken word segments
WHISPER_VAD_MIN_SILENCE_MS = 500  # Silence gap thickness required before triggering split boundaries
WHISPER_VAD_SPEECH_PAD_MS = 300   # Padding attached around text fragments to avoid chopping words

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
KOKORO_SAMPLE_RATE = 24_000   # Kokoro synthesizes native 24kHz audio outputs
KOKORO_WARMUP_WORD = "Hi."    # String used to pre-cache compilation layers on initialization

# =====================================================================
# Hardware Acceleration
# =====================================================================
# Runtime execution layout discovery mapping
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"  # Fast half-precision inference execution on CUDA GPUs
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"     # Quantized 8-bit inference execution conserving CPU efficiency
