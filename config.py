import os
import warnings
import torch

# Disable Hugging Face hub symlinks warning for a cleaner console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ignore specific deprecation and model warnings from underlying libraries
warnings.filterwarnings("ignore", message="dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

# =====================================================================
# Language Pair Configuration
# =====================================================================
# Define your language learning pair
NATIVE_LANGUAGE = "Russian"
TARGET_LANGUAGE = "English"
TARGET_LANG_CODE = "en"  # Used for Whisper transcription filtering

# =====================================================================
# Model & Server Parameters
# =====================================================================
# LM Studio local server configuration API parameters
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL = "local-model"

# Speech-to-Text (STT) model settings
WHISPER_MODEL = "small"
WHISPER_SAMPLE_RATE = 16_000  # Whisper expects 16kHz audio input

# Text-to-Speech (TTS) model settings
# Note: Kokoro pipeline lang_code 'a' = American English, 'b' = British English
KOKORO_LANG_CODE = "a" 
KOKORO_VOICE = "af_heart"
KOKORO_SAMPLE_RATE = 24_000  # Kokoro outputs 24kHz audio

# =====================================================================
# UI & Hardware Settings
# =====================================================================
# Safety threshold to prevent infinite recording if spacebar gets stuck
MAX_RECORD_SECONDS = 20

# System prompt shaping the LLM behavior into a specific persona
SYSTEM_PROMPT = (
    f"You are a friendly {TARGET_LANGUAGE} tutor named Emma. "
    f"The user's native language is {NATIVE_LANGUAGE}, but you should talk to them in simple {TARGET_LANGUAGE}. "
    "Keep responses very short. "
    "Use simple spoken sentences. "
    "Avoid idioms, abbreviations, complex punctuation, and compressed phrases."
)

# Hardware acceleration setup: Use CUDA (GPU) if available, otherwise fallback to CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"  # Fast FP16 inference for execution on GPU
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"  # Quantized INT8 execution for CPU efficiency
	