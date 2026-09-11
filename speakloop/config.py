# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""All configuration of the application, frozen once at import.

Values are layered, lowest priority first:

  1. built-in defaults - the literals in this file;
  2. config/hardware_config.json, "config" section - machine-derived values
     written by `python -m speakloop.detect_hardware`;
  3. config/settings.json - hand-edited user preferences.

A missing or broken file leaves the lower layers in effect: problems are
reported to stderr instead of crashing startup, because both files are
optional and settings.json is edited by hand.

This module also prepares the Hugging Face environment (HF_HOME and the offline
switch). Those variables are read when huggingface_hub is imported, so this
module has to be imported BEFORE faster_whisper and kokoro - app.py imports it
first. For the same reason the fetchers and the hardware probe must never
import it.
"""

import os
import sys
import threading
from functools import partial
from pathlib import Path

from speakloop import loader, model_fetch, models_info, paths

# What this machine writes (settings, downloads, logs). In a clone this is the
# project directory; installed as a package it is the OS user-data directory.
BASE_DIR = paths.data_root()
CONFIG_DIR = paths.config_dir()

# Before the first read below: in package mode none of these directories exists
# on a first run. Cheap and idempotent, so it runs unconditionally.
paths.ensure_dirs()

# =====================================================================
# Settings files (optional overrides)
# =====================================================================

# Machine-derived overrides. Only the "config" section is consumed here; the
# "hardware" section is diagnostics for humans.
_HW = loader.read_json(CONFIG_DIR / "hardware_config.json").get("config")
if not isinstance(_HW, dict):
    _HW = {}

# User preferences. Keys starting with "_" are skipped so they can serve as
# comments (plain JSON has no comment syntax).
_USER = loader.read_json(CONFIG_DIR / "settings.json")

# Every key settings.json may hold. A key outside this set is reported, because
# a typo in a hand-edited file otherwise changes nothing and says nothing.
_KNOWN_USER_KEYS = {
    "max_record_seconds",
}
for _key in _USER:
    if not _key.startswith("_") and _key not in _KNOWN_USER_KEYS:
        print(f"[config] settings.json: unknown key {_key!r} ignored",
              file=sys.stderr)

# Validated accessor for numeric settings.json values (reports and falls back
# instead of raising).
_num = partial(loader.user_number, _USER)

# =====================================================================
# Local model cache (Hugging Face) - download once, then load offline
# =====================================================================
# faster-whisper and Kokoro load their weights through huggingface_hub, which
# reads these variables at IMPORT time. setdefault(), so an externally set
# HF_HOME wins. The path comes from model_fetch, the module that downloads into
# it, so the app and the installer cannot look in different places.
MODEL_CACHE_DIR = model_fetch.MODEL_CACHE_DIR
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

# Offline gate. Once every repo THIS run loads is cached, the Hub is switched
# off and every start loads straight from disk with no network request. The set
# is all-or-nothing, so it must name only what the run really loads: a repo the
# run never touches would keep the Hub online for every model.
_CACHED_REPOS = (
    models_info.WHISPER_SMALL.repo_id,
    models_info.KOKORO.repo_id,
)
if loader.models_cached(Path(os.environ["HF_HOME"]) / "hub", _CACHED_REPOS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
else:
    # Something is missing, so this run downloads it through the libraries.
    # The Windows download workarounds (copy instead of symlink, no hf-xet)
    # have to be in place before huggingface_hub is imported, or a parallel
    # download can fail with WinError 1314.
    model_fetch.prepare_hf_env()

# =====================================================================
# Language Pair & Persona Configuration
# =====================================================================
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
# Controls
# =====================================================================
# Safety limit for one recording, read from settings.json
# ("max_record_seconds"). minimum=1: a value of 0 would end every recording
# at once.
MAX_RECORD_SECONDS = _num("max_record_seconds", 20, minimum=1)

# =====================================================================
# Compute devices
# =====================================================================
# A detected "cpu" wins outright (and costs no torch import); a detected "cuda"
# is re-checked against the installed torch, because hardware_config.json can
# outlive the environment that wrote it (see loader.detect_device).
DEVICE = loader.detect_device(_HW.get("DEVICE"))


def _model_device(hw_key: str, default: str) -> str:
    """A per-model device from hardware detection, capped by DEVICE.

    DEVICE is the whole answer to "can torch use CUDA here". A per-model value
    from the same file can be stale in the same way, so a stored "cuda" must
    not reach a model after DEVICE has stepped down to "cpu". A stored "cpu"
    while DEVICE is "cuda" is kept: that is a decision the detector made.
    """
    if DEVICE != "cuda":
        return "cpu"
    return _HW.get(hw_key) or default


# Device of faster-whisper. detect_hardware writes "cuda" only when torch sees
# CUDA and ctranslate2 reports a CUDA device, so the cap by DEVICE matches its
# own rule.
STT_DEVICE = _model_device("STT_DEVICE", DEVICE)

# =====================================================================
# LLM Backend Settings
# =====================================================================
# Backend selection:
#   "lm-studio"    - external LM Studio app (must be running separately)
#   "local_server" - llm_server/server.py started automatically as a subprocess
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

# The server script. It exists only in a source checkout, next to the package:
# llm_server/ is not part of the package.
LOCAL_SERVER_SCRIPT = str(Path(__file__).resolve().parent.parent
                          / "llm_server" / "server.py")

# =====================================================================
# GGUF Model Settings (for the "local_server" backend)
# =====================================================================
# The file gguf_fetch downloads; models_info is the single place its name is
# written down.
EXTERNAL_MODEL_PATH = str(paths.models_dir() / models_info.GGUF_CHAT.filename)
# GPU offload and context size from hardware detection; the literals are the
# conservative fallbacks for a machine without hardware_config.json.
EXTERNAL_N_GPU_LAYERS = _HW.get("EXTERNAL_N_GPU_LAYERS", 20)  # -1 = all layers
EXTERNAL_N_CTX = _HW.get("EXTERNAL_N_CTX", 2048)

# Generation tuning parameters
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 50
LLM_TOP_P = 0.9

# Context buffer constraints
LLM_HISTORY_MAX_PAIRS = 4  # Number of full conversation turns kept in short-term memory

# =====================================================================
# Speech-to-Text (Whisper) Settings
# =====================================================================
# Loaded by repo id, the same one model_fetch downloads, so the load cannot go
# to a repo the installer never fetched.
WHISPER_MODEL = models_info.WHISPER_SMALL.repo_id
WHISPER_BEAM_SIZE = 1         # Beam size 1 provides optimal speed at temperature 0.0
WHISPER_NO_SPEECH_THRESHOLD = 0.45
WHISPER_CPU_THREADS = 4       # CPU inference threads (tune to available core count)

# Context conditioning instruction guiding Whisper's spelling logic
WHISPER_INITIAL_PROMPT = (
    f"This is a conversation with a {TARGET_LANGUAGE} tutor. "
    f"The speaker is practicing simple phrases."
)

# =====================================================================
# Text-to-Speech (Kokoro) Settings
# =====================================================================
KOKORO_REPO_ID = models_info.KOKORO.repo_id
KOKORO_LANG_CODE = "a"        # 'a' = American English, 'b' = British English
KOKORO_VOICE = "af_heart"     # Voice model identifier

# =====================================================================
# Shared Audio Device Settings
# =====================================================================
# Single lock coordinates PortAudio access between the mic (app.py) and
# speaker (tts.py) streams. Both modules import this object - do not create
# separate Lock instances or they will not mutually exclude each other.
AUDIO_LOCK = threading.Lock()

AUDIO_CHANNELS = 1           # Mono for both recording and playback
AUDIO_LATENCY = None         # None -> OS default shared-mode latency
# Device indices from hardware detection; None -> OS default microphone/speaker.
AUDIO_INPUT_DEVICE = _HW.get("AUDIO_INPUT_DEVICE")
AUDIO_OUTPUT_DEVICE = _HW.get("AUDIO_OUTPUT_DEVICE")

# =====================================================================
# Logging Settings
# =====================================================================
# logs/ is created by paths.ensure_dirs() at the top of this module, so the
# handlers can open their files at once.
LOG_DIR = paths.log_dir()
LOG_FILE = str(LOG_DIR / "main.log")
# Output of the llm_server subprocess (see app.py _start_llm_server).
LLM_SERVER_LOG_FILE = str(LOG_DIR / "llm_server.log")
