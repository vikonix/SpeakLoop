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
import shutil
import sys
import threading
from functools import partial
from pathlib import Path

from speakloop import llama_server_fetch, loader, model_fetch, models_info, paths

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
    "color_theme",
    "llm_backend",
    "lm_studio_host",
    "llama_server_path",
    "external_model_path",
    "external_n_ctx",
}
for _key in _USER:
    if not _key.startswith("_") and _key not in _KNOWN_USER_KEYS:
        print(f"[config] settings.json: unknown key {_key!r} ignored",
              file=sys.stderr)

# Validated accessors for settings.json values (they report the problem and
# fall back instead of raising). _path resolves a relative value against
# CONFIG_DIR, the directory settings.json itself is in.
_num = partial(loader.user_number, _USER)
_path = partial(loader.user_path, _USER, CONFIG_DIR)

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
# Backend selection, read from settings.json ("llm_backend"):
#   "llama-server" - the official llama.cpp binary started automatically as a
#                    subprocess (see resolve_llama_server_path() below)
#   "lm-studio"    - external LM Studio app (must be running separately)
# There is no "no model" choice: a dialogue lesson cannot run without one.
LLM_BACKEND_CHOICES = ("llama-server", "lm-studio")
LLM_BACKEND = _USER.get("llm_backend", "llama-server")
if LLM_BACKEND not in LLM_BACKEND_CHOICES:
    print(f"[config] settings.json: unknown llm_backend {LLM_BACKEND!r} "
          f"(expected one of {', '.join(LLM_BACKEND_CHOICES)}); "
          f"using 'llama-server'", file=sys.stderr)
    LLM_BACKEND = "llama-server"

# =====================================================================
# LM Studio backend (for the "lm-studio" backend)
# =====================================================================
# Server address, read from settings.json ("lm_studio_host") so LM Studio can
# run on another machine in the local network. Accepts "host", "host:port" or
# a full "http://host:port" URL; the port defaults to LM Studio's 1234
# (loader.server_url normalizes every spelling to the same base URL).
LM_STUDIO_DEFAULT_PORT = 1234
LM_STUDIO_HOST = _USER.get("lm_studio_host", "localhost:1234")
if not isinstance(LM_STUDIO_HOST, str) or not LM_STUDIO_HOST.strip():
    print(f"[config] settings.json: lm_studio_host must be a non-empty "
          f"string, got {LM_STUDIO_HOST!r}; using 'localhost:1234'",
          file=sys.stderr)
    LM_STUDIO_HOST = "localhost:1234"
LM_STUDIO_URL = loader.server_url(LM_STUDIO_HOST, LM_STUDIO_DEFAULT_PORT)
# LM Studio checks no key, but the OpenAI client refuses to send an empty one.
LM_STUDIO_API_KEY = "lm-studio"
# The model name is not here: LM Studio serves whatever is loaded and ignores
# the field, so it configures nothing - see llm.PLACEHOLDER_MODEL.

# =====================================================================
# Local LLM Server (the "llama-server" backend)
# =====================================================================
# Address and credentials of the server SpeakLoop starts itself. It is
# OpenAI-compatible and ignores the model name, so the client path is the same
# one the "lm-studio" backend uses - only the address and the key differ.
LLM_SERVER_HOST = "127.0.0.1"
LLM_SERVER_PORT = 8765
LLM_SERVER_URL = f"http://{LLM_SERVER_HOST}:{LLM_SERVER_PORT}/v1"
# Shared secret between the two halves of this backend, NOT a placeholder:
# llm_server_ctl passes it to the binary as --api-key and llm.py sends it back
# as the bearer token. Without a key llama-server accepts every CORS origin, so
# any page open in a browser could call 127.0.0.1:8765 and read the answer.
LLM_SERVER_API_KEY = "local"

# How long (seconds) to wait for the server to become ready after launching
LLM_SERVER_STARTUP_TIMEOUT = 60

# =====================================================================
# llama-server binary (for the "llama-server" backend)
# =====================================================================
# The official llama.cpp server is launched as a subprocess and uses the
# LLM_SERVER_* constants above plus the GGUF settings below
# (speakloop/llm_server_ctl.py builds its command line).
#
# settings.json ("llama_server_path") names the binary. An empty value - the
# default - resolves in this order:
#   1. bin/llama/llama-server[.exe], i.e. whatever
#      speakloop/llama_server_fetch.py installed from the pinned llama.cpp
#      release;
#   2. "llama-server" on PATH, for a build the user manages themselves.
# An empty result is NOT reported here: the binary only matters when this
# backend is actually selected, and LLMServerController says so at start time.
#
# Deliberately a function and NOT a module constant, unlike every other path in
# this file: the binary can be installed while the app is not running, and a
# value frozen at import would also hide a binary removed since, so the answer
# is taken from the disk at the moment the server is started.
def _resolve_llama_server(setting) -> str:
    """Absolute path of the llama-server binary to launch, or "" if none."""
    if setting is None:
        setting = ""
    if not isinstance(setting, str):
        print(f"[config] settings.json: llama_server_path must be a string, "
              f"got {setting!r}; searching for the binary instead",
              file=sys.stderr)
        setting = ""
    if setting.strip():
        # A relative path resolves against the directory settings.json is in,
        # like every other path setting (pathlib keeps an absolute value
        # unchanged). Spelled out here rather than taken from loader.user_path
        # because this setting has its own fallback chain below, not a single
        # default path.
        return str(CONFIG_DIR / setting.strip())
    bundled = llama_server_fetch.installed_exe()
    if bundled is not None:
        return str(bundled)
    return shutil.which("llama-server") or ""


def resolve_llama_server_path() -> str:
    """Where the llama-server binary is right now, or "" if there is none.

    The single answer to that question in the app: llm_server_ctl builds its
    command line from it.
    """
    return _resolve_llama_server(_USER.get("llama_server_path", ""))


# =====================================================================
# GGUF Model Settings (used by the "llama-server" backend)
# =====================================================================
# The model llama-server loads and the parameters it loads it with. Unused by
# "lm-studio", which manages its own model.
# GGUF file, read from settings.json ("external_model_path"); a relative path
# resolves against the directory settings.json is in. The default is the file
# gguf_fetch downloads, and models_info is the single place its name is written
# down.
EXTERNAL_MODEL_PATH = _path(
    "external_model_path",
    paths.models_dir() / models_info.GGUF_CHAT.filename,
)
# GPU offload and context size from hardware detection; the literals are the
# conservative fallbacks for a machine without hardware_config.json.
EXTERNAL_N_GPU_LAYERS = _HW.get("EXTERNAL_N_GPU_LAYERS", 20)  # -1 = all layers
# Context window size (n_ctx). settings.json ("external_n_ctx") wins over the
# detected value, per the usual layering. int() because the value is passed to
# the server as a command-line argument (a float would break it). minimum=256:
# anything below breaks generation outright (the system prompt alone would not
# fit), so treat it as a typo rather than passing it through.
EXTERNAL_N_CTX = int(_num("external_n_ctx",
                          _HW.get("EXTERNAL_N_CTX", 2048), minimum=256))

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
# Color Theme (UI palette)
# =====================================================================
# UI colors, read from settings.json ("color_theme") at startup; changing the
# theme needs a restart. Each theme is one <name>_schema.json, a flat map of
# semantic color names to hex values, so adding a theme is just adding a file.
# The file is looked for in TWO places (see _theme_file): the user's
# config/themes/ first, then the schemas shipped inside the package. That is
# what lets a user add a theme without touching the installation, and override a
# shipped one by reusing its name. The built-in palette below doubles as the
# complete list of valid keys and as the fallback: a missing or broken schema
# file, or a missing key inside one, falls back to these values, so the app
# always starts with a usable (dark) palette.
#
# The values are the colors the window had when they were still literals in
# app.py, so the default look did not change when the view layer was split out.
_DARK_THEME = {
    # Surfaces
    "bg_main": "#121214",            # window background (darkest surface)
    "bg_panel": "#1a1a1e",           # chat, status bar, language chip
    "bg_accent": "#1f1430",          # accent-tinted fill: the idle mic button
    "border": "#25252a",             # chat outline
    "accent": "#8a2be2",             # brand purple: title, focus highlight
    # Text
    "text": "#f8f8f2",               # chat body
    "text_emph": "#f1f1f6",          # the partner's reply
    "text_bright": "#ffffff",        # the learner's own line, mic glyph, caret
    "text_dim": "#a0a0a5",           # secondary labels: stats, instruction
    "text_muted": "#6272a4",         # [System] lines
    # Status / feedback
    "good": "#50fa7b",               # mic outline while the partner speaks
    "ready": "#00e676",              # "Ready"
    "bad": "#ff5555",                # errors, mic outline while recording
    "warn": "#ffb86c",               # loading, processing
    "info": "#8be9fd",               # the learner's name, "Thinking (LLM)..."
    "partner": "#ff79c6",            # the partner's name and speaking status
    # Mic button per-state inner fill (outlines reuse accent/bad/warn/good)
    "mic_loading_bg": "#1e1e24",
    "mic_loading_outline": "#44475a",
    "mic_recording_bg": "#3a0c10",
    "mic_processing_bg": "#36220f",
    "mic_speaking_bg": "#0f2c1d",
}


def _theme_file(name: str) -> Path:
    """The schema file for theme *name*, user copy preferred over the shipped one.

    Returns the shipped path when neither exists: it is the one worth naming in
    an error message, and read_json answers the same way for a file that is not
    there as for one that is unreadable.
    """
    filename = f"{name}_schema.json"
    user_file = paths.themes_dir() / filename
    return user_file if user_file.is_file() else paths.shipped_themes_dir() / filename


COLOR_THEME = _USER.get("color_theme", "dark")
if not isinstance(COLOR_THEME, str) or not COLOR_THEME.strip():
    print(f"[config] settings.json: color_theme must be a non-empty string, "
          f"got {COLOR_THEME!r}; using 'dark'", file=sys.stderr)
    COLOR_THEME = "dark"

# Resolved palette consumed by ui_theme.py. Starts as a copy of the built-in
# dark palette so every key is always present, whatever the schema file holds.
THEME = dict(_DARK_THEME)

_THEME_FILE = _theme_file(COLOR_THEME)
_SCHEMA = loader.read_json(_THEME_FILE)
if not _SCHEMA:
    # For "dark" a missing file is fine - the built-in palette IS dark.
    if COLOR_THEME != "dark":
        print(f"[config] theme file {_THEME_FILE.name} is missing or invalid; "
              f"using the built-in dark palette", file=sys.stderr)
else:
    for _key, _value in _SCHEMA.items():
        if _key.startswith("_"):
            continue  # comment keys, same convention as settings.json
        if _key not in _DARK_THEME:
            print(f"[config] {_THEME_FILE.name}: unknown color {_key!r} ignored",
                  file=sys.stderr)
        elif isinstance(_value, str) and _value.strip():
            THEME[_key] = _value
        else:
            print(f"[config] {_THEME_FILE.name}: {_key} must be a color string, "
                  f"got {_value!r}; using {_DARK_THEME[_key]!r}", file=sys.stderr)
    _missing = sorted(set(_DARK_THEME) - set(_SCHEMA))
    if _missing:
        print(f"[config] {_THEME_FILE.name}: missing colors filled from the "
              f"built-in dark palette: {', '.join(_missing)}", file=sys.stderr)

# =====================================================================
# Logging Settings
# =====================================================================
# logs/ is created by paths.ensure_dirs() at the top of this module, so the
# handlers can open their files at once.
LOG_DIR = paths.log_dir()
LOG_FILE = str(LOG_DIR / "main.log")
# Output of the llama-server subprocess (see speakloop/llm_server_ctl.py).
LLM_SERVER_LOG_FILE = str(LOG_DIR / "llm_server.log")
