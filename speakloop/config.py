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

The language of the lesson is resolved here too: speakloop/languages/ holds one
pure-data profile per language, and every per-run language constant
(TARGET_LANGUAGE, WHISPER_LANGUAGE, TTS_*) is derived from the active profile
and its variant. Adding a language is a new profile module plus one entry in
LANGUAGE_PROFILES, never a branch in the code.

This module also prepares the model-download environment (HF_HOME, the
Supertonic cache and the offline switch). Those variables are read when
huggingface_hub is imported, so this module has to be imported BEFORE
faster_whisper, kokoro and supertonic - app.py imports it first. For the same
reason the fetchers and the hardware probe must never import it.
"""

import os
import sys
from functools import partial
from pathlib import Path

from speakloop import loader, model_fetch, models_info, paths
from speakloop.languages import english, spanish

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
SETTINGS_FILE = CONFIG_DIR / "settings.json"
_USER = loader.read_json(SETTINGS_FILE)

# Every key settings.json may hold. A key outside this set is reported, because
# a typo in a hand-edited file otherwise changes nothing and says nothing.
_KNOWN_USER_KEYS = {
    "max_record_seconds",
    "silence_timeout",
    "silence_threshold",
    "stt_device",
    "accent",
    "voice",
    "color_theme",
    "llm_backend",
    "lm_studio_host",
    "llama_server_path",
    "external_model_path",
    "external_n_ctx",
    "external_n_gpu_layers",
    "first_topic",
    "prompt_file",
    "show_notes",
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
_flag = partial(loader.user_bool, _USER)


def save_user_setting(key: str, value) -> bool:
    """Write one settings.json key from the running application.

    The only value this module writes instead of reading: the window saves the
    state of its Notes switch here, so the next lesson opens the way the last
    one was left. Every other key stays hand-edited. The file is re-read and
    rewritten atomically by the loader, so the comment keys and hand-made
    values survive, and a failure is reported on stderr and never raised.

    The constants above stay frozen at import: the caller already knows the
    value it saved, and nothing else in this run reads the key again.
    """
    return loader.save_setting(SETTINGS_FILE, key, value, _USER)

# =====================================================================
# Language of the lesson (profiles in speakloop/languages/)
# =====================================================================
# Resolved before the download section below, which needs to know the active
# synthesis backend: only that backend's model has to be cached before the Hub
# can be switched off.
#
# The profile format is documented in speakloop/languages/__init__.py.
LANGUAGE_PROFILES = {
    "english": english.PROFILE,
    "spanish": spanish.PROFILE,
}

# The practiced language is fixed to English: only the English lesson is
# checked. Both profiles are complete. No settings key, so nothing promises a
# choice that does not work.
PRACTICE_LANGUAGE = "english"
_LANG_PROFILE = LANGUAGE_PROFILES[PRACTICE_LANGUAGE]

# The language as the window and the system prompt name it.
TARGET_LANGUAGE = _LANG_PROFILE["display_name"]

# ISO code faster-whisper transcribes with: the learner speaks the practiced
# language, so this follows the profile.
WHISPER_LANGUAGE = _LANG_PROFILE["whisper_language"]

# Variant within the language, read from settings.json ("accent"). Changing it
# needs a restart: the synthesis backend, its language code and the selectable
# voices are all wired from the variant when the model is loaded.
#   English variants: "american" (General American), "british" (RP).
_VARIANT_MAP = _LANG_PROFILE["variants"]
ACCENT = _USER.get("accent", _LANG_PROFILE["default_variant"])
if not isinstance(ACCENT, str) or ACCENT not in _VARIANT_MAP:
    # (isinstance guards the dict lookup: an unhashable value such as a list
    # would raise TypeError instead of falling back.)
    print(f"[config] settings.json: unknown accent {ACCENT!r} for "
          f"{PRACTICE_LANGUAGE} (expected one of {sorted(_VARIANT_MAP)}); "
          f"using {_LANG_PROFILE['default_variant']!r}", file=sys.stderr)
    ACCENT = _LANG_PROFILE["default_variant"]
_VARIANT = _VARIANT_MAP[ACCENT]

# Synthesis backend of the active variant - one of the keys of
# speakloop.tts.TTS_BACKENDS ("kokoro" = Kokoro-82M on torch at 24 kHz,
# "supertonic" = Supertonic 3 on ONNX at 44.1 kHz). Data, not a language
# branch: Spanish selects another engine by profile alone.
TTS_BACKEND_CHOICES = ("kokoro", "supertonic")
TTS_BACKEND = _VARIANT.get("tts_backend", "kokoro")
if TTS_BACKEND not in TTS_BACKEND_CHOICES:
    print(f"[config] profile {PRACTICE_LANGUAGE}/{ACCENT}: unknown tts_backend "
          f"{TTS_BACKEND!r} (expected one of {', '.join(TTS_BACKEND_CHOICES)}); "
          f"using 'kokoro'", file=sys.stderr)
    TTS_BACKEND = "kokoro"

# Language code of the active variant, in the spelling its backend uses: Kokoro
# takes single letters ("a" American, "b" British), Supertonic takes ISO codes.
TTS_LANG_CODE = _VARIANT["tts_lang_code"]

# Voice: the variant's default, unless settings.json names another voice OF THE
# SAME VARIANT ("voice"). A voice of a different variant is rejected - it does
# not match this variant's backend or language code.
TTS_VOICE = _VARIANT["default_voice"]
_user_voice = _USER.get("voice")
if _user_voice is not None:
    if _user_voice in _VARIANT["voices"]:
        TTS_VOICE = _user_voice
    else:
        print(f"[config] settings.json: voice {_user_voice!r} is not a known "
              f"{ACCENT} voice; using {TTS_VOICE!r}", file=sys.stderr)

# Every voice of the active variant. They share its backend and language code,
# so the Kokoro backend can pre-fetch them all in one online run.
TTS_VOICES = _VARIANT["voices"]

# Supertonic quality and speed trade-off: refinement steps per synthesis (more
# is better and slower). The package range is wider, but 5..12 is the useful
# band for sentence-length audio; a value outside it is clamped, not fatal.
# Ignored by the Kokoro backend.
TTS_TOTAL_STEPS = int(_VARIANT.get("total_steps", 8))
if not 5 <= TTS_TOTAL_STEPS <= 12:
    print(f"[config] profile {PRACTICE_LANGUAGE}/{ACCENT}: total_steps "
          f"{TTS_TOTAL_STEPS} outside 5..12; clamping", file=sys.stderr)
    TTS_TOTAL_STEPS = min(12, max(5, TTS_TOTAL_STEPS))

# Word spoken by the synthesis warm-up pass, in the practiced language.
TTS_WARMUP = _LANG_PROFILE["tts_warmup"]

# =====================================================================
# Local model cache (Hugging Face) - download once, then load offline
# =====================================================================
# faster-whisper and Kokoro load their weights through huggingface_hub, which
# reads these variables at IMPORT time. setdefault(), so an externally set
# HF_HOME wins. The path comes from model_fetch, the module that downloads into
# it, so the app and the installer cannot look in different places.
MODEL_CACHE_DIR = model_fetch.MODEL_CACHE_DIR
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

# Supertonic 3 (the Spanish backend) keeps its model in its own directory, NOT
# under HF_HOME/hub: the package downloads with snapshot_download(local_dir=...)
# into the directory named by SUPERTONIC_CACHE_DIR (whose own default would be
# ~/.cache/supertonic3). Pinning it under model_cache/ keeps the weights next to
# the code, like HF_HOME above. The package reads the variable on every call
# rather than at import, but setting it here - before any supertonic import -
# keeps the timing rule the same. model_fetch sets the same variable, so the
# installer's download lands exactly where the app reads.
os.environ.setdefault("SUPERTONIC_CACHE_DIR",
                      str(model_fetch.DEFAULT_SUPERTONIC_CACHE_DIR))
SUPERTONIC_CACHE_DIR = Path(os.environ["SUPERTONIC_CACHE_DIR"])

# Offline gate. Once every model THIS run loads is cached, the Hub is switched
# off and every start loads straight from disk with no network request. The set
# is all-or-nothing, so it must name only what the run really loads: a repo the
# run never touches would keep the Hub online for every model. That is why the
# synthesis model follows the active backend - requiring Kokoro under Supertonic
# (or the other way round) would keep the Hub online for weights nobody loads.
_CACHED_REPOS = (
    (models_info.WHISPER,)
    + ((models_info.KOKORO,) if TTS_BACKEND == "kokoro" else ())
)
# model_fetch.supertonic_cached() owns this check: it reads the same variable
# set above and is pure filesystem work, so calling it here pulls in no
# huggingface_hub import. It matters because the Supertonic download itself goes
# through huggingface_hub - switching HF_HUB_OFFLINE on before the model exists
# would block that first download.
_TTS_MODEL_CACHED = (model_fetch.supertonic_cached()
                     if TTS_BACKEND == "supertonic"
                     else True)  # Kokoro is covered by _CACHED_REPOS above
if _TTS_MODEL_CACHED and loader.models_cached(
        Path(os.environ["HF_HOME"]) / "hub", _CACHED_REPOS):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
else:
    # Something is missing, so this run downloads it through the libraries.
    # The Windows download workarounds (copy instead of symlink, no hf-xet)
    # have to be in place before huggingface_hub is imported, or a parallel
    # download can fail with WinError 1314.
    model_fetch.prepare_hf_env()

# =====================================================================
# Lesson
# =====================================================================
# The language of corrections and of the summary. Fixed, with no settings key:
# the NOTE example in the prompt is written in Russian, and a prompt that asks
# for another language with a Russian example gives mixed corrections.
EXPLANATION_LANGUAGE = "Russian"

# First topic of the lesson, read from settings.json ("first_topic"). Empty
# means the default the prompt itself names in its SETTINGS line.
FIRST_TOPIC = _USER.get("first_topic", "")
if not isinstance(FIRST_TOPIC, str):
    print(f"[config] settings.json: first_topic must be a string, got "
          f"{FIRST_TOPIC!r}; using the prompt default", file=sys.stderr)
    FIRST_TOPIC = ""

# Are the NOTE lines shown in the chat? The window's Notes switch writes this
# key back (save_user_setting), so a lesson opens the way the last one ended.
# The corrections are always written into the transcript; the switch only
# hides them (speakloop/ui.py).
SHOW_NOTES = _flag("show_notes", True)

# The prompt body (speakloop/prompt.py builds the system message from it).
# settings.json ("prompt_file") can point at a copy to try changes on; the
# file is read once, when the models are loaded.
PROMPT_FILE = _path("prompt_file", paths.shipped_root() / "prompts" / "free_talk.md")

# =====================================================================
# Controls
# =====================================================================
# Safety limit for one recording, read from settings.json
# ("max_record_seconds"). minimum=1: a value of 0 would end every recording
# at once.
MAX_RECORD_SECONDS = _num("max_record_seconds", 20, minimum=1)

# A take starts on one press and ends by itself once the speaker falls silent
# (speakloop/recorder.py runs the detection on the capture thread). These two
# tune that automatic stop; both come from settings.json, so a room or a
# microphone can be accommodated without a code change.
#   silence_timeout   - seconds of continuous silence, after speech has begun,
#                       before the take is finished automatically.
#   silence_threshold - RMS level (0..1) strictly above which a block of audio
#                       counts as speech. Kept low: quiet speech may only reach
#                       about 0.04 and the level is averaged over a whole
#                       block, so too high a value never arms the timer. Raise
#                       it only when a noisy room stops the take from ending.
#                       The minimum is above zero because at 0 the noise floor
#                       of the microphone counts as speech and every take runs
#                       to MAX_RECORD_SECONDS.
SILENCE_TIMEOUT = _num("silence_timeout", 3.0, minimum=0.5)
SILENCE_THRESHOLD = _num("silence_threshold", 0.01, minimum=0.001)

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


# Device of faster-whisper, from settings.json ("stt_device"):
#   "auto" - the detected value. detect_hardware writes "cuda" whenever torch
#            sees CUDA and ctranslate2 reports a CUDA device, whatever the size
#            of the card, so the cap by DEVICE matches its own rule.
#   "cuda" - the GPU, set by hand. Still capped by DEVICE: on Windows the CUDA
#            build of torch provides the libraries ctranslate2 needs on the
#            GPU (stt.py), so without it the model load fails.
#   "cpu"  - the CPU, for a card where Whisper leaves the chat model too few
#            layers.
STT_DEVICE_CHOICES = ("auto", "cuda", "cpu")


def _stt_device_setting(value) -> str:
    """The "stt_device" value, or "auto" when it is not one of the choices.

    "auto" and not "cpu" on a typo: the detected device is the one the
    installation was checked with.
    """
    if value in STT_DEVICE_CHOICES:
        return value
    print(f"[config] settings.json: stt_device must be one of "
          f"{', '.join(repr(choice) for choice in STT_DEVICE_CHOICES)}, got "
          f"{value!r}; using 'auto'", file=sys.stderr)
    return "auto"


def _stt_device(setting: str) -> str:
    """The device of faster-whisper for a validated "stt_device" value."""
    if setting == "cpu":
        return "cpu"
    if setting == "cuda":
        if DEVICE != "cuda":
            print("[config] settings.json: stt_device is 'cuda', but torch "
                  "cannot use CUDA here; using 'cpu'", file=sys.stderr)
            return "cpu"
        return "cuda"
    return _model_device("STT_DEVICE", DEVICE)


STT_DEVICE = _stt_device(_stt_device_setting(_USER.get("stt_device", "auto")))

# Device of Kokoro. A key of its own and not DEVICE: detect_hardware moves
# Kokoro to the CPU when the chat model needs the whole card, and
# writing "cpu" into DEVICE instead would make warn_if_gpu_unused report a
# CUDA problem that does not exist.
TTS_DEVICE = _model_device("TTS_DEVICE", DEVICE)

# =====================================================================
# LLM Backend Settings
# =====================================================================
# Backend selection, read from settings.json ("llm_backend"):
#   "llama-server" - the official llama.cpp binary started automatically as a
#                    subprocess (llm_server_ctl.find_llama_server finds it)
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


def _client_address(backend: str) -> tuple:
    """URL and key of the chat server of *backend*."""
    if backend == "llama-server":
        return LLM_SERVER_URL, LLM_SERVER_API_KEY
    return LM_STUDIO_URL, LM_STUDIO_API_KEY


# The address and the key the chat client uses: app.py points LLMManager at
# them once, for either backend.
LLM_URL, LLM_API_KEY = _client_address(LLM_BACKEND)

# How long (seconds) to wait for the server to become ready after launching.
# The measured load of the 7 GB Gemma file was 16 s with the file in the OS
# cache; a cold disk and the memory fit (-fit) come on top of that, and a
# timeout here terminates a server that was about to answer.
LLM_SERVER_STARTUP_TIMEOUT = 120

# =====================================================================
# llama-server binary (for the "llama-server" backend)
# =====================================================================
# The official llama.cpp server is launched as a subprocess and uses the
# LLM_SERVER_* constants above plus the GGUF settings below
# (speakloop/llm_server_ctl.py builds its command line).
#
# settings.json ("llama_server_path") names a binary the user manages. Empty
# (the default) means "find it": llm_server_ctl.find_llama_server looks in
# bin/llama/ and then on PATH when the server starts.
def _llama_server_setting(setting) -> str:
    """The "llama_server_path" value as a path, or "" when it is not set.

    Not a validated default path like the other path keys: an empty value has
    its own meaning (search for the binary). A relative path resolves against
    the directory settings.json is in, like every other path setting.
    """
    if setting is None:
        return ""
    if not isinstance(setting, str):
        print(f"[config] settings.json: llama_server_path must be a string, "
              f"got {setting!r}; searching for the binary instead",
              file=sys.stderr)
        return ""
    if not setting.strip():
        return ""
    return str(CONFIG_DIR / setting.strip())


LLAMA_SERVER_PATH = _llama_server_setting(_USER.get("llama_server_path", ""))


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
# Neither value below is read from hardware_config.json: an old file can still
# hold EXTERNAL_N_GPU_LAYERS and EXTERNAL_N_CTX, and reading them would pass
# -ngl (the memory fit goes off) and cut the lesson context to 2048 tokens.

# Words llama-server's --n-gpu-layers accepts besides a number. "auto" is
# the default and passes no argument at all, so llama.cpp fits the layers
# into the free VRAM itself (-fit); "all" offloads every layer.
GPU_LAYERS_WORDS = ("auto", "all")


def _gpu_layers_setting(value) -> str:
    """The "external_n_gpu_layers" value as llama-server takes it, or "auto".

    A string because the result goes to the command line unchanged. A number
    or "all" is a manual override: it switches the memory fit off, for a
    machine where more layers fit than the fit chooses
    (docs/model-parameters.md).
    """
    if value in GPU_LAYERS_WORDS:
        return value
    # bool is a subclass of int - exclude it so `true` is not taken as 1.
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    print(f"[config] settings.json: external_n_gpu_layers must be "
          f"{' or '.join(repr(word) for word in GPU_LAYERS_WORDS)} or a whole "
          f"number from 0, got {value!r}; using 'auto'", file=sys.stderr)
    return "auto"


EXTERNAL_N_GPU_LAYERS = _gpu_layers_setting(
    _USER.get("external_n_gpu_layers", "auto"))

# Context window size (n_ctx), from settings.json ("external_n_ctx"). The
# default holds the lesson prompt (about 2300 tokens) and a long conversation.
# It is also passed as -fitc, so the memory fit takes GPU layers away rather
# than shrink the context. int() because the value is passed to the server as
# a command-line argument (a float would break it). minimum=256: anything
# below breaks generation outright (the system prompt alone would not fit), so
# treat it as a typo rather than passing it through.
EXTERNAL_N_CTX = int(_num("external_n_ctx", 16384, minimum=256))

# Sampling: the values Google recommends for Gemma, which its GGUF also carries
# as the server defaults.
LLM_TEMPERATURE = 1.0
LLM_TOP_P = 0.95
# Not a field of the OpenAI API: llm.py sends it only to llama-server. Without
# it the server takes the default of the loaded GGUF or its own, which is not
# 64 for every model.
LLM_TOP_K = 64
# Long enough for the multi-line SUMMARY. At about 5 tokens per
# second on a weak machine a reply this long takes close to two minutes, so
# the limit is a safety stop and not the expected length.
LLM_MAX_TOKENS = 512

# =====================================================================
# Speech-to-Text (Whisper) Settings
# =====================================================================
# Loaded by repo id, the same one model_fetch downloads, so the load cannot go
# to a repo the installer never fetched. The language is WHISPER_LANGUAGE,
# resolved from the language profile above.
WHISPER_MODEL = models_info.WHISPER.repo_id
WHISPER_BEAM_SIZE = 1         # Greedy decoding: the fastest
WHISPER_NO_SPEECH_THRESHOLD = 0.45
WHISPER_CPU_THREADS = 4       # CPU inference threads (tune to available core count)

# Context conditioning instruction guiding Whisper's spelling logic
WHISPER_INITIAL_PROMPT = (
    f"This is a conversation with a {TARGET_LANGUAGE} tutor. "
    f"The speaker is practicing simple phrases."
)

# =====================================================================
# Shared Audio Device Settings
# =====================================================================
# Rate of the audio pipeline: the recorder downsamples every take to it and the
# recognizer needs exactly this rate. NOT the synthesis rate, which belongs to
# the active TTS backend (TTSManager.sample_rate).
AUDIO_SAMPLE_RATE = 16_000

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
_DARK_THEME = {
    # Surfaces
    "bg_main": "#121214",            # window background (darkest surface)
    "bg_panel": "#1a1a1e",           # chat, status bar, control panel
    "bg_accent": "#1f1430",          # accent-tinted fill: the idle mic button
    "border": "#25252a",             # chat outline
    "accent": "#8a2be2",             # brand purple: title, focus highlight
    # Text
    "text": "#f8f8f2",               # chat body
    "text_emph": "#f1f1f6",          # the partner's reply
    "text_bright": "#ffffff",        # the learner's own line, mic glyph, caret
    "text_dim": "#a0a0a5",           # secondary labels: instruction, notes
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

# =====================================================================
# Lesson transcript
# =====================================================================
# transcript/ is created by paths.ensure_dirs() with the other directories, so
# the first record of a lesson has somewhere to go. The files inside it are
# named per lesson by speakloop/transcript.py, which is why only the directory
# is named here.
TRANSCRIPT_DIR = paths.transcript_dir()
