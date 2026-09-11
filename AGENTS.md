# AGENTS.md

This file provides guidance to agents when working with code in this repository.

It is a map, not a manual: each entry says what a module is for and what would
break if it were changed carelessly. The reasoning behind a specific line lives
in the comment next to that line.

## Project Overview

SpeakLoop is a local desktop **voice dialogue tutor** for language learning
(Python 3.11/3.12, Tkinter GUI). Push-to-talk (Space or the mic button) ->
faster-whisper speech recognition -> a local LLM (GGUF model through
`llm_server/`, or LM Studio) -> Kokoro speech synthesis.

The project is in the middle of a planned refactoring into a voice dialogue
trainer. The plan, the step order and the open questions are in
[`docs/refactoring.md`](docs/refactoring.md) (Russian). Many modules are
copied from the sibling project Mimora and kept close to the original; do not
rewrite them without a reason from the plan.

## Running the App

```bash
python install.py        # guided setup: dependencies, models, llama-server, hardware probe
pip install -r llm_server/requirements.txt   # until llm_server/ is replaced
python main.py
```

Three launch forms, identical after `speakloop/cli.py`'s `main()`: the root
`main.py` shim, `python -m speakloop`, and the `speakloop` console script
(after `pip install -e .`).

**Default backend**: `local_server` - `llm_server/server.py` (FastAPI +
llama-cpp-python) is launched as a subprocess. **Alternative**:
`LLM_BACKEND = "lm-studio"` in `speakloop/config.py`, with LM Studio on
`http://localhost:1234`.

## Architecture

### Entry and process

- [`main.py`](main.py) - root shim over `cli.main()`. Imports nothing else.
- [`speakloop/cli.py`](speakloop/cli.py) - console-script entry point.
  **Stdlib-only at module level**: `--version` must answer without loading
  torch, and `bootstrap.early_init()` must run before the heavy imports. The
  function-local `from speakloop import app` is load-bearing
  (`tests/test_cli.py` pins it). Also turns a missing tkinter or PortAudio into
  an install hint.
- [`speakloop/__main__.py`](speakloop/__main__.py) - `python -m speakloop` shim.
- [`speakloop/bootstrap.py`](speakloop/bootstrap.py) - stdlib-only early setup.
  `early_init()` runs before the heavy imports, `setup_logging()` after them
  (from `app.run()`, `force=True`). Do not merge or reorder the two.
  `open_log_section()` writes the run header of append-only logs.
- [`speakloop/lifecycle.py`](speakloop/lifecycle.py) - `hard_exit()` (used by
  `quit_app`: ends the process without the CUDA teardown that can crash on
  Windows) and `spawn_replacement()` (no caller yet).

### Application

- [`speakloop/app.py`](speakloop/app.py) - `VoiceTutorGUI`: Tkinter window,
  audio recording, threading orchestration, llm_server subprocess management.
  Module-level `run(append_log)` configures logging, logs
  `detect_hardware.warn_if_gpu_unused`, and opens the window. **Imports
  `speakloop.config` before `stt`/`tts`** (see config below).
- [`speakloop/stt.py`](speakloop/stt.py) - `STTManager`: faster-whisper with
  VAD filtering, on `config.STT_DEVICE`. **Imports torch before
  faster_whisper**: on Windows the CUDA build of torch provides the cuBLAS and
  cuDNN libraries ctranslate2 needs.
- [`speakloop/llm.py`](speakloop/llm.py) - `LLMManager`: OpenAI-compatible
  streaming client with conversation history; used by both backends.
- [`speakloop/tts.py`](speakloop/tts.py) - `TTSManager`: Kokoro on
  `config.DEVICE`, winsound playback on Windows, sounddevice elsewhere.
- [`llm_server/server.py`](llm_server/server.py) - standalone FastAPI server
  loading the GGUF with llama-cpp-python; a separate process to avoid GPU
  contention with Kokoro. To be replaced by the official llama-server.

### Configuration and paths

- [`speakloop/config.py`](speakloop/config.py) - all configuration, frozen at
  import. Layers, lowest first: literals in the file ->
  `config/hardware_config.json` (`"config"` section) ->
  `config/settings.json`. Known settings.json keys are listed in
  `_KNOWN_USER_KEYS` (today only `max_record_seconds`); keep
  `config/settings.example.json` in step (a test checks it). Also sets
  `HF_HOME` to `model_cache/` and switches `HF_HUB_OFFLINE=1` once the repos
  this run loads are cached - which is why it must be imported before anything
  imports huggingface_hub. `_model_device()` caps a per-model device by
  `DEVICE`.
- [`speakloop/loader.py`](speakloop/loader.py) - pure config-loading helpers
  (JSON read, validated values, atomic save, cache predicate,
  `detect_device`).
- [`speakloop/paths.py`](speakloop/paths.py) - where every file lives, and the
  only module that knows. `data_root()` (what this machine writes; the clone in
  repo mode, the OS user-data directory for an installed package,
  `SPEAKLOOP_HOME` overrides) and `shipped_root()` (read-only files inside the
  package). **Stdlib-only** - install.py reads it before the requirements
  exist.
- [`speakloop/detect_hardware.py`](speakloop/detect_hardware.py) - machine probe
  writing `config/hardware_config.json`: `DEVICE` (torch CUDA), `STT_DEVICE`
  (torch CUDA and a ctranslate2 CUDA device), `EXTERNAL_N_GPU_LAYERS` and
  `EXTERNAL_N_CTX` (from VRAM and the installed llama-server's
  `--list-devices`), audio devices. **Must not import config.**

### Downloads

Three fetchers, all **forbidden to import `config`** (config switches the Hub
offline once models are cached, exactly when a download would be wanted) and
all keeping huggingface_hub out of their module-level imports so `install.py`
can use them before the requirements step:

- [`speakloop/model_fetch.py`](speakloop/model_fetch.py) - faster-whisper small
  and Kokoro (hub cache), Supertonic 3 (own cache directory). Owns
  `prepare_hf_env()` (Windows symlink and hf-xet workarounds).
- [`speakloop/gguf_fetch.py`](speakloop/gguf_fetch.py) - the GGUF chat model
  into `models/`.
- [`speakloop/llama_server_fetch.py`](speakloop/llama_server_fetch.py) - the
  pinned llama.cpp release into `bin/llama/`, sha256 per asset, then
  `--version` and `--list-devices` probes. Not used by the app yet.

- [`speakloop/models_info.py`](speakloop/models_info.py) - model catalogue,
  the single place a repo id is written. Imports only `typing` (a test
  enforces it). Sizes are re-snapped with
  [`tools/measure_model_sizes.py`](tools/measure_model_sizes.py).
- [`install.py`](install.py) - interactive installer, a thin wrapper around the
  fetchers; dependency list read from `pyproject.toml`; logs to
  `logs/install.log`.

## Key Patterns & Gotchas

- **Threading**: recording, TTS queue processing and model loading run in
  daemon threads. Always update the GUI with `root.after()`.
- **TTS sentinel pattern** (`app.py`): LLM sentences are buffered in the TTS
  queue processor and only played after `_TTS_START_SENTINEL` arrives, so the
  model server has finished before Kokoro synthesis starts. Do not remove it
  while the LLM and Kokoro share one GPU.
- **LLM history rollback** (`llm.py`): the user message is appended inside
  `try`; on exception it is popped to keep user/assistant pairs consistent.
  History is trimmed to `LLM_HISTORY_MAX_PAIRS` pairs after each exchange.
- **Sentence streaming** (`llm.py`): output is split on sentence-ending
  punctuation followed by whitespace and an uppercase letter
  (`(?<=[.!?])\s+(?=[A-ZА-Я])`); the rest is flushed at the end of the stream.
- **Audio normalization** (`app.py`): peaks are normalized before STT; a peak
  below 0.01 is not boosted.
- **Windows audio**: TTS plays through `winsound` to bypass PortAudio/MME
  driver issues, with a 150 ms silence lead-in. `config.AUDIO_LOCK`
  serialises PortAudio init/teardown between the recording and playback paths.
- **LLM server subprocess**: started in `_start_llm_server()` with layers and
  context from `config` (hardware detection), polled with
  `LLMManager.check_connection()`, terminated in `quit_app()` with a 5-second
  kill fallback, output in `logs/llm_server.log`.
- **Known v0 issues** (interrupt race, invisible LLM errors, stream not closed
  on interrupt, orphan server, 20 s record limit, recording during loading) are
  listed in `docs/refactoring.md` with the step that fixes each. Do not fix
  them outside their step.

## Testing

```bash
python -m unittest discover -s tests -v
```

Test files are named after the module they cover (`tests/test_paths.py` for
`speakloop/paths.py`). The suite stubs subprocesses, the OS and the network and
downloads nothing. `tests/test_config.py` imports the real `config` of the
checkout (and so torch).

## Working Rules

From `docs/refactoring.md`, section 4:

- Every step ends with an application that starts and works; the owner checks
  it against the step's check list.
- Claude writes code and tests without running them; the owner runs them.
- Before changing existing files, describe the change and get permission.
- git operations and file deletion are done by the owner; every step ends with
  a list of files to delete.
- At the end of every step `AGENTS.md` and `README.md` describe the current
  code.

## Code Style (Python)

- No linting/formatting config - follow PEP 8.
- **Imports inside `speakloop/` are absolute** (`from speakloop import config`),
  never relative.
- Type hints where they help; logging via `logging` with
  `%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s`.
- Explicit `RuntimeError` with a descriptive message for runtime validation,
  not `assert`.
- Library warning filters live in `speakloop/bootstrap.py`.
- Code, comments, logs and messages in English (ASD-STE100 Simplified English).
- **Comments say what breaks if the code changes, not how the code got here.**
