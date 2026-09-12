# AGENTS.md

This file provides guidance to agents when working with code in this repository.

It is a map, not a manual: each entry says what a module is for and what would
break if it were changed carelessly. The reasoning behind a specific line lives
in the comment next to that line.

## Project Overview

SpeakLoop is a local desktop **voice dialogue tutor** for language learning
(Python 3.11/3.12, Tkinter with ttkbootstrap). One press (Space or the mic
button) opens the microphone and the take ends on silence -> faster-whisper
speech recognition -> a local LLM (a GGUF model served by `llama-server`, or LM
Studio) -> speech synthesis (Kokoro for English, Supertonic for Spanish).

The language of the lesson is data: `speakloop/languages/` holds one profile per
language and `config.py` derives every per-run language constant from the active
one. English is fixed until stage 3 of the plan.

The window and the logic are **separate**: `speakloop/ui.py` owns every widget,
color and piece of wording, `speakloop/app.py` owns the threads and the voice
loop and drives the window through the view's intent methods.

The project is in the middle of a planned refactoring into a voice dialogue
trainer. The plan, the step order and the open questions are in
[`docs/refactoring.md`](docs/refactoring.md) (Russian). Many modules are
copied from the sibling project Mimora and kept close to the original; do not
rewrite them without a reason from the plan.

## Running the App

```bash
python install.py        # guided setup: dependencies, models, llama-server, hardware probe
python main.py
```

Three launch forms, identical after `speakloop/cli.py`'s `main()`: the root
`main.py` shim, `python -m speakloop`, and the `speakloop` console script
(after `pip install -e .`).

**Default backend**: `llama-server` - the official llama.cpp binary from
`bin/llama/` is launched as a subprocess. **Alternative**:
`"llm_backend": "lm-studio"` in `config/settings.json`, with LM Studio running
separately. Both speak the same OpenAI-compatible API, so only the address and
the key differ.

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

- [`speakloop/app.py`](speakloop/app.py) - `VoiceTutorController`: the routing
  between the audio modules, threading orchestration, the voice loop, and the
  owner of the `AudioRecorder`, the `PlaybackController`, the
  `LLMServerController` and the view. It creates the Tk root and **touches no
  widget**: every window change is `self.root.after(0, self.view.<intent>, ...)`,
  which is also the only thread-safe way into Tk. It no longer captures audio
  itself - `recorder.py` does. `_toggle_recording` is the whole input model:
  one press starts a take, the next one ends it, and a release only clears the
  key auto-repeat guard. `_exchange_lock` keeps one exchange at a time, so a
  take made during the previous exchange waits instead of being dropped.
  Module-level `run(append_log)` configures logging, logs
  `detect_hardware.warn_if_gpu_unused`, and opens the window. **Imports
  `speakloop.config` before `stt`/`tts`** (see config below).
- [`speakloop/ui.py`](speakloop/ui.py) - `TutorView`: the whole window (header,
  chat transcript, mic canvas, status bar) plus the *intent* methods the
  controller calls (`enter_recording`, `enter_thinking`, `enter_error`, …) and
  the `append_*` transcript writers. Widget bindings call only the callables in
  the `ViewCallbacks` passed in, so the view never references the controller.
  Every status string, instruction line and the partner's name (`PARTNER_NAME`)
  live here - do not move wording or colors back into the controller. Its
  methods must run on the Tk main thread. `centered_geometry()` is pure on
  purpose, so where the window opens is tested without a display
  (`tests/test_ui.py`). `set_record_level()` repaints the mic
  button from the live input level while a take runs; the recording state has no
  glyph of its own any more, because the level disc IS the indicator.
- [`speakloop/ui_theme.py`](speakloop/ui_theme.py) - the palette (`THEME` from
  config), the ttkbootstrap base theme, `FONT_FAMILY` per platform and the
  `FONT_SIZE_*` scale. Importing it also **disables ttkbootstrap's
  classic-widget autostyle hook**, which would otherwise repaint every `tk`
  widget with the base theme's colors; that is why ui.py imports it first.
- [`speakloop/themes/`](speakloop/themes) - `dark_schema.json` (the default,
  equal to the built-in palette) and `light_schema.json`. A user copy in
  `config/themes/` of the same name wins over these.
- [`speakloop/stt.py`](speakloop/stt.py) - `STTManager`: faster-whisper with
  VAD filtering, on `config.STT_DEVICE`, in the language of the active profile
  (`config.WHISPER_LANGUAGE`; never automatic detection). **Imports torch before
  faster_whisper**: on Windows the CUDA build of torch provides the cuBLAS and
  cuDNN libraries ctranslate2 needs.
- [`speakloop/recorder.py`](speakloop/recorder.py) - `AudioRecorder`: the
  capture thread, the input device choice and the chunk buffer, plus the pure
  `normalize_audio`. It captures at the device's own rate through WASAPI on
  Windows (MME drops samples) and downsamples the take to
  `config.AUDIO_SAMPLE_RATE` with librosa. Its four callbacks
  (`on_max_duration`, `on_silence_stop`, `on_stream_error`, `on_level`) all run
  on the capture thread. The realtime callback takes **no lock** - that was the
  source of dropped samples - so the buffer may only be read after `join()`.
- [`speakloop/audio_io.py`](speakloop/audio_io.py) - the device plumbing both
  the microphone and the speaker need: `reset_portaudio()` (Windows only, and
  skipped while any stream is open - it invalidates every stream in the
  process), the open-stream counter and `uses_winsound()`.
- [`speakloop/playback.py`](speakloop/playback.py) - `PlaybackController`: the
  stop event of the **current reply**. `new_event()` and `stop()` are Tk-thread
  only; workers get the event as an argument. An event is only ever set, never
  cleared, which is what makes an interrupt final.
- [`speakloop/languages/`](speakloop/languages) - one pure-data `PROFILE` per
  language (`english.py`, `spanish.py`), the format documented in its
  `__init__.py`. No imports and no side effects: `config.py` assembles them.
- [`speakloop/llm.py`](speakloop/llm.py) - `LLMManager`: OpenAI-compatible
  streaming client with conversation history; used by both backends. The
  response is streamed inside a `with`, so an interrupt closes it and the
  server stops generating. A failed request **raises** after rolling the user
  message back out of the history: the window owns what the user is told.
  `error_message()` is that text - the server's own sentence out of the JSON
  body, since `str()` of an API error is the whole HTTP problem.
- [`speakloop/llm_server_ctl.py`](speakloop/llm_server_ctl.py) -
  `LLMServerController`: starts and stops the llama-server subprocess (own
  process, so the model server and Kokoro do not contend for the GPU).
  `llama_server_command()` is pure and holds the tuning that must not be left
  to the binary's defaults (`--ctx-size`, `--parallel 1`, `--cache-reuse 256`,
  `--api-key`, `--no-ui`); `log_compute_devices()` records `--list-devices`
  before the launch, which is the only thing that reveals a silent CPU
  fallback. `start()` and `shutdown()` share one lock, so a quit during
  startup cannot orphan a server. `_use_running_server()` is the fork before
  the launch: a server already listening on the port is used as it is (and
  never terminated on exit), because an app that died without `quit_app` left
  it there and a second one could not bind the port; a port held by anything
  that does not answer the API is refused instead. Every refusal leaves one
  sentence in `last_error`, which is what the window shows. An adopted server
  is then described in the log: its model ids, and its per-slot context from
  `server_properties()` (a plain GET of llama.cpp's `/props`, which is outside
  the `/v1` prefix, hence not through the OpenAI client). Both are diagnostic
  and warn on a mismatch with the configured model or a smaller context; a
  server that answers is never refused over them.
- [`speakloop/tts.py`](speakloop/tts.py) - two roles: a **synthesis backend**
  per engine (`KokoroBackend` on torch at 24 kHz, `SupertonicBackend` on ONNX at
  44.1 kHz), selected from the `TTS_BACKENDS` registry by the active variant's
  data and never by an `if language` branch; and the **playback** path
  (`play_array`), winsound on Windows and sounddevice elsewhere. `TTSManager`
  is the facade: `synthesize()` then `play_array()`, with `sample_rate` taken
  from the active backend - callers must never assume a rate. The winsound path
  deliberately takes **no** `config.AUDIO_LOCK`: winsound does not touch
  PortAudio, and holding the lock there made a new recording wait for the speech
  to end, which cut the beginning off the take.

### Configuration and paths

- [`speakloop/config.py`](speakloop/config.py) - all configuration, frozen at
  import. Layers, lowest first: literals in the file ->
  `config/hardware_config.json` (`"config"` section) ->
  `config/settings.json`. Known settings.json keys are listed in
  `_KNOWN_USER_KEYS` (`max_record_seconds`, `silence_timeout`,
  `silence_threshold`, `accent`, `voice`, `color_theme`, `llm_backend`,
  `lm_studio_host`, `llama_server_path`, `external_model_path`,
  `external_n_ctx`); keep `config/settings.example.json` in step (a test checks
  it). The **language section comes before the download section on purpose**:
  the offline gate has to know the active synthesis backend, because only that
  backend's model has to be cached before the Hub is switched off. Every
  language constant (`TARGET_LANGUAGE`, `WHISPER_LANGUAGE`, `TTS_BACKEND`,
  `TTS_LANG_CODE`, `TTS_VOICE`, `TTS_VOICES`, `TTS_TOTAL_STEPS`, `TTS_WARMUP`)
  is derived from the profile and its variant - add a language by adding a
  profile, never a branch. `PRACTICE_LANGUAGE` is fixed to `"english"` and has
  no settings key until stage 3. The UI palette is resolved here too: `_DARK_THEME` is the built-in
  palette, the complete list of valid color keys and the fallback for a missing
  file or key, and `THEME` is it overlaid with the selected
  `<name>_schema.json` (tests pin that the shipped dark schema equals
  `_DARK_THEME`).
  `resolve_llama_server_path()` is a function and not a constant on purpose:
  the binary can be installed or removed while the app is not running, so the
  answer is taken from the disk when the server is started. Also sets
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
  `--version` and `--list-devices` probes. `installed_exe()`, `list_devices()`
  and `installed_variant()` are also what config and llm_server_ctl use at run
  time.

- [`speakloop/models_info.py`](speakloop/models_info.py) - model catalogue,
  the single place a repo id is written. Imports only `typing` (a test
  enforces it). Sizes are re-snapped with
  [`tools/measure_model_sizes.py`](tools/measure_model_sizes.py).
- [`install.py`](install.py) - interactive installer, a thin wrapper around the
  fetchers; dependency list read from `pyproject.toml`; logs to
  `logs/install.log`.

## Key Patterns & Gotchas

- **Threading**: recording, TTS queue processing and model loading run in
  daemon threads. Always update the window with
  `root.after(0, self.view.<intent>, ...)` - never call a view method straight
  from a worker thread, and never reach for a widget.
- **Reply marker pattern** (`app.py`): LLM sentences are buffered in the TTS
  queue processor and synthesized only when the `_ReplyEnd` marker of their
  reply arrives, so the model server has finished before the synthesis starts.
  Do not remove it while the model and the synthesis share one GPU. The marker
  **carries the stop event of its own reply**: that is what makes a buffered
  reply drop itself after an interrupt instead of being spoken over the next
  take.
- **One stop event per reply** (`playback.py`): a single application-wide event
  had to be cleared before the next reply, and the clear ran in the OLD
  exchange's worker after a new take had already set it - the interrupt was lost
  (problem 1 of the plan). Never reintroduce a shared, cleared event.
- **LLM history rollback** (`llm.py`): the user message is appended inside
  `try`; on exception it is popped to keep user/assistant pairs consistent.
  History is trimmed to `LLM_HISTORY_MAX_PAIRS` pairs after each exchange.
- **Sentence streaming** (`llm.py`): output is split on sentence-ending
  punctuation followed by whitespace and an uppercase letter
  (`(?<=[.!?])\s+(?=[A-ZА-Я])`); the rest is flushed at the end of the stream.
- **Audio normalization** (`recorder.py`): peaks are normalized before STT; a
  peak below 0.01 is not boosted.
- **Windows audio**: TTS plays through `winsound` to bypass PortAudio/MME
  driver issues, with a 150 ms silence lead-in. `config.AUDIO_LOCK` serialises
  PortAudio init and teardown between the recording and the sounddevice
  playback path only - the winsound path takes no lock at all (see tts.py).
  **Known limit**: `winsound.PlaySound(None, 0)` does not cut a synchronous
  playback started by another thread, it waits for it, so an interrupt costs up
  to the length of the sentence being spoken and the window is blocked for that
  time (problem 12 in `docs/refactoring.md`, with the two candidate fixes). The
  stop-guard thread in `play_array` cannot help with this.
- **Resampler warm-up** (`recorder.warm_up_resampler`, called from
  `load_components`): the first `librosa.resample` cost 9.6 s, paid by the
  learner between their first phrase and the answer. Do not drop the call
  without moving that cost somewhere else.
- **LLM server subprocess**: started in `LLMServerController.start()` with
  layers and context from `config` (hardware detection), polled with
  `LLMManager.check_connection()`, terminated in `quit_app()` with a 5-second
  kill fallback, output in `logs/llm_server.log`. An **adopted** server (one
  that was already listening) has no subprocess behind it, so `shutdown()` is
  a no-op for it by construction - do not "fix" that into terminating it.
- **Aborting speech** (`app.py` `_answer`): emptying the queue is not enough,
  because the TTS thread buffers the sentences it has already taken from it. A
  `_ReplyEnd` marker whose event is set is what makes that thread drop them;
  without it a failed exchange is spoken after the next reply. For the same
  reason every exchange that opened the model stream must always end with one
  marker.
- **A take during the previous exchange** (`app.py`): it is neither dropped nor
  run in parallel. `_finalize_recording` collects the take BEFORE it waits for
  `_exchange_lock` (a later take would otherwise replace the chunk buffer it is
  about to read), and the previous exchange gives the lock up quickly because
  the new take already set its stop event. An exchange whose event was set
  before the model call shows the recognized phrase and asks nothing.
- **Known v0 issues** are listed in `docs/refactoring.md` with the step that
  fixes each. Do not fix them outside their step.

## Testing

```bash
python -m unittest discover -s tests -v
```

Test files are named after the module they cover (`tests/test_paths.py` for
`speakloop/paths.py`). The suite stubs subprocesses, the OS, the audio devices
and the network, and downloads nothing. `tests/test_config.py` imports the real
`config` of the checkout (and so torch). `tests/test_recorder.py` runs the real
capture thread against a stand-in `sd.InputStream`, so it is the one file with
short waits in it; `tests/test_tts.py` replaces the synthesis backend and never
loads a model; `tests/test_languages.py` imports the profile modules alone and
needs neither config nor torch.

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
