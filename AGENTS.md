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
Studio) -> speech synthesis (Kokoro for English, Supertonic for Spanish). The
learner can also type the phrase in the control panel and send it with Enter,
which skips the two speech steps and is otherwise the same exchange.

The lesson runs on the free-talk prompt
([`speakloop/prompts/free_talk.md`](speakloop/prompts/free_talk.md), the main
copy of it). The model opens the lesson itself and answers in lines that begin
with `NOTE:` (a correction, shown only), `SAY:` (the partner's line, shown and
spoken) or `SUMMARY:` (the lesson summary on "finish", shown only). The prompt
text belongs to the owner: change it only on request.

Every lesson is also written to `transcript/` while it runs
([`speakloop/transcript.py`](speakloop/transcript.py)): a jsonl file with one
record per event, which is the form the learning system reads, and a markdown
view of the same records for reading by eye.

The language of the lesson is data: `speakloop/languages/` holds one profile per
language and `config.py` derives every per-run language constant from the active
one. English is fixed; Spanish is postponed (the owner's decision), although its
profile stays complete.

The chat model is Gemma 4 12B (stage 2). How its server parameters were chosen,
and what it does on a small card, is in
[`docs/model-parameters.md`](docs/model-parameters.md) (Russian): read it before
touching GPU layers, context size or the speech devices.

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
  key auto-repeat guard. `trigger_recording_start` claims the floor (stops the
  speech, which sets the stop event of the reply that is running) only AFTER
  `recorder.start()` has agreed to the take: a refused take that had already
  stopped the playback left the exchange in flight without an answer and the
  window in its processing state for good (problem 13 in
  `docs/refactoring.md`). A press within `LATE_STOP_WINDOW_SECONDS` (1.5 s)
  after an automatic stop (`_auto_stop`: a pause or the time limit) is the
  learner's late stop and opens no take. A typed phrase enters the same flow at the model
  request: `on_text_submitted` stops the speech of the previous reply, writes
  the phrase into the chat and gives `_run_typed_exchange` a stop event of its
  own, so voice and keyboard share every step from `_ask_model` on. A command
  button takes the same path through `_submit_phrase` (`on_command_pressed`):
  a command is an ordinary phrase of the learner and stays in the history as
  one. `on_notes_toggled` only remembers the choice
  (`config.save_user_setting`); the hiding itself is the view's.
  `_record` and `_system` are the two ways into the transcript: every event of
  the lesson becomes a record, and `_system` is the ONE place a `[System]`
  line is written, so the window and the file always say the same. A phrase
  of the learner gets its number from `_next_turn` (under a lock: typed
  phrases are numbered on the Tk thread, spoken ones on the exchange thread)
  and the number is passed on explicitly, so a reply carries the number of
  the phrase it answers, which is what joins a NOTE to the phrase it corrects.
  A phrase that gets no answer because a new take started (superseded before
  the model call, or its reply interrupted) gets the `UNANSWERED_MESSAGE`
  line (`_report_unanswered`, not during quit).
  `_record_reply` puts `llm_ms` and `tokens` on the first record of a reply
  (the same numbers on a NOTE and a SAY would read as two answers) and writes
  the markdown view when the summary arrives; `quit_app` writes it again, for
  a lesson that ended without one.
  `_exchange_lock` keeps one exchange at a time, so a
  take made during the previous exchange waits instead of being dropped.
  `load_components` builds the lesson (`prompt.build_system_prompt`, then
  `Lesson`) before it loads any model, so a bad prompt file stops the start at
  once. `make_app_ready` opens the lesson (`_open_lesson`): the first model
  request runs like an exchange, with its own stop event and under the same
  lock. `_ask_model` is the one path to the model: it shows the reply
  (`_show_reply`: NOTE, SAY, SUMMARY in this order, or the whole text of a
  reply outside the contract) and queues only the sentences of SAY, with the
  markdown taken out of them for the speech alone. A reply outside the
  contract also gets a `[System]` line, written after the reply itself so the
  window and the file both hold the two in that order; the model is not asked
  again.
  Every window state a worker sets goes through `_enter_later`, which runs
  the intent on the Tk thread through `_enter_if_current`: only for the reply
  that is still current, so a late worker cannot draw over a new take.
  Module-level `run(append_log)` configures logging, logs
  `detect_hardware.warn_if_gpu_unused`, and opens the window. **Imports
  `speakloop.config` before `stt`/`tts`** (see config below).
- [`speakloop/ui.py`](speakloop/ui.py) - `TutorView`: the whole window (header,
  the control panel at the top - mic canvas, text entry and instruction line -
  the chat transcript below it, and the status bar) plus the *intent* methods the
  controller calls (`enter_recording`, `enter_thinking`, `enter_error`, …) and
  the `append_*` transcript writers (`append_partner_msg`, `append_note`,
  `append_summary`, ...). The partner is labelled `PARTNER_NAME = "Tutor"`,
  a role and not a name, because the prompt gives the model no name. NOTE and
  SUMMARY use existing palette keys, so an older user theme still has every
  color. Widget bindings call only the callables in
  the `ViewCallbacks` passed in, so the view never references the controller.
  What the window shows from outside comes in `ViewSettings` (the lesson
  language, the first Notes state, the commands); the view imports neither
  `config` nor `prompt` (only `ui_theme` reads the palette from `config`).
  Every status string, instruction line and the partner's name (`PARTNER_NAME`)
  live here - do not move wording or colors back into the controller. Its
  methods must run on the Tk main thread. `centered_geometry()` is pure on
  purpose, so where the window opens is tested without a display
  (`tests/test_ui.py`). `set_record_level()` repaints the mic
  button from the live input level while a take runs; the recording state has no
  glyph of its own any more, because the level disc IS the indicator.
  The text entry is the second way to answer: `<Return>` on it sends
  `clean_input()` of its content through `on_text_submitted` (empty sends
  nothing), so the controller receives a phrase and never a widget. The space
  bindings sit on the root window and therefore also see keys typed in the
  entry - `_typing()` is what keeps a space inside a phrase from starting a
  take. `_set_input_enabled()` is called from the `enter_*` intents themselves
  (closed while recording, transcribing and answering; open while the partner
  speaks, where Enter interrupts it), so the controller never enables the entry
  by hand. The status bar shows the state alone: the STT and LLM durations are
  in `logs/main.log` and `update_stats` is gone.
  The second row of the panel holds the lesson commands and the Notes switch.
  The command buttons are built from `ViewSettings.commands`, which app.py
  fills with `prompt.LESSON_COMMANDS`, so a button sends the word of the
  prompt itself and is never spelled here.
  `_apply_notes_visibility()` is the whole Notes feature: it sets `elide` on
  the two NOTE tags, which hides the corrections already in the chat and every
  one that comes later; the text stays in the widget. SUMMARY has tags of its own and is never hidden. The
  switch itself is outside `_set_input_enabled`: it sends nothing to the model
  and stays usable in every state.
- [`speakloop/ui_theme.py`](speakloop/ui_theme.py) - the palette (`THEME` from
  config), the ttkbootstrap base theme, `FONT_FAMILY` per platform and the
  `FONT_SIZE_*` scale. Importing it also **disables ttkbootstrap's
  classic-widget autostyle hook**, which would otherwise repaint every `tk`
  widget with the base theme's colors; that is why ui.py imports it first.
- [`speakloop/themes/`](speakloop/themes) - `dark_schema.json` (the default,
  equal to the built-in palette) and `light_schema.json`. A user copy in
  `config/themes/` of the same name wins over these.
- [`speakloop/stt.py`](speakloop/stt.py) - `STTManager`: faster-whisper
  large-v3-turbo with VAD filtering, on `config.STT_DEVICE` (`int8_float16` on
  the GPU, which leaves more video memory to the chat model), in the language
  of the active profile
  (`config.WHISPER_LANGUAGE`; never automatic detection). Decoding falls back
  to higher temperatures (`WHISPER_TEMPERATURES`) when a segment loops; with
  0.0 alone the loop reaches the chat model. **Imports torch before
  faster_whisper**: on Windows the CUDA build of torch provides the cuBLAS and
  cuDNN libraries ctranslate2 needs. `load_model()` logs the device and the
  compute type, the only place the log names where recognition runs.
- [`speakloop/recorder.py`](speakloop/recorder.py) - `AudioRecorder`: the
  capture thread, the input device choice and the chunk buffer, plus the pure
  `normalize_audio`. It captures at the device's own rate through WASAPI on
  Windows (MME drops samples) and downsamples the take to
  `config.AUDIO_SAMPLE_RATE` with librosa. Its four callbacks
  (`on_max_duration`, `on_silence_stop`, `on_stream_error`, `on_level`) all run
  on the capture thread. Every take is a `Take` of its own (chunks, capture
  rate, thread): `stop()` returns it, and `join(take)` and `get_audio(take)`
  work on that object, so a new take cannot empty or resample a take that has
  not been read. The realtime callback takes **no lock** - that was the
  source of dropped samples - so a take may only be read after `join()`.
- [`speakloop/audio_io.py`](speakloop/audio_io.py) - the device plumbing both
  the microphone and the speaker need: `reset_portaudio()` (Windows only, and
  skipped while any stream is open - it invalidates every stream in the
  process), the open-stream counter with `AUDIO_LOCK`, the one lock that
  serializes every PortAudio open, close and reset, and `uses_winsound()`.
- [`speakloop/playback.py`](speakloop/playback.py) - `PlaybackController`: the
  stop event of the **current reply**. `new_event()` and `stop()` are Tk-thread
  only; workers get the event as an argument. An event is only ever set, never
  cleared, which is what makes an interrupt final.
- [`speakloop/languages/`](speakloop/languages) - one pure-data `PROFILE` per
  language (`english.py`, `spanish.py`), the format documented in its
  `__init__.py`. No imports and no side effects: `config.py` assembles them.
- [`speakloop/prompt.py`](speakloop/prompt.py) - builds the system message:
  reads the prompt file (`config.PROMPT_FILE`) and fills its three SETTINGS
  lines (`Target language: [...]` and so on). An empty value keeps the
  brackets, which the prompt reads as its own default. A missing or repeated
  SETTINGS line **raises**: otherwise the lesson silently runs on a default.
  No config import; the caller passes the values. The system message is built
  once and never changes during a session (prefix cache). `LESSON_COMMANDS`
  lives here too - the four command words of the prompt, which the window's
  buttons send as they are (`tests/test_prompt.py` checks that the shipped
  prompt still names each of them).
- [`speakloop/contract.py`](speakloop/contract.py) - pure code for the output
  contract: `parse_reply()` gives a `Reply` (`note`, `say`, `summary`, `raw`,
  `follows_contract`). SUMMARY takes everything from its line to the end,
  also after a `NOTE:` or `SAY:` prefix on that line (Gemma writes
  `NOTE: SUMMARY:`, because the prompt says every line begins with one of
  them), and a `SUMMARY:` label repeated at the start of the summary is
  dropped; above it the first NOTE and the first SAY are taken. Prefixes are matched
  in capitals at the start of a line only - a NOTE read as SAY would be spoken
  in the explanation language, and a label with markdown around it
  (`**SAY:**`) is therefore a reply outside the contract, not a line to speak.
  `split_sentences()` cuts SAY for speech, and `strip_markdown()` takes the
  inline markers out of it first: the synthesis reads a marker as a sound. A
  marker counts only as a pair with no space against the text, and an
  underscore only between word boundaries, so "2 * 3 * 5" and "read_file_name"
  stay whole. The cleaning is for speech alone - the chat and the transcript
  keep the line as the model wrote it. What to do with a reply that breaks the
  contract is the controller's (see app.py).
- [`speakloop/conversation.py`](speakloop/conversation.py) - `Lesson` over
  `LLMManager`: `open()` sends `OPENING_MESSAGE` ("Begin.", a user message the
  chat template needs before the first question; it stays in the history and
  is never shown), `answer()` sends the learner's phrase **as recognized**
  (the prompt commands get no special handling). Both return a parsed
  `Reply`, or None after an interrupt, and log a warning for a reply outside
  the contract. `context_level_reached()` is the pure rule of the context
  warning: the highest of `CONTEXT_WARNING_LEVELS` (80 and 90 percent) that
  the token count has reached above the level already shown; the last level
  also comes when fewer than `reserve` tokens are left (app.py passes
  `CONTEXT_RESERVE_TOKENS`, two reply limits), because on a small context 10
  percent is less than a summary needs.
- [`speakloop/transcript.py`](speakloop/transcript.py) - the two files of one
  lesson. The record functions are pure (a record is a dict, the clock is an
  argument), `TranscriptWriter` owns the files and one lock, because records
  arrive from the exchange threads and from the Tk thread alike. The jsonl
  line is appended as the event happens and the markdown view is built from
  the kept records at the end, which is why the machine form is the one that
  is written first: markdown can be built from jsonl, not the other way round.
  `SCHEMA_VERSION` is what a reader checks; raise it when a field changes its
  meaning, not when an optional one is added. No file is created before the
  first record (today that is the first `[System]` line at startup, E1 of
  `docs/review-2026-09-23.md`), and a file that cannot be written is logged once and then left
  alone - a lesson must not end because a disk is full.
- [`speakloop/llm.py`](speakloop/llm.py) - `LLMManager`: OpenAI-compatible
  client with the conversation history; used by both backends. app.py
  points it once at `config.LLM_URL` with `config.LLM_API_KEY`
  (`init_client`, no request); nothing else changes the address.
  `start_conversation()` sets the system message; `ask()` refuses to run
  without it. `ask()` returns the **whole** reply as one text (the contract
  needs all of it), but the response is still streamed inside a `with`: an
  interrupt closes it and the server stops generating. After an interrupt
  `ask()` returns None and removes the user message too, so the history is as
  before the call and never has two user messages in a row. A failed request
  or an empty reply **raises** after the same rollback: the window owns what
  the user is told.
  `error_message()` is that text - the server's own sentence out of the JSON
  body, since `str()` of an API error is the whole HTTP problem.
  `is_context_overflow()` recognizes a refusal because the context is full:
  llama-server's error type `exceed_context_size_error`, a message about
  the context length (LM Studio), or `EmptyCutReplyError`, which `ask()`
  raises for a reply cut off by the length limit before any text (the model
  used the last free tokens and wrote nothing). `last_reply_cut` is True when the server
  ended the last reply with `finish_reason: "length"` (the reply limit or the
  end of the context): such a reply looks complete, so app.py shows
  `CUT_REPLY_MESSAGE` under it.
  Every request asks for the usage report
  (`stream_options.include_usage`): the last chunk of the stream then has the
  usage and **no choices**, so the loop must skip it, and `usage_log_line()`
  writes the context size to `logs/main.log` after each reply ("unknown"
  after an interrupt, which ends the stream before the report).
  `last_total_tokens` keeps that number for the transcript and the context
  warning: it is cleared before every request, so a caller
  never reads the size of the reply before it.
  `LLM_TIMEOUT` (360 s) is the longest pause before the first token on a weak
  machine, and the client makes **no retries** (a retry repeats the prompt
  processing); `check_connection` has its own short timeout.
  `request_extra_body()` adds fields outside the OpenAI API **for llama-server
  only**: `chat_template_kwargs.enable_thinking=false` (llama-server turns
  Gemma's thinking on by default, and it cost 40 s per one-line reply in
  `reasoning_content`, which this module never reads) and `top_k`. Do not drop
  it or send it to LM Studio. Sampling values are Gemma's recommended ones
  (`LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_TOP_K` in config).
- [`speakloop/llm_server_ctl.py`](speakloop/llm_server_ctl.py) -
  `LLMServerController`: starts and stops the llama-server subprocess (own
  process, so the model server and Kokoro do not contend for the GPU).
  `llama_server_command()` is pure and holds the tuning that must not be left
  to the binary's defaults (`--ctx-size`, `-fitc`, `--parallel 1`,
  `--cache-reuse 256`, `--api-key`, `--no-ui`; llama.cpp disables
  `--cache-reuse` for Gemma with a warning, and it stays for the fallback
  model); `log_compute_devices()` records `--list-devices`
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
  server that answers is never refused over them. The context is read for a
  launched server too and kept in `served_n_ctx`, because llama.cpp's memory
  fit can shrink it silently; app.py shows a chat warning when it is smaller.
  GPU layers: with `EXTERNAL_N_GPU_LAYERS == "auto"` **no `--n-gpu-layers` is
  passed** - an explicit value switches the fit (`-fit`) off - and `-fitc`
  always equals `--ctx-size`, so the fit gives up layers, not context
  (`docs/model-parameters.md`). The subprocess runs with
  `server_environment()`: this process's environment plus
  `GGML_OP_OFFLOAD_MIN_BATCH=16` (`setdefault`, so an exported value wins), so
  a learner's reply of about 30 tokens is computed on the GPU under partial
  offload; the value is logged because it is not on the command line.
- [`speakloop/tts.py`](speakloop/tts.py) - two roles: a **synthesis backend**
  per engine (`KokoroBackend` on torch at 24 kHz, `SupertonicBackend` on ONNX at
  44.1 kHz), selected from the `TTS_BACKENDS` registry by the active variant's
  data and never by an `if language` branch; and the **playback** path
  (`play_array`), winsound on Windows and sounddevice elsewhere. `TTSManager`
  is the facade: `synthesize()` then `play_array()`, with `sample_rate` taken
  from the active backend - callers must never assume a rate. The winsound path
  deliberately takes **no** `audio_io.AUDIO_LOCK`: winsound does not touch
  PortAudio, and holding the lock there made a new recording wait for the speech
  to end, which cut the beginning off the take.

### Configuration and paths

- [`speakloop/config.py`](speakloop/config.py) - all configuration, frozen at
  import. Layers, lowest first: literals in the file ->
  `config/hardware_config.json` (`"config"` section) ->
  `config/settings.json`. Known settings.json keys are listed in
  `_KNOWN_USER_KEYS` (`max_record_seconds`, `silence_timeout`,
  `silence_threshold`, `stt_device`, `accent`, `voice`, `color_theme`, `llm_backend`,
  `lm_studio_host`, `llama_server_path`, `external_model_path`,
  `external_n_ctx`, `external_n_gpu_layers`, `first_topic`, `prompt_file`,
  `show_notes`);
  keep
  `config/settings.example.json` in step (a test checks it).
  `EXTERNAL_N_GPU_LAYERS` is a **string** (`"auto"`, `"all"` or digits, from
  `_gpu_layers_setting`) and `EXTERNAL_N_CTX` defaults to 16384; neither is
  read from hardware_config.json. `TTS_DEVICE` comes from `_model_device`,
  capped by `DEVICE`. `STT_DEVICE` comes from the `stt_device` key
  (`_stt_device`): `"auto"` is the detected value through `_model_device`, and
  a hand-set `"cuda"` is capped by `DEVICE` too. The **language section comes before
  the download section on purpose**: the offline gate has to know the active synthesis backend, because only that
  backend's model has to be cached before the Hub is switched off. Every
  language constant (`TARGET_LANGUAGE`, `WHISPER_LANGUAGE`, `TTS_BACKEND`,
  `TTS_LANG_CODE`, `TTS_VOICE`, `TTS_VOICES`, `TTS_TOTAL_STEPS`, `TTS_WARMUP`)
  is derived from the profile and its variant - add a language by adding a
  profile, never a branch. `PRACTICE_LANGUAGE` is fixed to `"english"` and has
  no settings key (Spanish is postponed). The lesson section holds
  `EXPLANATION_LANGUAGE` (fixed to Russian, no key: the NOTE example in the
  prompt is Russian), `FIRST_TOPIC` and `PROMPT_FILE` (default: the file in
  `speakloop/prompts/`). The UI palette is resolved here too: `_DARK_THEME` is the built-in
  palette, the complete list of valid color keys and the fallback for a missing
  file or key, and `THEME` is it overlaid with the selected
  `<name>_schema.json` (tests pin that the shipped dark schema equals
  `_DARK_THEME`).
  `TRANSCRIPT_DIR` names the transcript directory; the files inside it are
  named per lesson by `transcript.py`, which is why only the directory is
  resolved here.
  `SHOW_NOTES` (key `show_notes`) is the one value the application writes
  back: `save_user_setting()` is the only writer of `settings.json`
  (`SETTINGS_FILE`), through `loader.save_setting`, which re-reads the file and
  replaces it atomically so hand-edited and comment keys survive. The constants
  themselves stay frozen at import.
  `LLM_URL` and `LLM_API_KEY` are the address and the key of the selected
  backend (`_client_address`), the only ones the chat client uses.
  `LLAMA_SERVER_PATH` is only the `llama_server_path` setting (`""` when it
  is empty); the search for the binary is `llm_server_ctl.find_llama_server`,
  so config does not import the fetchers' `llama_server_fetch`. Also sets
  `HF_HOME` to `model_cache/` and switches `HF_HUB_OFFLINE=1` once the repos
  this run loads are cached - which is why it must be imported before anything
  imports huggingface_hub. `_model_device()` caps a per-model device by
  `DEVICE`.
- [`speakloop/loader.py`](speakloop/loader.py) - pure config-loading helpers
  (JSON read, validated values, atomic save, cache predicate,
  `detect_device`). The cache predicate `models_cached` takes a hub repo for
  complete only when its snapshot holds the record's `weights_file` and no
  `*.incomplete` blob is left: huggingface_hub 1.x deletes a partial file on
  a failed download, so without the first check a repo with only its small
  files is skipped by the installer and switched offline.
- [`speakloop/paths.py`](speakloop/paths.py) - where every file lives, and the
  only module that knows. `transcript_dir()` sits beside `log_dir()` rather
  than inside it, because a transcript is the result of a lesson and the logs
  directory is what somebody deletes after an investigation; `ensure_dirs()`
  creates it with the rest. `data_root()` (what this machine writes; the clone in
  repo mode, the OS user-data directory for an installed package,
  `SPEAKLOOP_HOME` overrides) and `shipped_root()` (read-only files inside the
  package). **Stdlib-only** - install.py reads it before the requirements
  exist.
- [`speakloop/detect_hardware.py`](speakloop/detect_hardware.py) - machine probe
  writing `config/hardware_config.json`: `DEVICE` (torch CUDA - only that,
  `warn_if_gpu_unused` reads it so), `STT_DEVICE` (a ctranslate2 CUDA device,
  on a card of any size) and `TTS_DEVICE` (Kokoro), audio devices. Kokoro
  drops to the CPU on a card below `TTS_GPU_MIN_VRAM_GB` (12) that the chat
  model uses (the installed llama-server's `--list-devices` decides that). It writes no LLM
  layers or context any more, and config ignores those keys in an older
  file. **Must not import config.**

### Downloads

Three fetchers, all **forbidden to import `config`** (config switches the Hub
offline once models are cached, exactly when a download would be wanted) and
all keeping huggingface_hub out of their module-level imports so `install.py`
can use them before the requirements step:

- [`speakloop/model_fetch.py`](speakloop/model_fetch.py) - faster-whisper large-v3-turbo
  and Kokoro (hub cache), Supertonic 3 (own cache directory). Owns
  `prepare_hf_env()` (Windows symlink and hf-xet workarounds).
- [`speakloop/gguf_fetch.py`](speakloop/gguf_fetch.py) - the GGUF chat model
  into `models/`; `--fallback` (or `ensure_gguf(model=...)`) fetches
  `GGUF_CHAT_FALLBACK` instead, which the installer never does.
- [`speakloop/llama_server_fetch.py`](speakloop/llama_server_fetch.py) - the
  pinned llama.cpp release into `bin/llama/`, sha256 per asset, then
  `--version` and `--list-devices` probes. `installed_exe()`, `list_devices()`
  and `installed_variant()` are also what llm_server_ctl uses at run time.
  config does not import this module.

- [`speakloop/models_info.py`](speakloop/models_info.py) - model catalogue,
  the single place a repo id is written. `WHISPER` is faster-whisper
  large-v3-turbo; every `HfRepo` names its `weights_file` (see loader).
  `GGUF_CHAT` is Gemma 4 12B QAT Q4_0,
  `GGUF_CHAT_FALLBACK` is Llama 3.2 3B. Imports only `typing` (a test
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
- **Reply marker pattern** (`app.py`): the sentences of SAY are queued only
  after the whole reply has arrived, then the `_ReplyEnd` marker of their
  reply; the TTS thread synthesizes nothing before the marker. So the model
  server has finished before the synthesis starts. Do not queue speech before
  the reply is complete while the model and the synthesis share one GPU. The marker
  **carries the stop event of its own reply**: that is what makes a buffered
  reply drop itself after an interrupt instead of being spoken over the next
  take.
- **One stop event per reply** (`playback.py`): a single application-wide event
  had to be cleared before the next reply, and the clear ran in the OLD
  exchange's worker after a new take had already set it - the interrupt was lost
  (problem 1 of the plan). Never reintroduce a shared, cleared event.
- **LLM history rollback** (`llm.py`): the user message is appended inside
  `try`; on exception, on an empty reply and on an interrupt it is popped to
  keep user/assistant pairs consistent.
- **Whole history** (`llm.py`): the conversation is never trimmed. The lesson
  SUMMARY needs its start, and a trimmed start changes the prompt prefix, so
  the server processes the whole history again on every request. A history
  that does not fit the context makes the server refuse the request. Before
  that, app.py (`_warn_if_context_fills`) shows one `[System]` line at 80 and
  one at 90 percent of the context (the server's `served_n_ctx`, or
  `external_n_ctx` when the server reports none, as LM Studio does) with the
  advice to say "finish" (not after a SUMMARY: the lesson is over); after a
  refusal it shows `CONTEXT_FULL_MESSAGE` instead of the server's text,
  because even "finish" no longer fits.
- **Sentence split** (`contract.split_sentences`): SAY is split on
  sentence-ending punctuation followed by whitespace and an uppercase letter
  (`(?<=[.!?])\s+(?=[A-ZА-Я])`), and each sentence is synthesized and played
  on its own.
- **Audio normalization** (`recorder.py`): peaks are normalized before STT; a
  peak below 0.01 is not boosted.
- **Windows audio**: TTS plays through `winsound` to bypass PortAudio/MME
  driver issues, with a 150 ms silence lead-in. `audio_io.AUDIO_LOCK` serialises
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
- **Load order of the GPU users** (`app.py` `load_components`): Whisper is
  loaded before llama-server starts, so the server's memory fit sees the video
  memory Whisper holds and gives the chat model fewer layers, instead of
  leaving Whisper without memory. Do not move the Whisper load after the
  server start.
- **LLM server subprocess**: started in `LLMServerController.start()` with
  layers and context from `config` (settings.json; see llm_server_ctl above),
  polled with `LLMManager.check_connection()`, then asked for its context
  (`/props`), terminated in `quit_app()` with a 5-second
  kill fallback, output in `logs/llm_server.log`. An **adopted** server (one
  that was already listening) has no subprocess behind it, so `shutdown()` is
  a no-op for it by construction - do not "fix" that into terminating it.
- **Aborting speech** (`app.py` `_ask_model`): emptying the queue is not
  enough, because the TTS thread buffers the sentences it has already taken
  from it. A `_ReplyEnd` marker whose event is set is what makes that thread
  drop them. For the same reason every reply that queued a sentence must end
  with one marker; a failed or interrupted request queues nothing.
- **A take during the previous exchange** (`app.py`): it is neither dropped nor
  run in parallel. `_finalize_recording` gets its own `Take` from
  `trigger_recording_stop` and waits for `_exchange_lock`, and the previous
  exchange gives the lock up quickly because
  the new take already set its stop event. An exchange whose event was set
  before the model call shows the recognized phrase and asks nothing. Only a
  take that really starts may set that event: see `trigger_recording_start`
  above.
- **The transcript follows the window** (`app.py`): a record is added in the
  same place the same text goes to the chat, and its jsonl line reaches the
  disk at once. That is what makes the file complete up to the last event of a
  lesson that ended in a crash, and what keeps the two from drifting apart. Do
  not collect the lesson from the chat widget or from the LLM history instead:
  the widget would make the view the source of the data, and the history holds
  raw replies, the service `Begin.` and no times.
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
needs neither config nor torch, and so do `tests/test_prompt.py` (which also
reads the shipped prompt file), `tests/test_contract.py` and
`tests/test_conversation.py` (a stand-in for `LLMManager`).
`tests/test_transcript.py` needs neither config nor torch either; its writer
tests run against a temporary directory, and the two failure tests replace
`open` and `Path.write_text` with an `OSError`.

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
