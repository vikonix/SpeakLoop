# Mimora -> SpeakLoop reuse inventory (2026-09-11)

Goal (owner): SpeakLoop becomes a voice dialogue trainer for the language-teacher workspace, running the
free-talk prompt (`language-teacher/tmp/PROMPT-free-talk-2026-09-10.md`). Take from Mimora (1.1.0, D:\LLM-work\Mimora)
everything that is already solved there. Static review only, nothing was run.

## Take as is (small adaptation: package name, config keys)
- `llm_server_ctl.py` - llama-server subprocess: race-safe start/shutdown (fixes SpeakLoop orphan server),
  flags `--ctx-size`, `--parallel 1`, `--cache-reuse 256`, `--api-key`, `--no-ui`; `--list-devices` probe
  against silent CPU fallback.
- `llama_server_fetch.py`, `gguf_fetch.py` - pinned llama.cpp release with sha256; GGUF download.
  Replaces SpeakLoop `llm_server/` (FastAPI + llama-cpp-python), same move Mimora made in 1.1.0.
- `audio_io.py` - open-stream counter, PortAudio reset only on Windows and only with no open stream.
- `recorder.py` - WASAPI native-rate capture + resample, lock-free callback (fixes clicks), silence
  auto-stop, level meter, max-duration through the normal stop path, refuse start while old thread alive.
- `tts.py` - backend registry (Kokoro, Supertonic), `.eval()` and `KPipeline(model=...)` fixes,
  winsound stop-guard race fix, `sample_rate` property, `loudness_envelope`.
- `playback.py` - per-playback stop events (the fix for SpeakLoop interrupt race).
- `lifecycle.py` - `hard_exit` via TerminateProcess (CUDA crash on exit), relaunch.
- `bootstrap.py`, `cli.py` - early_init before heavy imports, logging with run header, stdlib-only entry.
- `paths.py`, `loader.py` - data/shipped/resource roots, env override, atomic settings.json writes.
- `detect_hardware.py`, `models_info.py`, `model_fetch.py`, `first_run*.py` - hardware probe,
  model catalogue, first-run download window.
- `languages/` profile-as-data pattern (fixes SpeakLoop "4 manual edits per language").
- Packaging: pyproject with reasoned pins, Python 3.11-3.12, uv tool install, fast unit tests with stubs.
- UI patterns: `ui.py` facade with `enter_*` states + `ViewCallbacks`, ttkbootstrap themes, declarative settings.

## Not needed
pronunciation engines, wav2vec2, espeak, prosody, NLLB translator, phrase generation/sliding window
(ideas worth keeping: wordfreq level check, prefix-cache-friendly stable system prompt).

## Missing in Mimora, SpeakLoop must provide
- STT (faster-whisper, from SpeakLoop), streaming chat with history, sentence-level TTS.
- Output contract parser: `NOTE:` (screen only) / `SAY:` (TTS) / `SUMMARY:` (multi-line, screen only).
- Driver for LESSON ARC (later), learner commands (finish, simpler, hint, new topic) as buttons.
- Session header input (Target language, Explanation language, First topic) and transcript output.

## Facts that constrain the design
- Prompt body ~9 KB, ~2300 tokens: SpeakLoop `n_ctx=2048` and `LLM_MAX_TOKENS=50` cannot run it.
- Prompt was tested on Ornith-1.5-9B and Gemma-4-12B-QAT, not on llama-3.2-3b.
- Whisper forced to one language breaks mixed-language turns the prompt allows; Whisper may also
  silently repair learner grammar, which changes what the corrections see.
- language-teacher IDEA-voice-scenes: the early local-LLM partner (SpeakLoop) "worked but was boring";
  option C (free talk on local model without scene/driver) was marked "do not do". The free-talk prompt
  is the new, constrained version of that format.
