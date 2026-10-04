# SpeakLoop - status snapshot (2026-09-11, project revival)

Source: static review of `D:\LLM-work\SpeakLoop` (no code was run).

## What it is
Desktop voice tutor (Tkinter, Windows). Push-to-talk (Space) -> faster-whisper STT -> local LLM (llama-3.2-3b GGUF via own FastAPI `llm_server/` subprocess, or LM Studio) -> Kokoro TTS via winsound. Persona "Emma", Russian -> English. MIT license, remote github.com/vikonix/SpeakLoop.

## History
- 2026-05-17: repo created. 2026-06-05: last code commits ("fix 3"). 2026-06-06: two specs in `proposals/`. 2026-06-13: last commit ("fix readme").
- Log of 2026-06-05 shows a working end-to-end loop: STT ~0.8 s, LLM ~0.5-1 s.

## Planned directions (proposals/, Russian)
- A2 "ReadLoop": graded text bank (CEFR), lesson phases, tutor with initiative.
- B "EchoLoop": pronunciation drill (OpenPronounce core, Wav2Vec2, DTW, espeak-ng).
- A3 (mentioned only): change or fine-tune the LLM.

## Findings from review
Bugs / risks:
1. Race: Space pressed during STT of previous turn -> `tts_stop_event.clear()` in `process_audio` cancels the interrupt; reply plays while user records; second utterance can be dropped by `is_processing_audio` guard with only a log warning.
2. LLM error path: `stream_and_queue_tts` returns "Sorry..." but it is not shown or spoken; UI shows empty "Emma:" line.
3. OpenAI stream is not closed on interrupt -> server keeps generating (small impact with max_tokens=50, but breaks sentinel GPU assumption).
4. Orphan `llm_server` process if app crashes or quits during startup; port 8765 stays busy, next start may connect to the old server.
5. Whisper is forced to `language=en`; Russian speech from the learner is not handled.
Config / docs drift:
- `EXTERNAL_N_GPU_LAYERS=20` = partial offload (model has 28 layers); on RTX 3090 full offload (-1) fits.
- `KOKORO_WARMUP_WORDS` has "g" (German) and "r" (Russian); Kokoro has no such languages.
- Language switch needs 4 manual edits (TARGET_LANG_CODE, KOKORO_LANG_CODE, KOKORO_VOICE, prompt).
- Root `requirements.txt`: `llama-cpp-python` (belongs to server only), `keyboard` (unused).
- `llm_fallback_warning` is dead code; `config.py` mentions removed `local_gguf` backend.
- AGENTS.md: split regex and `sound_lock` name are outdated.
- README clone URL is a placeholder; `.roomodes` is a C++ template.
- No tests.
