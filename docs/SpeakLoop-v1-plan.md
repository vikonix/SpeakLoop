# SpeakLoop v1 plan (status: proposed, waiting for owner approval, 2026-09-11)

## Owner decisions
- Reuse: copy needed Mimora modules into SpeakLoop and adapt them. No shared library, not a Mimora mode.
- Integration: standalone for v1. Header entered in the app, transcript saved to a file. Contract with
  language-teacher (registry row, review of transcripts) decided later.
- Behaviour: free-talk prompt as is. LESSON ARC stays in the prompt. Program parses NOTE/SAY/SUMMARY and
  gives command buttons. Driver and misunderstanding budget later, with room for them in the design.
- Model: Gemma-4-12B-QAT is the main model.

## Proposed defaults (not yet confirmed)
- Full conversation history (no pair trimming: SUMMARY and "fact from earlier" need it), n_ctx 16384.
- Wait for the complete model message, validate the contract, then TTS the SAY line (simplest, allows a
  fallback when the form breaks). Sentence streaming is a later optimisation.
- Whisper forced to the target language (safer for accented speech); mixed-language turns are a known v1 limit.
- Text input field next to push-to-talk (debugging and the spelling rules of the prompt).
- Prompt body stored in SpeakLoop as a text file, path configurable; header placeholders and the NOTE example filled by code.
- TTS: Kokoro only (English), backend registry kept for Supertonic later.
- Transcript: plain text copy of the chat per session (NOTE, SAY, learner turns, SUMMARY).
- No first-run window and no wheel for v1: `pip install -e .`, fetch commands from the CLI.

## Target layout (package `speakloop/`)
- From Mimora: bootstrap, cli, lifecycle, paths, loader, audio_io, recorder, playback (no face), tts (Kokoro),
  llm_server_ctl, llama_server_fetch, gguf_fetch.
- From SpeakLoop: stt (faster-whisper), reworked.
- New: config (Mimora pattern, small), prompt (load + fill header), contract (NOTE/SAY/SUMMARY parser, pure),
  conversation (history, commands, LLM calls), transcript, app (controller), ui (facade with enter_* states).
- Removed: llm_server/, requirements.txt, old root modules (owner deletes files; git operations are the owner's).

## Milestones
1. Skeleton: pyproject, package, bootstrap/cli/paths/loader/config, logging.
2. Audio in/out and STT.
3. llama-server control and fetch, Gemma GGUF.
4. Prompt, contract parser, conversation (pure code, unit tests with stubs).
5. UI: header dialog, chat, NOTE panel, command buttons, transcript save.
6. AGENTS.md and README.

## Risks to verify
- Gemma chat template and the system role through llama-server (jinja template).
- Whisper may repair learner grammar before the model sees it.
- Prompt ~2300 tokens: keep system prompt stable per session for prefix cache reuse.
