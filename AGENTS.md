# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Overview
SpeakLoop is a Python desktop voice tutor app (Tkinter GUI) for language learning using local AI models.

## Running the App
```bash
pip install -r requirements.txt
python main.py
```
**Requires**: LM Studio running on `http://localhost:1234` (configured in [`config.py`](config.py)).

## Architecture
- [`main.py`](main.py) — `VoiceTutorGUI` class: Tkinter GUI, audio recording loop, threading orchestration
- [`stt.py`](stt.py) — `STTManager`: faster-whisper speech-to-text with VAD filtering
- [`llm.py`](llm.py) — `LLMManager`: OpenAI-compatible client streaming to LM Studio local server
- [`tts.py`](tts.py) — `TTSManager`: Kokoro text-to-synthesis with stream playback
- [`config.py`](config.py): all configuration (languages, model settings, API endpoints)

## Key Patterns & Gotchas
- **Threading**: Recording, TTS queue processing, and model loading run in separate daemon threads. Use `root.after()` to update GUI from threads.
- **Audio normalization** ([`main.py:423`](main.py:423)): peaks are normalized before STT; silence below `AUDIO_MIN_PEAK_THRESHOLD=0.01` is skipped.
- **Streaming pipeline**: LLM output is sentence-split by regex `(?<=[.!?])\s+` and queued to TTS in real-time.
- **Conversation history** ([`llm.py:107`](llm.py:107)): trimmed to `LLM_HISTORY_MAX_PAIRS=4` turns (configurable).
- **Device detection**: CUDA auto-detection via `torch.cuda.is_available()` — set compute type accordingly (`float16` for CUDA, `int8` for CPU).
- **Kokoro warm-up words** ([`tts.py:14`](tts.py:14)): language-specific words prevent OOV phoneme warnings during first synthesis.

## Code Style (Python)
- No linting/formatting config files exist — follow PEP 8 conventions.
- Type hints used throughout (`from typing import Optional`).
- Logging via `logging` module with format: `%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s`.
- Warnings filtered for library deprecation noise (see [`main.py:19`](main.py:19)).
