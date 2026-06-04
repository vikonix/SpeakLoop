# SpeakLoop

AI-powered voice tutor for practicing foreign languages through real conversation.

## About

SpeakLoop is a desktop application (Windows) for practicing conversational foreign language with an AI partner. Hold Space to speak, release to get a response — the app transcribes your speech, sends it to an LLM, and reads the reply aloud.

## Tech Stack

- **GUI** — Tkinter
- **STT** — faster-whisper (Whisper small by default)
- **LLM** — local GGUF model via `llm_server/` or LM Studio
- **TTS** — Kokoro (hexgrad/Kokoro-82M)
- **Python** 3.11+

## Installation

```bash
git clone https://github.com/yourusername/speakloop.git
cd speakloop

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

For the LLM server — separate dependencies:
```bash
pip install -r llm_server/requirements.txt
```

For CUDA-enabled `llama-cpp-python`, see [`llm_server/README.md`](llm_server/README.md).

## Configuration

All settings are in [`config.py`](config.py):

```python
# LLM backend selection
LLM_BACKEND = "local_server"   # recommended
# LLM_BACKEND = "lm-studio"   # if using LM Studio

# Path to the GGUF model file (for local_server)
EXTERNAL_MODEL_PATH = "models/llama-3.2-3b-instruct-q4_k_m.gguf"

# Language pair
NATIVE_LANGUAGE = "Russian"
TARGET_LANGUAGE = "English"
```

## Running

```bash
python main.py
```

With `LLM_BACKEND = "local_server"` the server starts automatically. With `LLM_BACKEND = "lm-studio"` start LM Studio first.

## Controls

- **Space (hold)** — record speech
- **ESC** — quit

## Project Structure

```
speakloop/
├── main.py          — GUI, thread orchestration
├── stt.py           — Speech-to-Text (Whisper)
├── llm.py           — LLM client (OpenAI-compatible)
├── tts.py           — Text-to-Speech (Kokoro)
├── config.py        — all configuration
├── models/          — GGUF model files
└── llm_server/      — standalone process for local LLM
    ├── server.py
    ├── requirements.txt
    └── README.md
```
