# SpeakLoop

AI-powered voice tutor for practicing foreign languages through real conversation (MVP).

## About

SpeakLoop is a desktop application for practicing conversational foreign language with an AI partner. Hold Space to speak and release it to get a response: the app transcribes your speech, sends it to an LLM, and reads the reply aloud.

The application is being moved to a new structure step by step. It runs from the `speakloop/` package and serves the local model with the official `llama-server` binary from llama.cpp.

## Tech Stack

- **GUI**: Tkinter
- **STT**: faster-whisper (Whisper small)
- **LLM**: local GGUF model via `llama-server` (llama.cpp) or LM Studio
- **TTS**: Kokoro (hexgrad/Kokoro-82M)
- **Python**: 3.11 or 3.12

## Requirements

- **Python 3.11 or 3.12.** Newer versions need a C++ compiler for some dependencies, so `pip` refuses them.
- **Windows, Linux or macOS.** The current application is used on Windows.
- A microphone and speakers.
- **NVIDIA GPU**: optional. It needs a CUDA build of PyTorch (see [Platform notes](#platform-notes)).
- **tkinter** (Linux only): the Tk GUI toolkit is packaged apart from the interpreter (`python3-tk` on Debian/Ubuntu). It is not on PyPI.
- **PortAudio** (Linux only): the native audio library (`libportaudio2` on Debian/Ubuntu). The Windows and macOS wheels of `sounddevice` include it, the Linux wheels do not.

## Installation

### With `install.py` (recommended)

```bash
git clone https://github.com/vikonix/SpeakLoop.git
cd SpeakLoop

# Create and activate a virtual environment, then run the installer INSIDE it
# (the script installs into the interpreter that runs it):
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

python install.py
```

The installer does these steps:

1. Checks the environment: a virtual environment, the Visual C++ runtime (Windows), tkinter and PortAudio (Linux).
2. Checks the Python version.
3. Finds an NVIDIA GPU with `nvidia-smi` and, if there is one, installs the CUDA build of `torch`.
4. Installs the Python dependencies from `pyproject.toml`.
5. Downloads the Hugging Face models (faster-whisper small, Kokoro) into `model_cache/`.
6. Downloads the Supertonic 3 speech model (Spanish) into `model_cache/supertonic3/`.
7. Installs the pinned `llama-server` binary into `bin/llama/`.
8. Downloads the GGUF chat model into `models/`.
9. Detects the hardware and writes `config/hardware_config.json`.
10. Writes a launcher: `run_speakloop.bat` (Windows) or `run_speakloop.sh` (Linux, macOS).

Before each step the installer shows the step and its command and asks: `Y` runs it, `n` stops the installer, `s` skips the step. A step that is already done is offered as skip or reinstall. The full run is written to `logs/install.log`.

Useful flags:

- `--yes`: run without questions (steps that are already done are skipped; add `--reinstall` to do them again)
- `--dry-run`: show the steps and commands, do not run them
- `--cpu` / `--gpu`: skip the CUDA installs / do them even if no GPU is found
- `--skip-models`, `--skip-gguf`, `--skip-llm`: skip the model downloads, the GGUF download, or the whole LLM part (for LM Studio)

On Windows, **Developer Mode** lets the model cache use symlinks. Without it the model downloads copy files instead, which uses more disk.

### Manual installation

Run these commands in the activated virtual environment:

```bash
# Python dependencies. Editable (-e), so the code stays in this directory and
# config/, models/ and logs/ stay in the project directory too.
pip install -e .

# Models and the llama-server binary
python -m speakloop.model_fetch         # faster-whisper small, Kokoro, Supertonic 3
python -m speakloop.llama_server_fetch  # pinned llama.cpp build into bin/llama/
python -m speakloop.gguf_fetch          # GGUF chat model into models/

# Hardware detection
python -m speakloop.detect_hardware
```

Each fetcher accepts `--list` to show what is present and what is missing.

The spaCy English pipeline for Kokoro is not installed by these commands: Kokoro's English text processing (misaki) downloads it with `pip` at the first English speech output, so that first run needs network access.

### Platform notes

**Windows with an NVIDIA GPU.** PyPI serves a CPU-only `torch` on Windows. Install a CUDA build that matches your driver (see [pytorch.org](https://pytorch.org/get-started/locally/)), for example:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

`install.py` does this in its step 3. On Linux PyPI already serves a CUDA build, and macOS has no CUDA.

**Linux.** Install tkinter and PortAudio with the system package manager:

```bash
sudo apt install python3-tk libportaudio2          # Debian / Ubuntu
sudo dnf install python3-tkinter portaudio         # Fedora
sudo pacman -S tk portaudio                        # Arch
```

If PortAudio is installed but no audio device is found (the installer shows `0 input / 0 output`), PortAudio is usually built without the PulseAudio backend (WSL is a known case). Install `libasound2-plugins` and set `pcm.!default pulse` in `~/.asoundrc`, or build PortAudio with `./configure --with-pulseaudio`.

The microphone button shows its state with emoji. Tk cannot draw color emoji fonts, so on a new Linux system some icons show as empty boxes. Install a monochrome emoji font:

```bash
sudo apt install fonts-symbola     # in the Ubuntu "universe" repository
fc-cache -f -v
```

The Linux GPU build of `llama-server` uses Vulkan (llama.cpp publishes no CUDA binary for Linux). If Vulkan finds no device (for example under WSL2), the fetcher installs the CPU build and writes the reason in the log.

**macOS.** Intel Macs get an older stack automatically (torch 2.2.2, transformers 4.x, NumPy 1.x), because PyTorch publishes no newer wheel for them. The pinned `llama-server` build needs macOS 26 on Apple Silicon and macOS 13.3 on Intel; on an older Mac the installer says so before it downloads anything. Homebrew Python does not include tkinter: install `python-tk@<version>` for your Python version (`install.py` does this).

### Models

| Model | Used for | Download | Command |
|---|---|---|---|
| `Systran/faster-whisper-small` | speech recognition | 486 MB | `python -m speakloop.model_fetch --hf` |
| Kokoro-82M (`hexgrad/Kokoro-82M`) | speech output (English) | 363 MB | `python -m speakloop.model_fetch --hf` |
| Supertonic 3 (`Supertone/supertonic-3`) | speech output (Spanish) | 404 MB | `python -m speakloop.model_fetch --supertonic`. The weights have the **OpenRAIL-M** license, so they are downloaded, not included |
| `llama-3.2-3b-instruct-q4_k_m.gguf` | conversation | 2019 MB | `python -m speakloop.gguf_fetch` |
| `llama-server` (pinned llama.cpp release) | runs the GGUF model | 641 MB CUDA, 18 MB CPU (Windows) | `python -m speakloop.llama_server_fetch` |

## Configuration

The configuration has three layers, lowest priority first:

1. **Built-in defaults** in [`speakloop/config.py`](speakloop/config.py): language pair, persona prompt, LLM backend and server address, generation parameters, Whisper and Kokoro settings.
2. **`config/hardware_config.json`**, written by the installer or by `python -m speakloop.detect_hardware`: compute devices (`DEVICE`, `STT_DEVICE`), GPU layers and context size of the local model, audio devices.
3. **`config/settings.json`**, edited by hand: user preferences. Copy [`config/settings.example.json`](config/settings.example.json) to start. Keys:
   - `max_record_seconds`: limit of one recording, in seconds (default 20).
   - `llm_backend`: `"llama-server"` (default) or `"lm-studio"`.
   - `lm_studio_host`: address of LM Studio, `"host"`, `"host:port"` or a full URL (default `"localhost:1234"`).
   - `llama_server_path`: the `llama-server` binary to start. Empty (default) means `bin/llama/`, then a `llama-server` on PATH.
   - `external_model_path`: a GGUF model of your own. Absent (default) means the downloaded model in `models/`.
   - `external_n_ctx`: context size of the local model. Absent (default) means the value hardware detection wrote.

Both files are optional. A broken or missing file leaves the lower layers in effect, and the problem is printed to the console. Restart the app to apply a change.

Models are loaded from `model_cache/`. When all models of a run are there, the app does not connect to the Hugging Face Hub at all.

## Running

In the activated virtual environment, any of these starts the same application:

```bash
python main.py
python -m speakloop
speakloop                # console script, after `pip install -e .`
```

`speakloop --version` prints the version, `speakloop --detect-hardware` rewrites `config/hardware_config.json`.

With `"llm_backend": "llama-server"` (the default) the model server starts automatically. With `"llm_backend": "lm-studio"` start LM Studio first.

If a `llama-server` already answers on `127.0.0.1:8765` (the usual case is one left behind by a session that ended abnormally), the app uses that server as it is, says so in the chat and does not stop it on exit. Such a server keeps the model, context size and GPU layers it was started with, not the ones in the current settings, so `logs/main.log` records which model it serves and how large its context is, and warns when the model is not the configured one or the context is smaller. If the port is held by a program that does not answer the API, the app says the port is busy instead of starting a second server.

Logs are in `logs/`: `main.log` (the application, replaced at each start), `llm_server.log` (the model server), `install.log` and `hwdetect.log` (kept across runs).

## Controls

- **Space (hold)**: record speech
- **ESC**: quit

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests download nothing and replace subprocesses and the network with stubs. `tests/test_config.py` imports the real configuration of the checkout, so it also imports torch, like the application does.

## Project Structure

```
SpeakLoop/
├── main.py              launcher shim for a source checkout
├── install.py           guided installer
├── pyproject.toml       package metadata, dependency list, console script
├── speakloop/           application package
│   ├── cli.py               entry point (--version, --detect-hardware)
│   ├── __main__.py          python -m speakloop
│   ├── app.py               GUI, thread orchestration, run()
│   ├── stt.py               Speech-to-Text (faster-whisper)
│   ├── llm.py               LLM client (OpenAI-compatible)
│   ├── llm_server_ctl.py    starts and stops the llama-server subprocess
│   ├── tts.py               Text-to-Speech (Kokoro)
│   ├── config.py            configuration layers
│   ├── bootstrap.py         early process setup, logging
│   ├── lifecycle.py         process exit and relaunch helpers
│   ├── paths.py             where every file lives (SPEAKLOOP_HOME)
│   ├── loader.py            JSON reading and atomic writing
│   ├── models_info.py       model catalogue
│   ├── model_fetch.py       faster-whisper, Kokoro, Supertonic downloads
│   ├── gguf_fetch.py        GGUF model download
│   ├── llama_server_fetch.py  pinned llama-server binary
│   └── detect_hardware.py   hardware probe, config/hardware_config.json
├── config/              settings.example.json (settings.json, hardware_config.json are local)
├── tests/               unit tests
├── tools/               maintainer tools (measure_model_sizes.py)
├── docs/                plans and reviews
├── models/              GGUF model files
└── bin/, model_cache/, logs/   created by the installer
```
