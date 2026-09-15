# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Hardware detection for SpeakLoop.

Probes the machine (RAM, CPU, GPU/VRAM, audio devices) and writes
config/hardware_config.json with two sections:

  "hardware" - raw facts about the machine, for diagnostics;
  "config"   - ready-to-use parameter values (DEVICE, STT_DEVICE,
               TTS_DEVICE, audio devices) picked from the detected hardware.
               The app reads these instead of the hard-coded defaults in
               config.py. The chat model's GPU layers and context are NOT
               here: llama.cpp fits them into the free VRAM itself at launch
               (docs/model-parameters.md).

Run it manually whenever the hardware changes:

    python -m speakloop.detect_hardware

It lives in the package rather than in tools/ because of who runs it: tools/
holds what the maintainer runs, and this has to execute on the user's machine.
Installed as a package there is no tools/ directory and no install.py, so a
probe left outside the package would simply never run and the app would sit on
its conservative defaults on a machine with a GPU.

**This module must not import speakloop.config.** ``config`` reads
hardware_config.json at import time, so importing it here would take a snapshot
of the very file this module is about to rewrite. The rule is the same as for
the three fetchers: ``speakloop.paths`` yes, ``speakloop.config`` no.

It only relies on packages the project already uses (torch, ctranslate2 via
faster-whisper, sounddevice) plus the stdlib-only speakloop.bootstrap,
speakloop.paths and speakloop.llama_server_fetch (bootstrap for the run header
its log file opens with); each probe degrades gracefully if its package is
missing or broken, and any such problem is recorded in the "warnings" list of
the output file.
"""

import ctypes
import json
import logging
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    # Executed as a plain script rather than with -m: that form puts THIS
    # directory on sys.path instead of the project root, so "import speakloop"
    # would not resolve. Same shim, and the same reason, as in the fetchers.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speakloop import bootstrap, paths

# The output is a config artifact read by speakloop/config.py, so it goes to the
# config directory rather than next to this module. Both locations come from
# paths.py, which is also what config.py binds to - the two cannot drift.
OUTPUT_FILE = paths.config_dir() / "hardware_config.json"

# A timestamped record of each run is kept in the project-wide logs/ directory
# (the same one config.py uses for main.log), alongside the human-friendly
# console print()s. The log file is the place to look when diagnosing why a
# given machine was detected the way it was - it captures the warnings too.
LOG_DIR = paths.log_dir()
LOG_FILE = LOG_DIR / "hwdetect.log"

logger = logging.getLogger("hwdetect")


def _setup_logging() -> None:
    """Attach a file handler appending to logs/hwdetect.log.

    Kept independent of the console output: the terminal stays concise while the
    log file preserves a timestamped, complete record for later inspection.
    Idempotent, so a second call in the same process (tests, a future GUI
    caller) does not attach a second handler.

    Appended rather than overwritten, and every run opens with the header
    bootstrap.open_log_section writes. The question this file answers is "why
    was this machine detected the way it was", and it is usually asked about a
    probe that has since been followed by another one - a driver was installed,
    a binary replaced, SPEAKLOOP_HOME moved. Truncating per run answered it for
    the last probe only, which is the one nobody is asking about. The volume
    makes it free: a probe writes about a dozen lines, and it runs once per
    install or when the user asks for it.

    An unwritable log directory costs the file and nothing else, and that
    tolerance is load-bearing. This runs before the probe, on a path that does
    NOT import config - so nothing has called paths.ensure_dirs() and its own
    reporting cannot help here - and an unreachable SPEAKLOOP_HOME would otherwise
    end `speakloop --detect-hardware` in a three-deep pathlib traceback about a
    drive letter, from a command the user was told to run by some other
    message. The probe itself needs no log file: every line below is printed
    as well.
    """
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    # Set before the file handler is attempted, so it holds on both paths.
    logger.propagate = False
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    except OSError as exc:
        print(f"Note: no log file this run - could not open {LOG_FILE} "
              f"({exc}). The probe still runs and prints its result below.",
              file=sys.stderr)
        # A NullHandler, and propagate=False above is NOT enough on its own:
        # when callHandlers finds no handler ANYWHERE it falls back to
        # logging.lastResort, which prints every warning and error to stderr.
        # main() already prints its own message there, so without this the
        # reader gets the same failure twice, in two slightly different
        # wordings, and reasonably wonders whether it happened twice.
        logger.addHandler(logging.NullHandler())
        return
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    # Before the handler is attached, so this run's section opens the file
    # rather than starting under the first line the probe happens to log.
    bootstrap.open_log_section(handler, LOG_FILE)
    logger.addHandler(handler)

# Smallest card, in GB as vram_gb reports it, on which the speech models stay
# on the GPU next to the chat model. Gemma 4 12B Q4_0 takes about 6.5 GiB, its
# 16k context and compute buffers add more, and faster-whisper and Kokoro each
# open a CUDA context of their own (about 1-1.5 GB together). Below this size
# the chat model needs the whole card, and the speech models lose little on
# the CPU next to its answer time (docs/model-parameters.md, section 7.2).
SPEECH_GPU_MIN_VRAM_GB = 12


# =====================================================================
# RAM / CPU
# =====================================================================

def detect_ram_gb(warnings: list) -> float | None:
    """Total physical RAM in GiB."""
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 1024**3, 1)
        warnings.append("GlobalMemoryStatusEx failed; RAM size unknown")
        return None

    # Linux/macOS fallback (sysconf is absent on Windows only).
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        warnings.append("Could not determine RAM size on this platform")
        return None


def detect_cpu_features(warnings: list) -> dict:
    """SIMD CPU features relevant to llama.cpp, read from numpy (already a dep).

    Diagnostics only, but the first thing to look at when the pinned
    llama-server build dies on startup: a binary compiled for an instruction
    set the CPU lacks crashes outright (0xC000001D on Windows) instead of
    degrading, and AVX512 is the usual culprit. numpy exposes a CPUID-based
    feature map, the cleanest cross-platform source without a new dependency
    or fragile Windows API calls.
    """
    features: dict = {}
    try:
        import numpy as np
    except ImportError:
        warnings.append("numpy is not installed; CPU features unknown")
        return features
    try:
        # numpy >= 2 moved the internal module to np._core.
        try:
            umath = np._core._multiarray_umath
        except AttributeError:
            umath = np.core._multiarray_umath
        raw = umath.__cpu_features__
        for key in ("AVX", "AVX2", "AVX512F", "FMA3", "F16C"):
            features[key] = bool(raw.get(key, False))
    except Exception as exc:  # noqa: BLE001 - any probe failure is non-fatal
        warnings.append(f"CPU feature probe failed: {exc}")
    return features


# =====================================================================
# GPU
# =====================================================================

def detect_gpu(warnings: list) -> dict:
    """GPU presence, name, VRAM, and which backends can actually use it.

    Three independent consumers, probed separately:

    - llama-server runs as its own process with its own CUDA runtime, fully
      independent of torch. ``llama_gpu_offload`` reflects the installed
      build's own device probe (None when there is no install to ask -
      physical GPU presence is the fallback signal).
    - torch (used by Kokoro speech synthesis) reports CUDA via ``torch_cuda``.
      A CPU-only torch build only means Kokoro runs on CPU; it says nothing
      about the LLM.
    - ctranslate2 (the engine of faster-whisper speech recognition) reports
      ``stt_cuda`` - see :func:`_probe_stt_cuda`.

    Physical presence/name/VRAM come from nvidia-smi first (works regardless
    of torch build), then torch.cuda. Non-NVIDIA adapters (AMD/Intel) are
    listed by name only, for diagnostics.
    """
    gpu = {
        "present": False,
        "name": None,
        "vram_gb": None,
        "torch_cuda": False,
        "stt_cuda": False,
        "llama_gpu_offload": None,
        "device_count": 0,
        "all_adapters": _list_video_adapters(),
    }

    smi = _query_nvidia_smi(warnings)
    if smi:
        gpu.update(present=True, name=smi["name"], vram_gb=smi["vram_gb"],
                   device_count=1)

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu.update(
                present=True,
                name=props.name,
                vram_gb=round(props.total_memory / 1024**3, 1),
                torch_cuda=True,
                device_count=torch.cuda.device_count(),
            )
        elif gpu["present"]:
            warnings.append(
                f"'{gpu['name']}' is present but torch has no CUDA (CPU-only "
                "build) - speech synthesis (Kokoro) and speech recognition "
                "(faster-whisper) will run on CPU; the LLM is unaffected, "
                "llama-server carries its own CUDA runtime"
            )
    except ImportError:
        warnings.append("torch is not installed; speech devices default to CPU")
    except Exception as exc:  # noqa: BLE001 - a broken CUDA runtime must not kill detection
        warnings.append(f"torch CUDA probe failed: {exc}")

    # After the torch probe, and only when it succeeded: see _probe_stt_cuda
    # for why torch has to be imported before ctranslate2.
    gpu["stt_cuda"] = _probe_stt_cuda(warnings, gpu["torch_cuda"])

    # Probed last, once presence is settled: the probe stays quiet on machines
    # that have no GPU at all, where "the LLM cannot use the GPU" is not news.
    gpu["llama_gpu_offload"] = _probe_llama_offload(warnings, gpu["present"])

    return gpu


def _probe_stt_cuda(warnings: list, torch_cuda: bool) -> bool:
    """Whether faster-whisper (ctranslate2) can run speech recognition on CUDA.

    Two conditions, both required:

    - torch sees CUDA. ctranslate2's CUDA path needs cuBLAS 12 and cuDNN 9 at
      runtime, and on Windows the CUDA build of torch is where they come from:
      importing torch first puts its library directory on the DLL search path.
      A CPU-only torch has no such libraries, so the answer is "cpu" without
      asking ctranslate2 at all. That also keeps the import order right,
      because this function is called only after the torch probe imported it.
    - ctranslate2 reports at least one CUDA device.

    The check does not run a model, so it cannot prove that cuDNN really loads:
    that failure shows up at the first transcription in the app, whose STT
    warm-up is the place it is reported.
    """
    if not torch_cuda:
        return False
    try:
        import ctranslate2
    except ImportError:
        warnings.append("ctranslate2 (faster-whisper) is not installed; speech "
                        "recognition device defaults to CPU")
        return False
    except Exception as exc:  # noqa: BLE001 - a broken CUDA runtime must not kill detection
        warnings.append(f"ctranslate2 import failed: {exc}")
        return False
    try:
        device_count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001 - same reason as above
        warnings.append(f"ctranslate2 CUDA probe failed: {exc}")
        return False
    if device_count > 0:
        return True
    warnings.append("torch sees CUDA, but ctranslate2 reports no CUDA device - "
                    "speech recognition (faster-whisper) will run on CPU")
    return False


def _probe_llama_offload(warnings: list, gpu_present: bool) -> bool | None:
    """Whether the installed llama-server binary can offload to the GPU.

    Asks the binary itself with `--list-devices` (loads no model, sub-second)
    and matches the answer against the device pattern of the variant it was
    installed as. That comparison is the point: a CUDA build whose cudart DLLs
    are missing or of the wrong major version starts fine, still logs
    "offloaded N/N layers to GPU", and simply runs about three times slower on
    the CPU. The same probe guards every app start in speakloop/llm_server_ctl.py
    (log_compute_devices).

    Only the project's own install in bin/llama/ is probed. A llama-server the
    user manages themselves carries no record of which backend it was built
    for, so there is nothing to compare its device list against; it yields None
    like an absent install does.

    Returns True/False, or None when the question cannot be answered - callers
    then fall back to physical GPU presence, which is what build_config does.
    *gpu_present* only decides whether a negative answer is worth a warning: on
    a machine without a GPU "the LLM cannot use the GPU" is not news, and
    build_config treats an absent GPU as one the LLM does not use anyway.
    """
    # Deferred rather than imported at the top: an unimportable fetcher is a
    # recorded warning here, not a failure to detect the rest of the machine.
    # llama_server_fetch is stdlib-only and side-effect-free on import, which
    # is why it is safe to pull in even before the requirements are installed.
    try:
        from speakloop import llama_server_fetch
    except ImportError as exc:
        warnings.append(f"speakloop.llama_server_fetch is not importable ({exc}); "
                        "GPU offload capability unknown")
        return None

    exe = llama_server_fetch.installed_exe()
    if exe is None:
        if gpu_present:
            warnings.append(
                "llama-server is not installed in bin/llama (run "
                "'python -m speakloop.llama_server_fetch'); GPU offload "
                "capability unknown")
        return None

    variant_name = llama_server_fetch.installed_variant(exe)
    if variant_name is None:
        # installed_exe() already required a readable stamp, so the only way
        # here is a stamp naming a variant this version of the module dropped.
        warnings.append(f"{exe} was installed as a variant this build does not "
                        "know; GPU offload capability unknown")
        return None

    pattern = llama_server_fetch.VARIANTS[variant_name].device_pattern
    if pattern is None:
        # A CPU build cannot offload, full stop. This is the one branch that
        # answers a definite False rather than None, which is what lets
        # build_config leave the whole card to the speech models.
        if gpu_present:
            # Deliberately not naming the build a retry would land on: that
            # depends on the platform (llama.cpp publishes CUDA binaries for
            # Windows only, so the GPU build under Linux is the Vulkan one)
            # and on whether its backend comes up at all, and guessing it here
            # would be a second copy of a decision llama_server_fetch already
            # owns. The command re-runs that decision and says what it chose.
            warnings.append(
                f"the installed llama-server is the '{variant_name}' build - "
                f"the LLM cannot use the GPU; reinstall it with "
                f"'python -m speakloop.llama_server_fetch --force' to re-run the "
                f"build selection for this machine")
        return False

    try:
        devices = llama_server_fetch.list_devices(exe)
    except llama_server_fetch.LlamaServerFetchError as exc:
        warnings.append(f"llama-server --list-devices failed: {exc}")
        return None

    if re.search(pattern, devices):
        return True
    warnings.append(
        f"the '{variant_name}' llama-server build lists no matching device, so "
        f"it would silently run on the CPU (about three times slower); check "
        f"that the backend's runtime is reachable - the cudart DLLs next to "
        f"{exe.name} with a major version matching the build on Windows, a "
        f"Vulkan loader and ICD on Linux. --list-devices said:"
        f"\n{devices.strip()}")
    return False


def _query_nvidia_smi(warnings: list) -> dict | None:
    """First GPU reported by nvidia-smi, or None if the tool is absent/fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    # OSError rather than FileNotFoundError alone: an nvidia-smi that exists
    # but cannot be executed (no permission bit, a broken driver package that
    # left a stub) raises PermissionError, and killing the whole probe over a
    # diagnostic optional command is the wrong trade in either case.
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        name, mem_mib = out.stdout.strip().splitlines()[0].rsplit(",", 1)
        return {"name": name.strip(), "vram_gb": round(int(mem_mib) / 1024, 1)}
    except ValueError:
        warnings.append(f"Could not parse nvidia-smi output: {out.stdout!r}")
        return None


def _list_video_adapters() -> list[str]:
    """Names of all video adapters (Windows WMI); empty list elsewhere/on failure."""
    if platform.system() != "Windows":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=20,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# =====================================================================
# Audio
# =====================================================================

def detect_audio(warnings: list) -> dict:
    """Input/output audio devices as seen by sounddevice (PortAudio)."""
    audio = {"input_devices": [], "output_devices": []}
    try:
        import sounddevice as sd
    except ImportError:
        warnings.append("sounddevice is not installed; audio devices unknown")
        return audio

    try:
        default_in, default_out = sd.default.device
        for index, dev in enumerate(sd.query_devices()):
            entry = {
                "index": index,
                "name": dev["name"],
                "hostapi": sd.query_hostapis(dev["hostapi"])["name"],
                "default_samplerate": dev["default_samplerate"],
            }
            if dev["max_input_channels"] > 0:
                audio["input_devices"].append(
                    {**entry, "channels": dev["max_input_channels"],
                     "default": index == default_in}
                )
            if dev["max_output_channels"] > 0:
                audio["output_devices"].append(
                    {**entry, "channels": dev["max_output_channels"],
                     "default": index == default_out}
                )
    except Exception as exc:  # noqa: BLE001 - PortAudio errors must not kill detection
        warnings.append(f"Audio device query failed: {exc}")

    if not audio["input_devices"]:
        warnings.append("No audio input device (microphone) found")
    if not audio["output_devices"]:
        warnings.append("No audio output device (speakers) found")
    return audio


# =====================================================================
# Parameter selection
# =====================================================================

def build_config(hardware: dict) -> dict:
    """Pick concrete app parameters from the detected hardware.

    The names match the constants in config.py so the app can apply them
    directly. The chat model gets no values here: llama-server fits its GPU
    layers into the free VRAM at launch, and its context is a setting.

    The chat model is still what decides the speech devices. It shares the
    card with them, and on a card below SPEECH_GPU_MIN_VRAM_GB it needs all of
    it, so faster-whisper and Kokoro go to the CPU there. A card the chat
    model cannot use (a CPU build of llama-server) is free for them whatever
    its size. The LLM side follows the physical GPU and llama-server's own
    device probe - NOT torch or ctranslate2, which are the speech stacks.
    """
    gpu = hardware["gpu"]

    # LLM side: usable unless llama-server explicitly reported no usable device
    # (None = nothing to ask, assume a present GPU is usable).
    llm_uses_gpu = gpu["present"] and gpu["llama_gpu_offload"] is not False
    speech_fits_on_gpu = (not llm_uses_gpu
                          or (gpu["vram_gb"] or 0) >= SPEECH_GPU_MIN_VRAM_GB)

    # DEVICE stays the plain answer to "does torch see CUDA": config caps the
    # per-model devices by it, and warn_if_gpu_unused reads it as that answer.
    return {
        "DEVICE": "cuda" if gpu["torch_cuda"] else "cpu",
        "STT_DEVICE": ("cuda" if gpu["stt_cuda"] and speech_fits_on_gpu
                       else "cpu"),
        "TTS_DEVICE": ("cuda" if gpu["torch_cuda"] and speech_fits_on_gpu
                       else "cpu"),
        # null = system default device, which is the right choice on most
        # machines; the indices of all devices are listed under "hardware".
        "AUDIO_INPUT_DEVICE": None,
        "AUDIO_OUTPUT_DEVICE": None,
    }


# =====================================================================
# Entry points
# =====================================================================

def warn_if_gpu_unused(device: str) -> None:
    """Log a warning when an NVIDIA GPU is present but torch is not using it.

    The counterpart of ``llm_server_ctl.log_compute_devices``, for the same
    class of failure one layer over. A CPU-only torch on a machine with a card
    raises nothing: the app works, speech recognition and synthesis are simply
    several times slower, and only a comparison anyone would have to think of
    making would reveal it. That is the same reason the LLM's device probe
    exists - see _probe_llama_offload above, which asks the identical question
    about the llama-server binary.

    It is reachable by accident because a package cannot name an index. PyPI
    serves CUDA-enabled torch on Linux and CUDA does not exist on macOS, so the
    gap is one platform wide: Windows with an NVIDIA card, where an install that
    skipped install.py's CUDA torch step gets the CPU wheel.

    *device* is passed in rather than read from ``config``, which this module
    may not import (see the module docstring). It is already the answer to "does
    torch see CUDA" - ``loader.detect_device`` probed it, or a previous run of
    this module wrote it - and no user setting overrides it, so "cpu" means
    torch could not use a GPU rather than that somebody asked for one. Which
    also means torch is never imported here, and nvidia-smi is only consulted in
    that case: on a machine already running on CUDA this costs nothing.

    A driver present while torch sits on the CPU is equally what a broken CUDA
    installation looks like, so the message names that way out too.

    **The message depends on whether hardware_config.json exists**, because the
    sentence above - "cpu means torch could not use a GPU" - stops being true
    once it does. ``loader.detect_device`` short-circuits on a stored ``"cpu"``
    without asking torch (deliberately: a stale ``"cpu"`` only makes things
    slower, while a stale ``"cuda"`` kills the app and is therefore re-checked),
    so with that file present ``cpu`` can equally mean "the file is older than
    the torch now installed". That is not a corner case: it is what installing
    the CUDA build of torch over a CPU one, without re-running the probe, looks
    like - and telling that user to reinstall torch is telling them to redo
    what they just did.
    """
    if device != "cpu":
        return
    try:
        from speakloop import llama_server_fetch
    except ImportError:
        return  # same degradation as _probe_llama_offload: diagnostics only
    if llama_server_fetch.detect_driver_cuda() is None:
        return  # no NVIDIA driver, so the CPU is the correct answer here

    # install.py's CUDA torch step is the Windows answer, for the reason the
    # docstring above already gives. Elsewhere the step rarely applies (PyPI's
    # torch is already a CUDA build on Linux), so the general PyTorch
    # instructions are named first.
    if sys.platform == "win32":
        reinstall = ("run `python install.py` from a clone and accept its "
                     "CUDA torch step")
    else:
        reinstall = ("reinstall torch from the CUDA index matching this driver "
                     "(https://pytorch.org/get-started/locally/), or run "
                     "`python install.py` from a clone")
    if _stored_device_may_be_stale():
        logging.warning(
            "An NVIDIA GPU is present, but this run is on the CPU - "
            "speech recognition and speech synthesis will be several times "
            "slower. Two things look like this. Either %s records a machine "
            "state older than the torch now installed, in which case run "
            "%s to refresh it; or torch really is a CPU-only build, in which "
            "case reinstall it: %s.",
            OUTPUT_FILE, _refresh_command(), reinstall)
        return
    logging.warning(
        "An NVIDIA GPU is present, but torch is a CPU-only build and will not "
        "use it - speech recognition and speech synthesis will run several "
        "times slower. To reinstall with a CUDA build: %s. If the install did "
        "name a CUDA backend, then the driver and the build disagree - see "
        "https://pytorch.org/get-started/locally/.", reinstall)


def _refresh_command() -> str:
    """How to re-run this probe, spelled the way THIS installation allows.

    The two forms are not interchangeable, and printing the wrong one is worse
    than printing nothing. An installed tool may have no interpreter on PATH
    that can ``import speakloop`` - only its console script - so
    ``python -m speakloop.detect_hardware`` either fails or, if
    the user happens to have a checkout's virtual environment active, quietly
    rewrites a DIFFERENT hardware_config.json. Out of a clone the console
    script may not exist at all, which is the mirror image.
    """
    if paths.repo_mode():
        return "`python -m speakloop.detect_hardware`"
    return "`speakloop --detect-hardware`"


def _stored_device_may_be_stale() -> bool:
    """True when a written hardware_config.json could be the reason for "cpu".

    One stat call, and deliberately not a read: what the file SAYS does not
    matter here. If it exists at all, ``loader.detect_device`` may have taken
    its word without consulting torch, and that is the whole of the ambiguity
    the message has to admit to.
    """
    try:
        return OUTPUT_FILE.is_file()
    except OSError:
        return False


def probe_and_write() -> dict:
    """Probe the machine, write hardware_config.json, return the whole result.

    Separated from :func:`main` so a caller inside the app can run the probe
    without the CLI's console output - for example after it has installed the
    llama-server binary, which is exactly what _probe_llama_offload needs in
    place to answer anything.

    Raises OSError if the file cannot be written. Everything else is degraded
    into the "warnings" list, which is also part of the returned dict.
    """
    warnings: list[str] = []
    hardware = {
        "platform": f"{platform.system()} {platform.release()}",
        "ram_total_gb": detect_ram_gb(warnings),
        "cpu_cores": os.cpu_count(),
        "cpu_features": detect_cpu_features(warnings),
        "gpu": detect_gpu(warnings),
        "audio": detect_audio(warnings),
    }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": hardware,
        "config": build_config(hardware),
        "warnings": warnings,
    }

    # config.py creates this directory at import, but this module runs from its
    # own CLI too, where nothing has imported config.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    _setup_logging()

    print("Detecting hardware...")
    logger.info("Detecting hardware...")
    try:
        result = probe_and_write()
    except OSError as exc:
        # The one failure probe_and_write does not degrade into a warning is
        # not being able to write its own output, and it reaches a user who
        # typed `speakloop --detect-hardware` because some other message told them
        # to. A traceback there answers a question they did not ask; the path
        # and the reason answer the one they did.
        print(f"ERROR: could not write {OUTPUT_FILE}: {exc}", file=sys.stderr)
        logger.error("Could not write %s: %s", OUTPUT_FILE, exc)
        return 1
    hardware = result["hardware"]
    warnings = result["warnings"]

    gpu = hardware["gpu"]
    ram_line = f"RAM: {hardware['ram_total_gb']} GB, CPU cores: {hardware['cpu_cores']}"
    print(f"  {ram_line}")
    logger.info(ram_line)

    feats = hardware["cpu_features"]
    if feats:
        enabled = [name for name, present in feats.items() if present]
        cpu_feat_line = "CPU features: " + (", ".join(enabled) if enabled else "none")
        print(f"  {cpu_feat_line}")
        logger.info(cpu_feat_line)

    if gpu["present"]:
        llama_state = {True: "yes", False: "NO", None: "unknown"}[gpu["llama_gpu_offload"]]
        gpu_line = (f"GPU: {gpu['name']} ({gpu['vram_gb']} GB VRAM, "
                    f"llama-server offload: {llama_state}, torch CUDA: "
                    f"{'yes' if gpu['torch_cuda'] else 'no'}, STT CUDA: "
                    f"{'yes' if gpu['stt_cuda'] else 'no'})")
    else:
        gpu_line = "GPU: none detected"
    print(f"  {gpu_line}")
    logger.info(gpu_line)

    audio_line = (f"Audio: {len(hardware['audio']['input_devices'])} input / "
                  f"{len(hardware['audio']['output_devices'])} output device(s)")
    print(f"  {audio_line}")
    logger.info(audio_line)

    print(f"  Config: {json.dumps(result['config'])}")
    logger.info("Config: %s", json.dumps(result["config"]))

    for w in warnings:
        print(f"  WARNING: {w}")
        logger.warning(w)

    print(f"\nWritten to {OUTPUT_FILE}")
    logger.info("Written to %s", OUTPUT_FILE)
    logger.info("Log written to %s", LOG_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
