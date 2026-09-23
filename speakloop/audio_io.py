# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Shared audio-device infrastructure.

This module owns the PortAudio and winsound plumbing that the microphone path
(``recorder.py``) and the speaker path (``tts.py``) both need but neither
should own. Both depend on this neutral layer instead of on each other: the
capture side must not reach into the synthesis side for device-level code.

The lock that coordinates the two streams lives here, next to the
open-stream counter it guards. The device settings (``AUDIO_SAMPLE_RATE`` and
the other ``AUDIO_*`` values) stay in ``config``.
"""

import logging
import sys
import threading

import sounddevice as sd

from speakloop import config

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# Silence padding prepended to every winsound playback block.
# The Windows audio session needs ~50-200 ms to initialize on the first call,
# which clips the very beginning of the first sentence without this buffer.
WINSOUND_LEAD_IN_SECONDS = 0.15

# The synthesis sample rate is NOT here: it is a property of the active TTS
# backend (Kokoro 24 kHz, Supertonic 44.1 kHz), so callers read
# TTSManager.sample_rate instead of a constant.


# Serializes every PortAudio stream open and close, and reset_portaudio(),
# between the microphone (recorder.py) and the speaker (tts.py). There must be
# only this one instance: two locks do not exclude each other.
AUDIO_LOCK = threading.Lock()

# Number of PortAudio streams currently open in this process. Guarded by
# AUDIO_LOCK: every stream open and close, and every reset_portaudio()
# call, happens while holding it, so a plain int is safe here.
# reset_portaudio() consults it to skip the reset while any stream is open: the
# reset invalidates every PortAudio stream in the process, so running it then
# would corrupt a live stream (for example a recording that starts while a
# playback thread is still inside stream.write on the sounddevice path).
_open_streams = 0


def stream_opened() -> None:
    """Record that a PortAudio stream was opened. Call under AUDIO_LOCK."""
    global _open_streams
    _open_streams += 1


def stream_closed() -> None:
    """Record that a PortAudio stream was closed. Call under AUDIO_LOCK."""
    global _open_streams
    _open_streams = max(0, _open_streams - 1)


def reset_portaudio():
    """Fully reinitialize PortAudio before opening a stream.

    Heals HDMI and NVIDIA device disconnects caused by CUDA power-state changes
    on Windows. ``sd._terminate`` and ``sd._initialize`` are *not* public
    sounddevice API (they exist in 0.4.x and 0.5.x); if an upgrade removes them
    this degrades to a logged no-op and the reset can simply be dropped. Call
    only while holding AUDIO_LOCK.

    The reset invalidates every existing PortAudio stream in the process, so it
    is skipped while any stream is still open (see stream_opened and
    stream_closed); an open stream also proves PortAudio is healthy at that
    moment, so the heal is not needed then.

    Windows only: the disconnect it heals is a Windows and NVIDIA phenomenon,
    and on macOS repeated terminate and initialize cycles are known to leave
    CoreAudio in a state where a later stream.stop() never returns.
    """
    if sys.platform != "win32":
        logging.debug("PortAudio reset skipped: non-Windows platform.")
        return
    if _open_streams > 0:
        logging.debug("PortAudio reset skipped: %d stream(s) still open.",
                      _open_streams)
        return
    try:
        sd._terminate()
        sd._initialize()
    except Exception as init_err:
        logging.debug(f"PortAudio reinitialization error: {init_err}")


def uses_winsound() -> bool:
    """True when playback will take the blocking winsound path.

    winsound can only target the default output device, so an explicit
    AUDIO_OUTPUT_DEVICE forces the sounddevice path even on Windows. One source
    of truth, so the playback path and the lead-in length
    (TTSManager.playback_lead_in_seconds) can never disagree.
    """
    return WINSOUND_AVAILABLE and config.AUDIO_OUTPUT_DEVICE is None
