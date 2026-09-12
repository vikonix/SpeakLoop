# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Text-to-speech: the synthesis backends plus the shared playback path.

Two roles live here, split on purpose:

* **Synthesis backends** - one class per engine, selected by the active
  language variant's data (``config.TTS_BACKEND``), never by an
  ``if language`` branch. Each backend has the same small surface:
  ``load_model()``, ``warm_up()``, ``synthesize(text, voice) -> np.ndarray``
  (mono float32 at the backend's own rate) and a ``sample_rate`` attribute.
    - ``KokoroBackend``     - Kokoro-82M (torch), 24 kHz. The English variants.
    - ``SupertonicBackend`` - Supertonic 3 (ONNX, no torch), 44.1 kHz. The
      Spanish variant; the weights are OpenRAIL-M licensed and are downloaded
      into ``model_cache/supertonic3/`` (env SUPERTONIC_CACHE_DIR, set in
      config.py and pre-fetched by install.py).

* **Playback** - ``TTSManager.play_array`` plays any waveform at any sample
  rate (winsound on Windows, sounddevice elsewhere), so it works unchanged
  with either backend.

``TTSManager`` is the facade app.py composes: it owns the playback path and
delegates synthesis to the selected backend, exposing that backend's rate as
``sample_rate`` - which the controller reads, never a constant.
"""

import io
import logging
import os
import time
import wave
from threading import Event, Thread
from typing import Optional

import numpy as np
import sounddevice as sd

from speakloop import config, models_info
from speakloop.audio_io import (
    WINSOUND_AVAILABLE,
    WINSOUND_LEAD_IN_SECONDS,
    reset_portaudio,
    stream_closed,
    stream_opened,
    uses_winsound,
)

# winsound is the module that drives Windows playback below; the availability
# flag and the path choice live in speakloop.audio_io.
if WINSOUND_AVAILABLE:
    import winsound

# A never-set event used as a default, so a caller may omit the stop events.
_NULL_EVENT = Event()

# How long the winsound stop guard keeps watching for a racing stop after the
# playback starts (see play_array). The race window is microseconds; 0.2 s
# covers it with a large margin and keeps no extra thread alive for the whole
# length of a long reply.
WINSOUND_STOP_GUARD_SECONDS = 0.2

# Kokoro synthesizes at 24 kHz; a property of the model, so the constant sits
# next to the backend rather than among the audio settings. Callers read
# TTSManager.sample_rate instead of importing it.
KOKORO_SAMPLE_RATE = 24_000

# Supertonic 3 synthesizes at 44.1 kHz.
SUPERTONIC_SAMPLE_RATE = 44_100

# The warm-up word is language text, so it comes from the active language
# profile (config.TTS_WARMUP), never from a table in code.


class KokoroBackend:
    """Kokoro-82M synthesis (torch), 24 kHz. Used by the English variants."""

    sample_rate = KOKORO_SAMPLE_RATE

    def __init__(self):
        self.model = None
        self.pipeline = None

    def load_model(self):
        """Instantiate the Kokoro network in memory."""
        # Imported here and not at module level: only the backend the active
        # variant selects should pull its machine-learning stack into the
        # process.
        from kokoro import KModel, KPipeline
        # The repo id is bound from the catalogue and never spelled out again:
        # a second copy here could send the download to one repo while the
        # synthesis loaded another.
        #
        # .eval() carries weight. KModel is a plain nn.Module built here rather
        # than loaded through from_pretrained, so it starts in TRAINING mode,
        # and kokoro applies nn.Dropout unconditionally in forward().
        self.model = KModel(repo_id=models_info.KOKORO.repo_id).to(
            config.DEVICE).eval()
        # Both keyword arguments carry weight:
        #
        # repo_id, because KPipeline otherwise substitutes a default of its
        # own - a second copy of the id, which decides where load_voice()
        # fetches from and happens to match ours today.
        #
        # model, because it defaults to True, which makes KPipeline build a
        # SECOND KModel on a device it picks itself (cuda when available), past
        # config.DEVICE. Nothing reads that copy - every call passes
        # model=self.model - so it is 82M parameters parked on the card for the
        # whole session, on the machine that also has to fit the model server
        # and the recognizer.
        self.pipeline = KPipeline(lang_code=config.TTS_LANG_CODE,
                                  repo_id=models_info.KOKORO.repo_id,
                                  model=self.model)
        self._prefetch_voices()

    def _prefetch_voices(self):
        """Download every selectable voice once, while the Hub is still online.

        Kokoro fetches the data of a voice when the voice is first used. In
        offline mode that late download fails, so a voice the user never tried
        would break. All configured voices are pulled during the first (online)
        run; on later offline runs they are cached and this is skipped.
        """
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            return  # offline: what is not cached cannot be fetched anyway
        for voice in config.TTS_VOICES:
            try:
                self.pipeline.load_voice(voice)
            except Exception as error:
                logging.debug(f"Could not prefetch Kokoro voice {voice!r}: {error}")

    def warm_up(self):
        """Run a dummy synthesis pass to remove the first-call latency.

        The word comes from the language profile (config.TTS_WARMUP): a short
        word of the practiced language, so the dummy pass raises no
        out-of-vocabulary phoneme warnings.
        """
        if self.model is None or self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")
        list(self.pipeline(config.TTS_WARMUP, voice=config.TTS_VOICE,
                           model=self.model))

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """Synthesize *text* with *voice*; mono float32 at 24 kHz.

        The caller (TTSManager) has normalized the text and resolved the voice,
        so both arguments are non-empty here.
        """
        if self.model is None or self.pipeline is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        generator = self.pipeline(text, voice=voice, model=self.model)

        audio_chunks = []
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(np.asarray(audio, dtype=np.float32))

        if not audio_chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(audio_chunks)


class SupertonicBackend:
    """Supertonic 3 synthesis (ONNX runtime, no torch), 44.1 kHz.

    Used by the Spanish variant: Kokoro's Spanish is trained on little data
    (audible artifacts), while Supertonic 3 is multilingual by design and
    offers ten voices (F1..F5, M1..M5) at a fraction of the runtime cost. The
    weights (about 400 MB) are OpenRAIL-M licensed and are therefore never
    bundled: they are downloaded by install.py or on the first online run into
    ``model_cache/supertonic3/`` (env SUPERTONIC_CACHE_DIR, set in config.py).
    The download is atomic (temporary directory, then a rename), so a cache
    directory that exists is a complete one and later runs are offline.
    """

    sample_rate = SUPERTONIC_SAMPLE_RATE

    def __init__(self):
        self._tts = None
        # Voice-style objects are read from the JSON files that come with the
        # model (a local read); cached so each voice is parsed only once.
        self._styles = {}

    def load_model(self):
        """Load the ONNX sessions, downloading the model on the first run."""
        # Imported here and not at module level: only the backend the active
        # variant selects should pull onnxruntime into the process.
        from supertonic import TTS
        # The cache location comes from SUPERTONIC_CACHE_DIR (config.py points
        # it at model_cache/supertonic3/, matching the HF_HOME policy).
        # auto_download only downloads when the ONNX files are missing; once
        # install.py or a first online run has fetched them, the start is fully
        # offline. The model name is bound from the catalogue for the same
        # reason the Kokoro repo id above is.
        self._tts = TTS(model=models_info.SUPERTONIC.name, auto_download=True)

    def _style(self, voice: str):
        """The cached voice-style object for *voice* (loaded on first use)."""
        style = self._styles.get(voice)
        if style is None:
            style = self._tts.get_voice_style(voice_name=voice)
            self._styles[voice] = style
        return style

    def warm_up(self):
        """Run a dummy synthesis pass to remove the first-call latency.

        The word comes from the language profile (config.TTS_WARMUP), in the
        practiced language.
        """
        if self._tts is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")
        self.synthesize(config.TTS_WARMUP, config.TTS_VOICE)

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """Synthesize *text* with *voice*; mono float32 at 44.1 kHz.

        The caller (TTSManager) has normalized the text and resolved the voice,
        so both arguments are non-empty here. Speed stays 1.0 (the package
        default is 1.05): the partner must speak at a neutral speed.
        """
        if self._tts is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        wav, _duration = self._tts.synthesize(
            text=text,
            voice_style=self._style(voice),
            lang=config.TTS_LANG_CODE,
            total_steps=config.TTS_TOTAL_STEPS,
            speed=1.0,
            verbose=False,
        )
        # The package returns shape (1, samples); flatten to the mono float32
        # contract shared with KokoroBackend.
        return np.asarray(wav, dtype=np.float32).reshape(-1)


# Backend registry: the active language variant selects by name
# (config.TTS_BACKEND, validated there against these keys through
# config.TTS_BACKEND_CHOICES). A new backend is one class plus one entry.
TTS_BACKENDS = {
    "kokoro": KokoroBackend,
    "supertonic": SupertonicBackend,
}


class TTSManager:
    """The facade app.py composes: synthesis (delegated) plus playback (owned)."""

    def __init__(self):
        self._backend = TTS_BACKENDS[config.TTS_BACKEND]()

    @property
    def sample_rate(self) -> int:
        """Own sample rate of the active synthesis backend (Hz)."""
        return self._backend.sample_rate

    def load_model(self):
        """Load the active backend's synthesis model into memory."""
        self._backend.load_model()

    def warm_up(self):
        """Run a dummy synthesis pass to remove the first-call latency."""
        self._backend.warm_up()

    def synthesize(self, text: str, voice: Optional[str] = None) -> np.ndarray:
        """Synthesize ``text`` and return the waveform (mono float32).

        The waveform is at the backend's own rate (``self.sample_rate``).
        ``voice`` selects the backend voice; without it the configured default
        is used. Any voice of the active variant works without a reload.

        Returns an empty array when there is nothing to say. The whitespace
        collapse and the empty-text short circuit live here, so every backend
        honors the same contract.
        """
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)
        voice = voice or config.TTS_VOICE
        return self._backend.synthesize(text, voice)

    def stop_playback(self):
        """Interrupt the winsound playback that runs right now (Windows only)."""
        if WINSOUND_AVAILABLE:
            winsound.PlaySound(None, 0)

    def playback_lead_in_seconds(self) -> float:
        """Silence play_array prepends before the audio, in seconds (0 if none).

        The Windows audio session needs that time to start, and the value is
        reported here so a caller can account for the delay.
        """
        return WINSOUND_LEAD_IN_SECONDS if uses_winsound() else 0.0

    def play_array(self, waveform: np.ndarray, sample_rate: int,
                   stop_event: Event = _NULL_EVENT,
                   shutdown_event: Event = _NULL_EVENT):
        """Play a synthesized or recorded waveform at the given sample rate.

        Blocking; call it from a background thread.
        """
        full_audio = np.asarray(waveform, dtype=np.float32)
        if full_audio.size == 0:
            return
        # Checked before any device work: a reply that was interrupted while it
        # was still being synthesized must open no stream at all.
        if stop_event.is_set() or shutdown_event.is_set():
            return

        # Normalize the peak to avoid clipping. It must stay ahead of the
        # platform branch, so the loudness is the same on both paths.
        peak = np.max(np.abs(full_audio))
        if peak > 0:
            full_audio = full_audio / peak * 0.9

        try:
            # Windows path: winsound bypasses the PortAudio MME error 6.
            # winsound can only target the default output device, so an
            # explicit AUDIO_OUTPUT_DEVICE forces the sounddevice path below;
            # otherwise that setting would be ignored on Windows.
            if uses_winsound():
                # No config.AUDIO_LOCK here (unlike the sounddevice branch and
                # the recorder): that lock guards the PortAudio init and
                # teardown, and winsound does not touch PortAudio at all.
                # Taking it here was what made a recording wait for the speech
                # to end, which cut the beginning of the take off.
                #
                # Prepend silence so the Windows audio session can start
                # without clipping the first 150 ms. The lead-in follows the
                # sample rate.
                lead_in = np.zeros(
                    int(sample_rate * WINSOUND_LEAD_IN_SECONDS),
                    dtype=np.float32)
                full_audio = np.concatenate([lead_in, full_audio])

                # Convert float32 (-1.0..1.0) to 16-bit PCM.
                pcm_data = (full_audio * 32767).astype(np.int16)

                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wav_file:
                    wav_file.setnchannels(1)   # Mono
                    wav_file.setsampwidth(2)   # 16-bit PCM
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm_data.tobytes())
                wav_bytes = wav_io.getvalue()

                if stop_event.is_set() or shutdown_event.is_set():
                    return
                # winsound cannot combine SND_MEMORY with SND_ASYNC (it raises
                # "Cannot play asynchronously from memory"), so the playback is
                # synchronous and a short-lived guard thread closes the race
                # instead: a stop that lands in the microseconds between the
                # check above and PlaySound taking the audio channel has fired
                # its PlaySound(None, 0) too early, and the reply would play to
                # the end anyway, possibly into an open microphone. The guard
                # re-checks the events for the first
                # WINSOUND_STOP_GUARD_SECONDS and repeats the interrupt, so
                # such a stop cuts the sound within about 10 ms. ``done`` keeps
                # a guard that outlives this playback from killing the next one.
                done = Event()

                def _stop_guard():
                    guard_deadline = time.monotonic() + WINSOUND_STOP_GUARD_SECONDS
                    while time.monotonic() < guard_deadline and not done.is_set():
                        if stop_event.is_set() or shutdown_event.is_set():
                            if not done.is_set():
                                winsound.PlaySound(None, 0)
                            return
                        time.sleep(0.01)

                Thread(target=_stop_guard, daemon=True).start()
                try:
                    # Synchronous; normally interrupted by stop_playback(),
                    # which calls PlaySound(None, 0).
                    winsound.PlaySound(wav_bytes,
                                       winsound.SND_MEMORY | winsound.SND_NODEFAULT)
                finally:
                    done.set()
                return

            # Every other platform goes through sounddevice.
            if full_audio.ndim == 1:
                full_audio = full_audio.reshape(-1, 1)

            with config.AUDIO_LOCK:
                reset_portaudio()
                stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=config.AUDIO_CHANNELS,
                        dtype="float32",
                        blocksize=0,
                        latency=config.AUDIO_LATENCY,
                        device=config.AUDIO_OUTPUT_DEVICE,
                )
                try:
                    stream.start()
                except Exception:
                    stream.close()  # do not leak a stream that never started
                    raise
                stream_opened()  # counted only once fully started (see finally)

            try:
                # Written in small blocks, so a stop is noticed during the
                # playback and not only at its end.
                chunk_size = 1024
                for i in range(0, len(full_audio), chunk_size):
                    if stop_event.is_set() or shutdown_event.is_set():
                        return
                    stream.write(full_audio[i:i + chunk_size])
            finally:
                with config.AUDIO_LOCK:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception as close_error:
                        logging.debug("Error during sound output stream close: "
                                      f"{close_error}")
                    stream_closed()

        except Exception:
            logging.exception("TTS playback error:")
