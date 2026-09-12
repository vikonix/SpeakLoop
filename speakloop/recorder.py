# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Microphone capture and recorded-signal helpers.

AudioRecorder owns the capture thread, the input device choice and the raw
chunk buffer; the controller only starts and stops takes and collects the
result as one 16 kHz numpy array. The pure signal helper (normalize_audio)
lives here too, so the whole microphone side of the audio path is in this
module.
"""

import logging
import threading
import time
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

from speakloop import config
from speakloop.audio_io import reset_portaudio, stream_closed, stream_opened

# Technical recording and signal processing parameters
RECORDING_BLOCKSIZE = 0  # 0 -> PortAudio picks an optimal block size. A small
                         # fixed size together with low-latency buffers caused
                         # capture underruns (silence gaps inserted by the
                         # driver, heard as clicks) on Windows MME.

# Signal gain normalization parameters
AUDIO_MIN_PEAK_THRESHOLD = 0.01      # Prevents boosting the background noise floor during silence
AUDIO_NORMALIZATION_CEILING = 0.9    # Scales the peak to 90% of full scale

# How long to wait for the capture thread to finish after a stop.
RECORD_THREAD_JOIN_TIMEOUT_SEC = 1.5


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Scale the waveform so its peak hits the normalization ceiling.

    A near-silent take is returned unchanged, so the background noise floor is
    not boosted into the recognizer.
    """
    peak = np.max(np.abs(audio))
    logging.debug(f"Normalizing audio. Peak signal level: {peak:.4f}")
    if peak < AUDIO_MIN_PEAK_THRESHOLD:
        logging.debug("Peak signal is too low (silence). Skipping gain adjustment.")
        return audio.astype(np.float32)
    audio = audio / peak * AUDIO_NORMALIZATION_CEILING
    return np.nan_to_num(audio).astype(np.float32)


class AudioRecorder:
    """One take of microphone capture running on its own daemon thread.

    Usage: start() opens the capture thread, stop() asks it to finish, join()
    waits for it, get_audio() returns the take as one 16 kHz float32 array.

    All callbacks are invoked on the capture thread, so a GUI caller must
    marshal every widget update onto the Tk main thread itself (root.after):
        on_max_duration  - the take reached config.MAX_RECORD_SECONDS; the
                           caller is expected to route this through its normal
                           stop path, so the take is finalized like a manual
                           stop.
        on_stream_error  - the input stream failed; recording is already
                           flagged off, the caller only restores its window.
        on_silence_stop  - the speaker was silent for config.SILENCE_TIMEOUT
                           after having started to speak; like on_max_duration,
                           the caller routes it through its normal stop path.
        on_level         - live input level (RMS, 0..1) while a take runs,
                           throttled to about 20 Hz. It lets the window show
                           that the microphone hears the speaker, so the
                           automatic stop is not a black box. Best effort: it
                           never blocks the capture loop and its exceptions are
                           logged and dropped.
    """

    # How often (seconds) the live input level is reported through on_level.
    # About 20 Hz is smooth enough for an indicator and does not flood the Tk
    # event queue.
    LEVEL_EMIT_INTERVAL_SEC = 0.05

    def __init__(self, on_max_duration: Callable[[], None],
                 on_stream_error: Callable[[], None],
                 on_silence_stop: Callable[[], None],
                 on_level: Callable[[float], None]):
        self._on_max_duration = on_max_duration
        self._on_stream_error = on_stream_error
        self._on_silence_stop = on_silence_stop
        self._on_level = on_level

        self.is_recording = False
        self.record_lock = threading.Lock()
        self.recorded_chunks: List[np.ndarray] = []
        self.record_thread: Optional[threading.Thread] = None
        # Rate the microphone is actually captured at. The device's own rate is
        # used (through WASAPI on Windows) to avoid the driver's low-quality
        # resampling; the take is downsampled to 16 kHz in get_audio().
        self.capture_sr: int = config.AUDIO_SAMPLE_RATE

    def is_active(self) -> bool:
        with self.record_lock:
            return self.is_recording

    def start(self) -> bool:
        """Begin a new take.

        Returns False if a take is already running, or if the previous take's
        capture thread has not exited yet (its join timed out): that thread's
        callback may still be appending, and because the callback reads
        ``self.recorded_chunks`` at call time, starting now would mix the stuck
        take's tail into the new take's buffer.
        """
        with self.record_lock:
            if self.is_recording:
                return False
            if self.record_thread is not None and self.record_thread.is_alive():
                logging.warning("Previous record thread is still alive; "
                                "refusing to start a new take.")
                return False
            logging.info("Starting audio recording...")
            self.is_recording = True
            self.recorded_chunks = []
            self.record_thread = threading.Thread(target=self._record_loop,
                                                  daemon=True)
            self.record_thread.start()
        return True

    def stop(self) -> bool:
        """Ask the capture thread to finish. Returns False if not recording."""
        with self.record_lock:
            if not self.is_recording:
                return False
            logging.info("Stopping audio recording...")
            self.is_recording = False
        return True

    def join(self, timeout: float = RECORD_THREAD_JOIN_TIMEOUT_SEC) -> bool:
        """Wait for the capture thread to finish.

        Returns True once the thread has exited (or never ran). Returns False
        if it is still alive after the timeout: its callback may then still be
        appending chunks, so the buffer must not be read.
        """
        if self.record_thread is None:
            return True
        self.record_thread.join(timeout=timeout)
        if self.record_thread.is_alive():
            logging.warning(f"Record thread still alive after {timeout}s; "
                            "its chunk buffer may still be written to.")
            return False
        return True

    def get_audio(self) -> Optional[np.ndarray]:
        """Return the finished take as a 16 kHz mono float32 array (or None).

        Only call after join() has confirmed that the capture thread exited:
        the chunk buffer has no other guard against a running writer.
        """
        with self.record_lock:
            if not self.recorded_chunks:
                return None
            chunks = list(self.recorded_chunks)
            self.recorded_chunks = []
        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32,
                                                                copy=False)

        # The take was captured at the device's own rate; the rest of the
        # pipeline (recognition, playback of the take) expects 16 kHz.
        if self.capture_sr != config.AUDIO_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=self.capture_sr,
                                     target_sr=config.AUDIO_SAMPLE_RATE)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _select_capture_device(self):
        """Choose the input device and the capture sample rate.

        On Windows the default PortAudio host API is MME, which drops samples
        (silence gaps inserted by the driver, heard as clicks). WASAPI is free
        of that, so its default input device is preferred and the capture runs
        at that device's own rate. Returns (device_index, sample_rate) and
        falls back to the configured device at 16 kHz when WASAPI or its device
        cannot be resolved.
        """
        # An explicit device from hardware detection always wins.
        if config.AUDIO_INPUT_DEVICE is not None:
            return config.AUDIO_INPUT_DEVICE, config.AUDIO_SAMPLE_RATE
        try:
            for api in sd.query_hostapis():
                if "wasapi" not in api["name"].lower():
                    continue
                dev_index = api.get("default_input_device", -1)
                if dev_index is None or dev_index < 0:
                    break
                native_sr = int(round(
                    sd.query_devices(dev_index)["default_samplerate"]))
                logging.info(f"Capturing via WASAPI device #{dev_index} "
                             f"at {native_sr} Hz.")
                return dev_index, native_sr
        except Exception:
            logging.exception("WASAPI device selection failed; using defaults.")
        return config.AUDIO_INPUT_DEVICE, config.AUDIO_SAMPLE_RATE

    def _record_loop(self):
        start_time = time.time()
        logging.info("sd.InputStream thread started.")
        callback_warnings: List[str] = []
        capture_device, self.capture_sr = self._select_capture_device()

        def callback(indata, frames, time_info, status):
            # Runs on the realtime audio thread of PortAudio, which has a hard
            # deadline. It must never block, so it takes no lock: list.append
            # is atomic under the GIL, and recorded_chunks is only read after
            # the stream is closed and this thread is joined (see join and
            # get_audio), so there is no concurrent reader to guard against.
            # Holding record_lock here was the cause of dropped samples
            # (audible clicks) while the GUI thread held the same lock during a
            # start or a stop.
            if status:
                callback_warnings.append(str(status))
            self.recorded_chunks.append(indata.copy())

        try:
            with config.AUDIO_LOCK:
                reset_portaudio()
                stream = sd.InputStream(
                        samplerate=self.capture_sr,
                        channels=config.AUDIO_CHANNELS,
                        dtype="float32",
                        blocksize=RECORDING_BLOCKSIZE,
                        # "high" asks the host API for its larger, safer
                        # buffers, which stops the input underruns that are
                        # heard as silence gaps.
                        latency="high",
                        device=capture_device,
                        callback=callback,
                )
                try:
                    stream.start()
                except Exception:
                    stream.close()  # do not leak a stream that never started
                    raise
                stream_opened()  # counted only once fully started (see finally)

            # Voice-activity state of the automatic stop. The level of newly
            # arrived chunks is measured here, on the poll thread, never in the
            # realtime callback, which must not do this much work:
            #   processed       - how many chunks were measured already, so
            #                     only the ones appended since the last poll
            #                     are looked at.
            #   speech_started  - lead-in grace: silence is ignored until the
            #                     speaker first crosses the speech threshold,
            #                     so a slow start never cuts the take short.
            #   last_voice_time - time of the most recent chunk above the
            #                     threshold; the take stops once the gap since
            #                     it passes config.SILENCE_TIMEOUT.
            #   last_level_emit - throttles on_level to LEVEL_EMIT_INTERVAL_SEC.
            processed = 0
            speech_started = False
            last_voice_time = start_time
            last_level_emit = 0.0
            try:
                while True:
                    while callback_warnings:
                        logging.warning(
                            f"Audio input warning: {callback_warnings.pop(0)}")

                    with self.record_lock:
                        still_recording = self.is_recording
                    if not still_recording:
                        break

                    # Measure the chunks that arrived since the last poll. The
                    # slice is safe against the appending callback under the
                    # GIL (at worst the newest chunk is not seen yet).
                    total = len(self.recorded_chunks)
                    if total > processed:
                        block = np.concatenate(
                            self.recorded_chunks[processed:total], axis=0)
                        processed = total
                        rms = (float(np.sqrt(np.mean(np.square(block))))
                               if block.size else 0.0)
                        now = time.time()
                        # Strictly above: with ">=" a zero threshold would make
                        # even digital silence count as speech and disarm the
                        # automatic stop for good. The validated minimum in
                        # config.py is above zero for the same reason; both
                        # layers have to fail before that happens.
                        if rms > config.SILENCE_THRESHOLD:
                            speech_started = True
                            last_voice_time = now
                        if now - last_level_emit >= self.LEVEL_EMIT_INTERVAL_SEC:
                            last_level_emit = now
                            try:
                                self._on_level(rms)
                            except Exception:
                                logging.exception("on_level callback failed:")
                        # Stop only after the speaker has actually begun, so
                        # the pause before the first word is never counted.
                        if speech_started and \
                                now - last_voice_time >= config.SILENCE_TIMEOUT:
                            logging.info("Silence timeout reached; "
                                         "auto-stopping take.")
                            # Routed through the caller's normal stop path,
                            # exactly like on_max_duration, so the take is
                            # finalized instead of being left in the recording
                            # state.
                            self._on_silence_stop()
                            break

                    if time.time() - start_time >= config.MAX_RECORD_SECONDS:
                        logging.info("Maximum recording duration reached.")
                        # The caller routes this through its normal stop path
                        # too. Do not clear is_recording here: that leaves the
                        # take unprocessed and the window stuck in the
                        # recording state.
                        self._on_max_duration()
                        break

                    time.sleep(0.01)
            finally:
                with config.AUDIO_LOCK:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception as close_error:
                        logging.debug("Error during sound input stream close: "
                                      f"{close_error}")
                    stream_closed()

        except Exception:
            logging.exception("Recording InputStream error:")
            with self.record_lock:
                self.is_recording = False
            self._on_stream_error()
