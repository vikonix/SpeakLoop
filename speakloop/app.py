# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""The controller: the voice loop and the threads behind it, plus run().

Started only through speakloop/cli.py, which calls bootstrap.early_init()
before this module is imported (UTF-8 console, warning filters) - that is why
nothing of the kind is done here. Logging is configured in run(), after the
heavy imports below, for the reason given in bootstrap.setup_logging.

The window lives in speakloop/ui.py. This module creates the Tk root, composes
a TutorView on it (``self.view``) and drives it through the view's intent
methods; it owns no widget, no color and no interface wording. Every worker
thread reaches the window through ``self.root.after()``, the only thread-safe
way into Tk - a direct call from a thread is the one mistake this split cannot
prevent on its own.
"""

import time
import queue
import threading
from typing import Optional
import os
import logging
import tkinter as tk
import numpy as np
import sounddevice as sd

# config first: it sets HF_HOME and the offline switch, which huggingface_hub
# reads when stt and tts import faster_whisper and kokoro below.
from speakloop import config
from speakloop import bootstrap, detect_hardware, lifecycle
from speakloop.stt import STTManager, WHISPER_SAMPLE_RATE
from speakloop.llm import LLMManager, error_message
from speakloop.llm_server_ctl import LLMServerController
from speakloop.tts import TTSManager
from speakloop.ui import TutorView, ViewCallbacks

# Sentinel object pushed to the TTS queue after LLM finishes streaming.
# The TTS thread buffers sentences and only starts playback when it sees this object,
# ensuring the LLM has released the GPU before Kokoro synthesis begins.
_TTS_START_SENTINEL = object()

# Technical recording & signal processing parameters
RECORDING_BLOCKSIZE = 1024  # Small block sizes maintain responsive streaming frame intervals

# Signal gain normalization parameters
AUDIO_MIN_PEAK_THRESHOLD = 0.01      # Prevents boosting pure background noise floor during silence
AUDIO_NORMALIZATION_CEILING = 0.9    # Scales the peak target output level directly to 90%

# How long to wait for the recording thread to finish after stopping.
# Covers the last InputStream callback flush; should be well under 1 second in normal use.
RECORD_THREAD_JOIN_TIMEOUT_SEC = 1.5


class VoiceTutorController:
    """The application: the voice loop, its threads, and the window's driver.

    Holds the controller logic (model loading, recording, transcription, the
    model exchange and speech output) and owns the view by composition:
    ``self.view`` is a TutorView (ui.py) that builds and renders the widgets.
    The controller drives the window through ``self.view.*`` and the view
    forwards its bindings back to the handlers passed in ViewCallbacks.

    Flow per exchange (state machine):
        Record     -> Space or the mic button is held, record_loop captures.
        Process    -> faster-whisper transcribes the take (stt.py).
        Think      -> the model answers as a stream (llm.py), tokens go to the
                      window and whole sentences to the TTS queue.
        Speak      -> the queue is played once the model is done (sentinel).
        Loop       -> back to idle; a new recording interrupts the speech.
    """

    def __init__(self):
        logging.info("Starting Voice Tutor GUI Application...")

        # Core Tkinter setup. Only the root is created here; its title, size,
        # colors and widgets are the view's (see TutorView below).
        self.root = tk.Tk()

        # Thread management events
        self.shutdown_event = threading.Event()
        self.tts_stop_event = threading.Event()

        # Recording state management
        self.is_recording = False
        self.space_is_held = False
        self.record_lock = threading.Lock()
        self.recorded_chunks: list[np.ndarray] = []
        self.record_thread: Optional[threading.Thread] = None

        # Audio processing guard - prevents concurrent process_audio() calls
        self.is_processing_audio = False
        self.processing_lock = threading.Lock()

        # TTS state tracking - avoids reading Tkinter widget from background thread
        self._tts_is_speaking = False
        self.tts_state_lock = threading.Lock()

        # Text-to-Speech background queue and thread
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.tts_thread: Optional[threading.Thread] = None

        # Initialize core modular sub-managers
        self.stt_mgr = STTManager()
        self.tts_mgr = TTSManager()

        # LLM backend. config validates the name and falls back to the default,
        # so only the two known values reach this point.
        self.llm_backend = config.LLM_BACKEND
        logging.info(f"Using the {self.llm_backend} LLM backend.")
        # One client for both backends: they speak the same OpenAI API and only
        # the address differs, which init_client is told at connection time.
        self.llm_mgr = LLMManager()
        # Owns the llama-server subprocess. Built for every backend and left
        # untouched by "lm-studio": all of its methods are no-ops until start().
        self._llm_server = LLMServerController()

        # The window. Built last of the members, because the loader thread
        # started below drives it at once.
        self.view = TutorView(self.root, ViewCallbacks(
            on_mic_pressed=self.on_mic_pressed,
            on_mic_released=self.on_mic_released,
            on_space_pressed=self.on_space_pressed,
            on_space_released=self.on_space_released,
            on_quit=self.quit_app,
        ))

        # Start loading models in a background thread to prevent UI freezing
        threading.Thread(target=self.load_components, daemon=True).start()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def load_components(self):
        logging.info("Starting model loading thread...")
        self.root.after(0, self.view.enter_loading)
        self.root.after(0, self.view.append_system_msg, "Loading Whisper (STT) and Kokoro (TTS) models...")

        try:
            self.stt_mgr.load_model()
            logging.info("STT Model loaded successfully.")

            self.tts_mgr.load_model()
            logging.info("TTS Model loaded successfully.")

            if self.llm_backend == "llama-server":
                # Neutral until start() has decided. It uses a server that
                # already listens instead of launching one, and a line that
                # named the model and the launch would then describe something
                # that did not happen. Both are said below, once it is known
                # which of the two it was.
                self.root.after(0, self.view.append_system_msg, "Connecting to the LLM server...")
                self.root.after(0, self.view.enter_connecting)
                ready = self._llm_server.start(self.llm_mgr)
                if not ready:
                    # The controller's own sentence, because only it knows which
                    # of several failures happened (no binary, a busy port, an
                    # early exit). Both logs are named: the reason is in
                    # main.log and the server's own output in llm_server.log,
                    # and a failure before the launch writes nothing to the
                    # second one.
                    reason = (self._llm_server.last_error
                              or "The LLM server did not start.")
                    self.root.after(0, self.view.append_system_msg, f"Error: {reason}")
                    self.root.after(0, self.view.append_system_msg, "See logs/main.log and logs/llm_server.log.")
                    self.root.after(0, self.view.server_failed)
                    # Do not make the window ready - there is nothing to answer
                    # a recording with.
                    return
                if self._llm_server.adopted:
                    # Said in the window and not only in the log: the answers
                    # now come from a server this run did not configure, which
                    # explains a model or a speed the settings do not.
                    self.root.after(0, self.view.append_system_msg,
                                    f"Using the llama-server already running on "
                                    f"{config.LLM_SERVER_HOST}:{config.LLM_SERVER_PORT}. "
                                    f"It keeps the model it was started with.")
                else:
                    model_name = os.path.basename(config.EXTERNAL_MODEL_PATH)
                    self.root.after(0, self.view.append_system_msg,
                                    f"llama-server is ready with {model_name}.")
            else:
                self.llm_mgr.init_client()
                if not self.llm_mgr.check_connection():
                    self.root.after(0, self.view.append_system_msg, "Warning: LM Studio is offline. Start it to use voice tutor!")
                    logging.warning("LM Studio is offline during initialization.")

            self.root.after(0, self.view.enter_warming_up)
            self.stt_mgr.warm_up()
            self.tts_mgr.warm_up()
            logging.info("Models warmed up successfully.")

            # Start TTS background thread
            self.tts_thread = threading.Thread(target=self.process_tts_queue, daemon=True)
            self.tts_thread.start()
            logging.info("TTS Queue processor thread started.")

            # Make App Ready
            self.root.after(0, self.make_app_ready)
            logging.info("Voice Tutor initialization fully completed.")

        except Exception as e:
            logging.exception("Error during initialization thread:")
            self.root.after(0, self.view.append_system_msg, f"Initialization Error: {e}")
            self.root.after(0, self.view.init_failed)

    def make_app_ready(self):
        with self.tts_state_lock:
            self._tts_is_speaking = False
        self.view.enter_app_ready()
        self.view.append_system_msg(f"Voice Tutor ready. Practice learning {config.TARGET_LANGUAGE}!")

    # ------------------------------------------------------------------
    # Push-to-talk handlers (called by the view's bindings)
    # ------------------------------------------------------------------
    def on_mic_pressed(self):
        # Click behavior (simulates holding space)
        if not self.space_is_held:
            logging.info("GUI microphone button clicked.")
            self.trigger_recording_start()

    def on_mic_released(self):
        with self.record_lock:
            currently_recording = self.is_recording
        if currently_recording and not self.space_is_held:
            logging.info("GUI microphone button released.")
            self.trigger_recording_stop()

    def on_space_pressed(self):
        # Holding the key repeats KeyPress; the flag keeps the first one.
        if not self.space_is_held:
            self.space_is_held = True
            logging.info("Spacebar keyboard press event.")
            self.trigger_recording_start()

    def on_space_released(self):
        if self.space_is_held:
            self.space_is_held = False
            logging.info("Spacebar keyboard release event.")
            self.trigger_recording_stop()

    def trigger_recording_start(self):
        with self.record_lock:
            if self.is_recording:
                return  # Safety guard

            logging.info("Starting audio recording...")
            self.stop_current_tts()
            self.is_recording = True
            self.recorded_chunks = []

            # All window updates are scheduled on the main thread.
            self.root.after(0, self.view.enter_recording)

            self.record_thread = threading.Thread(target=self.record_loop, daemon=True)
            self.record_thread.start()

    def trigger_recording_stop(self):
        with self.record_lock:
            if not self.is_recording:
                return
            logging.info("Stopping audio recording...")
            self.is_recording = False

        self.root.after(0, self.view.enter_processing)

        # Join and audio processing happen off the main thread to prevent UI freeze.
        # record_thread.join() can block up to RECORD_THREAD_JOIN_TIMEOUT_SEC -
        # running it on the main thread would make the window unresponsive.
        threading.Thread(target=self._finalize_recording, daemon=True).start()

    def _finalize_recording(self):
        """Joins the record thread, then starts audio processing - runs off the main thread."""
        if self.record_thread:
            self.record_thread.join(timeout=RECORD_THREAD_JOIN_TIMEOUT_SEC)

        with self.processing_lock:
            if self.is_processing_audio:
                logging.warning("process_audio already running, skipping duplicate.")
                return
            self.is_processing_audio = True

        self._process_audio_safe()

    def _process_audio_safe(self):
        """Wrapper that ensures process_audio runs exactly once and releases the guard."""
        try:
            self.process_audio()
        finally:
            with self.processing_lock:
                self.is_processing_audio = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_loop(self):
        start_time = time.time()
        logging.info("sd.InputStream thread started.")

        # Warnings from the realtime callback are buffered here and logged from
        # the main record loop - calling logging directly inside a sounddevice
        # callback can block on I/O and cause audio dropouts.
        callback_warnings: list[str] = []

        def callback(indata, frames, time_info, status):
            if status:
                callback_warnings.append(str(status))
            with self.record_lock:
                if self.is_recording:
                    self.recorded_chunks.append(indata.copy())

        try:
            with config.AUDIO_LOCK:
                try:
                    sd._terminate()
                    sd._initialize()
                except Exception as init_err:
                    logging.debug(f"PortAudio reinitialization error: {init_err}")

                stream = sd.InputStream(
                        samplerate=WHISPER_SAMPLE_RATE,
                        channels=config.AUDIO_CHANNELS,
                        dtype="float32",
                        blocksize=RECORDING_BLOCKSIZE,
                        latency=config.AUDIO_LATENCY,
                        device=config.AUDIO_INPUT_DEVICE,
                        callback=callback,
                )
                stream.start()

            try:
                while True:
                    # Flush warnings accumulated by the realtime callback
                    while callback_warnings:
                        logging.warning(f"Audio input warning: {callback_warnings.pop(0)}")

                    with self.record_lock:
                        still_recording = self.is_recording

                    if not still_recording:
                        break

                    if time.time() - start_time >= config.MAX_RECORD_SECONDS:
                        logging.info("Maximum recording duration reached.")
                        self.root.after(0, self.view.append_system_msg, "Reached maximum record limit.")
                        with self.record_lock:
                            self.is_recording = False
                        break

                    time.sleep(0.01)
            finally:
                with config.AUDIO_LOCK:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception as close_error:
                        logging.debug(f"Error during sound input stream close: {close_error}")

        except Exception:
            logging.exception("Recording InputStream error:")
            with self.record_lock:
                self.is_recording = False
            self.root.after(0, self.view.recording_failed)

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        logging.info(f"Normalizing audio. Peak signal level: {peak:.4f}")
        if peak < AUDIO_MIN_PEAK_THRESHOLD:
            logging.info("Peak signal is too low (silence). Skipping gain adjustment.")
            return audio.astype(np.float32)
        audio = audio / peak * AUDIO_NORMALIZATION_CEILING
        return np.nan_to_num(audio).astype(np.float32)

    def get_recorded_audio(self) -> Optional[np.ndarray]:
        with self.record_lock:
            if not self.recorded_chunks:
                return None
            chunks = list(self.recorded_chunks)
            self.recorded_chunks = []
        return np.concatenate(chunks, axis=0).flatten().astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # One exchange: transcribe, ask the model, queue the speech
    # ------------------------------------------------------------------
    def process_audio(self):
        try:
            audio = self.get_recorded_audio()
            if audio is None or len(audio) < WHISPER_SAMPLE_RATE * 0.2:
                logging.warning("Captured audio too short or empty.")
                self.root.after(0, self.view.append_system_msg, "Audio is too short. Try holding space longer.")
                self.root.after(0, self.view.enter_idle)
                return

            audio = self.normalize_audio(audio)

            # Speech-to-Text (STT)
            stt_start = time.perf_counter()
            user_text = self.stt_mgr.transcribe(audio)
            stt_ms = (time.perf_counter() - stt_start) * 1000
            logging.info(f"STT transcribed speech: {user_text!r} | Latency: {stt_ms:.0f}ms")

            if not user_text:
                logging.info("STT returned empty transcription.")
                self.root.after(0, self.view.append_system_msg, "Could not hear you clearly. Please try again.")
                self.root.after(0, self.view.enter_idle)
                return

            # Update User Speech to GUI
            self.root.after(0, self.view.append_user_msg, user_text)
            self.root.after(0, self.view.enter_thinking)

            # Start LLM stream feeding the TTS queue
            llm_start = time.perf_counter()
            self.clear_tts_queue()
            self.tts_stop_event.clear()

            self.root.after(0, self.view.append_reply_start)

            # Streaming callback to append tokens live
            def token_cb(token):
                self.root.after(0, self.view.append_reply_token, token)

            try:
                self.llm_mgr.stream_and_queue_tts(
                    user_text,
                    self.tts_queue,
                    self.tts_stop_event,
                    token_callback=token_cb
                )
            except Exception as llm_error:
                # Handled here and not by the outer handler, which cannot know
                # that a reply line is already open in the chat. Without this
                # the window keeps an empty partner line and a status bar that
                # still says "Thinking", and the failure is only in the log.
                # llm.py has logged the traceback already.
                self.root.after(0, self.view.append_reply_end)
                self.root.after(0, self.view.append_system_msg,
                                f"LLM error: {error_message(llm_error)}")
                self._abort_tts()
                self.root.after(0, self.view.enter_error, "LLM Error")
                return

            llm_ms = (time.perf_counter() - llm_start) * 1000
            logging.info(f"LLM complete streaming and queuing. Duration: {llm_ms:.0f}ms")

            self.root.after(0, self.view.append_reply_end)
            self.root.after(0, self.view.update_stats, stt_ms, llm_ms)

            # Signal TTS thread that LLM has finished and GPU is free.
            # The TTS thread buffers sentences until it receives this sentinel,
            # preventing GPU contention between the model server and Kokoro.
            if not self.tts_stop_event.is_set():
                self.tts_queue.put(_TTS_START_SENTINEL)
                with self.tts_state_lock:
                    self._tts_is_speaking = True
                self.root.after(0, self.view.enter_speaking)

        except Exception:
            logging.exception("Error in process_audio:")
            self.root.after(0, self.view.append_system_msg, "Processing Error. Please try again.")
            self.root.after(0, self.view.enter_error, "Error")

    # ------------------------------------------------------------------
    # Speech output
    # ------------------------------------------------------------------
    def stop_current_tts(self):
        logging.info("Stopping active text-to-speech output...")
        with self.tts_state_lock:
            self._tts_is_speaking = False
        self.tts_stop_event.set()
        self.clear_tts_queue()
        self.tts_mgr.stop_playback()

    def _abort_tts(self):
        """Drop everything queued for speech after a failed exchange.

        stop_current_tts() empties the queue, but sentences the TTS thread has
        already taken from it are buffered inside that thread and are dropped
        only when it sees another item. The sentinel is that item: with the
        stop event set it clears the buffer and plays nothing. Without it those
        sentences would be spoken after the NEXT reply.
        """
        self.stop_current_tts()
        self.tts_queue.put(_TTS_START_SENTINEL)

    def clear_tts_queue(self):
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break

    def process_tts_queue(self):
        # Sentences are buffered here while LLM is still running on the GPU.
        # Playback starts only after _TTS_START_SENTINEL arrives (LLM done, GPU free).
        pending_sentences: list[str] = []

        while not self.shutdown_event.is_set():
            try:
                item = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                # Transition to idle only when truly done: sentinel was received (pending
                # is empty) and there are no more items waiting in the queue.
                if not pending_sentences:
                    with self.record_lock:
                        currently_recording = self.is_recording
                    if not currently_recording and not self.tts_stop_event.is_set() and self.tts_queue.empty():
                        with self.tts_state_lock:
                            tts_speaking = self._tts_is_speaking
                        if tts_speaking:
                            self.root.after(0, self.view.enter_idle)
                            with self.tts_state_lock:
                                self._tts_is_speaking = False
                continue

            try:
                if item is _TTS_START_SENTINEL:
                    # LLM has finished - GPU is now free. Play all buffered sentences.
                    logging.info(f"TTS sentinel received. Playing {len(pending_sentences)} buffered sentence(s).")
                    remaining = list(pending_sentences)
                    pending_sentences.clear()
                    for sentence in remaining:
                        if self.tts_stop_event.is_set() or self.shutdown_event.is_set():
                            break
                        try:
                            logging.info(f"TTS playing synthesized block: {sentence!r}")
                            self.tts_mgr.play_stream(sentence, self.tts_stop_event, self.shutdown_event)
                        except Exception as play_err:
                            # Skip the failed sentence and continue with the rest
                            logging.exception(f"TTS playback error for {sentence!r}:")
                elif self.tts_stop_event.is_set():
                    # Stop was requested - discard buffered sentences and this one
                    pending_sentences.clear()
                else:
                    # LLM still running - buffer the sentence, do not synthesize yet
                    logging.info(f"TTS buffering sentence (waiting for LLM): {item!r}")
                    pending_sentences.append(item)
            except Exception:
                logging.exception("Error in TTS queue thread:")
            finally:
                self.tts_queue.task_done()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def quit_app(self):
        logging.info("Shutting down VoiceTutor App...")
        self.shutdown_event.set()
        self.stop_current_tts()

        # Terminate the llama-server subprocess if this app started one. A
        # no-op for the "lm-studio" backend, for a server that was adopted
        # rather than started, and while the loader thread is still in start():
        # see LLMServerController.shutdown.
        self._llm_server.shutdown()

        self.root.destroy()
        # Not the interpreter's normal exit: with CUDA torch loaded, tearing the
        # CUDA context down during finalization can crash the process on
        # Windows (0xC0000409). hard_exit flushes the logs and ends the process
        # without that teardown. Everything that needs a clean release (the
        # server subprocess, its log file) is released above.
        lifecycle.hard_exit()

    def run(self):
        self.root.mainloop()


def run(append_log: bool = False) -> None:
    """Start the application: logging, the startup checks, then the window.

    The only thing speakloop/cli.py calls. ``append_log`` continues
    logs/main.log instead of truncating it (see bootstrap.setup_logging).
    """
    bootstrap.setup_logging(config.LOG_FILE, append=append_log)
    # One log line when an NVIDIA GPU is present but torch runs on the CPU:
    # the app works then, only several times slower, and nothing else says so.
    detect_hardware.warn_if_gpu_unused(config.DEVICE)
    app = VoiceTutorController()
    app.run()
