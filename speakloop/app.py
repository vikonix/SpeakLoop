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

The audio itself belongs to three modules of its own: speakloop/recorder.py
captures a take, speakloop/tts.py synthesizes and plays a reply, and
speakloop/playback.py owns the stop event that says which reply may still be
heard. This module only routes between them.
"""

import logging
import os
import queue
import threading
import time
import tkinter as tk
from typing import NamedTuple, Optional

import numpy as np

# config first: it sets HF_HOME, the Supertonic cache and the offline switch,
# which huggingface_hub reads when stt and tts import their engines below.
from speakloop import config
from speakloop import bootstrap, detect_hardware, lifecycle
from speakloop.llm import LLMManager, error_message
from speakloop.llm_server_ctl import LLMServerController
from speakloop.playback import PlaybackController
from speakloop.recorder import AudioRecorder, normalize_audio, warm_up_resampler
from speakloop.stt import STTManager
from speakloop.tts import TTSManager
from speakloop.ui import TutorView, ViewCallbacks


class _ReplyEnd(NamedTuple):
    """Queue marker: the model has finished, its sentences may now be spoken.

    The speech thread buffers the sentences of a reply and synthesizes nothing
    until this marker arrives, so the model server has released the GPU before
    the synthesis starts (both share one card).

    The marker carries the stop event of ITS OWN reply. That is what makes an
    interrupt final: a take started meanwhile has already set that event, so the
    buffered sentences are dropped instead of being spoken over the new take.
    """
    stop_event: threading.Event


# Shortest take that is passed to recognition, in seconds. Anything below is a
# slip of the key rather than a phrase.
MIN_RECORD_SECONDS = 0.2


class VoiceTutorController:
    """The application: the voice loop, its threads, and the window's driver.

    Holds the controller logic (model loading, the exchange with the model and
    the routing of audio) and owns the view by composition: ``self.view`` is a
    TutorView (ui.py) that builds and renders the widgets. The controller drives
    the window through ``self.view.*`` and the view forwards its bindings back
    to the handlers passed in ViewCallbacks.

    Flow per exchange (state machine):
        Record     -> one press opens the microphone; the take ends by itself
                      after a pause, or on the next press (recorder.py).
        Process    -> faster-whisper transcribes the take (stt.py).
        Think      -> the model answers as a stream (llm.py), tokens go to the
                      window and whole sentences to the speech queue.
        Speak      -> the queue is spoken once the model is done (_ReplyEnd).
        Loop       -> back to idle; a new take interrupts the speech.
    """

    def __init__(self):
        logging.info("Starting Voice Tutor GUI Application...")

        # Core Tkinter setup. Only the root is created here; its title, size,
        # colors and widgets are the view's (see TutorView below).
        self.root = tk.Tk()

        # Thread management
        self.shutdown_event = threading.Event()
        # One exchange at a time. A take made while the previous exchange still
        # runs waits for this lock instead of being dropped - losing it was the
        # second half of the interrupt race (problem 1 in docs/refactoring.md).
        self._exchange_lock = threading.Lock()

        # True once the models are loaded: nothing may be recorded before that.
        self.app_ready = False
        # Holding a key makes Tk repeat KeyPress. This flag keeps the first one
        # and is cleared on the matching KeyRelease; it is NOT a "hold to
        # record" state.
        self._record_key_held = False

        # Text-to-speech background queue and thread
        self.tts_queue: queue.Queue = queue.Queue()
        self.tts_thread: Optional[threading.Thread] = None

        # Initialize core modular sub-managers
        self.stt_mgr = STTManager()
        self.tts_mgr = TTSManager()
        # Owns the stop event of the current reply (see speakloop/playback.py).
        self.playback = PlaybackController(self.tts_mgr)
        # Owns the capture thread. All four callbacks run on that thread, so
        # each of them marshals its window work onto the Tk thread.
        self.recorder = AudioRecorder(
            on_max_duration=self._on_record_max_duration,
            on_stream_error=self._on_record_stream_error,
            on_silence_stop=self._on_record_silence_stop,
            on_level=self._on_record_level,
        )

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
        self.root.after(0, self.view.append_system_msg,
                        "Loading the speech models...")

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
            # Warmed up here with the models, and for the same reason: the first
            # call compiles, and left to the first take the learner pays those
            # seconds between their phrase and the answer.
            warm_up_resampler()
            logging.info("Models warmed up successfully.")

            # Start TTS background thread
            self.tts_thread = threading.Thread(target=self.process_tts_queue,
                                               daemon=True)
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
        self.app_ready = True
        self.view.enter_app_ready()
        self.view.append_system_msg(
            f"Voice Tutor ready. Practice learning {config.TARGET_LANGUAGE}!")

    # ------------------------------------------------------------------
    # Press handlers (called by the view's bindings, on the Tk main thread)
    # ------------------------------------------------------------------
    def on_mic_pressed(self):
        logging.info("GUI microphone button clicked.")
        self._toggle_recording()

    def on_space_pressed(self):
        # Holding the key repeats KeyPress; the flag keeps the first one.
        if self._record_key_held:
            return
        self._record_key_held = True
        logging.info("Spacebar keyboard press event.")
        self._toggle_recording()

    def on_space_released(self):
        # Only clears the auto-repeat guard. The take keeps running until it
        # stops on silence, on the time limit, or on the next press.
        self._record_key_held = False

    def _toggle_recording(self):
        """One press starts a take, the next one ends it.

        trigger_recording_start and trigger_recording_stop keep their own
        guards, so this only routes.
        """
        if self.recorder.is_active():
            self.trigger_recording_stop()
        else:
            self.trigger_recording_start()

    def trigger_recording_start(self):
        if not self.app_ready:
            # Before the models are loaded there is nothing to answer a take
            # with, and the window still shows the loading button.
            return
        # The learner has the floor: stop the reply that is being spoken. This
        # also sets the stop event of that reply for good, so its speech cannot
        # come back after the new take.
        self.playback.stop()
        if not self.recorder.start():
            return
        self.view.enter_recording()

    def trigger_recording_stop(self):
        if not self.recorder.stop():
            return

        self.view.enter_processing()
        # The stop event of the reply to THIS take is installed here, on the Tk
        # main thread (see PlaybackController.new_event); the worker only
        # receives it.
        stop_event = self.playback.new_event()
        # The join and the exchange run off the main thread: the join can block
        # for up to RECORD_THREAD_JOIN_TIMEOUT_SEC, which would freeze the
        # window.
        threading.Thread(target=self._finalize_recording, args=(stop_event,),
                         daemon=True).start()

    # ------------------------------------------------------------------
    # Recorder callbacks (all called on the capture thread)
    # ------------------------------------------------------------------
    def _on_record_max_duration(self):
        """The take reached MAX_RECORD_SECONDS.

        Routed through the normal stop path on the main thread, so the take is
        finalized exactly like a manual stop.
        """
        self.root.after(0, self.view.append_system_msg,
                        "Reached maximum record limit.")
        self.root.after(0, self.trigger_recording_stop)

    def _on_record_silence_stop(self):
        """The take ended after a pause. This is the designed ending, so the
        chat says nothing about it; the same stop path finalizes the take."""
        self.root.after(0, self.trigger_recording_stop)

    def _on_record_level(self, level: float):
        """Live microphone level during a take, forwarded to the Tk thread."""
        self.root.after(0, self._apply_record_level, level)

    def _apply_record_level(self, level: float):
        """Repaint the level indicator, but only while the take runs. (Tk thread.)

        A level report can be queued just before the take stops; applying it
        afterwards would draw the red recording disc back on top of the
        processing glyph. The capture thread clears its recording flag before
        the stop path repaints, so this check drops such late reports.
        """
        if self.recorder.is_active():
            self.view.set_record_level(level)

    def _on_record_stream_error(self):
        """The input stream failed; the recorder has already flagged itself off."""
        self.root.after(0, self.view.append_system_msg,
                        "The microphone could not be opened. See logs/main.log.")
        self.root.after(0, self.view.recording_failed)

    # ------------------------------------------------------------------
    # One exchange: transcribe, ask the model, queue the speech
    # ------------------------------------------------------------------
    def _finalize_recording(self, stop_event: threading.Event):
        """Collect the take, then run the exchange - off the main thread."""
        if not self.recorder.join():
            # The capture thread is stuck (a device that hangs on close) and its
            # callback may still be appending chunks. Reading them now would
            # race the writer, so the take is dropped.
            self.root.after(0, self.view.append_system_msg,
                            "The audio device did not stop in time. "
                            "The take was dropped, please try again.")
            self.root.after(0, self.view.enter_idle)
            return
        # Collected BEFORE the lock below: a take started while this thread
        # waits would otherwise replace the chunk buffer it is about to read.
        audio = self.recorder.get_audio()
        with self._exchange_lock:
            self._run_exchange(audio, stop_event)

    def _run_exchange(self, audio: Optional[np.ndarray],
                      stop_event: threading.Event):
        try:
            if audio is None or len(audio) < (config.AUDIO_SAMPLE_RATE
                                              * MIN_RECORD_SECONDS):
                logging.warning("Captured audio too short or empty.")
                self.root.after(0, self.view.append_system_msg,
                                "The recording is too short. Please speak a little longer.")
                self.root.after(0, self.view.enter_idle)
                return

            audio = normalize_audio(audio)

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

            if stop_event.is_set():
                # A new take began while this one was being transcribed. The
                # phrase stays in the chat, but it gets no answer: the learner
                # is already speaking again, and the answer would arrive on top
                # of the next one.
                logging.info("The exchange was superseded by a new take; "
                             "the model is not asked.")
                return

            self._answer(user_text, stt_ms, stop_event)

        except Exception:
            logging.exception("Error in the exchange:")
            self.root.after(0, self.view.append_system_msg, "Processing Error. Please try again.")
            self.root.after(0, self.view.enter_error, "Error")

    def _answer(self, user_text: str, stt_ms: float,
                stop_event: threading.Event):
        """Ask the model and hand its sentences to the speech thread."""
        self.root.after(0, self.view.enter_thinking)

        llm_start = time.perf_counter()
        self.root.after(0, self.view.append_reply_start)

        # Streaming callback to append tokens live
        def token_cb(token):
            self.root.after(0, self.view.append_reply_token, token)

        try:
            self.llm_mgr.stream_and_queue_tts(
                user_text,
                self.tts_queue,
                stop_event,
                token_callback=token_cb
            )
        except Exception as llm_error:
            # Handled here and not by the caller, which cannot know that a reply
            # line is already open in the chat. Without this the window keeps an
            # empty partner line and a status bar that still says "Thinking",
            # and the failure is only in the log. llm.py has logged the
            # traceback already.
            self.root.after(0, self.view.append_reply_end)
            self.root.after(0, self.view.append_system_msg,
                            f"LLM error: {error_message(llm_error)}")
            # Half a reply must not be spoken. Setting the event of this reply
            # from here is safe: it belongs to this exchange, and only the
            # reference is main-thread state (see playback.py). The marker is
            # what makes the speech thread drop the sentences it has already
            # taken out of the queue.
            stop_event.set()
            self.tts_queue.put(_ReplyEnd(stop_event))
            self.root.after(0, self.view.enter_error, "LLM Error")
            return

        llm_ms = (time.perf_counter() - llm_start) * 1000
        logging.info(f"LLM complete streaming and queuing. Duration: {llm_ms:.0f}ms")

        self.root.after(0, self.view.append_reply_end)
        self.root.after(0, self.view.update_stats, stt_ms, llm_ms)

        # The model has released the GPU, so the sentences may be synthesized.
        # The marker carries this reply's stop event: an interrupt that arrived
        # meanwhile drops them instead of speaking them over the new take.
        self.tts_queue.put(_ReplyEnd(stop_event))
        if not stop_event.is_set():
            self.root.after(0, self.view.enter_speaking)

    # ------------------------------------------------------------------
    # Speech output
    # ------------------------------------------------------------------
    def process_tts_queue(self):
        """Buffer the sentences of a reply, then speak them when it is over.

        The queue holds the sentences llm.py streams into it (plain strings)
        and one _ReplyEnd marker per reply. Nothing is synthesized before that
        marker: the model server and the synthesis share one GPU.
        """
        pending_sentences: list = []

        while not self.shutdown_event.is_set():
            try:
                item = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if not isinstance(item, _ReplyEnd):
                    logging.info(f"TTS buffering sentence (waiting for the model): {item!r}")
                    pending_sentences.append(item)
                    continue
                sentences = list(pending_sentences)
                pending_sentences.clear()
                self._speak_reply(sentences, item.stop_event)
            except Exception:
                logging.exception("Error in TTS queue thread:")
            finally:
                self.tts_queue.task_done()

    def _speak_reply(self, sentences: list, stop_event: threading.Event):
        """Synthesize and play one reply, sentence by sentence."""
        logging.info(f"TTS reply ready: {len(sentences)} sentence(s).")
        for sentence in sentences:
            if stop_event.is_set() or self.shutdown_event.is_set():
                logging.info("Speech stopped; the rest of the reply is dropped.")
                break
            try:
                waveform = self.tts_mgr.synthesize(sentence)
                if stop_event.is_set() or self.shutdown_event.is_set():
                    break
                logging.info(f"TTS playing synthesized block: {sentence!r}")
                self.tts_mgr.play_array(waveform, self.tts_mgr.sample_rate,
                                        stop_event, self.shutdown_event)
            except Exception:
                # Skip the failed sentence and continue with the rest
                logging.exception(f"TTS playback error for {sentence!r}:")

        if self.shutdown_event.is_set():
            return
        # Back to waiting for the learner - but only for the reply that is still
        # the current one. A reply that was interrupted must not overwrite the
        # window state of the take that interrupted it.
        if not stop_event.is_set() and self.playback.is_current(stop_event):
            self.root.after(0, self.view.enter_idle)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    def quit_app(self):
        logging.info("Shutting down VoiceTutor App...")
        self.shutdown_event.set()
        self.playback.stop()

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
