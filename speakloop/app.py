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

The lesson is written to disk while it runs: speakloop/transcript.py owns the
files, and every event of the lesson passes one record to it (_record, and
_system for the service lines). A record is added where the same text goes to
the window, so the transcript cannot fall behind what the learner sees.
"""

import logging
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from typing import NamedTuple, Optional

import numpy as np

# config first: it sets HF_HOME, the Supertonic cache and the offline switch,
# which huggingface_hub reads when stt and tts import their engines below.
from speakloop import config
from speakloop import (bootstrap, detect_hardware, lifecycle, prompt,
                       transcript)
from speakloop.contract import Reply, split_sentences, strip_markdown
from speakloop.conversation import Lesson
from speakloop.llm import LLMManager, error_message
from speakloop.llm_server_ctl import LLMServerController
from speakloop.playback import PlaybackController
from speakloop.recorder import AudioRecorder, normalize_audio, warm_up_resampler
from speakloop.stt import STTManager
from speakloop.tts import TTSManager
from speakloop.ui import TutorView, ViewCallbacks, ViewSettings


class _ReplyEnd(NamedTuple):
    """Queue marker: the last sentence of a reply is in the queue.

    The speech thread buffers the sentences of a reply and synthesizes nothing
    until this marker arrives. The sentences are queued only after the model
    has finished, so the model server has released the GPU before the
    synthesis starts (both share one card).

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

    The lesson opens by itself once everything is loaded: the model asks the
    first question (_open_lesson).

    Flow per exchange (state machine):
        Record     -> one press opens the microphone; the take ends by itself
                      after a pause, or on the next press (recorder.py).
        Process    -> faster-whisper transcribes the take (stt.py).
        Think      -> the model writes its whole reply (conversation.py), which
                      is split into NOTE / SAY / SUMMARY (contract.py).
        Show       -> NOTE, SAY and SUMMARY go to the window.
        Speak      -> the sentences of SAY go to the speech queue (_ReplyEnd).
        Loop       -> back to idle; a new take interrupts the speech.

    A phrase written in the window enters this flow at Think - typed and sent
    with Enter (on_text_submitted), or chosen with a command button
    (on_command_pressed). There is nothing to record and nothing to recognize,
    and everything from the model request on is the same code.
    """

    def __init__(self):
        logging.info("Starting Voice Tutor GUI Application...")

        # Core Tkinter setup. Only the root is created here; its title, size,
        # colors and widgets are the view's (see TutorView below).
        self.root = tk.Tk()

        # Thread management
        self.shutdown_event = threading.Event()
        # One exchange at a time. A take made while the previous exchange still
        # runs waits for this lock instead of being dropped.
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
        # One client for both backends: they speak the same OpenAI API, and
        # config gives the address and the key of the selected one.
        self.llm_mgr = LLMManager()
        self.llm_mgr.init_client(config.LLM_URL, config.LLM_API_KEY)
        # The lesson over llm_mgr. Built by load_components, from the prompt
        # file; no exchange can start before that (app_ready).
        self.lesson: Optional[Lesson] = None
        # Owns the llama-server subprocess. Built for every backend and left
        # untouched by "lm-studio": all of its methods are no-ops until start().
        self._llm_server = LLMServerController()

        # The transcript of this lesson (speakloop/transcript.py), with the
        # settings of the run. Its files are created with the first record.
        self._lesson_start = datetime.now()
        # The phrase of the learner the records belong to. The opening question
        # of the model answers no phrase of the learner and keeps 0.
        self._turn = 0
        self.transcript = transcript.TranscriptWriter(
            config.TRANSCRIPT_DIR, self._lesson_start, transcript.meta_record(
                started_at=self._lesson_start,
                target_language=config.TARGET_LANGUAGE,
                explanation_language=config.EXPLANATION_LANGUAGE,
                first_topic=config.FIRST_TOPIC,
                llm_model=self._chat_model_name(),
                stt_model=config.WHISPER_MODEL,
                tts_voice=config.TTS_VOICE))

        # The window. Built last of the members, because the loader thread
        # started below drives it at once.
        self.view = TutorView(self.root, ViewCallbacks(
            on_mic_pressed=self.on_mic_pressed,
            on_space_pressed=self.on_space_pressed,
            on_space_released=self.on_space_released,
            on_text_submitted=self.on_text_submitted,
            on_command_pressed=self.on_command_pressed,
            on_notes_toggled=self.on_notes_toggled,
            on_quit=self.quit_app,
        ), ViewSettings(
            lesson_language=config.TARGET_LANGUAGE,
            show_notes=config.SHOW_NOTES,
            # The buttons send the words of the prompt, so the model reads a
            # pressed button like the spoken command.
            commands=prompt.LESSON_COMMANDS,
        ))

        # Start loading models in a background thread to prevent UI freezing
        threading.Thread(target=self.load_components, daemon=True).start()

    # ------------------------------------------------------------------
    # The transcript and the service lines
    # ------------------------------------------------------------------
    def _chat_model_name(self) -> str:
        """The chat model, as the transcript names it.

        The GGUF file for the own server; the name of the backend for LM
        Studio, where the model is chosen in that application and this process
        cannot read which one it is.
        """
        if self.llm_backend == "llama-server":
            return os.path.basename(config.EXTERNAL_MODEL_PATH)
        return self.llm_backend

    def _record(self, record_type: str, text: str, **extra) -> None:
        """Add one event of the lesson to the transcript. (Any thread.)

        The writer has a lock of its own, so the exchange threads and the Tk
        thread use this the same way.
        """
        self.transcript.add(transcript.event_record(
            datetime.now(), self._turn, record_type, text, **extra))

    def _system(self, text: str) -> None:
        """Show a [System] line and keep it in the transcript. (Any thread.)

        Every service message of the application goes through here: the window
        is reached on the Tk thread as always, and the same sentence becomes a
        "system" record, which the reader of the file can skip or read.
        """
        self.root.after(0, self.view.append_system_msg, text)
        self._record(transcript.TYPE_SYSTEM, text)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    def load_components(self):
        logging.info("Starting model loading thread...")
        self.root.after(0, self.view.enter_loading)
        self._system("Loading the speech models...")

        try:
            # First of all: a missing or edited prompt file stops the start at
            # once, and not after a minute of model loading.
            system_prompt = prompt.build_system_prompt(
                config.PROMPT_FILE, config.TARGET_LANGUAGE,
                config.EXPLANATION_LANGUAGE, config.FIRST_TOPIC)
            self.lesson = Lesson(self.llm_mgr, system_prompt)
            logging.info(f"Lesson prompt loaded from {config.PROMPT_FILE} "
                         f"({len(system_prompt)} characters).")

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
                self._system("Connecting to the LLM server...")
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
                    self._system(f"Error: {reason}")
                    self._system("See logs/main.log and logs/llm_server.log.")
                    self.root.after(0, self.view.server_failed)
                    # Do not make the window ready - there is nothing to answer
                    # a recording with.
                    return
                if self._llm_server.adopted:
                    # Said in the window and not only in the log: the answers
                    # now come from a server this run did not configure, which
                    # explains a model or a speed the settings do not.
                    self._system(f"Using the llama-server already running "
                                 f"on {config.LLM_SERVER_HOST}:"
                                 f"{config.LLM_SERVER_PORT}. It keeps the "
                                 f"model it was started with.")
                else:
                    model_name = os.path.basename(config.EXTERNAL_MODEL_PATH)
                    self._system(f"llama-server is ready with {model_name}.")
                served_n_ctx = self._llm_server.served_n_ctx
                if (served_n_ctx is not None
                        and served_n_ctx < config.EXTERNAL_N_CTX):
                    # A warning and not a refusal: a short lesson still
                    # works. In the chat and not in the status bar, which the
                    # next state change overwrites at once.
                    self._system(f"Warning: the model has a context of "
                                 f"{served_n_ctx} tokens instead of "
                                 f"{config.EXTERNAL_N_CTX}. A long lesson "
                                 f"may not fit. See logs/main.log.")
            else:
                if not self.llm_mgr.check_connection():
                    self._system("Warning: LM Studio is offline. Start "
                                 "it to use voice tutor!")
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
            self._system(f"Initialization Error: {e}")
            self.root.after(0, self.view.init_failed)

    def make_app_ready(self):
        self.app_ready = True
        self.view.enter_app_ready()
        self._system(f"Ready. The lesson is in {config.TARGET_LANGUAGE}. "
                     f"Use the command buttons above, or say the same words: "
                     f"{', '.join(prompt.LESSON_COMMANDS)}.")
        self._open_lesson()

    def _open_lesson(self):
        """Ask the model for the first question of the lesson. (Tk thread.)

        Runs like an exchange: with a stop event of its own and under the
        exchange lock, so a take started meanwhile interrupts it the same way.
        """
        stop_event = self.playback.new_event()
        threading.Thread(target=self._run_opening, args=(stop_event,),
                         daemon=True).start()

    def _run_opening(self, stop_event: threading.Event):
        with self._exchange_lock:
            if stop_event.is_set():
                return
            try:
                self._ask_model(None, stop_event)
            except Exception:
                logging.exception("Error while opening the lesson:")
                self._system("The lesson could not be opened. "
                             "Speak to start it.")
                self.root.after(0, self.view.enter_error, "Error")

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

    def on_text_submitted(self, learner_text: str):
        """The learner sent a typed phrase with Enter. (Tk thread.)"""
        logging.info(f"The learner typed: {learner_text!r}")
        self._submit_phrase(learner_text, transcript.SOURCE_TEXT)

    def on_command_pressed(self, command: str):
        """A command button was pressed. (Tk thread.)

        A command is an ordinary phrase of the learner: the button sends the
        word of the prompt, and the model reads it exactly as it reads the
        spoken command. It goes into the chat and into the history like any
        other phrase, so the transcript says why the lesson changed course.
        """
        logging.info(f"The learner pressed the command button {command!r}.")
        self._submit_phrase(command, transcript.SOURCE_BUTTON)

    def on_notes_toggled(self, show_notes: bool):
        """The Notes switch was pressed. (Tk thread.)

        The window has already hidden or shown the corrections; the controller
        only remembers the choice for the next lesson. A file that cannot be
        written is reported by the loader on stderr, and the lesson goes on.
        """
        logging.info(f"Notes are now {'shown' if show_notes else 'hidden'}.")
        config.save_user_setting("show_notes", show_notes)

    def _submit_phrase(self, learner_text: str, source: str):
        """Send a phrase the learner wrote or chose. (Tk thread.)

        *source* says how the phrase was given (see speakloop/transcript.py):
        the reader of the transcript needs it, because only a spoken phrase can
        carry a recognition error.

        A written phrase is the take of this exchange: it stops the speech of
        the previous reply exactly as the start of a recording does, so the
        learner has the floor from the moment they press Enter or a button. The
        view keeps the entry and the buttons closed while the microphone is
        open and while the model answers, so there is no take of the other kind
        to cancel here.
        """
        if not self.app_ready:
            return
        self.playback.stop()
        self.view.append_user_msg(learner_text)
        self._turn += 1
        self._record(transcript.TYPE_LEARNER, learner_text, source=source)
        # Installed on the Tk thread, like the stop event of a recorded take
        # (see PlaybackController.new_event).
        stop_event = self.playback.new_event()
        threading.Thread(target=self._run_typed_exchange,
                         args=(learner_text, stop_event), daemon=True).start()

    def _run_typed_exchange(self, learner_text: str,
                            stop_event: threading.Event):
        """Ask the model about a typed phrase - off the main thread.

        Under the same lock as a recorded exchange, so a take made while the
        previous reply still runs waits instead of being answered in parallel.
        """
        with self._exchange_lock:
            if stop_event.is_set():
                # A recording was started while this phrase waited for the
                # lock; that take owns the window now.
                logging.info("The typed phrase was superseded by a new take; "
                             "the model is not asked.")
                return
            try:
                self._ask_model(learner_text, stop_event)
            except Exception:
                logging.exception("Error in the typed exchange:")
                self._system("Processing Error. Please try again.")
                self.root.after(0, self.view.enter_error, "Error")

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
        self._system("Reached maximum record limit.")
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
        self._system("The microphone could not be opened. See logs/main.log.")
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
            self._system("The audio device did not stop in time. "
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
                self._system("The recording is too short. "
                             "Please speak a little longer.")
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
                self._system("Could not hear you clearly. Please try again.")
                self.root.after(0, self.view.enter_idle)
                return

            # Update User Speech to GUI
            self.root.after(0, self.view.append_user_msg, user_text)
            self._turn += 1
            self._record(transcript.TYPE_LEARNER, user_text,
                         source=transcript.SOURCE_VOICE,
                         stt_ms=round(stt_ms))

            if stop_event.is_set():
                # A new take began while this one was being transcribed. The
                # phrase stays in the chat, but it gets no answer: the learner
                # is already speaking again, and the answer would arrive on top
                # of the next one.
                logging.info("The exchange was superseded by a new take; "
                             "the model is not asked.")
                return

            self._ask_model(user_text, stop_event)

        except Exception:
            logging.exception("Error in the exchange:")
            self._system("Processing Error. Please try again.")
            self.root.after(0, self.view.enter_error, "Error")

    def _ask_model(self, learner_text: Optional[str],
                   stop_event: threading.Event):
        """Get one reply of the model, show it and queue its speech.

        *learner_text* None opens the lesson. The phrase itself is already in
        the chat: a recorded one is written by _run_exchange after recognition,
        a typed one by on_text_submitted.
        """
        self.root.after(0, self.view.enter_thinking)
        llm_start = time.perf_counter()

        try:
            if learner_text is None:
                reply = self.lesson.open(stop_event)
            else:
                reply = self.lesson.answer(learner_text, stop_event)
        except Exception as llm_error:
            # Handled here and not by the caller: without this the status bar
            # keeps saying "Thinking" and the failure is only in the log.
            # llm.py has logged the traceback already. Nothing of the reply
            # was queued for speech, so there is nothing to drop.
            self._system(f"LLM error: {error_message(llm_error)}")
            self.root.after(0, self.view.enter_error, "LLM Error")
            return

        if reply is None:
            # Interrupted by a new take, which owns the window from now on.
            logging.info("The reply was interrupted and is not shown.")
            return

        llm_ms = (time.perf_counter() - llm_start) * 1000
        logging.info(f"LLM reply received. Duration: {llm_ms:.0f}ms")

        self.root.after(0, self._show_reply, reply)
        self._record_reply(reply, llm_ms)
        if not reply.follows_contract:
            # After the reply itself: the learner reads what the model wrote
            # and then why it stays silent. _show_reply is already queued on
            # the Tk thread, so this line lands under it. No second request:
            # a retry costs another 15-50 s and changes the history.
            self._system("The model did not answer in the lesson format. "
                         "The reply is shown in full and is not spoken.")

        # The markers go before the split, and for speech only: the synthesis
        # reads them aloud, while the chat and the transcript keep the line as
        # the model wrote it.
        spoken = strip_markdown(reply.say) if reply.say else ""
        sentences = split_sentences(spoken)
        if not sentences:
            # A SUMMARY, a reply outside the contract, or a SAY of markers
            # alone: shown, not spoken.
            self.root.after(0, self._enter_if_current, self.view.enter_idle,
                            stop_event)
            return
        for sentence in sentences:
            logging.info(f"Queued sentence to TTS: {sentence!r}")
            self.tts_queue.put(sentence)
        # The marker carries this reply's stop event: an interrupt that
        # arrives before the speech drops the sentences instead of speaking
        # them over the new take.
        self.tts_queue.put(_ReplyEnd(stop_event))
        self.root.after(0, self._enter_if_current, self.view.enter_speaking,
                        stop_event)

    def _show_reply(self, reply: Reply):
        """Write one reply into the chat. (Tk thread.)

        NOTE first, as the prompt orders the lines: the learner reads the
        correction before the question. A reply outside the contract is shown
        whole, so the learner still sees what the model wrote.
        """
        if not reply.follows_contract:
            self.view.append_partner_msg(reply.raw)
            return
        if reply.note:
            self.view.append_note(reply.note)
        if reply.say:
            self.view.append_partner_msg(reply.say)
        if reply.summary:
            self.view.append_summary(reply.summary)

    def _record_reply(self, reply: Reply, llm_ms: float) -> None:
        """Write one reply of the model into the transcript. (Exchange thread.)

        The duration and the size of the context go on the FIRST record of the
        reply: one reply can hold a NOTE and a SAY, and the same two numbers on
        both records would read as two answers of the model.

        A summary ends the lesson, so the readable view is written here. The
        window still works afterwards, and quit_app writes the view again.
        """
        stats = {"llm_ms": round(llm_ms),
                 "tokens": self.llm_mgr.last_total_tokens}
        if not reply.follows_contract:
            self._record(transcript.TYPE_BROKEN, reply.raw, **stats)
            return
        for record_type, text in ((transcript.TYPE_NOTE, reply.note),
                                  (transcript.TYPE_SAY, reply.say),
                                  (transcript.TYPE_SUMMARY, reply.summary)):
            if text:
                self._record(record_type, text, **stats)
                stats = {}
        if reply.summary:
            self.transcript.save_markdown()

    def _enter_if_current(self, intent, stop_event: threading.Event):
        """Run a view intent only for the reply that is still current. (Tk thread.)

        Checked on the Tk thread, where a new take changes the window: a check
        in the worker could pass just before a take starts, and the intent
        would then draw over the recording state.
        """
        if not stop_event.is_set() and self.playback.is_current(stop_event):
            intent()

    # ------------------------------------------------------------------
    # Speech output
    # ------------------------------------------------------------------
    def process_tts_queue(self):
        """Buffer the sentences of a reply, then speak them when it is over.

        The queue holds the sentences of each SAY line (plain strings) and one
        _ReplyEnd marker per reply. Nothing is synthesized before that marker.
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

        # The readable view of a lesson that ended without a summary. The jsonl
        # file needs nothing here: every record went to disk when it happened.
        self.transcript.save_markdown()

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
