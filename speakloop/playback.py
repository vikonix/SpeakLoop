# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Lifecycle of the stop event that governs one reply's speech.

The whole "which reply may still be spoken" state lives here: installing a
fresh stop event for a new reply, and stopping the current one. The controller
composes a PlaybackController with the TTSManager; worker threads receive a
stop event as a plain argument and never touch the shared reference.

Why an event per reply and not one event for the application: with a single
event the stop had to be cleared again before the next reply, and the clear
happened in the worker of the OLD exchange, after a new recording had already
set it. The interrupt was then lost and the abandoned reply was spoken over the
new take. An event that belongs to one reply is only ever set, never cleared,
so nothing can revoke an interrupt.

Threading contract:
  * ``new_event()`` and ``stop()`` run on the Tk main thread only: they replace
    and read the shared "current reply" reference.
  * ``is_current()`` is safe from any thread (one attribute read).
  * Setting an event that a worker already holds is safe from that worker: an
    Event is thread-safe, and only the reference is main-thread state.
"""

import threading


class PlaybackController:
    """The stop event of the current reply, and the way to stop that reply."""

    def __init__(self, tts_mgr):
        self.tts_mgr = tts_mgr
        # Pre-installed event, so stop() is safe before the first reply.
        self._current_event = threading.Event()

    def new_event(self) -> threading.Event:
        """Install a fresh stop event for a new reply and return it.

        Must be called on the Tk main thread: it replaces the reference stop()
        reads, so installing it from a worker would race the stop. A stop
        issued between the install and the start of the speech is not lost -
        the playback path checks the event before and during playback.
        """
        event = threading.Event()
        self._current_event = event
        return event

    def stop(self):
        """Stop the speech of the current reply. (Tk main thread.)

        Sets the current reply's event and interrupts the audio that is playing
        right now. The event stays set: it belongs to a reply that is over.
        """
        self._current_event.set()
        self.tts_mgr.stop_playback()

    def is_current(self, stop_event: threading.Event) -> bool:
        """True when *stop_event* still governs the newest reply.

        A worker uses it to decide whether its own finish may still change the
        window: the worker of a superseded reply must not overwrite a status
        set by the reply that replaced it.
        """
        return stop_event is self._current_event
