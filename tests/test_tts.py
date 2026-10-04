# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/tts.py.

Synthesis itself needs a model and a sound card, so what is tested here is the
part around it: the backend registry the language profiles select from, the
text contract every backend relies on, and the playback guards that must hold
before any device is touched. The backend is replaced by a stand-in, so nothing
here loads a model or opens a stream.

Run from the project root with:

    python -m unittest tests.test_tts
"""

import threading
import unittest
from unittest import mock

import numpy as np

from speakloop import config, models_info, tts


class FakeBackend:
    """Stand-in for a synthesis backend: records the calls, returns a tone."""

    sample_rate = 12_345

    def __init__(self):
        self.calls = []
        self.loaded = False
        self.warmed = False

    def load_model(self):
        self.loaded = True

    def warm_up(self):
        self.warmed = True

    def synthesize(self, text, voice):
        self.calls.append((text, voice))
        return np.zeros(8, dtype=np.float32)


def manager_with_fake_backend():
    """A TTSManager whose backend is a FakeBackend (no model is loaded)."""
    manager = tts.TTSManager.__new__(tts.TTSManager)
    manager._backend = FakeBackend()
    return manager


class BackendRegistryTests(unittest.TestCase):
    """What the language profiles are allowed to name."""

    def test_the_registry_matches_the_validated_choices(self):
        # config.py validates a profile's tts_backend against
        # TTS_BACKEND_CHOICES; a name accepted there but missing here is a
        # KeyError when the manager is built.
        self.assertEqual(set(tts.TTS_BACKENDS), set(config.TTS_BACKEND_CHOICES))

    def test_the_active_backend_can_be_built(self):
        self.assertIn(config.TTS_BACKEND, tts.TTS_BACKENDS)

    def test_every_backend_has_the_same_surface(self):
        for name, backend_class in tts.TTS_BACKENDS.items():
            with self.subTest(backend=name):
                for attribute in ("sample_rate", "load_model", "warm_up",
                                  "synthesize"):
                    self.assertTrue(hasattr(backend_class, attribute))

    def test_every_backend_reports_its_own_rate(self):
        # The manager exposes the rate of the active backend and the controller
        # passes it to play_array. Two backends that reported the same rate
        # would play one of them at the wrong speed.
        self.assertEqual(tts.KokoroBackend.sample_rate, tts.KOKORO_SAMPLE_RATE)
        self.assertEqual(tts.SupertonicBackend.sample_rate,
                         tts.SUPERTONIC_SAMPLE_RATE)
        self.assertNotEqual(tts.KOKORO_SAMPLE_RATE, tts.SUPERTONIC_SAMPLE_RATE)

    def test_the_supertonic_model_name_comes_from_the_catalogue(self):
        # models_info is the single place a model name is written down, so the
        # download and the load cannot go to different models.
        self.assertTrue(models_info.SUPERTONIC.name.strip())


class SynthesizeContractTests(unittest.TestCase):
    """The text and voice contract every backend can rely on."""

    def setUp(self):
        self.manager = manager_with_fake_backend()

    def test_the_backend_rate_is_reported(self):
        self.assertEqual(self.manager.sample_rate, FakeBackend.sample_rate)

    def test_empty_text_never_reaches_the_backend(self):
        result = self.manager.synthesize("   \n  ")
        self.assertEqual(result.size, 0)
        self.assertEqual(self.manager._backend.calls, [])

    def test_whitespace_is_collapsed_once_for_every_backend(self):
        self.manager.synthesize("  Hello   there\n friend ", voice="v1")
        self.assertEqual(self.manager._backend.calls,
                         [("Hello there friend", "v1")])

    def test_the_configured_voice_is_used_by_default(self):
        self.manager.synthesize("Hello.")
        self.assertEqual(self.manager._backend.calls[0][1], config.TTS_VOICE)


class PlaybackGuardTests(unittest.TestCase):
    """Nothing is played, and no device is touched, when there is a reason not to."""

    def setUp(self):
        self.manager = manager_with_fake_backend()

    def test_an_empty_waveform_opens_no_stream(self):
        with mock.patch.object(tts, "uses_winsound", return_value=False), \
                mock.patch.object(tts.sd, "OutputStream") as stream:
            self.manager.play_array(np.zeros(0, dtype=np.float32), 16_000)
        stream.assert_not_called()

    def test_a_stop_that_arrived_first_opens_no_stream(self):
        # The interrupt can land while the reply is still being synthesized.
        # Opening the output device then would play the first blocks of a reply
        # nobody is waiting for, into an already open microphone.
        stop_event = threading.Event()
        stop_event.set()
        with mock.patch.object(tts, "uses_winsound", return_value=False), \
                mock.patch.object(tts.sd, "OutputStream") as stream:
            self.manager.play_array(np.ones(64, dtype=np.float32), 16_000,
                                    stop_event)
        stream.assert_not_called()

    def test_a_shutdown_opens_no_stream(self):
        shutdown_event = threading.Event()
        shutdown_event.set()
        with mock.patch.object(tts, "uses_winsound", return_value=False), \
                mock.patch.object(tts.sd, "OutputStream") as stream:
            self.manager.play_array(np.ones(64, dtype=np.float32), 16_000,
                                    shutdown_event=shutdown_event)
        stream.assert_not_called()


class LeadInTests(unittest.TestCase):
    """The lead-in is reported only for the path that actually prepends it."""

    def setUp(self):
        self.manager = manager_with_fake_backend()

    def test_no_lead_in_without_winsound(self):
        with mock.patch.object(tts, "uses_winsound", return_value=False):
            self.assertEqual(self.manager.playback_lead_in_seconds(), 0.0)

    def test_the_winsound_lead_in_is_the_shared_constant(self):
        with mock.patch.object(tts, "uses_winsound", return_value=True):
            self.assertEqual(self.manager.playback_lead_in_seconds(),
                             tts.WINSOUND_LEAD_IN_SECONDS)


if __name__ == "__main__":
    unittest.main()
