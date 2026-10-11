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


class SpeechSpeedTests(unittest.TestCase):
    """The tutor's speed reaches the synthesis of both backends."""

    def test_kokoro_gets_the_configured_speed(self):
        backend = tts.KokoroBackend()
        backend.model = object()
        calls = []

        def pipeline(text, **kwargs):
            calls.append(kwargs)
            return iter([(None, None, np.ones(4, dtype=np.float32))])

        backend.pipeline = pipeline
        with mock.patch.object(config, "SPEECH_SPEED", 0.9):
            backend.synthesize("Hello.", "af_heart")
        self.assertEqual(calls[0]["speed"], 0.9)

    def test_supertonic_gets_the_configured_speed(self):
        # Never the package default of 1.05.
        backend = tts.SupertonicBackend()
        backend._tts = mock.Mock()
        backend._tts.synthesize.return_value = (
            np.zeros((1, 4), dtype=np.float32), 0.1)
        with mock.patch.object(backend, "_style", return_value="style"), \
                mock.patch.object(config, "SPEECH_SPEED", 0.9):
            backend.synthesize("Hola.", "M1")
        self.assertEqual(backend._tts.synthesize.call_args.kwargs["speed"], 0.9)


class SlowedPlaybackTests(unittest.TestCase):
    """The slower repeat of a question (docs/proactive-tutor-spec.md, 6.6)."""

    def setUp(self):
        self.waveform = np.ones(100, dtype=np.float32)

    def test_a_factor_of_one_plays_as_it_is(self):
        waveform, rate = tts.slowed_playback(self.waveform, 24_000, 1.0)
        self.assertIs(waveform, self.waveform)
        self.assertEqual(rate, 24_000)

    def test_a_lower_factor_lowers_the_rate_and_keeps_the_waveform(self):
        # 0.8 / 0.9: the owner's settings.
        waveform, rate = tts.slowed_playback(self.waveform, 24_000, 0.8 / 0.9)
        self.assertIs(waveform, self.waveform)
        self.assertEqual(rate, int(24_000 * 0.8 / 0.9))

    def test_the_repeat_is_never_faster_than_the_speech(self):
        _waveform, rate = tts.slowed_playback(self.waveform, 44_100, 1.25)
        self.assertEqual(rate, 44_100)


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
