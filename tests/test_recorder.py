# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/recorder.py.

No microphone is opened: sounddevice's InputStream is replaced by a stand-in
and the PortAudio reset is patched out, so what runs here is the capture
thread's own logic - the state machine, the automatic stop and the take that
comes out at the end.

Run from the project root with:

    python -m unittest tests.test_recorder
"""

import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

from speakloop import config, recorder


class _StubStream:
    """Stand-in for sd.InputStream: records what the loop did to it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def silent_recorder(**callbacks):
    """An AudioRecorder whose unspecified callbacks do nothing."""
    handlers = {
        "on_max_duration": lambda: None,
        "on_stream_error": lambda: None,
        "on_silence_stop": lambda: None,
        "on_level": lambda level: None,
    }
    handlers.update(callbacks)
    return recorder.AudioRecorder(**handlers)


def finish(audio_recorder):
    """Stop the take that runs, if any, and wait for its thread (cleanup)."""
    take = audio_recorder.stop() or audio_recorder._take
    if take is not None:
        audio_recorder.join(take)


class NormalizeAudioTests(unittest.TestCase):
    """The one pure signal helper of the capture path."""

    def test_a_quiet_take_is_lifted_to_the_ceiling(self):
        audio = np.array([0.0, 0.2, -0.1], dtype=np.float32)
        result = recorder.normalize_audio(audio)
        self.assertAlmostEqual(float(np.max(np.abs(result))),
                               recorder.AUDIO_NORMALIZATION_CEILING, places=5)

    def test_silence_is_left_alone(self):
        # Boosting the noise floor of a silent take would feed the recognizer
        # amplified room noise instead of speech.
        audio = np.full(16, 0.001, dtype=np.float32)
        result = recorder.normalize_audio(audio)
        np.testing.assert_allclose(result, audio)

    def test_the_result_is_float32(self):
        audio = np.array([0.0, 0.5], dtype=np.float64)
        self.assertEqual(recorder.normalize_audio(audio).dtype, np.float32)


class ResamplerWarmUpTests(unittest.TestCase):
    """The start-up warm-up must prepare the conversion a take really needs."""

    def test_the_warm_up_prepares_the_pipeline_rate(self):
        # librosa is replaced rather than imported: the real import is the very
        # cost this warm-up exists to move out of the first phrase, and the test
        # suite must not pay it either.
        fake_librosa = mock.Mock()
        with mock.patch.dict(sys.modules, {"librosa": fake_librosa}):
            recorder.warm_up_resampler(48_000)
        fake_librosa.resample.assert_called_once()
        arguments = fake_librosa.resample.call_args.kwargs
        self.assertEqual(arguments["orig_sr"], 48_000)
        self.assertEqual(arguments["target_sr"], config.AUDIO_SAMPLE_RATE)


class CaptureDeviceTests(unittest.TestCase):
    """Which device and which rate the take is captured with."""

    def test_a_detected_device_wins_over_the_host_api_search(self):
        # hardware_config.json named a device on purpose; the WASAPI search
        # would silently record from another one.
        with mock.patch.object(config, "AUDIO_INPUT_DEVICE", 7):
            device, rate = silent_recorder()._select_capture_device()
        self.assertEqual(device, 7)
        self.assertEqual(rate, config.AUDIO_SAMPLE_RATE)

    def test_a_failed_host_api_search_falls_back_to_the_defaults(self):
        # assertLogs does two things here: it pins that the failure is written
        # to the log (silent guesswork about the input device is how a take ends
        # up recorded from the wrong microphone), and it keeps the traceback out
        # of the suite's output, where it reads like a broken test.
        with mock.patch.object(config, "AUDIO_INPUT_DEVICE", None), \
                mock.patch.object(recorder.sd, "query_hostapis",
                                  side_effect=OSError("no host api")), \
                self.assertLogs(level="ERROR"):
            device, rate = silent_recorder()._select_capture_device()
        self.assertIsNone(device)
        self.assertEqual(rate, config.AUDIO_SAMPLE_RATE)


class TakeLifecycleTests(unittest.TestCase):
    """start, stop, join and the take that comes out."""

    def setUp(self):
        self.streams = []

        def make_stream(**kwargs):
            stream = _StubStream(**kwargs)
            self.streams.append(stream)
            return stream

        patches = [
            mock.patch.object(recorder, "reset_portaudio", lambda: None),
            mock.patch.object(recorder.sd, "InputStream", make_stream),
            mock.patch.object(config, "AUDIO_INPUT_DEVICE", 0),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_a_take_starts_stops_and_closes_its_stream(self):
        audio_recorder = silent_recorder()
        self.assertTrue(audio_recorder.start())
        self.assertTrue(audio_recorder.is_active())
        take = audio_recorder.stop()
        self.assertIsNotNone(take)
        self.assertTrue(audio_recorder.join(take))
        self.assertFalse(audio_recorder.is_active())
        self.assertEqual(len(self.streams), 1)
        self.assertTrue(self.streams[0].started)
        self.assertTrue(self.streams[0].closed)

    def test_a_second_start_while_recording_is_refused(self):
        audio_recorder = silent_recorder()
        self.addCleanup(finish, audio_recorder)
        self.assertTrue(audio_recorder.start())
        self.assertFalse(audio_recorder.start())

    def test_a_stop_without_a_take_is_refused(self):
        # The controller routes every ending through stop(); one that reports
        # no take is what keeps a second ending from finalizing it twice.
        self.assertIsNone(silent_recorder().stop())

    def test_the_take_is_returned_once_and_then_cleared(self):
        take = recorder.Take()
        take.chunks = [np.ones((4, 1), dtype=np.float32),
                       np.zeros((2, 1), dtype=np.float32)]
        audio = recorder.AudioRecorder.get_audio(take)
        self.assertEqual(list(audio), [1, 1, 1, 1, 0, 0])
        self.assertIsNone(recorder.AudioRecorder.get_audio(take))

    def test_an_empty_take_is_reported_as_nothing(self):
        self.assertIsNone(recorder.AudioRecorder.get_audio(recorder.Take()))

    def test_a_new_take_leaves_the_stopped_take_alone(self):
        # The controller reads a stopped take on another thread; a take that
        # starts in between must not empty or replace it.
        audio_recorder = silent_recorder()
        self.addCleanup(finish, audio_recorder)
        audio_recorder.start()
        first = audio_recorder.stop()
        self.assertTrue(audio_recorder.join(first))
        first.chunks.append(np.ones((4, 1), dtype=np.float32))
        self.assertTrue(audio_recorder.start())
        self.assertIsNot(audio_recorder._take, first)
        self.assertEqual(list(recorder.AudioRecorder.get_audio(first)),
                         [1, 1, 1, 1])

    def test_a_failing_stream_reports_the_error_and_stops_recording(self):
        reported = threading.Event()
        # assertLogs captures the traceback the capture thread writes, so it
        # stays out of the suite's output while still being required: an input
        # stream that cannot be opened must leave a reason in logs/main.log.
        with mock.patch.object(recorder.sd, "InputStream",
                               side_effect=OSError("device is busy")), \
                self.assertLogs(level="ERROR"):
            audio_recorder = silent_recorder(
                on_stream_error=reported.set)
            audio_recorder.start()
            audio_recorder.join(audio_recorder._take)
        self.assertTrue(reported.wait(timeout=2))
        self.assertFalse(audio_recorder.is_active())


class AutomaticStopTests(unittest.TestCase):
    """The take ends by itself once the speaker falls silent."""

    def setUp(self):
        patches = [
            mock.patch.object(recorder, "reset_portaudio", lambda: None),
            mock.patch.object(recorder.sd, "InputStream",
                              lambda **kwargs: _StubStream(**kwargs)),
            mock.patch.object(config, "AUDIO_INPUT_DEVICE", 0),
            # Short enough to keep the test quick, long enough to survive a
            # busy machine (the capture loop polls every 10 ms).
            mock.patch.object(config, "SILENCE_TIMEOUT", 0.1),
            mock.patch.object(config, "SILENCE_THRESHOLD", 0.01),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_silence_before_the_first_word_never_stops_the_take(self):
        # The pause while the learner thinks must not cut the take: the timer
        # is armed only once speech has been heard.
        stopped = threading.Event()
        audio_recorder = silent_recorder(on_silence_stop=stopped.set)
        self.addCleanup(finish, audio_recorder)
        audio_recorder.start()
        # Only silence arrives, for several times the silence timeout.
        for _ in range(10):
            audio_recorder._take.chunks.append(
                np.zeros((256, 1), dtype=np.float32))
            time.sleep(0.05)
        self.assertFalse(stopped.is_set())

    def test_silence_after_speech_ends_the_take(self):
        stopped = threading.Event()
        audio_recorder = silent_recorder(on_silence_stop=stopped.set)
        self.addCleanup(finish, audio_recorder)
        audio_recorder.start()
        # One loud block arms the timer. The silent blocks after it are what a
        # microphone really delivers once the speaker stops talking: the stream
        # keeps running, so the capture loop keeps measuring (see the comment on
        # the silence check in _record_loop).
        audio_recorder._take.chunks.append(
            np.full((256, 1), 0.5, dtype=np.float32))
        deadline = time.monotonic() + 3
        while not stopped.is_set() and time.monotonic() < deadline:
            audio_recorder._take.chunks.append(
                np.zeros((256, 1), dtype=np.float32))
            time.sleep(0.02)
        self.assertTrue(stopped.is_set())

    def test_the_live_level_is_reported_while_the_take_runs(self):
        levels = []
        audio_recorder = silent_recorder(on_level=levels.append)
        self.addCleanup(finish, audio_recorder)
        audio_recorder.start()
        audio_recorder._take.chunks.append(
            np.full((256, 1), 0.5, dtype=np.float32))
        deadline = time.monotonic() + 2
        while not levels and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(levels)
        self.assertAlmostEqual(levels[0], 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
