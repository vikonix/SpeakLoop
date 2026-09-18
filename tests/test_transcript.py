# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the lesson transcript (speakloop/transcript.py).

The module has two halves and they are tested differently. The record
functions are pure, so they are checked against the fields the learning system
reads. The writer touches the disk, so it runs against a temporary directory:
the module imports neither config nor torch, which keeps this file fast.

What is worth pinning here, in order of what would hurt most if it broke:

* the jsonl line is one line of valid JSON, with the letters of both languages
  in it and not as escapes - the file is read by a program and by a person;
* the meta line comes first and exactly once, because the reader takes the
  settings of the lesson from it;
* a measurement that is missing is absent and not null, while a zero stays;
* the markdown view shows the lesson and leaves the service lines out;
* a file that cannot be written does not raise: a lesson must not end because
  a disk is full.

Run from the project root with:

    python -m unittest tests.test_transcript
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from speakloop import transcript

# One fixed moment for every record here, so a test never depends on the clock.
_START = datetime(2026, 9, 18, 21, 40, 12)
_LATER = datetime(2026, 9, 18, 21, 41, 3)


def _meta():
    return transcript.meta_record(
        started_at=_START,
        target_language="English",
        explanation_language="Russian",
        first_topic="shopping",
        llm_model="gemma-4-12B-it-QAT-Q4_0.gguf",
        stt_model="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        tts_voice="af_heart")


class FileNameTests(unittest.TestCase):
    """The two files of a lesson are named after its start."""

    def test_stem_holds_the_date_and_the_time(self):
        self.assertEqual(transcript.file_stem(_START),
                         "dialog-2026-09-18_21-40")

    def test_two_lessons_of_one_day_get_different_names(self):
        # The minute is part of the name; without it the second lesson of a day
        # would append itself to the first one's file.
        other = datetime(2026, 9, 18, 22, 5, 0)
        self.assertNotEqual(transcript.file_stem(_START),
                            transcript.file_stem(other))


class TimestampTests(unittest.TestCase):
    """Every record says when it happened, and in which zone."""

    def test_the_time_carries_an_offset(self):
        # Without the offset a file read on another machine is ambiguous.
        parsed = datetime.fromisoformat(transcript.timestamp(_START))
        self.assertIsNotNone(parsed.tzinfo)

    def test_the_time_is_the_moment_it_is_given(self):
        parsed = datetime.fromisoformat(transcript.timestamp(_START))
        self.assertEqual(parsed.replace(tzinfo=None), _START)


class MetaRecordTests(unittest.TestCase):
    """The first line: what this lesson was."""

    def test_it_names_its_type_and_schema(self):
        record = _meta()
        self.assertEqual(record["type"], transcript.TYPE_META)
        self.assertEqual(record["schema"], transcript.SCHEMA_VERSION)

    def test_it_holds_the_settings_of_the_lesson(self):
        record = _meta()
        self.assertEqual(record["target_language"], "English")
        self.assertEqual(record["explanation_language"], "Russian")
        self.assertEqual(record["first_topic"], "shopping")
        self.assertEqual(record["llm_model"], "gemma-4-12B-it-QAT-Q4_0.gguf")
        self.assertEqual(record["tts_voice"], "af_heart")


class EventRecordTests(unittest.TestCase):
    """One event of the lesson."""

    def test_the_base_fields_are_always_there(self):
        record = transcript.event_record(_LATER, 2, transcript.TYPE_SAY, "Hi.")
        self.assertEqual(record["turn"], 2)
        self.assertEqual(record["type"], transcript.TYPE_SAY)
        self.assertEqual(record["text"], "Hi.")
        self.assertIn("ts", record)

    def test_a_missing_measurement_is_absent_rather_than_null(self):
        # The reply of an interrupted request has no token count. A null in the
        # file would make every reader test for it.
        record = transcript.event_record(_LATER, 1, transcript.TYPE_SAY, "Hi.",
                                         llm_ms=1200, tokens=None)
        self.assertEqual(record["llm_ms"], 1200)
        self.assertNotIn("tokens", record)

    def test_a_zero_measurement_is_kept(self):
        # Zero is a measurement, unlike a missing one.
        record = transcript.event_record(_LATER, 1, transcript.TYPE_LEARNER,
                                         "Hi.", stt_ms=0)
        self.assertEqual(record["stt_ms"], 0)

    def test_the_source_of_a_phrase_is_carried(self):
        # Only a spoken phrase can hold a recognition error, so the reader of
        # the lesson needs to know which one this was.
        record = transcript.event_record(_LATER, 1, transcript.TYPE_LEARNER,
                                         "I like it.",
                                         source=transcript.SOURCE_VOICE)
        self.assertEqual(record["source"], "voice")


class JsonlTests(unittest.TestCase):
    """The line that goes into the file."""

    def test_one_record_is_one_line(self):
        line = transcript.to_jsonl(transcript.event_record(
            _LATER, 1, transcript.TYPE_SAY, "First.\nSecond."))
        self.assertTrue(line.endswith("\n"))
        self.assertEqual(line.count("\n"), 1)

    def test_the_line_reads_back_as_the_record(self):
        record = transcript.event_record(_LATER, 3, transcript.TYPE_NOTE,
                                         "«closing» -> «clothing»")
        self.assertEqual(json.loads(transcript.to_jsonl(record)), record)

    def test_other_alphabets_stay_readable(self):
        # A person opens this file too, and \\u043a\\u0430\\u043a is not text.
        line = transcript.to_jsonl(transcript.event_record(
            _LATER, 1, transcript.TYPE_NOTE, "как"))
        self.assertIn("как", line)


class MarkdownTests(unittest.TestCase):
    """The readable view built at the end of the lesson."""

    def _view(self):
        records = [
            transcript.event_record(_START, 0, transcript.TYPE_SYSTEM,
                                    "Loading the speech models..."),
            transcript.event_record(_START, 0, transcript.TYPE_SAY,
                                    "What do you buy online?"),
            transcript.event_record(_LATER, 1, transcript.TYPE_LEARNER,
                                    "I buy closing.",
                                    source=transcript.SOURCE_VOICE),
            transcript.event_record(_LATER, 1, transcript.TYPE_NOTE,
                                    "«closing» -> «clothing»"),
            transcript.event_record(_LATER, 1, transcript.TYPE_SAY,
                                    "Which shop do you like?"),
            transcript.event_record(_LATER, 2, transcript.TYPE_BROKEN,
                                    "I am a helpful assistant."),
            transcript.event_record(_LATER, 3, transcript.TYPE_SUMMARY,
                                    "You spoke about shopping."),
        ]
        return transcript.to_markdown(_meta(), records)

    def test_the_header_names_the_lesson(self):
        view = self._view()
        self.assertIn("2026-09-18 21:40", view)
        self.assertIn("English", view)
        self.assertIn("shopping", view)

    def test_the_lesson_itself_is_shown(self):
        view = self._view()
        self.assertIn("I buy closing.", view)
        self.assertIn("«closing» -> «clothing»", view)
        self.assertIn("Which shop do you like?", view)
        self.assertIn("You spoke about shopping.", view)

    def test_the_service_records_stay_in_the_jsonl(self):
        # The view is the lesson, not the run of the application. Both of these
        # are kept in the machine file and skipped here.
        view = self._view()
        self.assertNotIn("Loading the speech models", view)
        self.assertNotIn("I am a helpful assistant.", view)

    def test_the_file_ends_with_one_line_break(self):
        self.assertTrue(self._view().endswith("\n"))
        self.assertFalse(self._view().endswith("\n\n"))


class WriterTests(unittest.TestCase):
    """The files themselves, against a temporary directory."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.directory = Path(self._directory.name)
        self.writer = transcript.TranscriptWriter(self.directory, _START,
                                                  _meta())

    def _lines(self):
        return self.writer.jsonl_path.read_text(
            encoding="utf-8").splitlines()

    def test_a_lesson_without_events_leaves_no_file(self):
        # A session that is closed while the models load must not fill the
        # directory with empty lessons.
        self.writer.save_markdown()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_the_meta_line_comes_first_and_once(self):
        self.writer.add(transcript.event_record(_START, 0,
                                                transcript.TYPE_SAY, "Hi."))
        self.writer.add(transcript.event_record(
            _LATER, 1, transcript.TYPE_LEARNER, "Hi."))
        lines = self._lines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[0])["type"], transcript.TYPE_META)
        self.assertEqual(json.loads(lines[1])["text"], "Hi.")

    def test_every_record_reaches_the_disk_at_once(self):
        # The point of the jsonl file: a lesson that ends in a crash is still
        # there, up to its last event.
        self.writer.add(transcript.event_record(_START, 0,
                                                transcript.TYPE_SAY, "Hi."))
        self.assertEqual(len(self._lines()), 2)

    def test_the_markdown_view_is_written_from_the_records(self):
        self.writer.add(transcript.event_record(_LATER, 1,
                                                transcript.TYPE_LEARNER,
                                                "I buy clothing."))
        self.writer.save_markdown()
        view = self.writer.markdown_path.read_text(encoding="utf-8")
        self.assertIn("I buy clothing.", view)

    def test_the_view_is_replaced_and_not_appended(self):
        # It is written at the summary and again on the way out, and the second
        # write must not repeat the lesson.
        self.writer.add(transcript.event_record(_LATER, 1,
                                                transcript.TYPE_LEARNER,
                                                "I buy clothing."))
        self.writer.save_markdown()
        self.writer.save_markdown()
        view = self.writer.markdown_path.read_text(encoding="utf-8")
        self.assertEqual(view.count("I buy clothing."), 1)

    def test_a_file_that_cannot_be_written_does_not_end_the_lesson(self):
        # The lesson is worth more than its transcript: the failure is logged
        # and the application goes on. assertLogs is what checks that the
        # failure IS reported, and it also keeps the message out of the output
        # of the test run, where it would read as a broken test.
        with self.assertLogs(level="ERROR"), \
                mock.patch("builtins.open", side_effect=OSError("full")):
            self.writer.add(transcript.event_record(
                _START, 0, transcript.TYPE_SAY, "Hi."))
        with self.assertLogs(level="ERROR"), \
                mock.patch.object(Path, "write_text",
                                  side_effect=OSError("full")):
            self.writer.save_markdown()  # must not raise

    def test_a_failed_file_is_not_tried_again_on_every_event(self):
        # One message in the log per lesson, not one per phrase.
        with self.assertLogs(level="ERROR"), \
                mock.patch("builtins.open",
                           side_effect=OSError("full")) as opened:
            for _ in range(3):
                self.writer.add(transcript.event_record(
                    _START, 0, transcript.TYPE_SAY, "Hi."))
        self.assertEqual(opened.call_count, 1)


if __name__ == "__main__":
    unittest.main()
