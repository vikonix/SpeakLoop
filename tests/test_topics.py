# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/topics.py.

The program chooses every topic of the lesson, so a topic that repeats, a
spoken command that is not recognized as one, or a list that does not load
would bring back the same three questions the model asked by itself. The last
tests read the shipped list, because the application cannot start without it.

topics.py imports no config, so this file needs neither config nor torch.

Run from the project root with:

    python -m unittest tests.test_topics
"""

import random
import tempfile
import unittest
from pathlib import Path

from speakloop import prompt, topics

SHIPPED_TOPICS = (Path(topics.__file__).resolve().parent
                  / "prompts" / "topics.txt")


class LoadTopicsTests(unittest.TestCase):
    def _load(self, text: str):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "topics.txt"
            path.write_text(text, encoding="utf-8")
            return topics.load_topics(path)

    def test_one_topic_per_line_in_file_order(self):
        self.assertEqual(self._load("travel\ncooking\n"),
                         ("travel", "cooking"))

    def test_comments_and_empty_lines_are_skipped(self):
        self.assertEqual(self._load("# Home\n\n  \nyour home\n# end\n"),
                         ("your home",))

    def test_spaces_around_and_inside_a_topic_are_tidied(self):
        self.assertEqual(self._load("  a   broken phone \n"),
                         ("a broken phone",))

    def test_a_repeated_topic_counts_once(self):
        self.assertEqual(self._load("travel\ncooking\ntravel\n"),
                         ("travel", "cooking"))

    def test_windows_line_ends_are_read_as_plain_ones(self):
        self.assertEqual(self._load("travel\r\ncooking\r\n"),
                         ("travel", "cooking"))

    def test_a_list_without_topics_stops_the_start(self):
        with self.assertRaises(RuntimeError):
            self._load("# only a comment\n\n")

    def test_a_missing_file_is_named_in_the_error(self):
        missing = Path(tempfile.gettempdir()) / "no-such-topics.txt"
        with self.assertRaises(RuntimeError) as raised:
            topics.load_topics(missing)
        self.assertIn("no-such-topics.txt", str(raised.exception))


class PickTopicTests(unittest.TestCase):
    TOPICS = ("travel", "cooking", "music")

    def test_a_used_topic_is_not_chosen_again(self):
        rng = random.Random(1)
        for _ in range(50):
            self.assertEqual(
                topics.pick_topic(self.TOPICS, ["travel", "music"], rng),
                "cooking")

    def test_every_topic_comes_once_before_any_repeats(self):
        rng = random.Random(2)
        used = []
        for _ in self.TOPICS:
            used.append(topics.pick_topic(self.TOPICS, used, rng))
        self.assertCountEqual(used, self.TOPICS)

    def test_after_the_whole_list_the_last_topic_is_not_repeated(self):
        rng = random.Random(3)
        used = ["travel", "cooking", "music"]
        for _ in range(50):
            self.assertNotEqual(topics.pick_topic(self.TOPICS, used, rng),
                                "music")

    def test_a_list_of_one_topic_still_gives_a_topic(self):
        self.assertEqual(topics.pick_topic(("travel",), ["travel"]), "travel")

    def test_random_seed_in_another_library_does_not_repeat_the_topics(self):
        # A library that calls random.seed() with a fixed value must not give
        # every lesson the same topics. Ten picks from a thousand topics
        # repeat by chance with a probability of 1e-30.
        many = tuple(f"topic {number}" for number in range(1000))
        random.seed(0)
        first = [topics.pick_topic(many, ()) for _ in range(10)]
        random.seed(0)
        second = [topics.pick_topic(many, ()) for _ in range(10)]
        self.assertNotEqual(first, second)

    def test_a_set_first_topic_outside_the_list_changes_nothing(self):
        # first_topic from settings.json need not be in the list.
        rng = random.Random(4)
        self.assertIn(topics.pick_topic(self.TOPICS, ["my garden"], rng),
                      self.TOPICS)


class NewTopicCommandTests(unittest.TestCase):
    def test_the_command_as_a_button_sends_it(self):
        self.assertTrue(topics.is_new_topic_command(topics.NEW_TOPIC_COMMAND))

    def test_the_command_as_recognition_writes_it(self):
        for text in ("New topic.", "new topic!", "  NEW   TOPIC  ",
                     "New, topic"):
            with self.subTest(text=text):
                self.assertTrue(topics.is_new_topic_command(text))

    def test_a_longer_phrase_is_not_the_command(self):
        for text in ("new topic, please", "a new topic", "new topics",
                     "new topic: travel", "topic", ""):
            with self.subTest(text=text):
                self.assertFalse(topics.is_new_topic_command(text))

    def test_the_phrase_names_the_topic(self):
        self.assertEqual(topics.topic_command("a trip you remember"),
                         "new topic: a trip you remember")

    def test_the_button_sends_the_same_command(self):
        self.assertIn(topics.NEW_TOPIC_COMMAND, prompt.LESSON_COMMANDS)


class ShippedTopicsTests(unittest.TestCase):
    def test_the_shipped_list_loads(self):
        self.assertGreaterEqual(len(topics.load_topics(SHIPPED_TOPICS)), 50)

    def test_no_shipped_topic_has_a_line_break_or_a_colon(self):
        # A topic goes into one line of the prompt and after "new topic:".
        for topic in topics.load_topics(SHIPPED_TOPICS):
            with self.subTest(topic=topic):
                self.assertNotIn(":", topic)
                self.assertEqual(topic, " ".join(topic.split()))


if __name__ == "__main__":
    unittest.main()
