# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/prompt.py.

The SETTINGS lines are the only part of the prompt the program changes, and a
mistake there is silent: the model then takes the default in brackets and the
lesson still looks correct. The last tests read the shipped prompt file
itself, because the application cannot start without it.

prompt.py imports no config, so this file needs neither config nor torch.

Run from the project root with:

    python -m unittest tests.test_prompt
"""

import tempfile
import unittest
from pathlib import Path

from speakloop import prompt

BODY = (
    "ROLE\n"
    "You are a tutor.\n"
    "\n"
    "SETTINGS\n"
    "Target language: [English]\n"
    "Explanation language: [Russian]\n"
    "First topic: [choose one everyday situation]\n"
    "Use the value in brackets when a line is empty.\n"
)

ALL_VALUES = {
    prompt.SETTING_TARGET_LANGUAGE: "Spanish",
    prompt.SETTING_EXPLANATION_LANGUAGE: "English",
    prompt.SETTING_FIRST_TOPIC: "my garden",
}

SHIPPED_PROMPT = Path(prompt.__file__).resolve().parent / "prompts" / "free_talk.md"


class FillSettingsTests(unittest.TestCase):
    def test_every_value_replaces_its_brackets(self):
        filled = prompt.fill_settings(BODY, ALL_VALUES)
        self.assertIn("Target language: Spanish\n", filled)
        self.assertIn("Explanation language: English\n", filled)
        self.assertIn("First topic: my garden\n", filled)
        self.assertNotIn("[", filled)

    def test_an_empty_value_keeps_the_default_in_brackets(self):
        # The prompt reads the brackets as the value of an empty line.
        values = dict(ALL_VALUES, **{prompt.SETTING_FIRST_TOPIC: ""})
        filled = prompt.fill_settings(BODY, values)
        self.assertIn("First topic: [choose one everyday situation]\n", filled)

    def test_a_missing_value_keeps_the_default_in_brackets(self):
        filled = prompt.fill_settings(BODY, {})
        self.assertEqual(filled, BODY)

    def test_a_value_of_spaces_counts_as_empty(self):
        values = {prompt.SETTING_FIRST_TOPIC: "   "}
        filled = prompt.fill_settings(BODY, values)
        self.assertIn("First topic: [choose one everyday situation]\n", filled)

    def test_a_value_stays_on_one_line(self):
        # A line break in the value would add a line the prompt does not know.
        values = {prompt.SETTING_FIRST_TOPIC: "my\n  morning   routine"}
        filled = prompt.fill_settings(BODY, values)
        self.assertIn("First topic: my morning routine\n", filled)

    def test_a_backslash_in_a_value_is_kept_as_it_is(self):
        values = {prompt.SETTING_FIRST_TOPIC: r"files in C:\work\1"}
        filled = prompt.fill_settings(BODY, values)
        self.assertIn("First topic: files in C:\\work\\1\n", filled)

    def test_the_rest_of_the_body_does_not_change(self):
        filled = prompt.fill_settings(BODY, ALL_VALUES)
        self.assertTrue(filled.startswith("ROLE\nYou are a tutor.\n\nSETTINGS\n"))
        self.assertTrue(filled.endswith(
            "Use the value in brackets when a line is empty.\n"))

    def test_a_missing_setting_line_stops_the_start(self):
        body = BODY.replace("First topic: [choose one everyday situation]\n", "")
        with self.assertRaises(RuntimeError) as caught:
            prompt.fill_settings(body, ALL_VALUES)
        self.assertIn("First topic", str(caught.exception))

    def test_a_setting_line_without_brackets_stops_the_start(self):
        # Somebody wrote the value into the file by hand; the program would
        # otherwise ignore the setting without a word.
        body = BODY.replace("Target language: [English]", "Target language: English")
        with self.assertRaises(RuntimeError):
            prompt.fill_settings(body, ALL_VALUES)

    def test_a_repeated_setting_line_stops_the_start(self):
        body = BODY + "Target language: [English]\n"
        with self.assertRaises(RuntimeError):
            prompt.fill_settings(body, ALL_VALUES)


class LoadBodyTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "prompt.md"

    def test_surrounding_blank_lines_are_dropped(self):
        self.path.write_text("\n\nROLE\nText.\n\n", encoding="utf-8")
        self.assertEqual(prompt.load_body(self.path), "ROLE\nText.")

    def test_windows_line_ends_are_read_as_plain_ones(self):
        # The file is checked out with CRLF on Windows.
        self.path.write_bytes(b"ROLE\r\nText.\r\n")
        self.assertEqual(prompt.load_body(self.path), "ROLE\nText.")

    def test_a_missing_file_is_named_in_the_error(self):
        with self.assertRaises(RuntimeError) as caught:
            prompt.load_body(self.path)
        self.assertIn(str(self.path), str(caught.exception))

    def test_an_empty_file_is_an_error(self):
        self.path.write_text(" \n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            prompt.load_body(self.path)


class ShippedPromptTests(unittest.TestCase):
    """The prompt file the application uses by default."""

    def setUp(self):
        self.system_prompt = prompt.build_system_prompt(
            SHIPPED_PROMPT, "English", "Russian", "")

    def test_the_settings_are_filled_in(self):
        self.assertIn("\nTarget language: English\n", self.system_prompt)
        self.assertIn("\nExplanation language: Russian\n", self.system_prompt)

    def test_an_empty_first_topic_keeps_the_prompt_default(self):
        self.assertIn("\nFirst topic: [choose one everyday situation]\n",
                      self.system_prompt)

    def test_the_body_is_the_whole_prompt(self):
        # Copied from the prompt block alone, without the notes around it.
        self.assertTrue(self.system_prompt.startswith("ROLE\n"))
        self.assertTrue(self.system_prompt.endswith(
            "Begin the lesson now with your first question."))

    def test_the_prompt_names_the_voice_commands(self):
        # The command buttons of the window send exactly these strings, so a
        # command renamed in the prompt has to be renamed in LESSON_COMMANDS.
        for command in prompt.LESSON_COMMANDS:
            with self.subTest(command=command):
                self.assertIn(f'"{command}"', self.system_prompt)

    def test_the_commands_are_written_as_the_learner_says_them(self):
        # They go to the model as they are: lower case, no full stop, the way
        # the prompt's Commands line spells them.
        for command in prompt.LESSON_COMMANDS:
            with self.subTest(command=command):
                self.assertEqual(command, command.strip().lower())

    def test_the_prompt_defines_the_three_prefixes(self):
        # speakloop/contract.py parses exactly these.
        for prefix in ("NOTE:", "SAY:", "SUMMARY:"):
            self.assertIn(prefix, self.system_prompt)


if __name__ == "__main__":
    unittest.main()
