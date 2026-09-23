# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/conversation.py.

Lesson is driven with a stand-in for LLMManager that records what it is
asked and answers from a list, so no client, server or config is needed.

Run from the project root with:

    python -m unittest tests.test_conversation
"""

import threading
import unittest

from speakloop.conversation import (
    OPENING_MESSAGE, Lesson, context_level_reached)


class FakeLLM:
    """The part of LLMManager that Lesson uses."""

    def __init__(self, answers=()):
        self.system_prompt = None
        self.requests = []
        self._answers = list(answers)

    def start_conversation(self, system_prompt):
        self.system_prompt = system_prompt

    def ask(self, user_text, stop_event):
        self.requests.append((user_text, stop_event))
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class ContextLevelTests(unittest.TestCase):
    """When the learner is told that the context fills up."""

    def test_below_the_first_level_nothing_is_said(self):
        self.assertIsNone(context_level_reached(799, 1000, 0.0))

    def test_the_first_level_is_reached(self):
        self.assertEqual(context_level_reached(800, 1000, 0.0), 0.8)

    def test_a_level_is_said_once(self):
        self.assertIsNone(context_level_reached(850, 1000, 0.8))

    def test_the_second_level_follows_the_first(self):
        self.assertEqual(context_level_reached(900, 1000, 0.8), 0.9)

    def test_a_jump_past_both_levels_gives_the_higher_one(self):
        self.assertEqual(context_level_reached(950, 1000, 0.0), 0.9)

    def test_no_token_count_says_nothing(self):
        # An interrupted reply has no usage report.
        self.assertIsNone(context_level_reached(None, 1000, 0.0))

    def test_no_context_size_says_nothing(self):
        self.assertIsNone(context_level_reached(900, 0, 0.0))

    def test_a_small_reserve_left_reaches_the_last_level_early(self):
        # 2560 tokens: 10 percent is 256, less than a summary may need.
        self.assertEqual(context_level_reached(1600, 2560, 0.8, 1024), 0.9)

    def test_near_the_end_the_last_level_comes_at_once(self):
        self.assertEqual(context_level_reached(2100, 2560, 0.0, 1024), 0.9)

    def test_enough_room_left_says_nothing(self):
        self.assertIsNone(context_level_reached(1000, 2560, 0.0, 1024))

    def test_a_large_context_is_not_affected_by_the_reserve(self):
        self.assertIsNone(context_level_reached(14000, 16384, 0.8, 1024))


class LessonTests(unittest.TestCase):
    def _lesson(self, *answers):
        self.llm = FakeLLM(answers)
        return Lesson(self.llm, "You are a tutor.")

    def test_a_new_lesson_starts_the_conversation_with_the_prompt(self):
        self._lesson()
        self.assertEqual(self.llm.system_prompt, "You are a tutor.")

    def test_the_opening_sends_the_opening_message(self):
        lesson = self._lesson("SAY: What do you build?")
        stop_event = threading.Event()
        with self.assertLogs(level="INFO"):
            lesson.open(stop_event)
        self.assertEqual(self.llm.requests, [(OPENING_MESSAGE, stop_event)])

    def test_the_opening_returns_the_first_question(self):
        lesson = self._lesson("SAY: What do you build?")
        with self.assertLogs(level="INFO"):
            reply = lesson.open(threading.Event())
        self.assertEqual(reply.say, "What do you build?")

    def test_an_answer_sends_the_learner_text_as_it_is(self):
        # Commands are not changed in this version (section 10.4).
        lesson = self._lesson("SUMMARY: Итог.")
        stop_event = threading.Event()
        lesson.answer("Finish.", stop_event)
        self.assertEqual(self.llm.requests, [("Finish.", stop_event)])

    def test_an_answer_returns_the_parsed_reply(self):
        lesson = self._lesson('NOTE: "a" -> "b". Причина.\nSAY: Why?')
        reply = lesson.answer("text", threading.Event())
        self.assertEqual(reply.note, '"a" -> "b". Причина.')
        self.assertEqual(reply.say, "Why?")

    def test_an_interrupted_exchange_gives_no_reply(self):
        lesson = self._lesson(None)
        self.assertIsNone(lesson.answer("text", threading.Event()))

    def test_a_reply_outside_the_contract_is_returned_with_a_warning(self):
        lesson = self._lesson("Where do you work?")
        with self.assertLogs(level="WARNING"):
            reply = lesson.answer("text", threading.Event())
        self.assertFalse(reply.follows_contract)
        self.assertEqual(reply.raw, "Where do you work?")

    def test_a_failed_request_is_raised(self):
        # The controller tells the learner; the lesson does not hide it.
        lesson = self._lesson(RuntimeError("server down"))
        with self.assertRaises(RuntimeError):
            lesson.answer("text", threading.Event())


if __name__ == "__main__":
    unittest.main()
