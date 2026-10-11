# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/proactive.py.

The ladder decides when the program speaks to a learner who has not answered.
A wrong step asks for a simpler question after a hint, keeps helping after the
learner answered, or never stops; these tests pin each rule of
docs/proactive-tutor-spec.md, section 6.

proactive.py imports no config, so this file needs neither config nor torch.

Run from the project root with:

    python -m unittest tests.test_proactive
"""

import unittest

from speakloop import proactive, topics


class IsEnabledTests(unittest.TestCase):
    def test_auto_follows_the_lesson_file(self):
        self.assertFalse(proactive.is_enabled("auto", has_lesson_file=False))
        self.assertTrue(proactive.is_enabled("auto", has_lesson_file=True))

    def test_a_flag_wins_over_the_lesson_file(self):
        self.assertTrue(proactive.is_enabled(True, has_lesson_file=False))
        self.assertFalse(proactive.is_enabled(False, has_lesson_file=True))


class LessonCommandTests(unittest.TestCase):
    def test_the_commands_are_recognized_as_spoken(self):
        # Recognition adds a capital letter and a full stop.
        self.assertEqual(proactive.lesson_command("Simpler."), "simpler")
        self.assertEqual(proactive.lesson_command("hint"), "hint")
        self.assertEqual(proactive.lesson_command("Finish!"), "finish")

    def test_both_forms_of_new_topic_are_the_command(self):
        # The program sends "new topic: <topic>" in place of the bare command.
        self.assertEqual(proactive.lesson_command("new topic"),
                         topics.NEW_TOPIC_COMMAND)
        self.assertEqual(
            proactive.lesson_command(topics.topic_command("my hobby")),
            topics.NEW_TOPIC_COMMAND)

    def test_the_example_command_is_recognized(self):
        self.assertEqual(proactive.lesson_command("Example."), "example")

    def test_a_phrase_that_only_contains_a_command_is_not_one(self):
        self.assertIsNone(proactive.lesson_command("Give me a hint please"))
        self.assertIsNone(proactive.lesson_command("I went to Madrid."))


class CompareWithExampleTests(unittest.TestCase):
    """The repeat of the model answer (docs/proactive-tutor-spec.md, step 5)."""

    def test_the_same_words_pass(self):
        match = proactive.compare_with_example("I went to Madrid by train.",
                                               "I went to Madrid by train")
        self.assertEqual((match.matched, match.total), (6, 6))
        self.assertEqual(match.missing, ())
        self.assertTrue(match.passed)

    def test_case_and_punctuation_do_not_count(self):
        match = proactive.compare_with_example("I don't know.",
                                               "i DON\u2019T know!")
        self.assertTrue(match.passed)

    def test_the_missing_words_are_named_once_in_order(self):
        match = proactive.compare_with_example(
            "I took the train to the coast.", "I went to the coast.")
        # "the" is twice in the sample and once in the answer, so one "the"
        # is missing too; it is named once, where it is first missing.
        self.assertEqual(match.missing, ("took", "train", "the"))
        self.assertEqual((match.matched, match.total), (4, 7))

    def test_extra_words_of_the_learner_do_not_count_against_them(self):
        match = proactive.compare_with_example(
            "I like tea.", "Well, I like tea very much.")
        self.assertEqual(match.ratio, 1.0)

    def test_the_threshold_decides(self):
        # 4 of 5 words is 0.8: passed.
        self.assertTrue(proactive.compare_with_example(
            "one two three four five", "one two three four").passed)
        # 3 of 5 is 0.6: not passed.
        self.assertFalse(proactive.compare_with_example(
            "one two three four five", "one two three").passed)

    def test_an_empty_example_never_passes(self):
        match = proactive.compare_with_example("", "anything")
        self.assertEqual(match.total, 0)
        self.assertFalse(match.passed)


class CountdownTests(unittest.TestCase):
    def test_the_last_three_seconds_are_counted(self):
        self.assertEqual(proactive.countdown_marks(10),
                         [(7, 3), (8, 2), (9, 1)])

    def test_the_shortest_wait_shows_the_whole_countdown(self):
        # config allows 3 seconds at least.
        self.assertEqual(proactive.countdown_marks(3),
                         [(0, 3), (1, 2), (2, 1)])

    def test_a_shorter_wait_shows_only_what_fits(self):
        self.assertEqual(proactive.countdown_marks(2), [(0, 2), (1, 1)])


class LadderStepTests(unittest.TestCase):
    def test_the_steps_come_in_order_then_the_ladder_ends(self):
        ladder = proactive.Ladder()
        self.assertEqual(ladder.step, proactive.STEP_WAIT)
        self.assertEqual(ladder.advance(), proactive.STEP_REPEAT)
        self.assertEqual(ladder.advance(), proactive.STEP_SIMPLER)
        self.assertEqual(ladder.advance(), proactive.STEP_HINT)
        self.assertEqual(ladder.advance(), proactive.STEP_EXAMPLE)
        self.assertIsNone(ladder.advance())

    def test_the_end_does_not_move_the_step(self):
        ladder = proactive.Ladder()
        for _ in range(5):
            ladder.advance()
        self.assertEqual(ladder.step, proactive.LAST_HELP_STEP)

    def test_a_shorter_ladder_ends_earlier(self):
        ladder = proactive.Ladder(last_help_step=proactive.STEP_HINT)
        steps = [ladder.advance() for _ in range(4)]
        self.assertEqual(steps, [1, 2, 3, None])

    def test_reset_returns_to_the_wait(self):
        ladder = proactive.Ladder()
        ladder.advance()
        ladder.reset()
        self.assertEqual(ladder.step, proactive.STEP_WAIT)


class LadderPhraseTests(unittest.TestCase):
    def _ladder_at(self, step: int) -> proactive.Ladder:
        ladder = proactive.Ladder()
        for _ in range(step):
            ladder.advance()
        return ladder

    def test_an_answer_resets_the_ladder(self):
        for answer in ("I went to Madrid.", "I don't know", "Не знаю"):
            with self.subTest(answer=answer):
                ladder = self._ladder_at(2)
                ladder.on_learner_phrase(answer)
                self.assertEqual(ladder.step, proactive.STEP_WAIT)

    def test_a_manual_simpler_moves_to_its_step(self):
        ladder = self._ladder_at(0)
        ladder.on_learner_phrase("simpler")
        self.assertEqual(ladder.step, proactive.STEP_SIMPLER)
        # The next step after the timeout is the hint, not a second "simpler".
        self.assertEqual(ladder.advance(), proactive.STEP_HINT)

    def test_a_manual_hint_moves_to_the_example_next(self):
        ladder = self._ladder_at(1)
        ladder.on_learner_phrase("Hint.")
        self.assertEqual(ladder.step, proactive.STEP_HINT)
        self.assertEqual(ladder.advance(), proactive.STEP_EXAMPLE)

    def test_a_manual_example_ends_the_ladder_next(self):
        # The model answer is the last help step: the pause comes next.
        ladder = self._ladder_at(0)
        ladder.on_learner_phrase("Example.")
        self.assertEqual(ladder.step, proactive.STEP_EXAMPLE)
        self.assertIsNone(ladder.advance())

    def test_a_manual_command_never_moves_the_ladder_back(self):
        ladder = self._ladder_at(3)
        ladder.on_learner_phrase("simpler")
        self.assertEqual(ladder.step, proactive.STEP_HINT)

    def test_new_topic_resets_the_ladder(self):
        ladder = self._ladder_at(3)
        ladder.on_learner_phrase(topics.topic_command("travel"))
        self.assertEqual(ladder.step, proactive.STEP_WAIT)

    def test_finish_stops_the_ladder(self):
        ladder = self._ladder_at(1)
        ladder.on_learner_phrase("finish")
        self.assertTrue(ladder.stopped)

    def test_a_phrase_after_finish_starts_it_again(self):
        # The model may ask "Do you want to finish?" instead of the summary.
        ladder = proactive.Ladder()
        ladder.on_learner_phrase("finish")
        ladder.on_learner_phrase("No, let's go on.")
        self.assertFalse(ladder.stopped)
        self.assertEqual(ladder.step, proactive.STEP_WAIT)


class LadderRepeatTests(unittest.TestCase):
    """Repeat by hand is help step 1 (the rule of the manual commands)."""

    def test_a_repeat_moves_the_wait_to_step_one(self):
        ladder = proactive.Ladder()
        ladder.on_repeat()
        self.assertEqual(ladder.step, proactive.STEP_REPEAT)
        # The next step after the timeout is the simpler question.
        self.assertEqual(ladder.advance(), proactive.STEP_SIMPLER)

    def test_a_repeat_never_moves_the_ladder_back(self):
        ladder = proactive.Ladder()
        ladder.advance()
        ladder.advance()
        ladder.on_repeat()
        self.assertEqual(ladder.step, proactive.STEP_SIMPLER)


class LadderTakeTests(unittest.TestCase):
    def test_a_take_starts_the_wait_again(self):
        ladder = proactive.Ladder()
        ladder.advance()
        ladder.advance()
        ladder.begin_take()
        self.assertEqual(ladder.step, proactive.STEP_WAIT)

    def test_an_empty_take_gives_the_step_back(self):
        # Space resets the step, but a take without speech is no answer.
        ladder = proactive.Ladder()
        ladder.advance()
        ladder.advance()
        ladder.begin_take()
        ladder.take_was_empty()
        self.assertEqual(ladder.step, proactive.STEP_SIMPLER)

    def test_a_recognized_take_keeps_the_reset(self):
        ladder = proactive.Ladder()
        ladder.advance()
        ladder.begin_take()
        ladder.on_learner_phrase("I went to Madrid.")
        # A late "empty" report must not bring the old step back.
        ladder.take_was_empty()
        self.assertEqual(ladder.step, proactive.STEP_WAIT)

    def test_an_empty_take_without_a_take_changes_nothing(self):
        ladder = proactive.Ladder()
        ladder.advance()
        ladder.take_was_empty()
        self.assertEqual(ladder.step, proactive.STEP_REPEAT)


if __name__ == "__main__":
    unittest.main()
