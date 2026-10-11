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

    def test_a_phrase_that_only_contains_a_command_is_not_one(self):
        self.assertIsNone(proactive.lesson_command("Give me a hint please"))
        self.assertIsNone(proactive.lesson_command("I went to Madrid."))


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
        self.assertIsNone(ladder.advance())

    def test_the_end_does_not_move_the_step(self):
        ladder = proactive.Ladder()
        for _ in range(5):
            ladder.advance()
        self.assertEqual(ladder.step, proactive.LAST_HELP_STEP)

    def test_a_longer_ladder_has_more_steps(self):
        # P2 adds the model answer as step 4.
        ladder = proactive.Ladder(last_help_step=4)
        steps = [ladder.advance() for _ in range(5)]
        self.assertEqual(steps, [1, 2, 3, 4, None])

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

    def test_a_manual_hint_ends_the_ladder_next(self):
        # In P1 the hint is the last help step: the pause comes next.
        ladder = self._ladder_at(1)
        ladder.on_learner_phrase("Hint.")
        self.assertEqual(ladder.step, proactive.STEP_HINT)
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
