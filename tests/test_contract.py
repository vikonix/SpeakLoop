# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/contract.py.

parse_reply() decides what the learner reads and what the learner hears: a
NOTE taken for SAY would be read aloud, in the explanation language. The
replies below have the shapes the prompt asks for, plus the near misses,
which the controller shows in full and does not speak. strip_markdown()
cleans the SAY line for the synthesis alone.

contract.py is pure code, so this file needs neither config nor torch.

Run from the project root with:

    python -m unittest tests.test_contract
"""

import unittest

from speakloop.contract import parse_reply, split_sentences, strip_markdown

NOTE_LINE = ('NOTE: "I fix my bicycle with a scotch" -> "with tape". '
             'Scotch это виски, а лента это tape.')
SAY_LINE = "SAY: Tape on a bicycle. How long does it hold?"


class ContractReplyTests(unittest.TestCase):
    """Replies that follow the contract."""

    def test_a_reply_with_a_correction_has_note_and_say(self):
        reply = parse_reply(f"{NOTE_LINE}\n{SAY_LINE}")
        self.assertEqual(reply.note,
                         '"I fix my bicycle with a scotch" -> "with tape". '
                         'Scotch это виски, а лента это tape.')
        self.assertEqual(reply.say, "Tape on a bicycle. How long does it hold?")
        self.assertIsNone(reply.summary)
        self.assertTrue(reply.follows_contract)

    def test_a_reply_without_a_correction_has_only_say(self):
        reply = parse_reply("SAY: A parrot in the kitchen. Who feeds it?")
        self.assertIsNone(reply.note)
        self.assertEqual(reply.say, "A parrot in the kitchen. Who feeds it?")
        self.assertTrue(reply.follows_contract)

    def test_the_summary_keeps_all_its_lines(self):
        text = ("SUMMARY: Итог урока.\n"
                "- \"I build a tool for learners\"\n"
                "\n"
                "- Практиковать: past simple.")
        reply = parse_reply(text)
        self.assertEqual(reply.summary,
                         "Итог урока.\n"
                         "- \"I build a tool for learners\"\n"
                         "\n"
                         "- Практиковать: past simple.")
        self.assertIsNone(reply.say)
        self.assertTrue(reply.follows_contract)

    def test_a_summary_may_start_on_the_next_line(self):
        reply = parse_reply("SUMMARY:\nИтог урока.")
        self.assertEqual(reply.summary, "Итог урока.")

    def test_the_summary_ends_the_parsing(self):
        # A "SAY:" inside the summary is part of the summary, not a line to
        # speak.
        reply = parse_reply("SUMMARY: Итог.\nSAY: this is not spoken")
        self.assertIsNone(reply.say)
        self.assertEqual(reply.summary, "Итог.\nSAY: this is not spoken")

    def test_lines_above_the_summary_are_still_read(self):
        reply = parse_reply("SAY: Thank you.\nSUMMARY: Итог.")
        self.assertEqual(reply.say, "Thank you.")
        self.assertEqual(reply.summary, "Итог.")

    def test_a_note_prefix_before_the_summary_is_allowed(self):
        # The prompt says every line begins with NOTE: or SAY:, and Gemma
        # wrote "NOTE: SUMMARY:" at "finish".
        reply = parse_reply('NOTE: SUMMARY:\n"I work in the king" -> "with a team".\n'
                            "Практикуйте артикли.")
        self.assertIsNone(reply.note)
        self.assertEqual(reply.summary,
                         '"I work in the king" -> "with a team".\n'
                         "Практикуйте артикли.")
        self.assertTrue(reply.follows_contract)

    def test_a_say_prefix_before_the_summary_is_allowed(self):
        reply = parse_reply("SAY: SUMMARY: Итог урока.")
        self.assertIsNone(reply.say)
        self.assertEqual(reply.summary, "Итог урока.")

    def test_a_repeated_summary_label_is_dropped(self):
        # Gemma wrote "NOTE: SUMMARY:" and then "SUMMARY:" on its own line.
        reply = parse_reply("NOTE: SUMMARY:\nSUMMARY:\nВы не успели ответить.\n"
                            "Обсудим магазины.")
        self.assertEqual(reply.summary,
                         "Вы не успели ответить.\nОбсудим магазины.")

    def test_a_repeated_label_on_the_same_line_is_dropped(self):
        reply = parse_reply("SUMMARY: SUMMARY: Итог урока.")
        self.assertEqual(reply.summary, "Итог урока.")

    def test_summary_inside_the_summary_text_stays(self):
        # Only a label at the very start is dropped.
        reply = parse_reply("SUMMARY: Итог.\nSUMMARY: второй раз")
        self.assertEqual(reply.summary, "Итог.\nSUMMARY: второй раз")

    def test_a_summary_of_labels_only_counts_as_absent(self):
        reply = parse_reply("SUMMARY:\nSUMMARY:")
        self.assertIsNone(reply.summary)

    def test_summary_inside_a_note_text_is_not_a_summary(self):
        # Only SUMMARY: right after the prefix starts the summary.
        reply = parse_reply('NOTE: "the SUMMARY: word" -> "x". Причина.\n'
                            "SAY: Why?")
        self.assertIsNone(reply.summary)
        self.assertEqual(reply.note, '"the SUMMARY: word" -> "x". Причина.')
        self.assertEqual(reply.say, "Why?")

    def test_spaces_and_blank_lines_around_the_lines_are_ignored(self):
        reply = parse_reply(f"\n  {NOTE_LINE}  \n\n   {SAY_LINE}\n")
        self.assertIsNotNone(reply.note)
        self.assertEqual(reply.say, "Tape on a bicycle. How long does it hold?")

    def test_raw_is_the_whole_reply_stripped(self):
        reply = parse_reply(f"\n{NOTE_LINE}\n{SAY_LINE}\n")
        self.assertEqual(reply.raw, f"{NOTE_LINE}\n{SAY_LINE}")


class ContractBreakTests(unittest.TestCase):
    """Replies outside the contract: reported, never spoken by mistake."""

    def test_a_reply_without_prefixes_does_not_follow_the_contract(self):
        reply = parse_reply("Tape on a bicycle. How long does it hold?")
        self.assertIsNone(reply.say)
        self.assertFalse(reply.follows_contract)

    def test_a_note_alone_does_not_follow_the_contract(self):
        # The learner would get a correction and no question.
        reply = parse_reply(NOTE_LINE)
        self.assertIsNotNone(reply.note)
        self.assertFalse(reply.follows_contract)

    def test_an_empty_say_counts_as_absent(self):
        reply = parse_reply("SAY:   ")
        self.assertIsNone(reply.say)
        self.assertFalse(reply.follows_contract)

    def test_an_empty_summary_counts_as_absent(self):
        reply = parse_reply("SUMMARY:")
        self.assertIsNone(reply.summary)
        self.assertFalse(reply.follows_contract)

    def test_only_the_first_say_is_taken(self):
        reply = parse_reply("SAY: First question?\nSAY: Second question?")
        self.assertEqual(reply.say, "First question?")

    def test_only_the_first_note_is_taken(self):
        reply = parse_reply("NOTE: first\nNOTE: second\nSAY: Why?")
        self.assertEqual(reply.note, "first")
        self.assertEqual(reply.say, "Why?")

    def test_prefixes_in_small_letters_are_not_read(self):
        reply = parse_reply("say: Where do you work?")
        self.assertIsNone(reply.say)

    def test_a_prefix_inside_a_line_is_not_read(self):
        # A prefix is read at the start of a line only, so "**SAY:**" is a
        # reply outside the contract: shown in full and not spoken (step 5b).
        # Markdown is removed from the text of a SAY line, never around its
        # label.
        reply = parse_reply("**SAY:** Where do you work?")
        self.assertIsNone(reply.say)

    def test_an_empty_reply_has_nothing(self):
        reply = parse_reply("")
        self.assertEqual((reply.note, reply.say, reply.summary, reply.raw),
                         (None, None, None, ""))


class StripMarkdownTests(unittest.TestCase):
    """The markers go before the synthesis; the words stay as they are."""

    def test_bold_markers_are_removed(self):
        self.assertEqual(strip_markdown("Use **tape**, not scotch."),
                         "Use tape, not scotch.")

    def test_italic_markers_are_removed(self):
        self.assertEqual(strip_markdown("It is *very* long."),
                         "It is very long.")

    def test_underscore_emphasis_is_removed(self):
        self.assertEqual(strip_markdown("It is _very_ long."),
                         "It is very long.")

    def test_a_word_with_underscores_is_kept(self):
        # The learner writes code, so snake_case is a word of the lesson.
        self.assertEqual(strip_markdown("I name it read_file_name."),
                         "I name it read_file_name.")

    def test_code_markers_are_removed(self):
        self.assertEqual(strip_markdown("Type `pip install` first."),
                         "Type pip install first.")

    def test_strikethrough_markers_are_removed(self):
        self.assertEqual(strip_markdown("~~scotch~~ tape"), "scotch tape")

    def test_several_markers_in_one_line_are_removed(self):
        self.assertEqual(strip_markdown("**Tape** is *not* `scotch`."),
                         "Tape is not scotch.")

    def test_a_marker_without_its_pair_stays(self):
        self.assertEqual(strip_markdown("2 * 3 is six."), "2 * 3 is six.")

    def test_two_markers_with_spaces_against_them_stay(self):
        # Emphasis has no space between the marker and the word, so this is
        # arithmetic and not a pair.
        self.assertEqual(strip_markdown("2 * 3 * 5 is thirty."),
                         "2 * 3 * 5 is thirty.")

    def test_plain_text_is_unchanged(self):
        self.assertEqual(strip_markdown("Where do you work?"),
                         "Where do you work?")

    def test_the_sentence_split_sees_the_clean_text(self):
        # The order app.py uses: a marker must not hide the end of a
        # sentence from the split.
        self.assertEqual(split_sentences(strip_markdown("**Tape.** *Why?*")),
                         ["Tape.", "Why?"])


class SplitSentencesTests(unittest.TestCase):
    def test_sentences_are_split_at_their_ends(self):
        self.assertEqual(
            split_sentences("A parrot in the kitchen. Who feeds it? Tell me!"),
            ["A parrot in the kitchen.", "Who feeds it?", "Tell me!"])

    def test_a_number_with_a_dot_is_not_split(self):
        self.assertEqual(split_sentences("It costs 1.5 euros. Is it much?"),
                         ["It costs 1.5 euros.", "Is it much?"])

    def test_a_small_letter_after_a_dot_is_not_a_new_sentence(self):
        self.assertEqual(split_sentences("Open the file e.g. notes.txt now."),
                         ["Open the file e.g. notes.txt now."])

    def test_one_sentence_stays_whole(self):
        self.assertEqual(split_sentences("Who feeds it?"), ["Who feeds it?"])

    def test_empty_text_has_no_sentences(self):
        self.assertEqual(split_sentences("   "), [])


if __name__ == "__main__":
    unittest.main()
