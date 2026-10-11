# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""The help ladder of the proactive tutor, without Tk and without config.

When the learner does not answer, the program helps step by step: it repeats
the question slower (step 1), asks a simpler question (step 2), gives a hint
(step 3), gives a model answer for the learner to say (step 4), and then
pauses the lesson. compare_with_example() checks how the learner said the
model answer. The timers and the window belong to
speakloop/app.py; this module only decides which step comes next. It is pure,
so every rule of the ladder is tested without a window
(docs/proactive-tutor-spec.md, section 6).

Imports only the pure prompt and topics modules: a command of the learner is
recognized here with the same rules the rest of the lesson uses.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from speakloop import prompt, topics

# The steps of the ladder. Step 0 is the wait after a question; 1-4 are the
# help steps.
STEP_WAIT = 0
STEP_REPEAT = 1
STEP_SIMPLER = 2
STEP_HINT = 3
STEP_EXAMPLE = 4
LAST_HELP_STEP = STEP_EXAMPLE

# The share of the words of the model answer the learner must say for the
# repeat to count as done (docs/proactive-tutor-spec.md, 5.2).
EXAMPLE_MATCH_THRESHOLD = 0.8

# A word of an answer, as the repeat check compares it: letters and digits,
# with an apostrophe inside ("don't", "I'm"). Case and punctuation do not
# count, because recognition adds both.
_ANSWER_WORD = re.compile(r"\w+(?:'\w+)*")

# The countdown before a step is shown for this many seconds only: a longer
# one would press the learner while they think.
COUNTDOWN_SECONDS = 3

# The value of the "proactive" setting that follows the lesson file.
PROACTIVE_AUTO = "auto"

# The lesson commands that are also help steps. A manual command moves the
# ladder to its own step, so the program never asks for a simpler question
# after the learner already asked for a hint.
_COMMAND_STEPS = {"simpler": STEP_SIMPLER, "hint": STEP_HINT,
                  prompt.EXAMPLE_COMMAND: STEP_EXAMPLE}
_FINISH_COMMAND = "finish"


def is_enabled(setting, has_lesson_file: bool) -> bool:
    """True when the ladder runs in this lesson.

    *setting* is the validated "proactive" value: "auto" follows the lesson
    file (there is no lesson file yet, so "auto" means off), True and False
    are the learner's own choice.
    """
    if setting == PROACTIVE_AUTO:
        return has_lesson_file
    return bool(setting)


def lesson_command(text: str) -> Optional[str]:
    """The lesson command that *text* is, or None for an ordinary phrase.

    The program sends "new topic: <topic>" in place of a bare "new topic",
    so both forms are the new topic command.
    """
    if text.strip().lower().startswith(topics.topic_command("")):
        return topics.NEW_TOPIC_COMMAND
    words = topics.command_words(text)
    for command in (*prompt.LESSON_COMMANDS, prompt.EXAMPLE_COMMAND):
        if words == command.split():
            return command
    return None


@dataclass(frozen=True)
class ExampleMatch:
    """How the learner said the model answer.

    matched - words of the model answer the learner said;
    total   - words of the model answer;
    missing - the words they left out, in the order of the model answer,
              each once;
    ratio   - matched / total (0.0 for an empty model answer);
    passed  - ratio at or above EXAMPLE_MATCH_THRESHOLD.
    """
    matched: int
    total: int
    missing: Tuple[str, ...]
    ratio: float
    passed: bool


def answer_words(text: str) -> List[str]:
    """The words of *text* in lower case, without punctuation.

    The typographic apostrophe counts as the plain one: recognition and the
    model do not always write the same one.
    """
    return _ANSWER_WORD.findall(text.lower().replace("\u2019", "'"))


def compare_with_example(example: str, answer: str) -> ExampleMatch:
    """Compare what the learner said with the model answer.

    Counts the words of the model answer found in the answer, each as many
    times as it occurs, in any order: the learner may change the order a
    little, and a recognition that drops a repeated word must not make the
    whole repeat fail. Extra words of the learner do not count against them.
    """
    sample = answer_words(example)
    said = Counter(answer_words(answer))
    matched = 0
    missing: List[str] = []
    for word in sample:
        if said[word] > 0:
            said[word] -= 1
            matched += 1
        elif word not in missing:
            missing.append(word)
    total = len(sample)
    ratio = matched / total if total else 0.0
    return ExampleMatch(matched=matched, total=total, missing=tuple(missing),
                        ratio=ratio, passed=ratio >= EXAMPLE_MATCH_THRESHOLD)


def countdown_marks(wait_seconds: float) -> List[Tuple[float, int]]:
    """When to show each number of the countdown, from the start of a wait.

    Pairs of (seconds after the start, number shown), the last number one
    second before the step. A mark before the start is dropped, so a wait
    shorter than the countdown shows only the numbers that fit.
    """
    marks = []
    for number in range(COUNTDOWN_SECONDS, 0, -1):
        at = wait_seconds - number
        if at >= 0:
            marks.append((at, number))
    return marks


class Ladder:
    """The current step of the ladder and the rules that move it.

    Not thread-safe: app.py uses it on the Tk thread only.
    """

    def __init__(self, last_help_step: int = LAST_HELP_STEP):
        self.step = STEP_WAIT
        # True after "finish": the learner asked to end the lesson, and the
        # program must not ask for an answer meanwhile.
        self.stopped = False
        self._last_help_step = last_help_step
        # The step before the open take, kept until the take is recognized.
        self._step_before_take: Optional[int] = None

    def reset(self) -> None:
        """Back to the wait after a question (step 0)."""
        self.step = STEP_WAIT

    def advance(self) -> Optional[int]:
        """Take the next help step and return it.

        None when the last help step is already done: the ladder is over and
        the lesson pauses. The step stays where it is then, and the resume
        resets it.
        """
        if self.step >= self._last_help_step:
            return None
        self.step += 1
        return self.step

    def begin_take(self) -> None:
        """A take opened: the learner answers, so the wait starts again.

        The step is kept: a take without speech is no answer, and
        take_was_empty() gives the step back.
        """
        self._step_before_take = self.step
        self.reset()

    def take_was_empty(self) -> None:
        """The take held no phrase: continue from the step before it."""
        if self._step_before_take is not None:
            self.step = self._step_before_take
            self._step_before_take = None

    def on_repeat(self) -> None:
        """The learner asked to hear the question again (Repeat, the R key).

        That is help step 1 done by hand, so the next step after the timeout
        is the simpler question and not a second repeat (the rule of the
        manual commands).
        """
        self._step_before_take = None
        self.step = max(self.step, STEP_REPEAT)

    def on_learner_phrase(self, text: str) -> None:
        """Move the ladder for a phrase of the learner (spoken, typed or a button).

        An ordinary phrase is an answer: back to step 0. "simpler" and "hint"
        move the ladder to their own step, so the next step after the timeout
        is the one after them. "new topic" asks a new question: step 0.
        "finish" stops the ladder; a phrase after it starts it again, because
        the model may ask a question instead of writing the summary.
        """
        self._step_before_take = None
        command = lesson_command(text)
        if command in _COMMAND_STEPS:
            self.step = max(self.step, _COMMAND_STEPS[command])
            self.stopped = False
        elif command == _FINISH_COMMAND:
            self.stopped = True
        else:
            self.reset()
            self.stopped = False
