# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""The topics of the lesson: the list, the choice and the "new topic" command.

The program chooses the topics, not the model. Left to itself, the model
opened almost every lesson with the same three questions (breakfast, work,
the morning), whatever the seed. A topic chosen here is also a topic the
window can show.

The list is a text file (speakloop/prompts/topics.txt). The first topic goes
into the "First topic" line of the prompt; every later one goes to the model
as the learner's command with the topic in it ("new topic: <topic>"), which
the Commands line of the prompt explains.

Pure functions and no config import: the caller passes the paths and the
random source, so the tests need neither config nor torch.
"""

import random
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

# The command as the learner says it and as the prompt names it. prompt.py
# builds LESSON_COMMANDS from it, so a button and a spoken command are the
# same string.
NEW_TOPIC_COMMAND = "new topic"

# Anything that is not a letter, a digit or a space. Recognition writes
# "New topic." and a learner may type "new topic!", and both are the command.
_NOT_A_WORD = re.compile(r"[^\w\s]")

# The random source of the topics: a generator of their own, fed by the
# operating system. The module-level functions of "random" share one state
# with every library in the process, and a library that calls random.seed()
# with a fixed value makes every lesson get the same topics (three lessons in
# a row all had "your family"). SystemRandom has no state to reset.
_RANDOM = random.SystemRandom()


def load_topics(path) -> Tuple[str, ...]:
    """The topics in the file *path*, in file order, each once.

    Empty lines and lines that start with "#" are skipped; spaces at the ends
    of a line are not part of the topic.

    Raises RuntimeError that names the file when it cannot be read or holds no
    topic: the lesson cannot change its topic without the list, so the start
    stops at once, like it does for the prompt file.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Cannot read the topic list {path}: {error}") from error
    topics = []
    for line in text.splitlines():
        topic = " ".join(line.split())
        if topic and not topic.startswith("#") and topic not in topics:
            topics.append(topic)
    if not topics:
        raise RuntimeError(f"The topic list {path} has no topics.")
    return tuple(topics)


def pick_topic(topics: Tuple[str, ...], used: Iterable[str],
               rng: Optional[random.Random] = None) -> str:
    """A random topic of *topics* that is not in *used*.

    When every topic is used, any topic but the last used one: a long lesson
    starts the list again rather than stop, and the learner never gets the
    topic they just asked to leave. *rng* is for the tests; None means
    _RANDOM.
    """
    rng = rng or _RANDOM
    used = list(used)
    fresh = [topic for topic in topics if topic not in used]
    if not fresh:
        last = used[-1] if used else None
        fresh = [topic for topic in topics if topic != last] or list(topics)
    return rng.choice(fresh)


def command_words(text: str) -> list:
    """The words of *text* as a command is compared: lower case, without
    punctuation. Recognition writes "New topic." and "Finish!", and both are
    the command."""
    return _NOT_A_WORD.sub(" ", text).lower().split()


def is_new_topic_command(text: str) -> bool:
    """True when *text* is the "new topic" command and nothing else.

    Case, punctuation and extra spaces do not count, because the command is
    also spoken and recognition adds a capital letter and a full stop. A
    longer phrase ("new topic, please") is not the command: it goes to the
    model as it is, and the model then chooses the topic itself.
    """
    return command_words(text) == NEW_TOPIC_COMMAND.split()


def topic_command(topic: str) -> str:
    """The phrase that asks the model for *topic*, as the prompt reads it."""
    return f"{NEW_TOPIC_COMMAND}: {topic}"
