# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""The output contract of the lesson prompt: NOTE, SAY and SUMMARY.

Every reply of the model is one or two lines, and each line begins with NOTE:
or SAY:. The last reply of a lesson begins with SUMMARY: and is the one reply
that may have several lines (speakloop/prompts/free_talk.md, sections OUTPUT
and SUMMARY). NOTE and SUMMARY are shown and never spoken; SAY is spoken.

Pure code: no config and no I/O. parse_reply() only reports what it found;
what to do with a reply that breaks the contract is the caller's decision.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

NOTE_PREFIX = "NOTE:"
SAY_PREFIX = "SAY:"
SUMMARY_PREFIX = "SUMMARY:"

# Sentence end: punctuation, whitespace, then a capital letter. A dot inside a
# number ("1.5") has no whitespace after it, so it does not split.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-Я])")

# Inline markdown Gemma sometimes puts inside a SAY line. Each pattern keeps
# the text and drops the markers. The two-character markers go first: the
# one-character rules below would otherwise leave a stray marker behind.
# A marker with a space against the text is not emphasis, which is what keeps
# "2 * 3 * 5" whole.
_MARKDOWN_SPANS = (
    re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*"),
    re.compile(r"~~(?!\s)(.+?)(?<!\s)~~"),
    re.compile(r"\*(?!\s)(.+?)(?<!\s)\*"),
    # An underscore counts as emphasis only between two word boundaries:
    # "read_file_name" is a word of the lesson and not a marker.
    re.compile(r"(?<!\w)_(.+?)_(?!\w)"),
    re.compile(r"`(.+?)`"),
)


@dataclass(frozen=True)
class Reply:
    """One reply of the model, split by the contract.

    note    - the correction, or None.
    say     - the partner's line, or None.
    summary - the lesson summary (may have several lines), or None.
    raw     - the whole reply as the model wrote it, stripped.
    """
    note: Optional[str]
    say: Optional[str]
    summary: Optional[str]
    raw: str

    @property
    def follows_contract(self) -> bool:
        """True when the lesson can use the reply: it has SAY or is the SUMMARY.

        A NOTE alone is not enough: the learner then gets no question, and the
        lesson stops without a word.
        """
        return self.say is not None or self.summary is not None


def _text_after(line: str, prefix: str) -> Optional[str]:
    """The text after *prefix* when *line* begins with it, else None."""
    if not line.startswith(prefix):
        return None
    return line[len(prefix):].strip()


def _summary_text(line: str) -> Optional[str]:
    """The text after SUMMARY: when *line* starts the summary, else None.

    A NOTE: or SAY: prefix just before SUMMARY: is allowed. The prompt says
    that every line begins with NOTE: or SAY:, so the model may write
    "NOTE: SUMMARY:"; without this rule the summary is taken for a NOTE and
    the reply is shown as a broken one.
    """
    for prefix in (NOTE_PREFIX, SAY_PREFIX):
        rest = _text_after(line, prefix)
        if rest is not None and rest.startswith(SUMMARY_PREFIX):
            line = rest
            break
    return _text_after(line, SUMMARY_PREFIX)


def _without_repeated_label(summary: str) -> str:
    """*summary* without SUMMARY: labels at its start.

    The model may repeat the label on the next line ("NOTE: SUMMARY:" and then
    "SUMMARY:"). The repeat is a label and not a part of the summary; left in,
    the window shows it as the first line of the summary.
    """
    while True:
        after = _text_after(summary, SUMMARY_PREFIX)
        if after is None:
            return summary
        summary = after


def parse_reply(text: str) -> Reply:
    """Split a reply of the model into NOTE, SAY and SUMMARY.

    The SUMMARY takes everything from its line to the end of the reply, also
    when that line is "NOTE: SUMMARY:" or "SAY: SUMMARY:", and without a
    SUMMARY: label repeated at its start. Above
    it, the first NOTE line and the first SAY line are taken; a second line of
    the same kind and any line without a prefix are ignored. A reply with
    neither SAY nor SUMMARY is reported by follows_contract, and the
    controller shows it in full. Spaces before a prefix are ignored. The
    prefixes are matched in capitals, as the prompt writes them. A prefix with
    no text after it counts as absent.
    """
    raw = text.strip()
    note = say = summary = None
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        line = line.strip()
        summary_text = _summary_text(line)
        if summary_text is not None:
            rest = [summary_text] + [later.rstrip()
                                     for later in lines[index + 1:]]
            summary = _without_repeated_label("\n".join(rest).strip())
            summary = summary or None
            break
        note_text = _text_after(line, NOTE_PREFIX)
        if note_text:
            if note is None:
                note = note_text
            continue
        say_text = _text_after(line, SAY_PREFIX)
        if say_text and say is None:
            say = say_text
    return Reply(note=note, say=say, summary=summary, raw=raw)


def strip_markdown(text: str) -> str:
    """*text* without the inline markdown markers, for speech only.

    The synthesis reads a marker as a sound, so "**tape**" is heard wrong.
    Only the markers go; the text between them stays, and what the learner
    reads in the chat is not changed. A marker that has no pair, or that
    stands against a space, stays: "2 * 3 * 5" is arithmetic.
    """
    for pattern in _MARKDOWN_SPANS:
        text = pattern.sub(r"\1", text)
    return text


def split_sentences(text: str) -> List[str]:
    """The sentences of a SAY line, in order, for speech one at a time.

    Speech goes sentence by sentence so that the first sentence is heard
    before the next one is synthesized, and an interrupt stops between two
    sentences.
    """
    return [part.strip() for part in _SENTENCE_END.split(text)
            if part.strip()]
