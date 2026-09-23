# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""The transcript of one lesson: a jsonl file as it happens, a markdown view at
the end.

Two forms on purpose. The jsonl file is the
main one: one record per line, so the learning system reads the lesson without
parsing prose, and the line is appended after every event, so a lesson that
ends in a crash is still on disk. The markdown file is built from the same
records when the lesson ends and is for reading by eye. Markdown can always be
built again from the jsonl; the jsonl cannot be built from markdown, which is
why the machine form is the one that is written first.

Everything above :class:`TranscriptWriter` is pure: a record is a dictionary
and the clock is an argument, so the format is tested without a disk. The
writer owns the two files and one lock, because records arrive both from the
exchange threads and from the Tk thread (speakloop/app.py).
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

# The record types of the jsonl file. The reader selects the lesson it wants by
# them, so each of them is written here once and never spelled at a call site.
TYPE_META = "meta"
TYPE_LEARNER = "learner"
TYPE_NOTE = "note"
TYPE_SAY = "say"
TYPE_SUMMARY = "summary"
TYPE_SYSTEM = "system"
TYPE_BROKEN = "broken"

# Where a phrase of the learner came from. Only a "voice" phrase can hold a
# recognition error, which is what tells a mistake of the learner from a
# mistake of Whisper when the lesson is read later.
SOURCE_VOICE = "voice"
SOURCE_TEXT = "text"
SOURCE_BUTTON = "button"

# The schema of the records. Raise it when a field changes its meaning or goes
# away; a new optional field does not need a new number.
SCHEMA_VERSION = 1

# The name of the two files of a lesson, without the extension. Minutes and no
# seconds: two lessons in one minute would need the same file, and that does
# not happen (a lesson opens with a model request of about a minute).
_FILE_STEM_FORMAT = "dialog-%Y-%m-%d_%H-%M"

# The header of the markdown view.
_MD_TIME_FORMAT = "%Y-%m-%d %H:%M"

# The labels of the markdown view. The partner is a role and not a name, as in
# the window; the window's own label is not imported, because the view must
# stay out of the modules that write files.
_MD_PARTNER = "Tutor"
_MD_LEARNER = "You"
_MD_NOTE = "Note"
_MD_SUMMARY = "Summary"

# The record types the markdown view shows, in the order they were written.
# The service lines ("system") and a reply outside the contract ("broken") stay
# in the jsonl alone: the view is the lesson, not the run of the application.
_MD_TYPES = (TYPE_LEARNER, TYPE_NOTE, TYPE_SAY, TYPE_SUMMARY)


def file_stem(started_at: datetime) -> str:
    """The name of both files of a lesson, without the extension."""
    return started_at.strftime(_FILE_STEM_FORMAT)


def timestamp(moment: datetime) -> str:
    """The time of one record: ISO 8601 with the offset of the local zone.

    With the offset, because a reader that gets the file from another machine
    must not guess whether the time is local or UTC. Seconds are enough: the
    durations of an exchange are measured in their own fields.
    """
    return moment.astimezone().isoformat(timespec="seconds")


def meta_record(started_at: datetime, target_language: str,
                explanation_language: str, first_topic: str, llm_model: str,
                stt_model: str, tts_voice: str) -> dict:
    """The first line of the file: what this lesson was.

    Everything here is fixed for the whole lesson, so it is written once and
    not repeated on every event.
    """
    return {
        "type": TYPE_META,
        "schema": SCHEMA_VERSION,
        "started_at": timestamp(started_at),
        "target_language": target_language,
        "explanation_language": explanation_language,
        "first_topic": first_topic,
        "llm_model": llm_model,
        "stt_model": stt_model,
        "tts_voice": tts_voice,
    }


def event_record(moment: datetime, turn: int, record_type: str, text: str,
                 **extra) -> dict:
    """One event of the lesson.

    *turn* is the number of the phrase of the learner the event belongs to, and
    the reply to that phrase carries the same number: that is what joins a
    correction to the phrase it corrects, without a reader counting lines. A
    system record about no phrase in particular carries the number of the
    latest phrase before it.

    An *extra* field that is None is dropped, so a record never carries an
    empty measurement. A zero is kept: it is a measurement.
    """
    record = {
        "ts": timestamp(moment),
        "turn": turn,
        "type": record_type,
        "text": text,
    }
    record.update({key: value for key, value in extra.items()
                   if value is not None})
    return record


def to_jsonl(record: dict) -> str:
    """One record as the line that goes into the file, with its line break.

    ensure_ascii=False: the lesson is text in two languages, and a person who
    opens the file has to be able to read it.
    """
    return json.dumps(record, ensure_ascii=False) + "\n"


def to_markdown(meta: dict, records: Iterable[dict]) -> str:
    """The lesson as a page to read.

    Shows the lesson itself (the phrases, the corrections and the summary) and
    leaves the service records to the jsonl. The result always ends with one
    line break, so the file has no unfinished last line.
    """
    lines = _markdown_header(meta)
    for record in records:
        block = _markdown_block(record)
        if block:
            lines.append(block)
    return "\n\n".join(lines) + "\n"


def _markdown_header(meta: dict) -> List[str]:
    """The title of the view and the settings of the lesson under it."""
    started = meta.get("started_at", "")
    title = f"# Lesson {_readable_time(started)}".rstrip()
    settings = [
        f"- Language: {meta.get('target_language', '')}",
        f"- First topic: {meta.get('first_topic') or '(the prompt default)'}",
        f"- Chat model: {meta.get('llm_model', '')}",
        f"- Recognition: {meta.get('stt_model', '')}",
    ]
    return [title, "\n".join(settings)]


def _readable_time(started: str) -> str:
    """The time of the header, from the ISO 8601 string of the meta record.

    The string itself when it cannot be read: the header is a title, and a
    title is not worth an exception that would cost the whole file.
    """
    try:
        return datetime.fromisoformat(started).strftime(_MD_TIME_FORMAT)
    except ValueError:
        return started


def _markdown_block(record: dict) -> Optional[str]:
    """One record as a paragraph of the view, or None when it is not shown."""
    record_type = record.get("type")
    if record_type not in _MD_TYPES:
        return None
    text = record.get("text", "")
    if record_type == TYPE_LEARNER:
        return f"**{_MD_LEARNER}:** {text}"
    if record_type == TYPE_NOTE:
        return f"*{_MD_NOTE}: {text}*"
    if record_type == TYPE_SAY:
        return f"**{_MD_PARTNER}:** {text}"
    return f"## {_MD_SUMMARY}\n\n{text}"


class TranscriptWriter:
    """The two files of one lesson: the jsonl as it happens, the md at the end.

    One writer per lesson. No file is created before the first record.

    :meth:`add` and :meth:`save_markdown` may be called from any thread: the
    records of an exchange are written by the exchange thread and the service
    lines by the Tk thread, and one lock covers both the list and the files.

    A file that cannot be written costs the transcript and not the lesson: the
    failure is logged once and the writer stays quiet afterwards, because one
    message per event would fill the log with the same sentence.
    """

    def __init__(self, directory, started_at: datetime, meta: dict):
        """Prepare the files of the lesson that starts at *started_at*."""
        stem = file_stem(started_at)
        self.jsonl_path = Path(directory) / f"{stem}.jsonl"
        self.markdown_path = Path(directory) / f"{stem}.md"
        self._meta = meta
        self._records: List[dict] = []
        self._lock = threading.Lock()
        self._failed = False

    def add(self, record: dict) -> None:
        """Keep one record and append it to the jsonl file. (Any thread.)

        The meta line is written together with the first record, and not at
        construction.
        """
        with self._lock:
            lines = [] if self._records else [to_jsonl(self._meta)]
            self._records.append(record)
            lines.append(to_jsonl(record))
            self._append(lines)

    def save_markdown(self) -> None:
        """Write the readable view of everything recorded so far. (Any thread.)

        Called when the lesson ends (the SUMMARY) and again when the window
        closes, because the learner can speak after the summary. The file is
        replaced each time, which is why it is built from the records and not
        appended to.
        """
        with self._lock:
            if not self._records:
                return
            text = to_markdown(self._meta, self._records)
            try:
                self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
                self.markdown_path.write_text(text, encoding="utf-8")
            except OSError:
                logging.exception(
                    f"The transcript {self.markdown_path} was not written.")
                return
        logging.info(f"Transcript saved to {self.markdown_path}.")

    def _append(self, lines: List[str]) -> None:
        """Add lines to the jsonl file. (Called under the lock.)

        newline="\\n" on purpose: the file is a machine format and its readers
        must not meet a Windows line break in the middle of a record.
        """
        if self._failed:
            return
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8",
                      newline="\n") as transcript_file:
                transcript_file.writelines(lines)
        except OSError:
            self._failed = True
            logging.exception(
                f"The transcript {self.jsonl_path} cannot be written. "
                f"The lesson goes on without it.")
