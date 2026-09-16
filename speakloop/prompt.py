# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""The lesson prompt: load the free-talk body and fill in its SETTINGS lines.

The body is speakloop/prompts/free_talk.md, the main copy of the prompt
(docs/refactoring.md, section 10.4). It is the system message of the whole
lesson and is built once per session: the server reuses the processed prompt
only while the start of the conversation stays the same.

The prompt has four places to fill in. Three are the SETTINGS lines, filled
here. The fourth is the NOTE example in OUTPUT, which is written in the
explanation language. The explanation language is fixed to Russian in this
version (config.EXPLANATION_LANGUAGE), so the example stays as the file has it.

No config import: the caller passes the values, so the tests need neither
config nor torch.
"""

import re
from pathlib import Path

# The SETTINGS lines, with the exact words of the prompt. A line renamed in the
# file must stop the start: otherwise the lesson silently runs on the default
# in brackets and looks correct.
SETTING_TARGET_LANGUAGE = "Target language"
SETTING_EXPLANATION_LANGUAGE = "Explanation language"
SETTING_FIRST_TOPIC = "First topic"
SETTING_NAMES = (
    SETTING_TARGET_LANGUAGE,
    SETTING_EXPLANATION_LANGUAGE,
    SETTING_FIRST_TOPIC,
)


def _setting_pattern(name: str) -> "re.Pattern":
    """A whole line "<name>: [<default>]"."""
    return re.compile(rf"^{re.escape(name)}: \[[^\]\n]*\][ \t]*$",
                      re.MULTILINE)


def fill_settings(body: str, values: dict) -> str:
    """The prompt body with the SETTINGS lines filled from *values*.

    *values* maps a name from SETTING_NAMES to its text. The value replaces the
    brackets, and does not stand beside them: the prompt reads the brackets as
    "use this when the line is empty". An empty or missing value leaves the
    line as it is, so the prompt takes its own default. Line breaks and runs of
    spaces in a value become one space, because each setting is one line.

    Raises RuntimeError when a SETTINGS line is missing from the body or is
    there more than once.
    """
    for name in SETTING_NAMES:
        pattern = _setting_pattern(name)
        found = len(pattern.findall(body))
        if found != 1:
            raise RuntimeError(
                f"The lesson prompt must have one line '{name}: [...]', "
                f"found {found}.")
        value = " ".join(str(values.get(name) or "").split())
        if value:
            # A function and not a replacement string: a backslash in the
            # value must not be read as a group reference.
            body = pattern.sub(lambda _match, line=f"{name}: {value}": line,
                               body)
    return body


def load_body(path) -> str:
    """The prompt body from *path*, without surrounding blank lines.

    Raises RuntimeError that names the file, because the window shows only
    the message of the error.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeError(
            f"Cannot read the lesson prompt {path}: {error}") from error
    body = text.strip()
    if not body:
        raise RuntimeError(f"The lesson prompt {path} is empty.")
    return body


def build_system_prompt(path, target_language: str,
                        explanation_language: str, first_topic: str) -> str:
    """The system message of this session: the body of *path*, filled in."""
    return fill_settings(load_body(path), {
        SETTING_TARGET_LANGUAGE: target_language,
        SETTING_EXPLANATION_LANGUAGE: explanation_language,
        SETTING_FIRST_TOPIC: first_topic,
    })
