# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""One lesson: the exchanges with the model and their parsed replies.

Lesson sits between the controller and LLMManager. It opens the lesson, sends
the learner's phrases, and returns every reply split by the contract
(speakloop/contract.py). The conversation history itself stays in LLMManager.

The commands of the prompt (simpler, hint, new topic, finish) get no special
handling: they go to the model as the learner gave them, like any other
phrase.
"""

import logging
import threading
from typing import Optional

from speakloop.contract import Reply, parse_reply

# The user message that opens the lesson. The prompt tells the model to begin
# with its first question, but the request still needs a user message after
# the system message: a chat template may refuse a conversation without one.
# It stays in the history like any other message and is never shown.
OPENING_MESSAGE = "Begin."


class Lesson:
    """The lesson of this session, over an LLMManager."""

    def __init__(self, llm, system_prompt: str):
        """Start a new conversation in *llm* with *system_prompt*."""
        self._llm = llm
        self._llm.start_conversation(system_prompt)

    def open(self, stop_event: threading.Event) -> Optional[Reply]:
        """Ask the model for the first question of the lesson."""
        logging.info("Opening the lesson.")
        return self._exchange(OPENING_MESSAGE, stop_event)

    def answer(self, learner_text: str,
               stop_event: threading.Event) -> Optional[Reply]:
        """Send a phrase of the learner and return the reply."""
        return self._exchange(learner_text, stop_event)

    def _exchange(self, text: str,
                  stop_event: threading.Event) -> Optional[Reply]:
        """One request. None when it was interrupted.

        A failed request raises what LLMManager.ask raises; the history is
        already rolled back then.
        """
        reply_text = self._llm.ask(text, stop_event)
        if reply_text is None:
            return None
        reply = parse_reply(reply_text)
        if not reply.follows_contract:
            logging.warning("The reply has neither SAY nor SUMMARY. "
                            "It is shown as it is and is not spoken.")
        return reply
