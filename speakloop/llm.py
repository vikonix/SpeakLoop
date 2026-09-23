# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

import logging
import threading
from threading import Event
from typing import Optional

from openai import OpenAI

from speakloop import config

# Seconds the client waits for the server. For a streamed reply this is the
# longest pause between two chunks, and the longest pause is the prompt
# processing before the first token. On a weak machine that is about 44 s for
# the lesson prompt, and a full re-read of a 16384-token context at the
# measured 52.8 tokens per second is about 310 s (docs/model-parameters.md).
# A shorter value ends a reply that was about to come.
LLM_TIMEOUT = 360.0

# The client repeats a timed-out or failed request twice by default. A repeat
# costs the whole prompt processing again, and the server-start poll in
# llm_server_ctl already repeats on its own schedule, so none is made here.
LLM_MAX_RETRIES = 0

# Seconds check_connection waits for the model list. Much shorter than
# LLM_TIMEOUT: a loading llama-server answers 503 at once, and a program on
# the port that accepts the connection but never answers must not hold the
# startup for the six minutes a reply may take.
LLM_CHECK_TIMEOUT = 10.0

# Model name sent in every request. Both backends ignore it - llama-server
# serves the one GGUF it was started with and LM Studio the one it has loaded -
# but the OpenAI client requires the field, so it is a placeholder and not a
# setting.
PLACEHOLDER_MODEL = "local-model"


def error_message(error: Exception) -> str:
    """Short text of a failed request, for the window.

    str() of an OpenAI API error is the whole HTTP problem with the JSON body
    appended as a Python dict, which is unreadable in a chat window. The
    server's own sentence ("No models loaded...") is the only part the user
    can act on, so it is what this returns; the full text stays in the log,
    where it is the thing that helps.

    The body is read by attribute and not by exception type: every error the
    OpenAI client raises for an HTTP status carries the parsed JSON as .body,
    and a transport failure (no server listening) carries none, which is
    exactly when str(error) is already the short answer.

    Two shapes are read, because the client unwraps the outer "error" key
    before storing the body: {"message": ..., "type": ...} is what actually
    arrives here, and {"error": {"message": ...}} is the same answer as the
    server wrote it. An unwrapped plain string is the third.
    """
    detail = getattr(error, "body", None)
    if isinstance(detail, dict):
        detail = detail.get("message", detail.get("error"))
        if isinstance(detail, dict):
            detail = detail.get("message")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return str(error) or error.__class__.__name__


# The error type llama-server gives a request that does not fit the context.
CONTEXT_OVERFLOW_TYPE = "exceed_context_size_error"


class EmptyCutReplyError(RuntimeError):
    """The server stopped the reply at the length limit before any text.

    Near the end of the context the model can use the last free tokens
    without writing any text, so the context is as full as after a refusal.
    """


def is_context_overflow(error: Exception) -> bool:
    """True when the context is too full for an answer.

    A refusal of the server, or a reply cut off before any text
    (EmptyCutReplyError). llama-server names the refusal in the error type.
    Other servers (LM Studio) only say it in words, so the message is read
    too.
    """
    if isinstance(error, EmptyCutReplyError):
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error")
        error_type = (inner.get("type") if isinstance(inner, dict)
                      else body.get("type"))
        if error_type == CONTEXT_OVERFLOW_TYPE:
            return True
    text = error_message(error).lower()
    return "context" in text and any(
        word in text for word in ("exceed", "too long", "length", "overflow"))


def request_extra_body(backend: str):
    """Fields outside the OpenAI API to send with a chat request, or None.

    Only llama-server gets them, and both are llama.cpp request fields:

    - chat_template_kwargs.enable_thinking=False. llama-server passes
      enable_thinking=true to Gemma's chat template by default, and the model
      then thinks for 40 s or more before a one-line reply; the thinking goes
      to reasoning_content, which this module never reads
      (docs/model-parameters.md). A template without the
      variable ignores it, so the fallback model is not affected.
    - top_k, which the OpenAI API does not have.

    LM Studio gets None: it has its own thinking switch and model settings,
    and an unknown field there is a risk with nothing to gain.

    A new dict on every call, so no caller can change what the next request
    sends.
    """
    if backend != "llama-server":
        return None
    return {
        "top_k": config.LLM_TOP_K,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def usage_log_line(usage) -> str:
    """Log line about the context a finished request used.

    total_tokens is the whole conversation the server holds after the reply
    (system prompt, history, new message and reply), which is the number to
    compare with the context size. usage is None when the stream ended before
    the server sent its report, which is what an interrupt does.
    """
    if usage is None:
        return "Context tokens: unknown (no usage report in the stream)."
    return (f"Context tokens: {usage.total_tokens} "
            f"(prompt {usage.prompt_tokens}, reply {usage.completion_tokens}).")


class LLMManager:
    """OpenAI-compatible client with the conversation history of the lesson.

    Used by both backends. The history holds the system message and then
    user/assistant pairs, and is never trimmed (see ask()).
    """

    def __init__(self, model: str = None):
        self.client = None
        # Model name sent in API requests; see PLACEHOLDER_MODEL
        self.model = model or PLACEHOLDER_MODEL
        # The conversation: empty until start_conversation() sets the system
        # message, so no request can go out without the lesson prompt.
        self.messages = []
        # Protects self.messages from concurrent reads/writes across threads
        self._messages_lock = threading.Lock()
        # The context size the server reported for the last finished reply:
        # the whole conversation after that reply, in tokens. None when no
        # reply has finished yet, or when the last one was interrupted.
        self.last_total_tokens: Optional[int] = None
        # True when the last finished reply was cut off: the server stopped
        # it at the reply limit (max_tokens) or at the end of the context
        # (finish_reason "length"). Such a reply looks complete otherwise.
        self.last_reply_cut = False

    def start_conversation(self, system_prompt: str):
        """Begin a new conversation with *system_prompt* as its system message."""
        with self._messages_lock:
            self.messages = [{"role": "system", "content": system_prompt}]

    def init_client(self, base_url: str, api_key: str):
        """Point the client at the chat server. Sends no request."""
        logging.info(f"Initializing LLM client → {base_url}")
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=LLM_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )

    def check_connection(self, silent: bool = False) -> bool:
        """
        Validates connectivity to the local LLM server.

        Pass silent=True during startup polling to suppress per-attempt error logs
        and avoid flooding the log with dozens of identical connection errors.
        """
        try:
            if self.client is None:
                raise RuntimeError("LLM client not initialized. Call init_client() first.")
            self.client.models.list(timeout=LLM_CHECK_TIMEOUT)
            logging.info("Successfully connected to LLM server.")
            return True
        except Exception as error:
            if silent:
                logging.debug(f"LLM server not yet available: {error}")
            else:
                # Connection failures are expected (e.g. LM Studio offline) -
                # log the message only, not the full traceback.
                logging.error(f"LLM server not available: {error}")
            return False

    def ask(self, user_text: str, stop_event: Event) -> Optional[str]:
        """Send one user message and return the whole reply.

        The reply is returned only when it is complete: the caller splits it
        into NOTE / SAY / SUMMARY, which needs the whole text. It is still
        streamed from the server, for two reasons: closing the stream is what
        stops the server on an interrupt, and the last chunk carries the usage.

        Returns None when stop_event is set before the reply is returned. The
        learner never saw that reply, so the user message leaves the history
        with it: the history is then as it was before the call, and it never
        holds two user messages in a row, which a chat template may refuse.

        Raises when the request fails or the reply is empty, with the history
        rolled back in the same way.
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Call init_client() first.")
        with self._messages_lock:
            if not self.messages:
                raise RuntimeError("No conversation. Call start_conversation() first.")

        logging.info(f"LLM request started for user input: {user_text!r}")
        # Cleared before the request: a caller reads this number after a reply,
        # and the number of the reply before it would be worse than none.
        self.last_total_tokens = None
        self.last_reply_cut = False

        try:
            # Append user message and snapshot history for the API call.
            # Snapshot prevents the lock being held during the entire streaming operation.
            with self._messages_lock:
                self.messages.append({"role": "user", "content": user_text})
                messages_snapshot = list(self.messages)

            # A context manager and not a bare call: the loop below breaks out
            # of a stream the server is still generating into. Closing the
            # response is what tells the server to stop - without it the model
            # keeps producing an answer nobody will see, and the next request
            # waits behind it.
            with self.client.chat.completions.create(
                model=self.model,
                messages=messages_snapshot,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                top_p=config.LLM_TOP_P,
                stream=True,
                # A streamed reply has no usage unless it is asked for. With
                # this option the server sends one more chunk at the end: the
                # usage and an empty choices list. Both backends support it.
                stream_options={"include_usage": True},
                timeout=LLM_TIMEOUT,
                extra_body=request_extra_body(config.LLM_BACKEND),
            ) as stream_response:
                reply = ""
                usage = None
                finish_reason = None

                for chunk in stream_response:
                    if stop_event.is_set():
                        break

                    if chunk.usage is not None:
                        usage = chunk.usage
                    # The usage chunk has no choices; choices[0] would raise
                    # IndexError after the whole reply had arrived.
                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    reply += choice.delta.content or ""
                    finish_reason = (getattr(choice, "finish_reason", None)
                                     or finish_reason)

            # Checked after the stream and not only inside it: an event set
            # during the last chunk also means that nobody waits for the reply.
            if stop_event.is_set():
                logging.info("LLM reply interrupted by a stop event; "
                             "the exchange is removed from the history.")
                self._roll_back_user_message()
                return None

            final_reply = reply.strip()
            if not final_reply:
                # An error and not a stand-in text: a stand-in in the history
                # would show the model a reply outside the lesson contract.
                if finish_reason == "length":
                    raise EmptyCutReplyError(
                        "The reply was cut off before any text: the context "
                        "or the reply limit is full.")
                raise RuntimeError("The model returned an empty reply.")

            # The whole history is kept, without trimming. The lesson SUMMARY
            # needs its start, and a trimmed start changes the prompt prefix,
            # so the server processes the whole history again on every
            # request. A history that no
            # longer fits the context makes the server refuse the request,
            # and that error goes to the window like any other.
            with self._messages_lock:
                self.messages.append({"role": "assistant", "content": final_reply})

            logging.info(f"LLM full response: {final_reply!r}")
            logging.info(usage_log_line(usage))
            if usage is not None:
                self.last_total_tokens = usage.total_tokens
            if finish_reason == "length":
                self.last_reply_cut = True
                logging.warning("The reply was cut off at the reply limit or "
                                "at the end of the context.")
            return final_reply

        except Exception:
            self._roll_back_user_message()
            logging.exception("LLM request error:")
            # Raised, not answered with an apology string: what the user is
            # told about a failure is the window's decision.
            raise

    def _roll_back_user_message(self):
        """Remove the user message of a failed or interrupted request.

        Keeps the history in user/assistant pairs.
        """
        with self._messages_lock:
            if self.messages and self.messages[-1].get("role") == "user":
                self.messages.pop()
