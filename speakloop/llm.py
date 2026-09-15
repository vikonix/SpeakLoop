# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

import re
import logging
import threading
from queue import Queue
from threading import Event
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

# Compiled once at import time - splits on sentence-ending punctuation only when
# followed by an uppercase letter, avoiding false splits on "Mr. Smith" or "1.5 sec".
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-Я])')


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


def request_extra_body(backend: str):
    """Fields outside the OpenAI API to send with a chat request, or None.

    Only llama-server gets them, and both are llama.cpp request fields:

    - chat_template_kwargs.enable_thinking=False. llama-server passes
      enable_thinking=true to Gemma's chat template by default, and the model
      then thinks for 40 s or more before a one-line reply; the thinking goes
      to reasoning_content, which this module never reads
      (docs/model-parameters.md, section 4.8). A template without the
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


class LLMManager:
    def __init__(self, model: str = None):
        self.client = None
        # Model name sent in API requests; see PLACEHOLDER_MODEL
        self.model = model or PLACEHOLDER_MODEL
        # Chat history buffer starting with the system instructions
        self.messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        # Protects self.messages from concurrent reads/writes across threads
        self._messages_lock = threading.Lock()

    def init_client(self, base_url: str = None, api_key: str = None):
        """
        Configure OpenAI-compatible client.

        Defaults to LM Studio settings from config when arguments are omitted,
        so existing "lm-studio" backend usage is unchanged.
        """
        url = base_url or config.LM_STUDIO_URL
        key = api_key or config.LM_STUDIO_API_KEY
        logging.info(f"Initializing LLM client → {url}")
        self.client = OpenAI(
            base_url=url,
            api_key=key,
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

    def stream_and_queue_tts(self, user_text: str, tts_queue: Queue, stop_event: Event, token_callback=None) -> str:
        """
        Streams text from the LLM, parses sentences using regex on-the-fly,
        and pushes completed strings into the TTS queue.

        Raises the API error when the request fails, with the conversation
        history rolled back to the state before the call.
        """
        if self.client is None:
            raise RuntimeError("LLM client not initialized. Call init_client() first.")

        logging.info(f"LLM request started for user input: {user_text!r}")

        try:
            # Append user message and snapshot history for the API call.
            # Snapshot prevents the lock being held during the entire streaming operation.
            with self._messages_lock:
                self.messages.append({"role": "user", "content": user_text})
                messages_snapshot = list(self.messages)

            # A context manager and not a bare call: the loop below breaks out
            # of a stream the server is still generating into. Closing the
            # response is what tells the server to stop - without it the model
            # keeps producing an answer nobody will hear, and the next request
            # waits behind it.
            with self.client.chat.completions.create(
                model=self.model,
                messages=messages_snapshot,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                top_p=config.LLM_TOP_P,
                stream=True,
                timeout=LLM_TIMEOUT,
                extra_body=request_extra_body(config.LLM_BACKEND),
            ) as stream_response:
                full_reply = ""
                sentence_buffer = ""

                for chunk in stream_response:
                    if stop_event.is_set():
                        logging.info("LLM streaming interrupted by user stop event.")
                        break

                    token = chunk.choices[0].delta.content or ""
                    if not token:
                        continue

                    if token_callback:
                        token_callback(token)
                    full_reply += token
                    sentence_buffer += token

                    parts = _SENTENCE_END.split(sentence_buffer)
                    if len(parts) > 1:
                        sentence_buffer = parts.pop()
                        for item in parts:
                            text_to_speak = item.strip()
                            if text_to_speak:
                                logging.info(f"Queued sentence to TTS: {text_to_speak!r}")
                                tts_queue.put(text_to_speak)

            # Flush any residual text remaining inside the buffer
            remaining_text = sentence_buffer.strip()
            if remaining_text and not stop_event.is_set():
                logging.info(f"Queued final sentence segment to TTS: {remaining_text!r}")
                tts_queue.put(remaining_text)

            final_reply = full_reply.strip() if full_reply.strip() else "Sorry, I did not get a response."
            with self._messages_lock:
                self.messages.append({"role": "assistant", "content": final_reply})
                self._trim_history()

            logging.info(f"LLM full response: {final_reply!r}")
            return final_reply

        except Exception:
            # Roll back the user message so history stays consistent (user/assistant pairs)
            with self._messages_lock:
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
            logging.exception("LLM Stream error:")
            # Raised, not answered with an apology string: the caller streams
            # the tokens into the window itself and would otherwise show an
            # empty reply and a status bar that still says "Thinking". What the
            # user is told about a failure is the window's decision.
            raise

    def _trim_history(self):
        """Prunes conversation history to the most recent LLM_HISTORY_MAX_PAIRS turns.

        Must be called with self._messages_lock held.
        """
        system_message = self.messages[0]
        conversation = self.messages[1:]
        max_messages = config.LLM_HISTORY_MAX_PAIRS * 2
        self.messages = [system_message] + conversation[-max_messages:]
