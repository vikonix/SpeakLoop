# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/llm.py.

error_message, request_extra_body and usage_log_line are the pure parts of
the module: the first decides what a user reads in the chat window when a
request fails, the second what a request sends outside the OpenAI API (the
thinking switch of Gemma), the third what the log says about the context.
The streaming tests use a stand-in client to see that those fields reach the
request, that the usage chunk at the end of the stream is read, and that the
history is kept whole; how a real server answers belongs to the manual check
list.

The stubs below stand in for the errors the OpenAI client raises, which carry
the parsed JSON body as .body - building a real one would need an httpx
response and would test that library instead of this rule.

Run from the project root with:

    python -m unittest tests.test_llm
"""

import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from speakloop import config
from speakloop.llm import (
    LLMManager, error_message, request_extra_body, usage_log_line)


class ApiError(Exception):
    """An OpenAI API error as error_message sees it: a message and a body."""

    def __init__(self, message, body):
        super().__init__(message)
        self.body = body


LONG = ("Error code: 400 - {'error': {'message': 'No models loaded.', "
        "'type': 'invalid_request_error', 'param': 'model', 'code': None}}")


class ErrorMessageTests(unittest.TestCase):
    def test_the_servers_own_sentence_is_taken_from_the_body(self):
        # What LM Studio and llama-server both answer with.
        error = ApiError(LONG, {"error": {"message": "No models loaded."}})
        self.assertEqual(error_message(error), "No models loaded.")

    def test_a_plain_string_error_field_is_taken_too(self):
        # Not every OpenAI-compatible server nests the message.
        error = ApiError(LONG, {"error": "context window exceeded"})
        self.assertEqual(error_message(error), "context window exceeded")

    def test_surrounding_whitespace_is_dropped(self):
        error = ApiError(LONG, {"error": {"message": "  Out of memory.\n"}})
        self.assertEqual(error_message(error), "Out of memory.")

    def test_an_error_without_a_body_keeps_its_own_text(self):
        # A transport failure (nothing listening on the port) has no body,
        # and its own text is already short.
        self.assertEqual(error_message(ApiError("Connection error.", None)),
                         "Connection error.")

    def test_an_unexpected_body_shape_keeps_the_full_text(self):
        # Better a long message than no message: the body is not understood,
        # so nothing may be dropped from it.
        error = ApiError(LONG, {"detail": "something else"})
        self.assertEqual(error_message(error), LONG)

    def test_an_empty_message_falls_back_to_the_error_class(self):
        # "LLM error: " with nothing after it would say nothing at all.
        self.assertEqual(error_message(TimeoutError()), "TimeoutError")

    def test_an_ordinary_exception_is_passed_through(self):
        self.assertEqual(error_message(RuntimeError("no client")), "no client")


class RequestExtraBodyTests(unittest.TestCase):
    def test_llama_server_gets_thinking_off(self):
        # Without it Gemma thinks for 40 s before a one-line reply
        # (docs/model-parameters.md, section 4.8).
        body = request_extra_body("llama-server")
        self.assertIs(body["chat_template_kwargs"]["enable_thinking"], False)

    def test_llama_server_gets_top_k(self):
        self.assertEqual(request_extra_body("llama-server")["top_k"],
                         config.LLM_TOP_K)

    def test_lm_studio_gets_nothing(self):
        # LM Studio has its own switch; an unknown field is only a risk there.
        self.assertIsNone(request_extra_body("lm-studio"))

    def test_every_call_returns_a_new_dict(self):
        first = request_extra_body("llama-server")
        first["chat_template_kwargs"]["enable_thinking"] = True
        second = request_extra_body("llama-server")
        self.assertIs(second["chat_template_kwargs"]["enable_thinking"], False)


class UsageLogLineTests(unittest.TestCase):
    def test_the_line_gives_the_total_and_its_parts(self):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                                total_tokens=120)
        self.assertEqual(usage_log_line(usage),
                         "Context tokens: 120 (prompt 100, reply 20).")

    def test_a_missing_report_is_said_so(self):
        # An interrupted stream ends before the usage chunk.
        self.assertIn("unknown", usage_log_line(None))


def _chunk(text):
    """One streamed chunk, shaped like the part of the OpenAI answer llm.py reads."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
        usage=None)


def _usage_chunk(prompt_tokens, completion_tokens):
    """The last chunk of a stream with include_usage: usage and no choices."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              total_tokens=prompt_tokens + completion_tokens))


def _stream_of(chunks):
    """A stand-in for the stream context manager the client returns."""
    stream = MagicMock()
    stream.__enter__.return_value = iter(chunks)
    return stream


REPLY_CHUNKS = [_chunk("Hello there. "), _chunk("How are you?")]


class StreamRequestTests(unittest.TestCase):
    """The request stream_and_queue_tts sends, with a stand-in client."""

    def _stream(self, backend, chunks=REPLY_CHUNKS, stop_event=None):
        manager = LLMManager()
        manager.client = MagicMock()
        manager.client.chat.completions.create.return_value = _stream_of(chunks)
        tts_queue = Queue()
        with patch.object(config, "LLM_BACKEND", backend), \
                self.assertLogs(level="INFO") as logs:
            reply = manager.stream_and_queue_tts(
                "Hi", tts_queue, stop_event or threading.Event())
        kwargs = manager.client.chat.completions.create.call_args.kwargs
        self.log_lines = logs.output
        return reply, kwargs, tts_queue

    def test_llama_server_request_carries_the_extra_fields(self):
        _, kwargs, _ = self._stream("llama-server")
        self.assertEqual(kwargs["extra_body"],
                         request_extra_body("llama-server"))

    def test_lm_studio_request_has_no_extra_fields(self):
        _, kwargs, _ = self._stream("lm-studio")
        self.assertIsNone(kwargs["extra_body"])

    def test_the_sampling_values_come_from_config(self):
        _, kwargs, _ = self._stream("llama-server")
        self.assertEqual(kwargs["temperature"], config.LLM_TEMPERATURE)
        self.assertEqual(kwargs["top_p"], config.LLM_TOP_P)
        self.assertEqual(kwargs["max_tokens"], config.LLM_MAX_TOKENS)

    def test_the_reply_is_still_split_into_sentences(self):
        reply, _, tts_queue = self._stream("llama-server")
        self.assertEqual(reply, "Hello there. How are you?")
        self.assertEqual([tts_queue.get_nowait(), tts_queue.get_nowait()],
                         ["Hello there.", "How are you?"])

    def test_the_request_asks_for_the_usage(self):
        # Without the option a streamed reply carries no usage.
        _, kwargs, _ = self._stream("llama-server")
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})

    def test_the_usage_chunk_does_not_change_the_reply(self):
        # Its choices list is empty; reading choices[0] would raise.
        reply, _, tts_queue = self._stream(
            "llama-server", REPLY_CHUNKS + [_usage_chunk(100, 20)])
        self.assertEqual(reply, "Hello there. How are you?")
        self.assertEqual(tts_queue.qsize(), 2)

    def test_the_context_size_is_logged(self):
        self._stream("llama-server", REPLY_CHUNKS + [_usage_chunk(100, 20)])
        self.assertTrue(any("Context tokens: 120 (prompt 100, reply 20)."
                            in line for line in self.log_lines))

    def test_an_interrupted_stream_logs_an_unknown_size(self):
        # The stop event ends the loop before the usage chunk arrives.
        stop_event = threading.Event()
        stop_event.set()
        self._stream("llama-server", REPLY_CHUNKS + [_usage_chunk(100, 20)],
                     stop_event)
        self.assertTrue(any("Context tokens: unknown" in line
                            for line in self.log_lines))


class HistoryTests(unittest.TestCase):
    """The conversation history is kept whole (step 2c)."""

    EXCHANGES = 10

    def _talk(self):
        manager = LLMManager()
        manager.client = MagicMock()
        # A new stream for every request: an iterator can be read only once.
        manager.client.chat.completions.create.side_effect = (
            lambda **kwargs: _stream_of([_chunk("Fine.")]))
        with self.assertLogs(level="INFO"):
            for number in range(self.EXCHANGES):
                manager.stream_and_queue_tts(
                    f"Message {number}", Queue(), threading.Event())
        return manager

    def test_no_exchange_is_dropped(self):
        manager = self._talk()
        # The system prompt, then one user and one assistant message each.
        self.assertEqual(len(manager.messages), 1 + 2 * self.EXCHANGES)

    def test_the_first_message_stays_after_the_system_prompt(self):
        # A moving start would change the prompt prefix on every request.
        manager = self._talk()
        self.assertEqual(manager.messages[0]["role"], "system")
        self.assertEqual(manager.messages[1],
                         {"role": "user", "content": "Message 0"})

    def test_the_last_request_carries_the_whole_history(self):
        manager = self._talk()
        kwargs = manager.client.chat.completions.create.call_args.kwargs
        # All earlier pairs plus the new user message.
        self.assertEqual(len(kwargs["messages"]), 2 * self.EXCHANGES)

    def test_a_failed_request_is_rolled_back(self):
        manager = LLMManager()
        manager.client = MagicMock()
        manager.client.chat.completions.create.side_effect = (
            RuntimeError("context window exceeded"))
        with self.assertLogs(level="ERROR"), \
                self.assertRaises(RuntimeError):
            manager.stream_and_queue_tts("Hi", Queue(), threading.Event())
        self.assertEqual(len(manager.messages), 1)


if __name__ == "__main__":
    unittest.main()
