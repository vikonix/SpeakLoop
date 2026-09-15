# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/llm.py.

error_message and request_extra_body are the pure parts of the module: the
first decides what a user reads in the chat window when a request fails, the
second what a request sends outside the OpenAI API (the thinking switch of
Gemma). One test drives the streaming path with a stand-in client, only to
see that those fields reach the request; how a real server answers belongs to
the manual check list.

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
from speakloop.llm import LLMManager, error_message, request_extra_body


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


def _chunk(text):
    """One streamed chunk, shaped like the part of the OpenAI answer llm.py reads."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class StreamRequestTests(unittest.TestCase):
    """The request stream_and_queue_tts sends, with a stand-in client."""

    def _stream(self, backend):
        stream = MagicMock()
        stream.__enter__.return_value = iter(
            [_chunk("Hello there. "), _chunk("How are you?")])
        manager = LLMManager()
        manager.client = MagicMock()
        manager.client.chat.completions.create.return_value = stream
        tts_queue = Queue()
        with patch.object(config, "LLM_BACKEND", backend), \
                self.assertLogs(level="INFO"):
            reply = manager.stream_and_queue_tts(
                "Hi", tts_queue, threading.Event())
        kwargs = manager.client.chat.completions.create.call_args.kwargs
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


if __name__ == "__main__":
    unittest.main()
