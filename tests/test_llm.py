# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/llm.py.

error_message, request_extra_body and usage_log_line are the pure parts of
the module: the first decides what a user reads in the chat window when a
request fails, the second what a request sends outside the OpenAI API (the
thinking switch of Gemma), the third what the log says about the context.
The ask() tests use a stand-in client to see that those fields reach the
request, that the usage chunk at the end of the stream is read, that an
interrupt leaves the history as it was, and that the history is kept whole;
how a real server answers belongs to the manual check list.

The stubs below stand in for the errors the OpenAI client raises, which carry
the parsed JSON body as .body - building a real one would need an httpx
response and would test that library instead of this rule.

Run from the project root with:

    python -m unittest tests.test_llm
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from speakloop import config
from speakloop.llm import (
    EmptyCutReplyError, LLMManager, error_message, is_context_overflow,
    request_extra_body, usage_log_line)


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


class ContextOverflowTests(unittest.TestCase):
    def test_the_llama_server_error_type_is_recognized(self):
        # The OpenAI client unwraps the outer "error" key.
        error = ApiError("400", {
            "code": 400, "type": "exceed_context_size_error",
            "message": "the request exceeds the available context size, "
                       "try increasing it"})
        self.assertTrue(is_context_overflow(error))

    def test_the_type_inside_an_error_key_is_recognized(self):
        error = ApiError("400", {"error": {
            "type": "exceed_context_size_error", "message": "full"}})
        self.assertTrue(is_context_overflow(error))

    def test_a_message_about_the_context_length_is_recognized(self):
        error = ApiError("400", {"error": {
            "message": "Context length exceeded. Trying to keep 20000 tokens"}})
        self.assertTrue(is_context_overflow(error))

    def test_another_error_is_not_an_overflow(self):
        error = ApiError(LONG, {"error": {"message": "No models loaded."}})
        self.assertFalse(is_context_overflow(error))

    def test_a_transport_error_is_not_an_overflow(self):
        self.assertFalse(is_context_overflow(ConnectionError("refused")))


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


def _chunk(text, finish_reason=None):
    """One streamed chunk, shaped like the part of the OpenAI answer llm.py reads."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text),
                                 finish_reason=finish_reason)],
        usage=None)


def _usage_chunk(prompt_tokens, completion_tokens):
    """The last chunk of a stream with include_usage: usage and no choices."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens,
                              total_tokens=prompt_tokens + completion_tokens))


def _stream_of(chunks, on_chunk=None):
    """A stand-in for the stream context manager the client returns.

    *on_chunk* runs before each chunk is given out, so a test can set a stop
    event in the middle of the stream.
    """
    def generate():
        for chunk in chunks:
            if on_chunk:
                on_chunk()
            yield chunk

    stream = MagicMock()
    stream.__enter__.return_value = generate()
    return stream


REPLY_CHUNKS = [_chunk("SAY: Hello there. "), _chunk("How are you?")]
SYSTEM_PROMPT = "You are a tutor."


def _manager():
    """An LLMManager with a stand-in client and a started conversation."""
    manager = LLMManager()
    manager.client = MagicMock()
    manager.start_conversation(SYSTEM_PROMPT)
    return manager


class StartConversationTests(unittest.TestCase):
    def test_the_history_starts_with_the_system_prompt(self):
        manager = LLMManager()
        manager.start_conversation(SYSTEM_PROMPT)
        self.assertEqual(manager.messages,
                         [{"role": "system", "content": SYSTEM_PROMPT}])

    def test_a_new_conversation_drops_the_old_one(self):
        manager = LLMManager()
        manager.start_conversation("first")
        manager.messages.append({"role": "user", "content": "Hi"})
        manager.start_conversation("second")
        self.assertEqual(manager.messages,
                         [{"role": "system", "content": "second"}])

    def test_ask_without_a_conversation_is_refused(self):
        # A request without the lesson prompt would be a lesson without rules.
        manager = LLMManager()
        manager.client = MagicMock()
        with self.assertRaises(RuntimeError):
            manager.ask("Hi", threading.Event())
        manager.client.chat.completions.create.assert_not_called()


class AskRequestTests(unittest.TestCase):
    """The request ask() sends and the text it returns, with a stand-in client."""

    def _ask(self, backend, chunks=REPLY_CHUNKS, stop_event=None,
             on_chunk=None):
        manager = _manager()
        manager.client.chat.completions.create.return_value = _stream_of(
            chunks, on_chunk)
        with patch.object(config, "LLM_BACKEND", backend), \
                self.assertLogs(level="INFO") as logs:
            reply = manager.ask("Hi", stop_event or threading.Event())
        kwargs = manager.client.chat.completions.create.call_args.kwargs
        self.log_lines = logs.output
        self.manager = manager
        return reply, kwargs

    def test_llama_server_request_carries_the_extra_fields(self):
        _, kwargs = self._ask("llama-server")
        self.assertEqual(kwargs["extra_body"],
                         request_extra_body("llama-server"))

    def test_lm_studio_request_has_no_extra_fields(self):
        _, kwargs = self._ask("lm-studio")
        self.assertIsNone(kwargs["extra_body"])

    def test_the_sampling_values_come_from_config(self):
        _, kwargs = self._ask("llama-server")
        self.assertEqual(kwargs["temperature"], config.LLM_TEMPERATURE)
        self.assertEqual(kwargs["top_p"], config.LLM_TOP_P)
        self.assertEqual(kwargs["max_tokens"], config.LLM_MAX_TOKENS)

    def test_the_whole_reply_is_returned_as_one_text(self):
        reply, _ = self._ask("llama-server")
        self.assertEqual(reply, "SAY: Hello there. How are you?")

    def test_the_request_asks_for_the_usage(self):
        # Without the option a streamed reply carries no usage.
        _, kwargs = self._ask("llama-server")
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})

    def test_the_request_starts_with_the_system_prompt(self):
        _, kwargs = self._ask("llama-server")
        self.assertEqual(kwargs["messages"],
                         [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": "Hi"}])

    def test_a_complete_reply_is_not_cut(self):
        self._ask("llama-server",
                  REPLY_CHUNKS + [_chunk("", finish_reason="stop")])
        self.assertFalse(self.manager.last_reply_cut)

    def test_a_reply_stopped_by_the_length_limit_is_cut(self):
        # The server stopped it at max_tokens or at the end of the context.
        self._ask("llama-server",
                  REPLY_CHUNKS + [_chunk("", finish_reason="length")])
        self.assertTrue(self.manager.last_reply_cut)

    def test_the_usage_chunk_does_not_change_the_reply(self):
        # Its choices list is empty; reading choices[0] would raise.
        reply, _ = self._ask("llama-server",
                             REPLY_CHUNKS + [_usage_chunk(100, 20)])
        self.assertEqual(reply, "SAY: Hello there. How are you?")

    def test_the_context_size_is_logged(self):
        self._ask("llama-server", REPLY_CHUNKS + [_usage_chunk(100, 20)])
        self.assertTrue(any("Context tokens: 120 (prompt 100, reply 20)."
                            in line for line in self.log_lines))

    def test_a_reply_without_a_usage_report_logs_an_unknown_size(self):
        self._ask("llama-server")
        self.assertTrue(any("Context tokens: unknown" in line
                            for line in self.log_lines))


class InterruptTests(unittest.TestCase):
    """A stop event set before the reply is complete."""

    def _interrupted(self, stop_event, on_chunk=None):
        manager = _manager()
        manager.client.chat.completions.create.return_value = _stream_of(
            REPLY_CHUNKS + [_usage_chunk(100, 20)], on_chunk)
        with self.assertLogs(level="INFO"):
            reply = manager.ask("Hi", stop_event)
        return manager, reply

    def test_an_event_set_before_the_request_gives_no_reply(self):
        stop_event = threading.Event()
        stop_event.set()
        _, reply = self._interrupted(stop_event)
        self.assertIsNone(reply)

    def test_an_event_set_during_the_stream_gives_no_reply(self):
        stop_event = threading.Event()
        _, reply = self._interrupted(stop_event, on_chunk=stop_event.set)
        self.assertIsNone(reply)

    def test_the_history_is_as_before_the_request(self):
        # The learner never saw the reply, and a user message left alone
        # would make two user messages in a row with the next request.
        stop_event = threading.Event()
        manager, _ = self._interrupted(stop_event, on_chunk=stop_event.set)
        self.assertEqual(manager.messages,
                         [{"role": "system", "content": SYSTEM_PROMPT}])


class FailureTests(unittest.TestCase):
    def test_a_failed_request_is_rolled_back_and_raised(self):
        manager = _manager()
        manager.client.chat.completions.create.side_effect = (
            RuntimeError("context window exceeded"))
        with self.assertLogs(level="ERROR"), \
                self.assertRaises(RuntimeError):
            manager.ask("Hi", threading.Event())
        self.assertEqual(len(manager.messages), 1)

    def test_an_empty_reply_is_an_error(self):
        # A stand-in text in the history would be a reply outside the
        # lesson contract.
        manager = _manager()
        manager.client.chat.completions.create.return_value = _stream_of(
            [_chunk("  "), _usage_chunk(100, 1)])
        with self.assertLogs(level="ERROR"), \
                self.assertRaises(RuntimeError):
            manager.ask("Hi", threading.Event())
        self.assertEqual(len(manager.messages), 1)

    def test_an_empty_reply_cut_off_is_a_full_context(self):
        # The model used the last free tokens and wrote no text: the window
        # must say that the context is full, not "empty reply".
        manager = _manager()
        manager.client.chat.completions.create.return_value = _stream_of(
            [_chunk(""), _chunk("", finish_reason="length")])
        with self.assertLogs(level="ERROR"), \
                self.assertRaises(EmptyCutReplyError) as caught:
            manager.ask("Hi", threading.Event())
        self.assertTrue(is_context_overflow(caught.exception))
        self.assertEqual(len(manager.messages), 1)


class HistoryTests(unittest.TestCase):
    """The conversation history is kept whole."""

    EXCHANGES = 10

    def _talk(self):
        manager = _manager()
        # A new stream for every request: a generator can be read only once.
        manager.client.chat.completions.create.side_effect = (
            lambda **kwargs: _stream_of([_chunk("SAY: Fine.")]))
        with self.assertLogs(level="INFO"):
            for number in range(self.EXCHANGES):
                manager.ask(f"Message {number}", threading.Event())
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
        # The system prompt, all earlier pairs and the new user message.
        self.assertEqual(len(kwargs["messages"]), 2 * self.EXCHANGES)


if __name__ == "__main__":
    unittest.main()
