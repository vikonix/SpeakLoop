# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/llm.py.

Only error_message is covered here: it is the pure part of the module, and it
decides what a user reads in the chat window when a request fails. The
streaming path needs a server and belongs to the manual check list.

The stubs below stand in for the errors the OpenAI client raises, which carry
the parsed JSON body as .body - building a real one would need an httpx
response and would test that library instead of this rule.

Run from the project root with:

    python -m unittest tests.test_llm
"""

import unittest

from speakloop.llm import error_message


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


if __name__ == "__main__":
    unittest.main()
