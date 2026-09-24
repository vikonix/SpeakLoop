# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/ui.py.

The window itself needs a display and is checked by hand. What is testable are
the pure helpers: the geometry one, which decides where the window opens, and
the one that cleans a typed phrase before it is sent. The Space rule
(_typing) is tested on stand-ins for the root and the entry.

Importing ui.py creates no widget, so this file runs without a display.

Run from the project root with:

    python -m unittest tests.test_ui
"""

import tkinter as tk
import unittest
from types import SimpleNamespace

from speakloop import ui


class ViewDependencyTests(unittest.TestCase):
    """The window gets its settings from the controller (ViewSettings)."""

    def test_the_view_does_not_read_the_settings_itself(self):
        self.assertFalse(hasattr(ui, "config"))

    def test_the_view_does_not_read_the_prompt_commands_itself(self):
        self.assertFalse(hasattr(ui, "LESSON_COMMANDS"))
        self.assertFalse(hasattr(ui, "prompt"))

    def test_the_commands_are_kept_as_given(self):
        settings = ui.ViewSettings(lesson_language="English",
                                   show_notes=False,
                                   commands=("hint", "finish"))
        self.assertEqual(settings.commands, ("hint", "finish"))


class CenteredGeometryTests(unittest.TestCase):
    """Where the window opens on the screen."""

    def test_the_window_is_centered_on_a_common_screen(self):
        self.assertEqual(ui.centered_geometry(1920, 1080, 500, 700),
                         "500x700+710+190")

    def test_the_default_size_is_the_window_size(self):
        # The helper is called with the screen alone, so its defaults must be
        # the same constants build_ui would otherwise pass.
        self.assertEqual(ui.centered_geometry(1920, 1080),
                         ui.centered_geometry(1920, 1080,
                                              ui.WINDOW_WIDTH,
                                              ui.WINDOW_HEIGHT))

    def test_a_window_wider_than_the_screen_starts_at_the_left_edge(self):
        # A negative offset would put the title bar outside the screen, and the
        # user could no longer drag the window back.
        self.assertEqual(ui.centered_geometry(400, 1080, 500, 700),
                         "500x700+0+190")

    def test_a_window_taller_than_the_screen_starts_at_the_top_edge(self):
        self.assertEqual(ui.centered_geometry(1920, 600, 500, 700),
                         "500x700+710+0")

    def test_the_result_is_a_tk_geometry_string(self):
        # Tk accepts "<width>x<height>+<x>+<y>"; anything else raises inside
        # geometry() while the window is being built.
        self.assertRegex(ui.centered_geometry(1366, 768), r"^\d+x\d+\+\d+\+\d+$")


class CleanInputTests(unittest.TestCase):
    """What the text entry sends to the controller."""

    def test_a_typed_phrase_is_sent_as_it_is(self):
        self.assertEqual(ui.clean_input("I go to the store"),
                         "I go to the store")

    def test_the_ends_are_trimmed(self):
        self.assertEqual(ui.clean_input("  I go to the store  "),
                         "I go to the store")

    def test_a_pasted_phrase_becomes_one_line(self):
        # A phrase copied from another window can carry line breaks; the model
        # gets one line, like a recognized take.
        self.assertEqual(ui.clean_input("I go\nto   the\tstore"),
                         "I go to the store")

    def test_an_empty_entry_gives_an_empty_phrase(self):
        # The caller drops it: an empty request would still cost a model call.
        self.assertEqual(ui.clean_input(""), "")

    def test_spaces_alone_give_an_empty_phrase(self):
        self.assertEqual(ui.clean_input("   \n\t "), "")


class _StubEntry:
    """A stand-in for the text entry: only the state is read."""

    def __init__(self, state: str):
        self._state = state

    def cget(self, option: str) -> str:
        if option != "state":
            raise KeyError(option)
        return self._state


class TypingTests(unittest.TestCase):
    """_typing decides whether Space goes to the entry or to the recording."""

    @staticmethod
    def _view(entry_state: str, focus_on_entry: bool):
        entry = _StubEntry(entry_state)
        other_widget = object()
        focused = entry if focus_on_entry else other_widget
        root = SimpleNamespace(focus_get=lambda: focused)
        return SimpleNamespace(root=root, text_entry=entry)

    def test_an_open_entry_with_the_focus_takes_the_space(self):
        view = self._view(tk.NORMAL, focus_on_entry=True)
        self.assertTrue(ui.TutorView._typing(view))

    def test_a_closed_entry_with_the_focus_leaves_the_space_to_the_recording(self):
        # The entry is closed while the microphone is open, and it keeps the
        # focus: Space must still end the take.
        view = self._view(tk.DISABLED, focus_on_entry=True)
        self.assertFalse(ui.TutorView._typing(view))

    def test_an_open_entry_without_the_focus_leaves_the_space_to_the_recording(self):
        view = self._view(tk.NORMAL, focus_on_entry=False)
        self.assertFalse(ui.TutorView._typing(view))


if __name__ == "__main__":
    unittest.main()
