# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/ui.py.

The window itself needs a display and is checked by hand. What is testable is
the pure geometry helper, which decides where the window opens.

Importing ui.py creates no widget, so this file runs without a display.

Run from the project root with:

    python -m unittest tests.test_ui
"""

import unittest

from speakloop import ui


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


if __name__ == "__main__":
    unittest.main()
