# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Shared building blocks of the view layer: the palette and the fonts.

Everything here is window-agnostic, so ui.py can import it and this module
imports nothing of the view in return.

Importing it also applies the ttkbootstrap "autostyle" patch (see the comment at
the import below), so it must be imported before any Tk widget is created -
which holds automatically, because the only view module imports it at module
level.
"""

import platform

# ttkbootstrap is a drop-in replacement for tkinter.ttk (same widget classes,
# modern flat themes). Imported here so the autostyle patch below runs exactly
# once, before any widget exists.
from ttkbootstrap.style import Bootstyle

# On import ttkbootstrap patches the constructors of the classic tk widgets
# ("autostyle"): right after creation every widget is repainted with the base
# theme's colors, discarding the explicit THEME bg/fg the view passes (labels
# turn theme-blue, panels grey). The view themes every classic widget itself, so
# the hook is unwanted globally - including for widgets created inside libraries,
# which the per-widget ``autostyle=False`` flag cannot reach. Only the
# classic-widget hook is disabled; ttk widgets keep their ttkbootstrap styling.
# Verified against ttkbootstrap 1.x internals - see the version pin in
# pyproject.toml.
Bootstyle.update_tk_widget_style = staticmethod(lambda widget=None: None)

from speakloop import config  # noqa: E402  (after the patch above, on purpose)

# Resolved UI color palette (semantic name -> hex), selected by the
# "color_theme" setting in settings.json; see config.py.
THEME = config.THEME

# ttkbootstrap base theme per SpeakLoop color theme. The base theme supplies
# only the ttk widget geometry and elements (focus behaviour, scrollbar parts);
# every visible color is still set from THEME, so the palette keeps coming from
# the theme schemas.
_BOOTSTRAP_THEMES = {"dark": "darkly", "light": "flatly"}
BOOTSTRAP_THEME = _BOOTSTRAP_THEMES.get(config.COLOR_THEME, "darkly")

# UI font family, chosen per platform. "Segoe UI" exists only on Windows;
# without an explicit choice Tk would silently substitute an arbitrary font on
# other systems, so each platform gets its standard UI face instead.
_FONT_FAMILIES = {
    "Windows": "Segoe UI",
    "Darwin": "Helvetica Neue",   # macOS
}
FONT_FAMILY = _FONT_FAMILIES.get(platform.system(), "DejaVu Sans")  # Linux/other

# Typographic scale (Tk points), ordered large -> small. This is the single
# source of truth for every font size in the window: widgets pull from these
# names instead of hard-coding numbers, so the whole interface can be rescaled
# from one place. Sizes that share a value are still kept as separate,
# role-named constants so each can be tuned independently later.
FONT_SIZE_TITLE = 16     # header brand title ("EMMA - Voice Tutor")
FONT_SIZE_EMOJI = 20     # emoji glyph drawn on the round mic button
FONT_SIZE_CHAT = 11      # chat transcript
FONT_SIZE_BODY = 10      # [System] lines, the instruction under the button
FONT_SIZE_SMALL = 9      # status bar, language chip
