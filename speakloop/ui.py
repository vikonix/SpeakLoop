# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""View layer of the voice tutor (facade).

The window lives here as a standalone, passive :class:`TutorView`, composed into
the controller (``self.view``) in app.py rather than inherited. The view owns
every widget and every color and text of the interface; the controller owns the
application logic. Both directions across the boundary are explicit contracts,
so the two sides share no namespace:

* view -> controller: widget bindings call only ``self._cb.<handler>`` on a
  :class:`ViewCallbacks` of plain callables passed in at construction. The view
  never references the controller object, so the two form no cycle and the view
  can be driven with stand-in callbacks.
* controller -> view: the controller drives the window through named *intent*
  methods (``enter_recording``, ``enter_thinking``, ``enter_error`` ...). It
  never touches a widget, a color or a status string - every interface state and
  its wording live here.

Every method here must run on the Tk main thread: the controller's worker
threads reach them through ``root.after()``, never directly.
"""

import tkinter as tk
from dataclasses import dataclass
from typing import Callable

# Importing ui_theme also disables ttkbootstrap's classic-widget "autostyle"
# hook (see the comment there), so it must stay the first view import.
from speakloop.ui_theme import (
    BOOTSTRAP_THEME,
    FONT_FAMILY,
    FONT_SIZE_BODY,
    FONT_SIZE_CHAT,
    FONT_SIZE_EMOJI,
    FONT_SIZE_SMALL,
    FONT_SIZE_TITLE,
    THEME,
)

# ttkbootstrap is a drop-in replacement for tkinter.ttk (same widget classes,
# modern flat themes). Aliased as ``ttk`` so ttk.Style keeps working unchanged.
import ttkbootstrap as ttk

# The application name, as the title bar and the header show it.
APP_NAME = "SpeakLoop"

# The dialogue partner's label in the chat. A role and not a person's name:
# the lesson prompt gives the model no name, and a name in the window that the
# model does not know would contradict its answers. The controller never
# writes the label at all.
PARTNER_NAME = "Tutor"

# Labels of the two lines that are shown and never spoken.
NOTE_LABEL = "Note"
SUMMARY_LABEL = "Summary"

# The switch that hides and shows the Note lines of the whole lesson.
NOTES_BUTTON_LABEL = "Notes"

# The tags of a NOTE line. Hiding a note means eliding both of them, so the
# line disappears with its own line break and leaves no empty row behind.
_NOTE_TAGS = ("note", "text_note")

# The instruction line beside the mic button, per state. A take starts on one
# press and ends by itself after a pause, so the wording says press, never hold.
# It names the two ways to answer, because the text entry has no placeholder of
# its own (Tk has none, and a fake one has to be cleared on every focus change).
INSTRUCTION_LOADING = "Loading components..."
INSTRUCTION_READY_FIRST = "Press SPACE to speak, or type and press Enter. ESC quits."
INSTRUCTION_READY = "Press SPACE to speak, or type a phrase and press Enter."
INSTRUCTION_RECORDING = "Speak. Recording stops after a pause, or press again."
INSTRUCTION_SERVER_FAILED = "LLM server failed to start. Check the log and restart."

# Window size.
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 700

# Mic button geometry (canvas is 72x72, so the center is at 36,36). Small
# enough to stand beside the text entry in the control panel.
_MIC_CANVAS_SIZE = 72
_MIC_CENTER = 36
_MIC_R_OUTER = 30
_MIC_R_INNER = 24
_MIC_RING_WIDTH = 3

# Live-level mapping of the recording indicator: the outer ring stays at full
# radius and a solid red disc inside it grows with the input level, from
# _MIC_LEVEL_MIN_R up to the inner radius (just short of the ring). An input RMS
# at or above _MIC_LEVEL_FULL_RMS fills it to the inner radius; it never shrinks
# below the minimum radius, so the microphone stays visibly open in silence.
_MIC_LEVEL_FULL_RMS = 0.08
_MIC_LEVEL_MIN_R = 7


def clean_input(text: str) -> str:
    """The typed phrase as it is sent to the model.

    Line breaks and runs of spaces become one space and the ends are trimmed,
    so a phrase pasted from another window arrives as one line. An empty result
    means there is nothing to send; the caller drops it.

    Pure, so the rule can be tested without a display.
    """
    return " ".join(text.split())


def centered_geometry(screen_width: int, screen_height: int,
                      width: int = WINDOW_WIDTH,
                      height: int = WINDOW_HEIGHT) -> str:
    """The Tk geometry string that puts a window of *width* x *height* in the
    middle of a screen of *screen_width* x *screen_height*.

    Pure, so the arithmetic can be tested without a display. The offsets never
    go below zero: a window larger than the screen would otherwise be placed at
    a negative offset, which moves its title bar off the top edge and leaves the
    window impossible to drag back.
    """
    x = max((screen_width - width) // 2, 0)
    y = max((screen_height - height) // 2, 0)
    return f"{width}x{height}+{x}+{y}"


@dataclass(frozen=True)
class ViewSettings:
    """What the window shows that comes from outside it.

    lesson_language - the language of the lesson, named in the title and the
                      header.
    show_notes      - the first state of the Notes switch.
    commands        - the lesson commands; each button shows and sends one
                      of them exactly as given.
    """
    lesson_language: str
    show_notes: bool
    commands: tuple


@dataclass(frozen=True)
class ViewCallbacks:
    """Typed view->controller contract: the handlers the bindings invoke.

    The view stores only this bundle of callables (never the controller object),
    so the two sides share no implicit namespace. None of them take the Tk event:
    each binding is specific enough that the event carries nothing the controller
    needs (the space bindings are already space-only), and a handler without an
    event argument can also be called from the controller itself.

    There is no button-release handler: one press starts a take and the next one
    ends it, so a release means nothing. The KEY release is still reported, and
    only because holding a key makes Tk repeat KeyPress - the controller uses it
    to tell one physical press from the repeats.

    Three handlers carry a value, and the view never passes a widget: the text
    entry is read and cleaned here (on_text_submitted), a command button sends
    its own prompt word (on_command_pressed), and the Notes switch reports the
    state it has just taken (on_notes_toggled). The view hides the notes
    itself.
    """
    on_mic_pressed: Callable[[], None]
    on_space_pressed: Callable[[], None]
    on_space_released: Callable[[], None]
    on_text_submitted: Callable[[str], None]
    on_command_pressed: Callable[[str], None]
    on_notes_toggled: Callable[[bool], None]
    on_quit: Callable[[], None]


class TutorView:
    """Passive view facade: builds the window and renders its states.

    Owns the header, the control panel at the top (the mic button, the text
    entry and the instruction line), the chat transcript below it and the
    status bar. Widget bindings forward to the :class:`ViewCallbacks` passed in
    (``self._cb``); the controller drives the window through the intent methods
    below. The view holds no reference to the controller.
    """

    def __init__(self, root, callbacks: ViewCallbacks, settings: ViewSettings):
        """Build the window under ``root``, wiring the bindings to ``callbacks``.

        Args:
            root: the Tk root window the widgets are placed in.
            callbacks: the view->controller handlers the bindings invoke.
            settings: the language, the Notes state and the commands to show.
        """
        self.root = root
        self._cb = callbacks
        self._settings = settings
        self._notes_shown = settings.show_notes
        self.setup_styles()
        self.build_ui()
        self.bind_events()
        # After the chat tags exist: it elides the NOTE tags when the switch
        # comes up off, and paints the switch either way.
        self._apply_notes_visibility()
        # Second palette pass: a ttk widget created in build_ui makes
        # ttkbootstrap build its default style, which can override the colors
        # applied in setup_styles (see _apply_ttk_palette).
        self._apply_ttk_palette()

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------
    def setup_styles(self):
        # The first Style() instantiation applies the ttkbootstrap base theme;
        # later calls return the same singleton. All visible colors are then
        # overridden from THEME.
        self.style = ttk.Style(theme=BOOTSTRAP_THEME)
        self._apply_ttk_palette()

    def _apply_ttk_palette(self):
        """Apply the THEME colors to the ttk widget styles.

        Called once before the widgets are built and once after: ttkbootstrap
        builds a widget class's default style lazily when the first widget of
        that class is created, which would overwrite configure() calls made
        beforehand. The second pass makes the THEME colors win.

        The scrollbar entry styles a ttk scrollbar, of which this window has
        none yet: the chat's own is a classic tk one (see _build_chat). It is
        kept so the first ttk scrollbar added here is themed instead of arriving
        in the base theme's colors.
        """
        self.style.configure("Vertical.TScrollbar",
                             gripcount=0,
                             background=THEME["bg_panel"],
                             troughcolor=THEME["bg_main"],
                             bordercolor=THEME["bg_main"],
                             arrowcolor=THEME["accent"])

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_ui(self):
        # The window itself: title, size, and the background that shows through
        # wherever no widget covers it. The view owns the chrome, so the
        # controller needs to know neither the palette nor the wording.
        self.root.title(
            f"{APP_NAME} - {self._settings.lesson_language} Voice Tutor")
        # Size and position in one call, before the window is shown: the
        # winfo_screen* values are already valid at this point, so the window
        # appears in the middle of the screen instead of being drawn in a corner
        # and then jumping. On several monitors Tk reports the primary one,
        # which is where a window the user did not place belongs.
        self.root.geometry(centered_geometry(self.root.winfo_screenwidth(),
                                             self.root.winfo_screenheight()))
        self.root.configure(bg=THEME["bg_main"])
        self._build_header()
        self._build_status_bar()
        # The controls stand at the top, under the header: the learner answers
        # from there and reads the lesson below it, so the panel never moves
        # when the transcript grows.
        self._build_controls()
        # Last, and packed with expand=True: it takes whatever space the fixed
        # parts above and below have left.
        self._build_chat()

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=THEME["bg_main"], height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(10, 5))

        # The language of the lesson is part of the title; the explanation
        # language is fixed and not shown.
        tk.Label(header_frame,
                 text=f"{APP_NAME.upper()} • "
                      f"{self._settings.lesson_language} Voice Tutor",
                 font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                 fg=THEME["accent"], bg=THEME["bg_main"]).pack(side=tk.LEFT)

    def _build_status_bar(self):
        # The state of the window alone. The STT and LLM durations are in
        # logs/main.log.
        status_bar = tk.Frame(self.root, bg=THEME["bg_panel"], height=30)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(
            status_bar, text="Status: Starting...",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg=THEME["ready"], bg=THEME["bg_panel"])
        self.status_label.pack(side=tk.LEFT, padx=15, pady=4)

    def _build_controls(self):
        """The control panel at the top of the window: the mic button and the
        text entry in the first row, the lesson commands in the second."""
        control_frame = tk.Frame(self.root, bg=THEME["bg_panel"],
                                 highlightthickness=1,
                                 highlightbackground=THEME["border"])
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=5)

        input_row = tk.Frame(control_frame, bg=THEME["bg_panel"])
        input_row.pack(side=tk.TOP, fill=tk.X)

        # A Canvas and not a button: the five states are drawn (two circles and
        # an emoji), which no Tk button can show.
        self.btn_canvas = tk.Canvas(
            input_row, width=_MIC_CANVAS_SIZE, height=_MIC_CANVAS_SIZE,
            bg=THEME["bg_panel"], highlightthickness=0, cursor="hand2")
        self.btn_canvas.pack(side=tk.LEFT, padx=12, pady=12)
        self.draw_mic_button("loading")

        # The entry and the instruction share the space right of the button.
        entry_frame = tk.Frame(input_row, bg=THEME["bg_panel"])
        entry_frame.pack(side=tk.LEFT, fill=tk.X, expand=True,
                         padx=(0, 12), pady=12)

        # bg_main and not bg_panel: the field has to be visible against the
        # panel it lies on.
        self.text_entry = tk.Entry(
            entry_frame,
            bg=THEME["bg_main"],
            fg=THEME["text_bright"],
            disabledbackground=THEME["bg_main"],
            disabledforeground=THEME["text_muted"],
            insertbackground=THEME["text_bright"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT),
            bd=0,
            highlightthickness=1,
            highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
            state=tk.DISABLED,
        )
        self.text_entry.pack(side=tk.TOP, fill=tk.X, ipady=5)

        self.instruction_label = tk.Label(
            entry_frame, text=INSTRUCTION_LOADING,
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg=THEME["text_dim"], bg=THEME["bg_panel"], anchor=tk.W)
        self.instruction_label.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        self._build_command_row(control_frame)

    def _build_command_row(self, parent):
        """The lesson commands and the Notes switch, in one row.

        Each command button sends its own word exactly as ViewSettings gives
        it. The
        Notes switch stands apart, behind a separator: it sends nothing and
        stays usable in every state of the window.
        """
        command_row = tk.Frame(parent, bg=THEME["bg_panel"])
        command_row.pack(side=tk.TOP, fill=tk.X, padx=12, pady=(0, 12))

        # Packed before the commands: a widget packed to the right keeps its
        # place while the buttons left of it share what is left.
        self.notes_button = self._panel_button(
            command_row, NOTES_BUTTON_LABEL, self._toggle_notes)
        self.notes_button.pack(side=tk.RIGHT, padx=(6, 0))
        tk.Frame(command_row, bg=THEME["border"], width=1).pack(
            side=tk.RIGHT, fill=tk.Y, padx=6, pady=2)

        self.command_buttons = []
        for command in self._settings.commands:
            button = self._panel_button(
                command_row, command,
                # command=command binds this loop value; without it every
                # button would send the last command of the loop.
                lambda text=command: self._cb.on_command_pressed(text))
            button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
            button.configure(state=tk.DISABLED)
            self.command_buttons.append(button)

    def _panel_button(self, parent, text: str, command) -> tk.Button:
        """One flat button of the control panel, in the colors of the theme."""
        return tk.Button(
            parent, text=text, command=command,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            bg=THEME["bg_accent"], fg=THEME["text_dim"],
            activebackground=THEME["bg_accent"],
            activeforeground=THEME["text_bright"],
            disabledforeground=THEME["text_muted"],
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=THEME["border"],
            padx=6, pady=4, cursor="hand2")

    def _build_chat(self):
        chat_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=5)

        # A Text and a Scrollbar of our own rather than scrolledtext.
        # ScrolledText: that class hides its frame and scrollbar behind
        # undocumented attributes, so the pair is assembled here instead (the
        # way the Mimora panels do it), where both widgets can be themed and
        # reached by name.
        self.chat_display = tk.Text(
            chat_frame,
            bg=THEME["bg_panel"],
            fg=THEME["text"],
            insertbackground=THEME["text_bright"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT),
            wrap=tk.WORD,
            bd=0,
            highlightthickness=1,
            highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
            padx=15,
            pady=15,
            spacing2=6,
            spacing3=10,
        )
        scrollbar = tk.Scrollbar(chat_frame, orient=tk.VERTICAL,
                                 command=self.chat_display.yview)
        self.chat_display.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Read-only: the transcript is written by _append below, which lifts the
        # flag for the insert and puts it back.
        self.chat_display.configure(state=tk.DISABLED)

        self.chat_display.tag_configure(
            "user", foreground=THEME["info"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT, "bold"))
        self.chat_display.tag_configure(
            "partner", foreground=THEME["partner"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT, "bold"))
        self.chat_display.tag_configure(
            "system", foreground=THEME["text_muted"],
            font=(FONT_FAMILY, FONT_SIZE_BODY, "italic"))
        self.chat_display.tag_configure(
            "text_user", foreground=THEME["text_bright"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT))
        self.chat_display.tag_configure(
            "text_partner", foreground=THEME["text_emph"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT))
        # NOTE and SUMMARY take colors of existing palette keys, so a user
        # theme written before them still has every color it needs.
        self.chat_display.tag_configure(
            "note", foreground=THEME["warn"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT, "bold"))
        self.chat_display.tag_configure(
            "text_note", foreground=THEME["text_dim"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT))
        self.chat_display.tag_configure(
            "summary", foreground=THEME["good"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT, "bold"))

    def bind_events(self):
        # One press starts a take, the next one ends it. The bindings are
        # space-only, so the handlers need no keysym check; holding the key
        # repeats KeyPress, which the controller filters with the release
        # binding below.
        self.root.bind("<KeyPress-space>", self._on_space_press)
        self.root.bind("<KeyRelease-space>", self._on_space_release)
        self.btn_canvas.bind("<ButtonPress-1>", lambda _e: self._cb.on_mic_pressed())
        # Enter sends what was typed. The binding is on the entry itself, so
        # Enter means nothing anywhere else in the window.
        self.text_entry.bind("<Return>", lambda _e: self._submit_text())
        # Both ways out of the application end in the same controller handler.
        self.root.bind("<Escape>", lambda _e: self._cb.on_quit())
        self.root.protocol("WM_DELETE_WINDOW", self._cb.on_quit)

    def _typing(self) -> bool:
        """True while the keyboard belongs to the text entry.

        The space bindings sit on the root window and therefore also see the
        keys pressed inside the entry. Without this check a space typed in a
        phrase would start a recording instead of a space.
        """
        return self.root.focus_get() is self.text_entry

    def _on_space_press(self, _event):
        if self._typing():
            return
        self._cb.on_space_pressed()

    def _on_space_release(self, _event):
        if self._typing():
            return
        self._cb.on_space_released()

    def _submit_text(self):
        """Send the typed phrase to the controller and empty the entry.

        An entry that holds only spaces sends nothing: the window stays as it
        is, which is what the learner sees anyway.
        """
        phrase = clean_input(self.text_entry.get())
        if not phrase:
            return
        self.text_entry.delete(0, tk.END)
        self._cb.on_text_submitted(phrase)

    def _set_input_enabled(self, enabled: bool):
        """Open or close the text entry and the command buttons together.

        Closed while the microphone is open and while the model answers: one
        phrase at a time reaches the lesson, by voice, by keyboard or by
        button. The text already typed is kept, so a draft survives an
        exchange. The Notes switch is not part of this: it sends nothing.
        """
        state = tk.NORMAL if enabled else tk.DISABLED
        self.text_entry.configure(state=state)
        for button in self.command_buttons:
            button.configure(state=state)

    # ------------------------------------------------------------------
    # The Notes switch
    # ------------------------------------------------------------------
    def _toggle_notes(self):
        """Hide or show the corrections of the whole lesson. (Tk thread.)"""
        self._notes_shown = not self._notes_shown
        self._apply_notes_visibility()
        self._cb.on_notes_toggled(self._notes_shown)

    def _apply_notes_visibility(self):
        """Elide or reveal every NOTE line, and repaint the switch.

        The corrections are always written into the transcript: hiding them is
        a property of their tags, so one call covers the notes already in the
        chat and every note that arrives later. SUMMARY has tags of its own
        and is never hidden.
        """
        for tag in _NOTE_TAGS:
            self.chat_display.tag_configure(tag, elide=not self._notes_shown)
        self.notes_button.configure(
            fg=THEME["text_bright"] if self._notes_shown else THEME["text_muted"],
            highlightbackground=(THEME["accent"] if self._notes_shown
                                 else THEME["border"]))

    # ------------------------------------------------------------------
    # Mic button
    # ------------------------------------------------------------------
    def draw_mic_button(self, state: str):
        """Draw the round mic button in one of its five states.

        An unknown state draws the loading look with the idle glyph, so a new
        caller cannot blank the button.
        """
        palette = {
            "loading": (THEME["mic_loading_bg"], THEME["mic_loading_outline"], "⌛"),
            "idle": (THEME["bg_accent"], THEME["accent"], "🎤"),
            "recording": (THEME["mic_recording_bg"], THEME["bad"], "🔴"),
            "processing": (THEME["mic_processing_bg"], THEME["warn"], "⚡"),
            "speaking": (THEME["mic_speaking_bg"], THEME["good"], "🔊"),
        }
        bg_color, outline_color, emoji = palette.get(
            state, (THEME["mic_loading_bg"], THEME["mic_loading_outline"], "🎤"))

        cx = cy = _MIC_CENTER
        self.btn_canvas.delete("all")
        # Outer ring: the state color, drawn as an outline only.
        self.btn_canvas.create_oval(
            cx - _MIC_R_OUTER, cy - _MIC_R_OUTER,
            cx + _MIC_R_OUTER, cy + _MIC_R_OUTER,
            fill="", outline=outline_color, width=_MIC_RING_WIDTH)
        # Solid inner disc, then the glyph on top of it.
        self.btn_canvas.create_oval(
            cx - _MIC_R_INNER, cy - _MIC_R_INNER,
            cx + _MIC_R_INNER, cy + _MIC_R_INNER,
            fill=bg_color, outline="")
        self.btn_canvas.create_text(
            cx, cy, text=emoji, font=(FONT_FAMILY, FONT_SIZE_EMOJI),
            fill=THEME["text_bright"])

    def set_record_level(self, level: float):
        """Redraw the recording button with a fill driven by the input level.

        The take ends by itself after a pause, so a static red glyph would not
        tell the learner that the microphone hears them. The outer ring is drawn
        exactly as in the recording state (full radius, red); inside it a solid
        red disc follows the live level (``level`` is the RMS in 0..1 reported
        by the recorder): quiet gives a small disc (the automatic stop is near),
        louder fills it up to the inner radius, just short of the ring. Leaving
        the recording state is the next draw_mic_button call, which repaints the
        button from scratch.
        """
        cx = cy = _MIC_CENTER
        # Map the RMS to a 0..1 fraction, then to the disc radius; clamped, so a
        # loud peak cannot grow the fill past the inner radius into the ring.
        fraction = max(0.0, min(1.0, level / _MIC_LEVEL_FULL_RMS))
        r_level = _MIC_LEVEL_MIN_R + fraction * (_MIC_R_INNER - _MIC_LEVEL_MIN_R)

        self.btn_canvas.delete("all")
        # Outer ring: identical to draw_mic_button("recording") - fixed, red.
        self.btn_canvas.create_oval(
            cx - _MIC_R_OUTER, cy - _MIC_R_OUTER,
            cx + _MIC_R_OUTER, cy + _MIC_R_OUTER,
            fill="", outline=THEME["bad"], width=_MIC_RING_WIDTH)
        # Dark inner track, so the red fill is visible when it shrinks.
        self.btn_canvas.create_oval(
            cx - _MIC_R_INNER, cy - _MIC_R_INNER,
            cx + _MIC_R_INNER, cy + _MIC_R_INNER,
            fill=THEME["mic_recording_bg"], outline="")
        # The level indicator itself: a solid red disc from the minimum radius
        # to the inner radius. No glyph on top - the disc is the indicator, and
        # the bounding box of an emoji does not share the disc's center.
        self.btn_canvas.create_oval(
            cx - r_level, cy - r_level, cx + r_level, cy + r_level,
            fill=THEME["bad"], outline="")

    # ------------------------------------------------------------------
    # Chat transcript
    # ------------------------------------------------------------------
    def _append(self, *pieces):
        """Append (text, tag) pieces to the transcript and scroll to the end.

        The widget is read-only, so every write lifts the DISABLED flag and puts
        it back: one place for that dance instead of one per message kind. A tag
        of None inserts untagged text - Tk takes no tag argument at all then.
        """
        self.chat_display.configure(state=tk.NORMAL)
        for text, tag in pieces:
            if tag is None:
                self.chat_display.insert(tk.END, text)
            else:
                self.chat_display.insert(tk.END, text, tag)
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def append_system_msg(self, text: str):
        """Add a [System] line: progress, a warning or an error for the user."""
        self._append((f"[System] {text}\n", "system"))

    def append_user_msg(self, text: str):
        """Add what the learner said, as recognized."""
        self._append(("You: ", "user"), (f"{text}\n", "text_user"))

    def append_partner_msg(self, text: str):
        """Add what the partner says (the SAY line of a reply).

        Also used for a reply outside the contract, which is shown whole.
        """
        self._append((f"{PARTNER_NAME}: ", "partner"),
                     (f"{text}\n", "text_partner"))

    def append_note(self, text: str):
        """Add a correction (the NOTE line of a reply). It is never spoken."""
        self._append((f"{NOTE_LABEL}: ", "note"), (f"{text}\n", "text_note"))

    def append_summary(self, text: str):
        """Add the lesson summary. It may have several lines and is never spoken.

        The label stands on a line of its own, so the summary lines start at
        the same edge.
        """
        self._append((f"{SUMMARY_LABEL}:\n", "summary"),
                     (f"{text}\n", "text_partner"))

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def update_status(self, text: str, color: str = THEME["text_dim"]):
        self.status_label.configure(text=f"Status: {text}", fg=color)

    def update_instruction(self, text: str):
        self.instruction_label.configure(text=text)

    # ------------------------------------------------------------------
    # Startup intents (status line only: the button stays in the loading
    # state until the application is ready, so these leave it alone)
    # ------------------------------------------------------------------
    def enter_loading(self):
        """Speech models are loading."""
        self.update_status("Loading models...", THEME["warn"])

    def enter_connecting(self):
        """The LLM server is being started or looked for."""
        self.update_status("Connecting to LLM server...", THEME["warn"])

    def enter_warming_up(self):
        """The models are loaded and are being warmed up."""
        self.update_status("Warming up models...", THEME["warn"])

    def enter_app_ready(self):
        """Everything is loaded: the first idle state of the session.

        Separate from enter_idle only for its instruction, which names ESC once,
        when the user first has a reason to read it.
        """
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.update_instruction(INSTRUCTION_READY_FIRST)
        self._set_input_enabled(True)

    # ------------------------------------------------------------------
    # Exchange intents
    # ------------------------------------------------------------------
    def enter_idle(self):
        """Waiting for the learner to speak or to type."""
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.update_instruction(INSTRUCTION_READY)
        self._set_input_enabled(True)

    def enter_recording(self):
        """The microphone is open.

        The level indicator is drawn at once (at zero), so the button shows the
        open microphone from the first moment instead of waiting for the first
        level report from the capture thread.
        """
        self.set_record_level(0.0)
        self.update_status("Recording...", THEME["bad"])
        self.update_instruction(INSTRUCTION_RECORDING)
        self._set_input_enabled(False)

    def enter_processing(self):
        """The recording is being transcribed."""
        self.draw_mic_button("processing")
        self.update_status("Processing Speech (STT)...", THEME["warn"])
        self._set_input_enabled(False)

    def enter_thinking(self):
        """The model is answering.

        The button keeps the processing look: to the user this is one wait, and
        two glyphs for it would only flicker. The entry is closed here as well,
        because a typed phrase reaches this state without passing through
        enter_processing.
        """
        self.update_status("Thinking (LLM)...", THEME["info"])
        self.draw_mic_button("processing")
        self._set_input_enabled(False)

    def enter_speaking(self):
        """The partner's reply is being spoken.

        The entry is open: typing and Enter interrupt the speech, exactly as
        pressing SPACE does.
        """
        self.draw_mic_button("speaking")
        self.update_status(f"{PARTNER_NAME} is speaking...", THEME["partner"])
        self.update_instruction(INSTRUCTION_READY)
        self._set_input_enabled(True)

    # ------------------------------------------------------------------
    # Failure intents
    # ------------------------------------------------------------------
    def enter_error(self, status: str):
        """A failed exchange: say so and return the window to idle.

        The status text comes from the controller because only it knows which
        step failed; the color and the button are this module's business. What
        the user is told in the chat is a separate append_system_msg - an error
        stays in the transcript, while the status line is overwritten by the
        next state.
        """
        self.draw_mic_button("idle")
        self.update_status(status, THEME["bad"])
        self.update_instruction(INSTRUCTION_READY)
        self._set_input_enabled(True)

    def server_failed(self):
        """The LLM server did not start: the session cannot continue.

        The button deliberately stays in the loading state - there is nothing to
        press, and an idle mic would invite a recording that cannot be answered.
        The entry stays closed for the same reason.
        """
        self.update_status("LLM Server Error", THEME["bad"])
        self.update_instruction(INSTRUCTION_SERVER_FAILED)
        self._set_input_enabled(False)

    def init_failed(self):
        """Startup stopped on an unexpected error."""
        self.update_status("Initialization Failed", THEME["bad"])
        self._set_input_enabled(False)

    def recording_failed(self):
        """The microphone input stream failed: the take is gone.

        The button leaves the recording look, because nothing is being
        recorded any more and the level indicator would keep the last disc it
        drew on the screen.
        """
        self.draw_mic_button("idle")
        self.update_status("Recording Error", THEME["bad"])
        self.update_instruction(INSTRUCTION_READY)
        self._set_input_enabled(True)
