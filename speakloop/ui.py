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

from speakloop import config

# The dialogue partner's name, as the chat labels it. One spelling, here: the
# persona itself is set in the system prompt (config.SYSTEM_PROMPT), and the
# controller never writes the name at all.
PARTNER_NAME = "Emma"

# The instruction line under the mic button, per state. A take starts on one
# press and ends by itself after a pause, so the wording says press, never hold.
INSTRUCTION_LOADING = "Loading components..."
INSTRUCTION_READY_FIRST = "Press SPACE or the button to speak. Press ESC to quit."
INSTRUCTION_READY = "Press SPACE or the button to speak."
INSTRUCTION_RECORDING = "Speak. Recording stops after a pause, or press again."
INSTRUCTION_SERVER_FAILED = "LLM server failed to start. Check the log and restart."

# Window title and size. Here with the rest of the wording: the title carries
# the partner's name, which has one spelling in this module.
WINDOW_TITLE = f"{PARTNER_NAME} - Voice Tutor"
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 700

# Mic button geometry (canvas is 100x100, so the center is at 50,50).
_MIC_CANVAS_SIZE = 100
_MIC_CENTER = 50
_MIC_R_OUTER = 42
_MIC_R_INNER = 34
_MIC_RING_WIDTH = 3

# Live-level mapping of the recording indicator: the outer ring stays at full
# radius and a solid red disc inside it grows with the input level, from
# _MIC_LEVEL_MIN_R up to the inner radius (just short of the ring). An input RMS
# at or above _MIC_LEVEL_FULL_RMS fills it to the inner radius; it never shrinks
# below the minimum radius, so the microphone stays visibly open in silence.
_MIC_LEVEL_FULL_RMS = 0.08
_MIC_LEVEL_MIN_R = 10


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
    """
    on_mic_pressed: Callable[[], None]
    on_space_pressed: Callable[[], None]
    on_space_released: Callable[[], None]
    on_quit: Callable[[], None]


class TutorView:
    """Passive view facade: builds the window and renders its states.

    Owns the header, the chat transcript, the mic button and the status bar.
    Widget bindings forward to the :class:`ViewCallbacks` passed in
    (``self._cb``); the controller drives the window through the intent methods
    below. The view holds no reference to the controller.
    """

    def __init__(self, root, callbacks: ViewCallbacks):
        """Build the window under ``root``, wiring the bindings to ``callbacks``.

        Args:
            root: the Tk root window the widgets are placed in.
            callbacks: the view->controller handlers the bindings invoke.
        """
        self.root = root
        self._cb = callbacks
        self.setup_styles()
        self.build_ui()
        self.bind_events()
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
        self.root.title(WINDOW_TITLE)
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
        self._build_controls()
        # Last, and packed with expand=True: it takes whatever space the fixed
        # parts above and below have left.
        self._build_chat()

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=THEME["bg_main"], height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        tk.Label(header_frame, text=f"{PARTNER_NAME.upper()} • Voice Tutor",
                 font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
                 fg=THEME["accent"], bg=THEME["bg_main"]).pack(side=tk.LEFT)

        tk.Label(header_frame,
                 text=f"{config.NATIVE_LANGUAGE} ➔ {config.TARGET_LANGUAGE}",
                 font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
                 fg=THEME["text_dim"], bg=THEME["bg_panel"],
                 padx=10, pady=4, bd=0).pack(side=tk.RIGHT)

    def _build_status_bar(self):
        status_bar = tk.Frame(self.root, bg=THEME["bg_panel"], height=30)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(
            status_bar, text="Status: Starting...",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg=THEME["ready"], bg=THEME["bg_panel"])
        self.status_label.pack(side=tk.LEFT, padx=15, pady=4)

        self.stats_label = tk.Label(
            status_bar, text="STT: --ms | LLM: --ms",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg=THEME["text_dim"], bg=THEME["bg_panel"])
        self.stats_label.pack(side=tk.RIGHT, padx=15, pady=4)

    def _build_controls(self):
        control_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

        # A Canvas and not a button: the five states are drawn (two circles and
        # an emoji), which no Tk button can show.
        self.btn_canvas = tk.Canvas(
            control_frame, width=_MIC_CANVAS_SIZE, height=_MIC_CANVAS_SIZE,
            bg=THEME["bg_main"], highlightthickness=0, cursor="hand2")
        self.btn_canvas.pack(pady=5)
        self.draw_mic_button("loading")

        self.instruction_label = tk.Label(
            control_frame, text=INSTRUCTION_LOADING,
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg=THEME["text_dim"], bg=THEME["bg_main"])
        self.instruction_label.pack(pady=5)

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

    def bind_events(self):
        # One press starts a take, the next one ends it. The bindings are
        # space-only, so the handlers need no keysym check; holding the key
        # repeats KeyPress, which the controller filters with the release
        # binding below.
        self.root.bind("<KeyPress-space>", lambda _e: self._cb.on_space_pressed())
        self.root.bind("<KeyRelease-space>", lambda _e: self._cb.on_space_released())
        self.btn_canvas.bind("<ButtonPress-1>", lambda _e: self._cb.on_mic_pressed())
        # Both ways out of the application end in the same controller handler.
        self.root.bind("<Escape>", lambda _e: self._cb.on_quit())
        self.root.protocol("WM_DELETE_WINDOW", self._cb.on_quit)

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

    def append_reply_start(self):
        """Open the partner's line, before the first token of the reply."""
        self._append((f"{PARTNER_NAME}: ", "partner"))

    def append_reply_token(self, token: str):
        """Add one streamed token to the open partner line."""
        self._append((token, "text_partner"))

    def append_reply_end(self):
        """Close the partner's line.

        Called for a failed exchange too: a line opened by append_reply_start
        must be closed, or the next message continues it.
        """
        self._append(("\n", None))

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def update_status(self, text: str, color: str = THEME["text_dim"]):
        self.status_label.configure(text=f"Status: {text}", fg=color)

    def update_instruction(self, text: str):
        self.instruction_label.configure(text=text)

    def update_stats(self, stt_ms: float, llm_ms: float):
        """Show the durations of the last exchange, in milliseconds."""
        self.stats_label.configure(
            text=f"STT: {stt_ms:.0f}ms | LLM: {llm_ms:.0f}ms")

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

    # ------------------------------------------------------------------
    # Exchange intents
    # ------------------------------------------------------------------
    def enter_idle(self):
        """Waiting for the learner to speak."""
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.update_instruction(INSTRUCTION_READY)

    def enter_recording(self):
        """The microphone is open.

        The level indicator is drawn at once (at zero), so the button shows the
        open microphone from the first moment instead of waiting for the first
        level report from the capture thread.
        """
        self.set_record_level(0.0)
        self.update_status("Recording...", THEME["bad"])
        self.update_instruction(INSTRUCTION_RECORDING)

    def enter_processing(self):
        """The recording is being transcribed."""
        self.draw_mic_button("processing")
        self.update_status("Processing Speech (STT)...", THEME["warn"])

    def enter_thinking(self):
        """The model is answering.

        The button keeps the processing look: to the user this is one wait, and
        two glyphs for it would only flicker.
        """
        self.update_status("Thinking (LLM)...", THEME["info"])

    def enter_speaking(self):
        """The partner's reply is being spoken."""
        self.draw_mic_button("speaking")
        self.update_status(f"{PARTNER_NAME} is speaking...", THEME["partner"])

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

    def server_failed(self):
        """The LLM server did not start: the session cannot continue.

        The button deliberately stays in the loading state - there is nothing to
        press, and an idle mic would invite a recording that cannot be answered.
        """
        self.update_status("LLM Server Error", THEME["bad"])
        self.update_instruction(INSTRUCTION_SERVER_FAILED)

    def init_failed(self):
        """Startup stopped on an unexpected error."""
        self.update_status("Initialization Failed", THEME["bad"])

    def recording_failed(self):
        """The microphone input stream failed: the take is gone.

        The button leaves the recording look, because nothing is being
        recorded any more and the level indicator would keep the last disc it
        drew on the screen.
        """
        self.draw_mic_button("idle")
        self.update_status("Recording Error", THEME["bad"])
        self.update_instruction(INSTRUCTION_READY)
