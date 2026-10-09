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
import tkinter.font as tkfont
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
    FONT_SIZE_QUESTION,
    FONT_SIZE_SMALL,
    THEME,
)

# ttkbootstrap is a drop-in replacement for tkinter.ttk (same widget classes,
# modern flat themes). Aliased as ``ttk`` so ttk.Style keeps working unchanged.
import ttkbootstrap as ttk

# The application name, as the title bar shows it.
APP_NAME = "SpeakLoop"

# The dialogue partner's label in the chat. A role and not a person's name:
# the lesson prompt gives the model no name, and a name in the window that the
# model does not know would contradict its answers. The controller never
# writes the label at all.
PARTNER_NAME = "Tutor"

# The learner's label and the label of a service line in the chat.
USER_LABEL = "You"
SYSTEM_LABEL = "System"

# Labels of the two lines that are shown and never spoken. A NOTE is a
# correction of the learner's last phrase, hence "Fix".
NOTE_LABEL = "Fix"
SUMMARY_LABEL = "Summary"

# Chat layout, in pixels. Every line starts with its label in one column and
# the text starts at one tab stop after the widest label, so the eye reads the
# text straight down. _BAND_PADDING is the inner left edge of a Fix or Summary
# band; _LABEL_GAP is the space between the widest label and the text.
_BAND_PADDING = 12
_LABEL_GAP = 12

# The question block above the control panel. The caption names who asks only
# while the block holds a question of the tutor; the other texts are states of
# the lesson, and "TUTOR ASKS" over them would be wrong.
QUESTION_CAPTION_ASKED = "TUTOR ASKS"
# Beside the caption of a question, when the program knows the topic.
TOPIC_CAPTION = "/  TOPIC: {topic}"
QUESTION_CAPTION_STATE = "LESSON"
QUESTION_LOADING = "Loading the lesson..."
QUESTION_FAILED = "The lesson cannot start."
QUESTION_FINISHED = "Lesson finished."

# The stages of one exchange, in the order the bar above the question shows
# them. Speech recognition and the model's answer are one wait for the learner,
# so both are Thinking. While the application loads, or after it failed to
# start, no stage is active.
STAGE_READY = "Ready"
STAGE_LISTENING = "Listening"
STAGE_THINKING = "Thinking"
STAGE_SPEAKING = "Speaking"
STAGES = (STAGE_READY, STAGE_LISTENING, STAGE_THINKING, STAGE_SPEAKING)

# The switch that hides and shows the Note lines of the whole lesson. The
# state is in the label itself (a ticked or an empty box): a tk.Button on
# Windows draws no highlight border, so a colored outline could not show it.
NOTES_LABEL_SHOWN = "\u2611 Notes"
NOTES_LABEL_HIDDEN = "\u2610 Notes"

# The tags of a NOTE line. Hiding a note means eliding both of them, so the
# line disappears with its own line break and leaves no empty row behind.
_NOTE_TAGS = ("note", "text_note")

# The instruction line beside the mic button, per state. A take starts on one
# press and ends by itself after a pause, so the wording says press, never hold.
# It names the two ways to answer, because the text entry has no placeholder of
# its own (Tk has none, and a fake one has to be cleared on every focus change).
# Empty while loading: the question block already says "Loading the
# lesson...", and a second loading line under the entry only repeated it.
# The label keeps its height, so the panel does not move when the ready
# instruction arrives.
INSTRUCTION_LOADING = ""
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

    Owns the question block at the top (the tutor's current question), the
    control panel under it (the mic button, the text
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
        # The topic the next question belongs to (set_topic). None until the
        # controller has one.
        self._topic = None
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

        The scrollbar entry styles the chat's scrollbar (see _build_chat).
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
        # No header: the title bar of the window already names the
        # application, and the space goes to the stage bar and the current
        # question.
        self._build_stage_bar()
        self._build_question()
        self._build_status_bar()
        # The controls stand under the question: the learner answers
        # from there and reads the lesson below it, so the panel never moves
        # when the transcript grows.
        self._build_controls()
        # Last, and packed with expand=True: it takes whatever space the fixed
        # parts above and below have left.
        self._build_chat()

    def _build_stage_bar(self):
        """The stages of the exchange in one row, the active one marked.

        Labels and not buttons: nothing here can be clicked, and the default
        cursor tells the learner so. The status bar at the bottom stays: it
        also shows the loading steps and the errors, which have no stage.
        """
        stage_bar = tk.Frame(self.root, bg=THEME["bg_main"])
        stage_bar.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(10, 0))

        self._stage_labels = {}
        self._stage_lines = {}
        last_column = len(STAGES) - 1
        for column, stage in enumerate(STAGES):
            # The same "uniform" group as the command buttons: equal columns
            # whatever the length of the word.
            stage_bar.columnconfigure(column, weight=1, uniform="stage")
            gap = (0, 0 if column == last_column else 6)
            label = tk.Label(stage_bar, text=stage, bg=THEME["bg_main"],
                             anchor=tk.W)
            label.grid(row=0, column=column, sticky="ew", padx=gap)
            line = tk.Frame(stage_bar, height=3)
            line.grid(row=1, column=column, sticky="ew", padx=gap, pady=(3, 0))
            self._stage_labels[stage] = label
            self._stage_lines[stage] = line

        self._set_stage(None)

    def _set_stage(self, active):
        """Mark the stage *active* in the stage bar; None marks no stage."""
        for stage in STAGES:
            is_active = stage == active
            self._stage_labels[stage].configure(
                fg=THEME["text_bright"] if is_active else THEME["text_dim"],
                font=(FONT_FAMILY, FONT_SIZE_SMALL,
                      "bold" if is_active else "normal"))
            self._stage_lines[stage].configure(
                bg=THEME["accent"] if is_active else THEME["border"])

    def _build_question(self):
        """The tutor's current question, large, above the control panel.

        In the chat the question is one line among many, and after a few
        exchanges the learner has to look for it. Here it stays in one place.
        """
        question_frame = tk.Frame(self.root, bg=THEME["bg_main"])
        question_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=(12, 6))

        # One row: who asks, in the accent, and the topic beside it, dim.
        caption_row = tk.Frame(question_frame, bg=THEME["bg_main"])
        caption_row.pack(side=tk.TOP, fill=tk.X)
        self.question_caption = tk.Label(
            caption_row, font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
            fg=THEME["accent"], bg=THEME["bg_main"], anchor=tk.W)
        self.question_caption.pack(side=tk.LEFT)
        self.topic_caption = tk.Label(
            caption_row, font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg=THEME["text_dim"], bg=THEME["bg_main"], anchor=tk.W)
        self.topic_caption.pack(side=tk.LEFT, padx=(6, 0))

        self.question_label = tk.Label(
            question_frame, font=(FONT_FAMILY, FONT_SIZE_QUESTION, "bold"),
            bg=THEME["bg_main"], anchor=tk.W, justify=tk.LEFT)
        self.question_label.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))
        # A Label wraps only at a fixed pixel width, so the width follows the
        # frame; without it a long question would be cut at the window edge.
        question_frame.bind(
            "<Configure>",
            lambda event: self.question_label.configure(wraplength=event.width))

        self._set_question(QUESTION_LOADING, asked=False)

    def _set_question(self, text: str, asked: bool):
        """Show a question of the tutor (asked=True) or a state of the lesson.

        The topic stands beside a question only: "Lesson finished." belongs to
        no topic.
        """
        self.question_caption.configure(
            text=QUESTION_CAPTION_ASKED if asked else QUESTION_CAPTION_STATE)
        self.topic_caption.configure(
            text=(TOPIC_CAPTION.format(topic=self._topic.upper())
                  if asked and self._topic else ""))
        self.question_label.configure(
            text=text, fg=THEME["text_bright"] if asked else THEME["text_dim"])

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

        # A grid and not pack: the "uniform" group gives every command column
        # the same width, while pack(expand=True) shares only the free space,
        # so a longer word ("new topic") made its button wider.
        self.command_buttons = []
        for column, command in enumerate(self._settings.commands):
            button = self._panel_button(
                command_row, command,
                # command=command binds this loop value; without it every
                # button would send the last command of the loop.
                lambda text=command: self._cb.on_command_pressed(text))
            button.grid(row=0, column=column, sticky="ew", padx=(0, 6))
            command_row.columnconfigure(column, weight=1, uniform="command")
            button.configure(state=tk.DISABLED)
            self.command_buttons.append(button)

        # The separator and the switch keep their own width; only the command
        # columns left of them grow with the window.
        notes_column = len(self.command_buttons)
        tk.Frame(command_row, bg=THEME["border"], width=1).grid(
            row=0, column=notes_column, sticky="ns", padx=6, pady=2)
        self.notes_button = self._panel_button(
            command_row, NOTES_LABEL_SHOWN, self._toggle_notes)
        self.notes_button.grid(row=0, column=notes_column + 1, padx=(6, 0))

    def _panel_button(self, parent, text: str, command) -> tk.Button:
        """One flat button of the control panel, in the colors of the theme.

        The neutral "border" fill stands out from the panel, which the
        accent-tinted fill did not: an enabled button looked disabled. The
        accent tint is kept for the hover and the press.
        """
        button = tk.Button(
            parent, text=text, command=command,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            bg=THEME["border"], fg=THEME["text"],
            activebackground=THEME["bg_accent"],
            activeforeground=THEME["text_bright"],
            disabledforeground=THEME["text_muted"],
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=THEME["border"],
            padx=6, pady=4, cursor="hand2")
        # Tk buttons have no hover color of their own. A disabled button gets
        # no hover, so it does not look clickable.
        button.bind("<Enter>", lambda _e: self._hover_button(button, True))
        button.bind("<Leave>", lambda _e: self._hover_button(button, False))
        return button

    @staticmethod
    def _hover_button(button: tk.Button, inside: bool):
        """Paint a panel button for the mouse over it, or back to its rest look."""
        if inside and str(button.cget("state")) == tk.DISABLED:
            return
        button.configure(bg=THEME["bg_accent"] if inside else THEME["border"],
                         fg=THEME["text_bright"] if inside else THEME["text"])

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
            # A wrapped line stays close to its first line (spacing2) and a new
            # line of the chat gets more room (spacing1 + spacing3): with the
            # two nearly equal a wrapped word read as a new line.
            spacing1=4,
            spacing2=2,
            spacing3=6,
        )
        # A ttk scrollbar: the classic tk one is drawn by Windows itself, light
        # grey whatever colors it is given. The style is set in
        # _apply_ttk_palette.
        scrollbar = ttk.Scrollbar(chat_frame, orient=tk.VERTICAL,
                                  command=self.chat_display.yview,
                                  style="Vertical.TScrollbar")
        self.chat_display.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Read-only: the transcript is written by _append below, which lifts the
        # flag for the insert and puts it back.
        self.chat_display.configure(state=tk.DISABLED)

        # The text column: one tab stop after the widest label (the Fix label
        # also has the band padding before it). Measured in the label font,
        # so it follows the font and the screen scaling. Tk sets the indent of
        # a wrapped line from the tag of its first character, which is always
        # text, so lmargin2 goes on the text tags and not on the labels.
        # Each label is measured in the font it is drawn with: the System label
        # is smaller and not bold, and measured in the chat label font it was
        # the widest one and made the column too wide.
        label_font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_CHAT,
                                 weight="bold")
        system_font = tkfont.Font(family=FONT_FAMILY, size=FONT_SIZE_BODY)
        label_width = max(
            max(label_font.measure(label)
                for label in (PARTNER_NAME, USER_LABEL, NOTE_LABEL)),
            system_font.measure(SYSTEM_LABEL))
        text_column = _BAND_PADDING + label_width + _LABEL_GAP
        self.chat_display.configure(tabs=(text_column,))

        self.chat_display.tag_configure(
            "user", foreground=THEME["info"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT, "bold"))
        # The accent and not the "partner" pink: one brand color for the tutor
        # in the question block and in the chat.
        self.chat_display.tag_configure(
            "partner", foreground=THEME["accent"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT, "bold"))
        # Upright and not italic: italic Segoe UI at the body size is hard to
        # read in a muted color.
        self.chat_display.tag_configure(
            "system_label", foreground=THEME["text_muted"],
            font=(FONT_FAMILY, FONT_SIZE_BODY))
        self.chat_display.tag_configure(
            "system", foreground=THEME["text_muted"],
            font=(FONT_FAMILY, FONT_SIZE_BODY), lmargin2=text_column)
        self.chat_display.tag_configure(
            "text_user", foreground=THEME["text_bright"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT), lmargin2=text_column)
        self.chat_display.tag_configure(
            "text_partner", foreground=THEME["text_emph"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT), lmargin2=text_column)
        # NOTE and SUMMARY take colors of existing palette keys, so a user
        # theme written before them still has every color it needs.
        # Both are drawn as bands: the label and the text share the
        # background, and lmargincolor paints the left margin too, so the band
        # starts at the edge and the text has room inside it.
        band = dict(background=THEME["bg_accent"],
                    lmargincolor=THEME["bg_accent"],
                    lmargin1=_BAND_PADDING, lmargin2=_BAND_PADDING,
                    rmargin=_BAND_PADDING)
        self.chat_display.tag_configure(
            "note", foreground=THEME["warn"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT, "bold"), **band)
        self.chat_display.tag_configure(
            "text_note", foreground=THEME["text"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT),
            **{**band, "lmargin2": text_column})
        # The Summary label stands on a line of its own, so its text keeps the
        # band padding and not the text column.
        self.chat_display.tag_configure(
            "summary", foreground=THEME["good"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT, "bold"), **band)
        self.chat_display.tag_configure(
            "text_summary", foreground=THEME["text_emph"],
            font=(FONT_FAMILY, FONT_SIZE_CHAT), **band)

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

        A closed entry keeps the focus it had (Tk does not move it), so the
        focus alone is not enough: without the state check Space could not
        end a take after a typed phrase, because the entry is closed while
        the microphone is open.
        """
        return (self.root.focus_get() is self.text_entry
                and str(self.text_entry.cget("state")) == tk.NORMAL)

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

        The focus goes back to the window after a phrase is sent, so the next
        Space starts a take or interrupts the speech instead of typing a space
        into the entry. To type the next phrase the learner clicks the entry.
        """
        phrase = clean_input(self.text_entry.get())
        if not phrase:
            return
        self.text_entry.delete(0, tk.END)
        self.root.focus_set()
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
            # A clicked button is disabled with the mouse still over it, and
            # its hover color would make it look active while the model
            # answers. The next <Enter> paints the hover again.
            self._hover_button(button, False)

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
            text=NOTES_LABEL_SHOWN if self._notes_shown else NOTES_LABEL_HIDDEN)

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
        """Add a System line: progress, a warning or an error for the user."""
        self._append((f"{SYSTEM_LABEL}\t", "system_label"),
                     (f"{text}\n", "system"))

    def append_user_msg(self, text: str):
        """Add what the learner said, as recognized."""
        self._append((f"{USER_LABEL}\t", "user"), (f"{text}\n", "text_user"))

    def append_partner_msg(self, text: str):
        """Add what the partner says (the SAY line of a reply).

        The line also becomes the current question above the control panel.
        """
        self._append((f"{PARTNER_NAME}\t", "partner"),
                     (f"{text}\n", "text_partner"))
        self._set_question(text, asked=True)

    def set_topic(self, topic: str):
        """Take *topic* for the next question of the tutor.

        Not shown at once: the question on the screen still belongs to the old
        topic until the model answers, and the topic beside it would be
        wrong for that second. The next append_partner_msg shows it.
        """
        self._topic = topic

    def append_raw_reply(self, text: str):
        """Add a reply outside the contract, shown whole.

        Chat only: such a reply can be long and is not a question, so the
        question block keeps the last real question.
        """
        self._append((f"{PARTNER_NAME}\t", "partner"),
                     (f"{text}\n", "text_partner"))

    def append_note(self, text: str):
        """Add a correction (the NOTE line of a reply). It is never spoken."""
        self._append((f"{NOTE_LABEL}\t", "note"), (f"{text}\n", "text_note"))

    def append_summary(self, text: str):
        """Add the lesson summary. It may have several lines and is never spoken.

        The label stands on a line of its own, so the summary lines start at
        the same edge.
        """
        self._append((f"{SUMMARY_LABEL}\n", "summary"),
                     (f"{text}\n", "text_summary"))
        # The last question would stay in the block and read as if the tutor
        # still waited for an answer. A later SAY replaces this again.
        self._set_question(QUESTION_FINISHED, asked=False)

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
        self._set_stage(STAGE_READY)

    # ------------------------------------------------------------------
    # Exchange intents
    # ------------------------------------------------------------------
    def enter_idle(self):
        """Waiting for the learner to speak or to type."""
        self.draw_mic_button("idle")
        self.update_status("Ready", THEME["ready"])
        self.update_instruction(INSTRUCTION_READY)
        self._set_input_enabled(True)
        self._set_stage(STAGE_READY)

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
        self._set_stage(STAGE_LISTENING)

    def enter_processing(self):
        """The recording is being transcribed."""
        self.draw_mic_button("processing")
        self.update_status("Processing Speech (STT)...", THEME["warn"])
        self._set_input_enabled(False)
        self._set_stage(STAGE_THINKING)

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
        self._set_stage(STAGE_THINKING)

    def enter_speaking(self):
        """The partner's reply is being spoken.

        The entry is open: typing and Enter interrupt the speech, exactly as
        pressing SPACE does.
        """
        self.draw_mic_button("speaking")
        self.update_status(f"{PARTNER_NAME} is speaking...", THEME["partner"])
        self.update_instruction(INSTRUCTION_READY)
        self._set_input_enabled(True)
        self._set_stage(STAGE_SPEAKING)

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
        self._set_stage(STAGE_READY)

    def server_failed(self):
        """The LLM server did not start: the session cannot continue.

        The button deliberately stays in the loading state - there is nothing to
        press, and an idle mic would invite a recording that cannot be answered.
        The entry stays closed for the same reason.
        """
        self.update_status("LLM Server Error", THEME["bad"])
        self.update_instruction(INSTRUCTION_SERVER_FAILED)
        self._set_input_enabled(False)
        self._set_question(QUESTION_FAILED, asked=False)
        self._set_stage(None)

    def init_failed(self):
        """Startup stopped on an unexpected error."""
        self.update_status("Initialization Failed", THEME["bad"])
        self._set_input_enabled(False)
        self._set_question(QUESTION_FAILED, asked=False)
        self._set_stage(None)

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
        self._set_stage(STAGE_READY)
