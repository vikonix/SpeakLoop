# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Per-language profile data for the languages the tutor can practice.

Each module here holds a single ``PROFILE`` dict: pure data, no imports and no
side effects. ``speakloop/config.py`` assembles them into ``LANGUAGE_PROFILES``
and derives every per-run constant from the active one, so a new language is a
new module plus one entry in that assembly - never an ``if language == ...``
branch somewhere in the code.

Top-level keys:

``display_name``
    The language as the window and the system prompt name it ("English").
``whisper_language``
    ISO code faster-whisper transcribes with ("en", "es"). The learner speaks
    the practiced language, so this follows the profile and nothing else.
``default_variant``
    Variant used when settings.json names none.
``tts_warmup``
    Short in-vocabulary word spoken by the synthesis warm-up pass
    (speakloop/tts.py). In the practiced language, so the dummy synthesis
    raises no out-of-vocabulary phoneme warnings.
``variants``
    Variant key -> synthesis wiring. Each variant names its backend
    (``tts_backend``, one of the keys of ``speakloop.tts.TTS_BACKENDS``), that
    backend's language code (``tts_lang_code``: Kokoro uses single letters,
    "a" American and "b" British; Supertonic uses ISO codes such as "es"), the
    selectable ``voices`` and the ``default_voice`` among them. Supertonic
    variants may add ``total_steps``, its quality and speed knob.

A voice belongs to exactly one variant: the backend and the language code come
from the variant, so a voice of another variant would be synthesized with the
wrong engine or the wrong language.
"""
