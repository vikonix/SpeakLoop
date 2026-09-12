# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Spanish profile (variant: castilian).

Pure data consumed by speakloop/config.py (assembled into LANGUAGE_PROFILES).
The profile format is documented in speakloop/languages/__init__.py.

The profile is complete, but the language cannot be selected yet: the tutor
runs on English until the lesson prompt arrives (stage 3 of the plan in
docs/refactoring.md).
"""

PROFILE = {
    "display_name": "Spanish",
    "whisper_language": "es",
    "default_variant": "castilian",
    "tts_warmup": "Hola.",
    "variants": {
        # Supertonic instead of Kokoro: Kokoro's Spanish is trained on little
        # data and has audible artifacts, while Supertonic 3 is multilingual by
        # design and offers ten voices. Choosing the engine is data here, not a
        # language branch in the code.
        "castilian": {
            "tts_backend": "supertonic",
            "tts_lang_code": "es",
            "default_voice": "F1",
            "voices": ["F1", "F2", "F3", "F4", "F5",
                       "M1", "M2", "M3", "M4", "M5"],
            # Supertonic quality and speed knob (5..12 is the useful band for
            # phrase-length audio); 8 is the value Mimora settled on.
            "total_steps": 8,
        },
    },
}
