# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""English profile (variants: american, british).

Pure data consumed by speakloop/config.py (assembled into LANGUAGE_PROFILES).
The profile format is documented in speakloop/languages/__init__.py.
"""

PROFILE = {
    "display_name": "English",
    "whisper_language": "en",
    "default_variant": "american",
    "tts_warmup": "Hi.",
    "variants": {
        # Both English variants run Kokoro: it is trained on English and is the
        # model the installer downloads for it.
        "american": {
            "tts_backend": "kokoro",
            "tts_lang_code": "a",
            "default_voice": "af_heart",
            # Kokoro voice names carry their variant and gender: "af_"/"am_"
            # are American female/male, "bf_"/"bm_" British.
            "voices": [
                "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
                "am_adam", "am_michael", "am_echo", "am_eric", "am_liam",
            ],
        },
        "british": {
            "tts_backend": "kokoro",
            "tts_lang_code": "b",
            "default_voice": "bf_emma",
            "voices": [
                "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
                "bm_george", "bm_daniel", "bm_fable", "bm_lewis",
            ],
        },
    },
}
