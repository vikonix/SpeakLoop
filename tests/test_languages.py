# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the language profiles in speakloop/languages/.

The profiles are pure data that config.py turns into per-run constants without
checking much: a missing key is a KeyError during startup, and a default voice
that is not in its own variant would be sent to the synthesis backend as is.
These tests are that check.

The profile modules are imported directly, so this file needs neither config
nor torch.

Run from the project root with:

    python -m unittest tests.test_languages
"""

import unittest

from speakloop.languages import english, spanish

# The backends speakloop/tts.py registers. Spelled out here rather than
# imported, so this test file stays free of numpy, sounddevice and config;
# tests/test_tts.py pins the same set against the registry itself.
KNOWN_BACKENDS = {"kokoro", "supertonic"}

PROFILES = {
    "english": english.PROFILE,
    "spanish": spanish.PROFILE,
}


class ProfileShapeTests(unittest.TestCase):
    """Every key config.py reads must be present in every profile."""

    def test_the_top_level_keys_are_there(self):
        for name, profile in PROFILES.items():
            with self.subTest(language=name):
                for key in ("display_name", "whisper_language",
                            "default_variant", "tts_warmup", "variants"):
                    self.assertIn(key, profile)

    def test_the_display_name_is_a_name(self):
        for name, profile in PROFILES.items():
            with self.subTest(language=name):
                self.assertTrue(profile["display_name"].strip())

    def test_the_whisper_language_is_a_short_code(self):
        # faster-whisper takes ISO codes such as "en" and "es"; a display name
        # here would be rejected at the first transcription, not at startup.
        for name, profile in PROFILES.items():
            with self.subTest(language=name):
                code = profile["whisper_language"]
                self.assertRegex(code, r"^[a-z]{2}$")

    def test_the_warmup_word_is_not_empty(self):
        # An empty word makes the warm-up synthesize nothing, and the first real
        # sentence of the lesson pays the whole start-up latency.
        for name, profile in PROFILES.items():
            with self.subTest(language=name):
                self.assertTrue(profile["tts_warmup"].strip())


class VariantTests(unittest.TestCase):
    """A variant carries the whole synthesis wiring and must be consistent."""

    def test_the_default_variant_exists(self):
        for name, profile in PROFILES.items():
            with self.subTest(language=name):
                self.assertIn(profile["default_variant"], profile["variants"])

    def test_every_variant_names_a_known_backend(self):
        for name, profile in PROFILES.items():
            for variant_name, variant in profile["variants"].items():
                with self.subTest(language=name, variant=variant_name):
                    self.assertIn(variant.get("tts_backend", "kokoro"),
                                  KNOWN_BACKENDS)

    def test_every_variant_names_its_language_code(self):
        for name, profile in PROFILES.items():
            for variant_name, variant in profile["variants"].items():
                with self.subTest(language=name, variant=variant_name):
                    self.assertTrue(variant["tts_lang_code"].strip())

    def test_the_default_voice_belongs_to_its_variant(self):
        # config.py only validates a voice from settings.json against this list.
        # A default that is not in it would reach the backend unchecked.
        for name, profile in PROFILES.items():
            for variant_name, variant in profile["variants"].items():
                with self.subTest(language=name, variant=variant_name):
                    self.assertIn(variant["default_voice"], variant["voices"])

    def test_the_voice_lists_hold_no_duplicates(self):
        for name, profile in PROFILES.items():
            for variant_name, variant in profile["variants"].items():
                with self.subTest(language=name, variant=variant_name):
                    voices = variant["voices"]
                    self.assertEqual(len(voices), len(set(voices)))

    def test_a_voice_belongs_to_exactly_one_variant_of_a_language(self):
        # The backend and the language code come from the variant, so the same
        # voice name under two variants would be synthesized differently
        # depending on the setting that selected it.
        for name, profile in PROFILES.items():
            seen = set()
            for variant_name, variant in profile["variants"].items():
                with self.subTest(language=name, variant=variant_name):
                    voices = set(variant["voices"])
                    self.assertFalse(voices & seen)
                    seen |= voices

    def test_total_steps_stays_in_the_useful_band(self):
        # config.py clamps a value outside 5..12 and prints a warning on every
        # start; a committed profile must not need that.
        for name, profile in PROFILES.items():
            for variant_name, variant in profile["variants"].items():
                if "total_steps" not in variant:
                    continue
                with self.subTest(language=name, variant=variant_name):
                    self.assertGreaterEqual(variant["total_steps"], 5)
                    self.assertLessEqual(variant["total_steps"], 12)


if __name__ == "__main__":
    unittest.main()
