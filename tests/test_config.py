# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for speakloop/config.py.

config.py is mostly constants built at import time, so what is testable here is
the small amount of logic that decides them, and the bindings that keep the app
loading exactly what the installer downloaded.

_model_device is the rule that keeps a per-model device from outliving the
DEVICE it is derived from. Both module globals it reads are patched per case,
so nothing here depends on this machine's hardware_config.json or on torch.
Importing config itself does read the real config/ directory of the checkout,
exactly as the application does.

Run from the project root with:

    python -m unittest tests.test_config
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from speakloop import config, models_info, paths


class ModelDeviceTests(unittest.TestCase):
    def _device(self, device: str, hw: dict, key: str, default: str) -> str:
        with mock.patch.object(config, "DEVICE", device), \
                mock.patch.object(config, "_HW", hw):
            return config._model_device(key, default)

    def test_stored_cuda_is_used_when_device_is_cuda(self):
        self.assertEqual(
            self._device("cuda", {"STT_DEVICE": "cuda"}, "STT_DEVICE", "cuda"),
            "cuda")

    def test_stale_cuda_cannot_exceed_a_cpu_device(self):
        # The case this helper exists for: DEVICE stepped down because torch
        # turned out to be a CPU build, while the same file still says "cuda"
        # for speech recognition. Believing it would move the crash from Kokoro
        # to faster-whisper rather than prevent it.
        self.assertEqual(
            self._device("cpu", {"STT_DEVICE": "cuda"}, "STT_DEVICE", "cpu"),
            "cpu")

    def test_stored_cpu_is_kept_on_a_cuda_machine(self):
        # A "cpu" written by the detector is its decision (ctranslate2 found no
        # CUDA device), not staleness.
        self.assertEqual(
            self._device("cuda", {"STT_DEVICE": "cpu"}, "STT_DEVICE", "cuda"),
            "cpu")

    def test_default_applies_when_the_key_is_absent(self):
        self.assertEqual(self._device("cuda", {}, "STT_DEVICE", "cuda"), "cuda")

    def test_cuda_default_is_capped_too(self):
        self.assertEqual(self._device("cpu", {}, "STT_DEVICE", "cuda"), "cpu")


class ModelBindingTests(unittest.TestCase):
    """The app loads by the same names the fetchers download by."""

    def test_whisper_is_loaded_by_the_catalogue_repo_id(self):
        self.assertEqual(config.WHISPER_MODEL, models_info.WHISPER.repo_id)

    def test_the_gguf_path_is_the_catalogue_file_in_models(self):
        self.assertEqual(Path(config.EXTERNAL_MODEL_PATH),
                         paths.models_dir() / models_info.GGUF_CHAT.filename)

    def test_the_offline_gate_names_the_model_of_the_active_backend(self):
        # The gate is all-or-nothing: a repo that this run never loads would
        # keep the Hub online forever, and a repo it does load but that is not
        # listed would be switched offline before it was downloaded.
        self.assertIn(models_info.WHISPER, config._CACHED_REPOS)
        if config.TTS_BACKEND == "kokoro":
            self.assertIn(models_info.KOKORO, config._CACHED_REPOS)
        else:
            # Supertonic keeps its weights outside the hub cache, so it is
            # checked by its own predicate instead.
            self.assertNotIn(models_info.KOKORO, config._CACHED_REPOS)


class EnvironmentTests(unittest.TestCase):
    """What the import leaves behind for the download libraries to read."""

    def test_hf_home_is_set(self):
        # setdefault in config: either the model cache or a value the user set
        # before the start. Either way the variable must exist once config is
        # imported, because huggingface_hub reads it at its own import.
        self.assertIn("HF_HOME", os.environ)

    def test_the_supertonic_cache_is_pinned(self):
        # The supertonic package reads this variable on every call; model_fetch
        # sets the same one, so the installer's download and the app's load
        # cannot end up in different directories.
        self.assertIn("SUPERTONIC_CACHE_DIR", os.environ)
        self.assertEqual(Path(os.environ["SUPERTONIC_CACHE_DIR"]),
                         config.SUPERTONIC_CACHE_DIR)

    def test_logs_go_to_the_log_directory(self):
        self.assertEqual(Path(config.LOG_FILE).parent, paths.log_dir())
        self.assertEqual(Path(config.LLM_SERVER_LOG_FILE).parent,
                         paths.log_dir())


class LanguageTests(unittest.TestCase):
    """Every per-run language constant comes from the active profile."""

    def setUp(self):
        self.profile = config.LANGUAGE_PROFILES[config.PRACTICE_LANGUAGE]
        self.variant = self.profile["variants"][config.ACCENT]

    def test_the_practiced_language_has_a_profile(self):
        self.assertIn(config.PRACTICE_LANGUAGE, config.LANGUAGE_PROFILES)

    def test_english_and_spanish_are_both_assembled(self):
        self.assertEqual(set(config.LANGUAGE_PROFILES), {"english", "spanish"})

    def test_the_language_is_english_in_this_version(self):
        # The lesson prompt and the language selector arrive together in stage
        # 3; until then a Spanish lesson would run on an English prompt.
        self.assertEqual(config.PRACTICE_LANGUAGE, "english")

    def test_the_display_name_comes_from_the_profile(self):
        self.assertEqual(config.TARGET_LANGUAGE, self.profile["display_name"])

    def test_the_recognition_language_comes_from_the_profile(self):
        self.assertEqual(config.WHISPER_LANGUAGE,
                         self.profile["whisper_language"])

    def test_the_warmup_word_comes_from_the_profile(self):
        self.assertEqual(config.TTS_WARMUP, self.profile["tts_warmup"])

    def test_the_variant_exists_in_the_profile(self):
        self.assertIn(config.ACCENT, self.profile["variants"])

    def test_the_synthesis_wiring_comes_from_the_variant(self):
        self.assertEqual(config.TTS_BACKEND,
                         self.variant.get("tts_backend", "kokoro"))
        self.assertEqual(config.TTS_LANG_CODE, self.variant["tts_lang_code"])
        self.assertEqual(list(config.TTS_VOICES), list(self.variant["voices"]))

    def test_the_backend_is_one_of_the_known_names(self):
        self.assertIn(config.TTS_BACKEND, config.TTS_BACKEND_CHOICES)

    def test_the_voice_belongs_to_the_active_variant(self):
        # A voice of another variant would be sent to another backend, or to
        # the same backend with the wrong language code.
        self.assertIn(config.TTS_VOICE, config.TTS_VOICES)

    def test_total_steps_is_inside_the_useful_band(self):
        self.assertGreaterEqual(config.TTS_TOTAL_STEPS, 5)
        self.assertLessEqual(config.TTS_TOTAL_STEPS, 12)

    def test_accent_and_voice_are_known_keys(self):
        self.assertIn("accent", config._KNOWN_USER_KEYS)
        self.assertIn("voice", config._KNOWN_USER_KEYS)


class UserSettingTests(unittest.TestCase):
    """The settings.json keys of this step, validated like Mimora's."""

    def test_max_record_seconds_is_a_known_key(self):
        self.assertIn("max_record_seconds", config._KNOWN_USER_KEYS)

    def test_max_record_seconds_is_at_least_one(self):
        self.assertGreaterEqual(config.MAX_RECORD_SECONDS, 1)

    def test_the_silence_keys_are_known(self):
        self.assertIn("silence_timeout", config._KNOWN_USER_KEYS)
        self.assertIn("silence_threshold", config._KNOWN_USER_KEYS)

    def test_the_silence_timeout_leaves_room_to_breathe(self):
        # Below half a second an ordinary pause between two words would end the
        # take in the middle of a sentence.
        self.assertGreaterEqual(config.SILENCE_TIMEOUT, 0.5)

    def test_the_silence_threshold_is_above_zero(self):
        # At zero the noise floor of the microphone counts as speech, the timer
        # is never armed, and every take runs to the time limit.
        self.assertGreater(config.SILENCE_THRESHOLD, 0)

    def test_the_example_file_names_only_known_keys(self):
        # settings.example.json is what a user copies; a key there that config
        # does not know would be reported as unknown on the first start.
        example = (Path(config.__file__).resolve().parent.parent
                   / "config" / "settings.example.json")
        data = json.loads(example.read_text(encoding="utf-8"))
        keys = {key for key in data if not key.startswith("_")}
        self.assertLessEqual(keys, config._KNOWN_USER_KEYS)


class LessonSettingTests(unittest.TestCase):
    """The lesson prompt settings (stage 3)."""

    def test_the_lesson_keys_are_known(self):
        self.assertIn("first_topic", config._KNOWN_USER_KEYS)
        self.assertIn("prompt_file", config._KNOWN_USER_KEYS)

    def test_the_explanation_language_is_russian(self):
        # The NOTE example in the prompt is in Russian; another explanation
        # language would give mixed corrections.
        self.assertEqual(config.EXPLANATION_LANGUAGE, "Russian")

    def test_the_first_topic_is_a_string(self):
        self.assertIsInstance(config.FIRST_TOPIC, str)

    def test_show_notes_is_a_known_key(self):
        self.assertIn("show_notes", config._KNOWN_USER_KEYS)

    def test_show_notes_is_a_flag(self):
        # ui.py uses it as one, and a string would silently mean "shown".
        self.assertIsInstance(config.SHOW_NOTES, bool)

    def test_the_prompt_file_is_an_absolute_path(self):
        self.assertTrue(Path(config.PROMPT_FILE).is_absolute())

    def test_the_default_prompt_file_is_shipped_with_the_package(self):
        # The default of PROMPT_FILE; without this file the app cannot start.
        shipped = (Path(config.__file__).resolve().parent
                   / "prompts" / "free_talk.md")
        self.assertTrue(shipped.is_file())


class SaveUserSettingTests(unittest.TestCase):
    """The one value the application writes back to settings.json."""

    def test_the_settings_file_is_the_one_config_reads(self):
        self.assertEqual(Path(config.SETTINGS_FILE).parent, paths.config_dir())
        self.assertEqual(Path(config.SETTINGS_FILE).name, "settings.json")

    def test_a_saved_setting_goes_through_the_loader(self):
        # The loader re-reads the file and writes it atomically, which is what
        # keeps the hand-edited keys and the comment keys of the user.
        with mock.patch.object(config.loader, "save_setting",
                               return_value=True) as save:
            self.assertTrue(config.save_user_setting("show_notes", False))
        save.assert_called_once_with(config.SETTINGS_FILE, "show_notes", False,
                                     config._USER)

    def test_a_failed_save_is_reported_and_not_raised(self):
        with mock.patch.object(config.loader, "save_setting",
                               return_value=False):
            self.assertFalse(config.save_user_setting("show_notes", True))


class GpuLayersSettingTests(unittest.TestCase):
    """external_n_gpu_layers: "auto", "all" or a whole number, as a string.

    Every rejected value prints a line to stderr; mock.patch keeps it out of
    the test report.
    """

    def _parse(self, value):
        with mock.patch("sys.stderr"):
            return config._gpu_layers_setting(value)

    def test_the_key_is_known(self):
        self.assertIn("external_n_gpu_layers", config._KNOWN_USER_KEYS)

    def test_the_two_words_are_kept(self):
        self.assertEqual(self._parse("auto"), "auto")
        self.assertEqual(self._parse("all"), "all")

    def test_a_whole_number_becomes_a_string(self):
        # The value goes to the command line unchanged.
        self.assertEqual(self._parse(20), "20")
        self.assertEqual(self._parse(0), "0")

    def test_invalid_values_fall_back_to_auto(self):
        # A typo must not switch the memory fit off: "auto" is the safe value.
        for value in (-1, 2.5, True, "20", "Auto", "", None, [20]):
            with self.subTest(value=value):
                self.assertEqual(self._parse(value), "auto")

    def test_the_resolved_value_is_a_valid_one(self):
        # Whatever the checkout's settings.json says.
        value = config.EXTERNAL_N_GPU_LAYERS
        self.assertTrue(value in config.GPU_LAYERS_WORDS or value.isdigit(),
                        value)


class SttDeviceSettingTests(unittest.TestCase):
    """stt_device: "auto", "cuda" or "cpu", and the device it resolves to.

    Every rejected value prints a line to stderr; mock.patch keeps it out of
    the test report. DEVICE and _HW are patched per case, so nothing here
    depends on this machine's hardware_config.json or on torch.
    """

    def _parse(self, value):
        with mock.patch("sys.stderr"):
            return config._stt_device_setting(value)

    def _resolve(self, setting: str, device: str, hw: dict) -> str:
        with mock.patch.object(config, "DEVICE", device), \
                mock.patch.object(config, "_HW", hw), \
                mock.patch("sys.stderr"):
            return config._stt_device(setting)

    def test_the_key_is_known(self):
        self.assertIn("stt_device", config._KNOWN_USER_KEYS)

    def test_the_three_choices_are_kept(self):
        for value in ("auto", "cuda", "cpu"):
            with self.subTest(value=value):
                self.assertEqual(self._parse(value), value)

    def test_invalid_values_fall_back_to_auto(self):
        # A typo gives the detected device, not a device nobody chose.
        for value in ("CUDA", "gpu", "", None, 1, True, ["cuda"]):
            with self.subTest(value=value):
                self.assertEqual(self._parse(value), "auto")

    def test_auto_takes_the_detected_device(self):
        self.assertEqual(
            self._resolve("auto", "cuda", {"STT_DEVICE": "cuda"}), "cuda")
        self.assertEqual(
            self._resolve("auto", "cuda", {"STT_DEVICE": "cpu"}), "cpu")

    def test_auto_is_capped_by_device(self):
        self.assertEqual(
            self._resolve("auto", "cpu", {"STT_DEVICE": "cuda"}), "cpu")

    def test_cpu_wins_over_a_detected_cuda(self):
        # The way out for a card where Whisper leaves the chat model too few
        # layers.
        self.assertEqual(
            self._resolve("cpu", "cuda", {"STT_DEVICE": "cuda"}), "cpu")

    def test_cuda_wins_over_a_detected_cpu(self):
        self.assertEqual(
            self._resolve("cuda", "cuda", {"STT_DEVICE": "cpu"}), "cuda")

    def test_cuda_is_capped_by_device(self):
        # Without the CUDA build of torch, ctranslate2 has no GPU libraries on
        # Windows and the model load would fail.
        self.assertEqual(
            self._resolve("cuda", "cpu", {"STT_DEVICE": "cuda"}), "cpu")

    def test_the_resolved_device_is_a_valid_one(self):
        # Whatever the checkout's settings.json says.
        self.assertIn(config.STT_DEVICE, ("cuda", "cpu"))


class LlmSettingTests(unittest.TestCase):
    """The stage 2 values of the chat model (docs/model-parameters.md)."""

    def test_the_context_is_a_whole_number_of_at_least_256(self):
        # It goes to the command line twice (--ctx-size and -fitc), where a
        # float would break the launch.
        self.assertIsInstance(config.EXTERNAL_N_CTX, int)
        self.assertGreaterEqual(config.EXTERNAL_N_CTX, 256)

    def test_the_reply_limit_leaves_room_for_a_summary(self):
        self.assertGreaterEqual(config.LLM_MAX_TOKENS, 512)

    def test_the_speech_devices_are_capped_by_device(self):
        # Both come from _model_device, so neither can be "cuda" on a machine
        # where torch runs on the CPU.
        if config.DEVICE != "cuda":
            self.assertEqual(config.STT_DEVICE, "cpu")
            self.assertEqual(config.TTS_DEVICE, "cpu")
        self.assertIn(config.TTS_DEVICE, ("cuda", "cpu"))


class AudioSettingTests(unittest.TestCase):
    """The rate of the audio pipeline, which two modules have to agree on."""

    def test_the_pipeline_rate_is_the_one_whisper_needs(self):
        # The recorder downsamples every take to this rate and faster-whisper
        # accepts no other one.
        self.assertEqual(config.AUDIO_SAMPLE_RATE, 16_000)


class ColorThemeTests(unittest.TestCase):
    """The palette the view layer reads: complete, and made of colors."""

    def test_color_theme_is_a_known_key(self):
        self.assertIn("color_theme", config._KNOWN_USER_KEYS)

    def test_the_resolved_palette_holds_every_built_in_key(self):
        # ui.py indexes THEME directly, so a missing key is a KeyError in the
        # middle of building the window.
        self.assertEqual(set(config.THEME), set(config._DARK_THEME))

    def test_every_resolved_color_is_a_hex_value(self):
        for name, value in config.THEME.items():
            with self.subTest(color=name):
                self.assertRegex(value, r"^#[0-9a-fA-F]{6}$")


class ThemeFileTests(unittest.TestCase):
    """Where a schema is looked for: the user's copy first, then the shipped one."""

    def test_a_user_schema_wins_over_the_shipped_one(self):
        # What lets a user edit a theme without touching the installation.
        with tempfile.TemporaryDirectory() as temporary:
            user_dir = Path(temporary)
            (user_dir / "dark_schema.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(paths, "themes_dir", return_value=user_dir):
                self.assertEqual(config._theme_file("dark"),
                                 user_dir / "dark_schema.json")

    def test_the_shipped_file_is_used_without_a_user_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(paths, "themes_dir",
                                   return_value=Path(temporary)):
                self.assertEqual(config._theme_file("dark"),
                                 paths.shipped_themes_dir() / "dark_schema.json")

    def test_an_unknown_theme_points_at_the_shipped_directory(self):
        # Nothing is there to read, and the shipped path is the one worth
        # naming in the warning that follows.
        self.assertEqual(config._theme_file("no-such-theme").parent,
                         paths.shipped_themes_dir())


class ShippedThemeTests(unittest.TestCase):
    """The schemas that travel with the code must need no fallback at all."""

    THEMES = ("dark", "light")

    def _colors(self, name: str) -> dict:
        """The color keys of a shipped schema, without the '_' comment keys."""
        path = paths.shipped_themes_dir() / f"{name}_schema.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return {key: value for key, value in data.items()
                if not key.startswith("_")}

    def test_both_themes_are_shipped(self):
        for name in self.THEMES:
            with self.subTest(theme=name):
                self.assertTrue(
                    (paths.shipped_themes_dir() / f"{name}_schema.json").is_file())

    def test_the_schemas_name_only_known_colors(self):
        # An unknown key is ignored with a message on every single start.
        for name in self.THEMES:
            with self.subTest(theme=name):
                self.assertLessEqual(set(self._colors(name)),
                                     set(config._DARK_THEME))

    def test_the_schemas_define_every_color(self):
        # A missing key falls back to the dark value, which inside a light
        # theme is an unreadable surprise rather than a safe default.
        for name in self.THEMES:
            with self.subTest(theme=name):
                self.assertEqual(set(self._colors(name)),
                                 set(config._DARK_THEME))

    def test_the_dark_schema_matches_the_built_in_palette(self):
        # The built-in palette IS the dark theme. Two spellings that disagree
        # would change the window's look depending on whether the file is found.
        self.assertEqual(self._colors("dark"), config._DARK_THEME)


if __name__ == "__main__":
    unittest.main()
