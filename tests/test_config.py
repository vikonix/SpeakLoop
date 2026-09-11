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
        self.assertEqual(config.WHISPER_MODEL, models_info.WHISPER_SMALL.repo_id)

    def test_kokoro_is_loaded_by_the_catalogue_repo_id(self):
        self.assertEqual(config.KOKORO_REPO_ID, models_info.KOKORO.repo_id)

    def test_the_gguf_path_is_the_catalogue_file_in_models(self):
        self.assertEqual(Path(config.EXTERNAL_MODEL_PATH),
                         paths.models_dir() / models_info.GGUF_CHAT.filename)


class EnvironmentTests(unittest.TestCase):
    """What the import leaves behind for huggingface_hub to read."""

    def test_hf_home_is_set(self):
        # setdefault in config: either the model cache or a value the user set
        # before the start. Either way the variable must exist once config is
        # imported, because huggingface_hub reads it at its own import.
        self.assertIn("HF_HOME", os.environ)

    def test_logs_go_to_the_log_directory(self):
        self.assertEqual(Path(config.LOG_FILE).parent, paths.log_dir())
        self.assertEqual(Path(config.LLM_SERVER_LOG_FILE).parent,
                         paths.log_dir())


class UserSettingTests(unittest.TestCase):
    """The one settings.json key of this step, validated like Mimora's."""

    def test_max_record_seconds_is_a_known_key(self):
        self.assertIn("max_record_seconds", config._KNOWN_USER_KEYS)

    def test_max_record_seconds_is_at_least_one(self):
        self.assertGreaterEqual(config.MAX_RECORD_SECONDS, 1)

    def test_the_example_file_names_only_known_keys(self):
        # settings.example.json is what a user copies; a key there that config
        # does not know would be reported as unknown on the first start.
        example = (Path(config.__file__).resolve().parent.parent
                   / "config" / "settings.example.json")
        data = json.loads(example.read_text(encoding="utf-8"))
        keys = {key for key in data if not key.startswith("_")}
        self.assertLessEqual(keys, config._KNOWN_USER_KEYS)


if __name__ == "__main__":
    unittest.main()
