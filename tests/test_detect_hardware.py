# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Unit tests for the hardware probe (speakloop/detect_hardware.py).

Three things are worth testing without a machine to probe: the decision table
of the llama-server offload probe (which of True/False/None each situation
deserves), the speech-recognition CUDA probe, and the speech-device rule
build_config turns the answers into. Everything
that would touch a disk or spawn a process is stubbed. Run from the project
root with:

    python -m unittest tests.test_detect_hardware
"""

import contextlib
import io
import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from speakloop import bootstrap, detect_hardware, llama_server_fetch

# Fake install location: the probe only ever reads its .name for a message.
EXE = Path("/opt/speakloop/bin/llama/llama-server")

# Variants the probe branches on. Named explicitly so that renaming one in
# llama_server_fetch.VARIANTS fails here loudly instead of silently skipping.
GPU_VARIANT = "win-cuda-12.4-x64"
CPU_VARIANT = "win-cpu-x64"

CUDA_LISTING = ("Available devices:\n"
                "  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB, 23000 MiB free)")
EMPTY_LISTING = "Available devices:\n  (none)"


def hardware(vram_gb=None, present=None, offload=None, torch_cuda=False,
             stt_cuda=False):
    """Minimal hardware dict in the shape build_config expects."""
    return {
        "gpu": {
            "present": bool(vram_gb) if present is None else present,
            "vram_gb": vram_gb,
            "torch_cuda": torch_cuda,
            "stt_cuda": stt_cuda,
            "llama_gpu_offload": offload,
        }
    }


class VariantTableTests(unittest.TestCase):
    """The probe's branching assumes these two shapes exist in the table."""

    def test_named_variants_still_exist(self):
        self.assertIn(GPU_VARIANT, llama_server_fetch.VARIANTS)
        self.assertIn(CPU_VARIANT, llama_server_fetch.VARIANTS)

    def test_gpu_variant_promises_a_device_and_cpu_variant_does_not(self):
        self.assertIsNotNone(
            llama_server_fetch.VARIANTS[GPU_VARIANT].device_pattern)
        self.assertIsNone(
            llama_server_fetch.VARIANTS[CPU_VARIANT].device_pattern)


class ProbeLlamaOffloadTests(unittest.TestCase):
    """Decision table of _probe_llama_offload.

    The verdict feeds build_config, so False is not a neutral answer: it zeroes
    the LLM's VRAM budget. It is therefore reserved for the two cases where the
    installed binary provably cannot offload, and everything else says None.
    """

    def _probe(self, gpu_present=True, exe=EXE, variant=GPU_VARIANT,
               listing=CUDA_LISTING, error=None):
        """Run the probe with the fetch module's three lookups stubbed out.

        Each stub stands in for a file on disk or a subprocess, neither of
        which a unit test should need.
        """
        warnings = []
        list_devices = (Mock(side_effect=error) if error
                        else Mock(return_value=listing))
        with patch.multiple(llama_server_fetch,
                            installed_exe=Mock(return_value=exe),
                            installed_variant=Mock(return_value=variant),
                            list_devices=list_devices):
            verdict = detect_hardware._probe_llama_offload(warnings, gpu_present)
        return verdict, warnings

    def test_promised_device_is_present(self):
        verdict, warnings = self._probe()
        self.assertIs(verdict, True)
        self.assertEqual(warnings, [])

    def test_promised_device_is_missing(self):
        # The silent CPU fallback: a CUDA build with the wrong cudart DLLs
        # still starts and still answers, about three times slower.
        verdict, warnings = self._probe(listing=EMPTY_LISTING)
        self.assertIs(verdict, False)
        self.assertEqual(len(warnings), 1)
        self.assertIn("cudart", warnings[0])

    def test_cpu_build_cannot_offload(self):
        verdict, warnings = self._probe(variant=CPU_VARIANT)
        self.assertIs(verdict, False)
        self.assertEqual(len(warnings), 1)
        self.assertIn(CPU_VARIANT, warnings[0])

    def test_cpu_build_is_not_worth_a_warning_without_a_gpu(self):
        # The overwhelmingly common case: no GPU, so the CPU build was the
        # right choice and there is nothing to tell the user about.
        verdict, warnings = self._probe(variant=CPU_VARIANT, gpu_present=False)
        self.assertIs(verdict, False)
        self.assertEqual(warnings, [])

    def test_missing_install_is_unknown_not_negative(self):
        # None keeps build_config on its "physical GPU presence" fallback,
        # which is what makes the tool usable before install.py has fetched
        # the llama-server binary (step_llama_server).
        verdict, warnings = self._probe(exe=None)
        self.assertIsNone(verdict)
        self.assertEqual(len(warnings), 1)
        self.assertIn("llama_server_fetch", warnings[0])

    def test_missing_install_is_silent_without_a_gpu(self):
        verdict, warnings = self._probe(exe=None, gpu_present=False)
        self.assertIsNone(verdict)
        self.assertEqual(warnings, [])

    def test_unknown_variant_is_unknown(self):
        # A stamp naming a variant this build dropped: there is no documented
        # expectation to compare a device listing against.
        verdict, warnings = self._probe(variant=None)
        self.assertIsNone(verdict)
        self.assertEqual(len(warnings), 1)

    def test_probe_failure_is_unknown(self):
        error = llama_server_fetch.LlamaServerFetchError("binary would not run")
        verdict, warnings = self._probe(error=error)
        self.assertIsNone(verdict)
        self.assertIn("binary would not run", warnings[0])


class BuildConfigLlmTests(unittest.TestCase):
    """The chat model gets no values from the probe any more."""

    def test_no_llm_values_are_written(self):
        # llama.cpp fits the GPU layers itself and the context is a setting.
        # A written value would be read by an older config.py and pass -ngl.
        config = detect_hardware.build_config(
            hardware(vram_gb=24.0, offload=True, torch_cuda=True,
                     stt_cuda=True))
        self.assertNotIn("EXTERNAL_N_GPU_LAYERS", config)
        self.assertNotIn("EXTERNAL_N_CTX", config)

    def test_unknown_verdict_trusts_the_physical_card(self):
        # None must behave exactly like True here - this is what lets
        # detect_hardware run before the binary is installed without changing
        # the values it writes.
        for vram in (4.0, 24.0):
            with self.subTest(vram=vram):
                self.assertEqual(
                    detect_hardware.build_config(hardware(
                        vram_gb=vram, offload=None, torch_cuda=True,
                        stt_cuda=True)),
                    detect_hardware.build_config(hardware(
                        vram_gb=vram, offload=True, torch_cuda=True,
                        stt_cuda=True)))


class ProbeSttCudaTests(unittest.TestCase):
    """Decision table of _probe_stt_cuda.

    ctranslate2 is replaced by a stub module in sys.modules, so the suite
    neither needs faster-whisper installed nor asks the real machine.
    """

    def _probe(self, torch_cuda=True, device_count=1, error=None,
               missing=False):
        warnings = []
        if missing:
            # None in sys.modules makes the import raise ImportError.
            stub = None
        else:
            count = (Mock(side_effect=error) if error
                     else Mock(return_value=device_count))
            stub = Mock(get_cuda_device_count=count)
        with patch.dict(sys.modules, {"ctranslate2": stub}):
            verdict = detect_hardware._probe_stt_cuda(warnings, torch_cuda)
        return verdict, warnings, stub

    def test_torch_cuda_and_a_ctranslate2_device_mean_cuda(self):
        verdict, warnings, _ = self._probe()
        self.assertIs(verdict, True)
        self.assertEqual(warnings, [])

    def test_cpu_only_torch_is_cpu_without_asking_ctranslate2(self):
        # A CPU-only torch brings no cuBLAS or cuDNN, so a CUDA device that
        # ctranslate2 might still report would fail at the first transcription.
        verdict, warnings, stub = self._probe(torch_cuda=False)
        self.assertIs(verdict, False)
        self.assertEqual(warnings, [])
        stub.get_cuda_device_count.assert_not_called()

    def test_no_ctranslate2_device_is_cpu_with_a_warning(self):
        verdict, warnings, _ = self._probe(device_count=0)
        self.assertIs(verdict, False)
        self.assertEqual(len(warnings), 1)
        self.assertIn("faster-whisper", warnings[0])

    def test_missing_ctranslate2_is_cpu_with_a_warning(self):
        verdict, warnings, _ = self._probe(missing=True)
        self.assertIs(verdict, False)
        self.assertEqual(len(warnings), 1)

    def test_probe_failure_is_cpu_with_the_reason(self):
        verdict, warnings, _ = self._probe(error=RuntimeError("driver broke"))
        self.assertIs(verdict, False)
        self.assertIn("driver broke", warnings[0])


class BuildConfigSpeechTests(unittest.TestCase):
    """The speech devices: their own probes, and the room the chat model
    leaves Kokoro. Whisper takes the GPU on a card of any size."""

    LARGE = detect_hardware.TTS_GPU_MIN_VRAM_GB
    SMALL = 4.0  # the reference laptop (docs/model-parameters.md, section 2)

    def _devices(self, **kwargs):
        config = detect_hardware.build_config(hardware(**kwargs))
        return config["DEVICE"], config["STT_DEVICE"], config["TTS_DEVICE"]

    def test_a_large_card_keeps_the_speech_models_on_the_gpu(self):
        self.assertEqual(
            self._devices(vram_gb=self.LARGE, offload=True, torch_cuda=True,
                          stt_cuda=True),
            ("cuda", "cuda", "cuda"))

    def test_a_small_card_keeps_whisper_and_moves_kokoro(self):
        # DEVICE stays "cuda": it answers whether torch sees CUDA, and a "cpu"
        # there would make warn_if_gpu_unused report a broken install.
        self.assertEqual(
            self._devices(vram_gb=self.SMALL, offload=True, torch_cuda=True,
                          stt_cuda=True),
            ("cuda", "cuda", "cpu"))

    def test_a_card_just_below_the_threshold_moves_kokoro(self):
        just_below = self.LARGE - 0.1
        self.assertEqual(
            self._devices(vram_gb=just_below, offload=True, torch_cuda=True,
                          stt_cuda=True)[2],
            "cpu")

    def test_whisper_does_not_depend_on_the_card_size(self):
        for vram_gb in (2.0, self.SMALL, self.LARGE, None):
            with self.subTest(vram_gb=vram_gb):
                self.assertEqual(
                    self._devices(vram_gb=vram_gb, offload=True,
                                  torch_cuda=True, stt_cuda=True)[1],
                    "cuda")

    def test_a_card_the_chat_model_cannot_use_is_free_for_speech(self):
        # A CPU build of llama-server leaves even a small card to the speech
        # models.
        self.assertEqual(
            self._devices(vram_gb=self.SMALL, offload=False, torch_cuda=True,
                          stt_cuda=True),
            ("cuda", "cuda", "cuda"))

    def test_cpu_only_torch_build_falls_back(self):
        self.assertEqual(
            self._devices(vram_gb=24.0, offload=True, torch_cuda=False),
            ("cpu", "cpu", "cpu"))

    def test_the_two_speech_devices_are_decided_separately(self):
        # torch can reach CUDA while ctranslate2 cannot; Kokoro should still
        # get the card.
        self.assertEqual(
            self._devices(vram_gb=24.0, offload=True, torch_cuda=True,
                          stt_cuda=False),
            ("cuda", "cpu", "cuda"))

    def test_no_gpu_means_cpu_everywhere(self):
        self.assertEqual(
            self._devices(vram_gb=None, present=False, offload=None),
            ("cpu", "cpu", "cpu"))


class UnwritableDataRootTests(unittest.TestCase):
    """`speakloop --detect-hardware` on a data root the machine cannot write to.

    This command is named to the user by warn_if_gpu_unused, so it is run by
    people following advice rather than by people debugging. It also does NOT
    import config, which means paths.ensure_dirs() never runs on this path and
    its own reporting cannot cover it: everything here has to be handled where
    it happens. An unreachable SPEAKLOOP_HOME would otherwise end the command in
    a three-deep pathlib traceback about a drive letter.
    """

    def setUp(self):
        # The module logger is process-global and this test empties it, which
        # is also what makes _setup_logging's idempotence guard let us in.
        logger = detect_hardware.logger
        saved = logger.handlers[:]
        logger.handlers.clear()
        self.addCleanup(lambda: (logger.handlers.clear(),
                                 logger.handlers.extend(saved)))

    def test_an_unwritable_log_directory_costs_the_file_and_nothing_else(self):
        stderr = io.StringIO()
        with patch.object(detect_hardware.Path, "mkdir",
                          side_effect=OSError(3, "no such drive")), \
                contextlib.redirect_stderr(stderr):
            detect_hardware._setup_logging()  # must not raise

        self.assertIn("no such drive", stderr.getvalue())
        # The probe is the point of the command and does not need the file.
        self.assertEqual(detect_hardware.logger.level, logging.INFO)
        # No file handler, but not an empty handler list either: logging falls
        # back to lastResort when it finds no handler anywhere, which printed
        # main()'s failure to stderr a second time in different words.
        self.assertFalse(any(isinstance(handler, logging.FileHandler)
                             for handler in detect_hardware.logger.handlers))
        self.assertTrue(any(isinstance(handler, logging.NullHandler)
                            for handler in detect_hardware.logger.handlers))

    def test_the_failure_is_not_reported_twice(self):
        # One failure, one message. The duplicate this guards against was not
        # cosmetic: two spellings of one error read as two errors.
        stderr = io.StringIO()
        with patch.object(detect_hardware.Path, "mkdir",
                          side_effect=OSError(3, "no such drive")), \
                patch.object(detect_hardware, "probe_and_write",
                             side_effect=OSError(3, "no such drive")), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr):
            detect_hardware.main()

        # Lower-cased before counting: the print and the log record differ in
        # capitalisation, and a case-sensitive count would match only one of
        # them and pass while the duplicate was still there.
        self.assertEqual(stderr.getvalue().lower().count("could not write"), 1)

    def test_an_unwritable_output_file_is_reported_not_raised(self):
        stderr = io.StringIO()
        # _setup_logging is stubbed rather than allowed to run: the real one
        # opens the project's own logs/hwdetect.log, and a test that adds a
        # run to a real log file is a test with a side effect.
        with patch.object(detect_hardware, "_setup_logging"), \
                patch.object(detect_hardware, "probe_and_write",
                             side_effect=OSError(3, "no such drive")), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr):
            exit_code = detect_hardware.main()

        self.assertEqual(exit_code, 1)
        self.assertIn("no such drive", stderr.getvalue())
        self.assertIn(str(detect_hardware.OUTPUT_FILE), stderr.getvalue())


class LogFileHistoryTests(unittest.TestCase):
    """logs/hwdetect.log keeps every probe, not only the last one.

    The file answers "why was this machine detected the way it was", and that
    is asked about a probe some later probe has already replaced - a driver was
    installed, a binary was swapped. Truncating per run answered it for the one
    run nobody is asking about.
    """

    def setUp(self):
        import tempfile

        # The module logger is process-global and this test empties it, which
        # is also what makes _setup_logging's idempotence guard let us in.
        logger = detect_hardware.logger
        saved = logger.handlers[:]
        logger.handlers.clear()
        self.addCleanup(lambda: (logger.handlers.clear(),
                                 logger.handlers.extend(saved)))
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log_file = Path(directory.name) / "hwdetect.log"

    def _probe_once(self, message):
        """One run's worth of logging, with the handler closed afterwards."""
        with patch.object(detect_hardware, "LOG_DIR", self.log_file.parent), \
                patch.object(detect_hardware, "LOG_FILE", self.log_file):
            detect_hardware._setup_logging()
            detect_hardware.logger.info(message)
        # Closed here rather than at cleanup: on Windows the directory cannot
        # be removed while the handler holds the file open, and the next call
        # needs the guard above to let it in.
        for handler in detect_hardware.logger.handlers[:]:
            detect_hardware.logger.removeHandler(handler)
            handler.close()

    def test_a_second_probe_is_added_below_the_first(self):
        self._probe_once("the first probe")
        self._probe_once("the second probe")

        lines = self.log_file.read_text(encoding="utf-8").splitlines()
        self.assertIn("the first probe", "\n".join(lines))
        self.assertIn("the second probe", "\n".join(lines))
        # Each run opens with the header, so the sections can be told apart.
        self.assertEqual(lines.count(bootstrap._HEADER_RULE), 2)
        self.assertEqual(lines[0], bootstrap._HEADER_RULE)


class WarnIfGpuUnusedTests(unittest.TestCase):
    """The startup warning for a CPU-only torch on a machine with a card.

    The failure it exists for is silent by nature, so what matters is that it
    fires in exactly one situation and stays quiet - and cheap - in the others.
    nvidia-smi is stubbed throughout; the real one would answer for whatever
    machine happens to run the suite.
    """

    def _run(self, device, driver_cuda, config_written=False,
             platform="win32"):
        # config_written is stubbed rather than left to the filesystem: the
        # machine running the suite may well have a hardware_config.json, and
        # then the branch under test would be whichever one it happens to have.
        #
        # platform is pinned for the same class of reason: the reinstall advice
        # differs by it. Left to the machine running the suite, the assertions
        # below would pass or fail according to who ran them. Windows is the
        # default here because it is the platform this warning exists for.
        with patch.object(detect_hardware.sys, "platform", platform), \
                patch.object(llama_server_fetch, "detect_driver_cuda",
                             return_value=driver_cuda) as detect, \
                patch.object(detect_hardware, "_stored_device_may_be_stale",
                             return_value=config_written):
            with self.assertLogs(level="WARNING") as captured:
                detect_hardware.warn_if_gpu_unused(device)
                # assertLogs fails an empty block, so every path needs one
                # record of its own to compare against.
                logging.warning("sentinel")
        return [line for line in captured.output if "sentinel" not in line], detect

    def test_cpu_torch_with_a_driver_present_is_warned_about(self):
        warnings, _ = self._run("cpu", (12, 4))
        self.assertEqual(len(warnings), 1)
        self.assertIn("install.py", warnings[0])
        self.assertIn("CPU-only build", warnings[0])

    def test_a_written_hardware_config_makes_the_message_name_both_causes(self):
        # With that file present, "cpu" can also mean detect_device took its
        # word without asking torch - which is what installing the CUDA build
        # over a CPU one without re-running the probe looks like. Telling such
        # a user to reinstall torch is telling them to redo what they just did,
        # so the refresh command has to be named as well.
        with patch.object(detect_hardware.paths, "repo_mode",
                          return_value=False):
            warnings, _ = self._run("cpu", (12, 4), config_written=True)
        self.assertEqual(len(warnings), 1)
        self.assertIn("speakloop --detect-hardware", warnings[0])
        self.assertIn("install.py", warnings[0])

    def test_other_platforms_get_the_general_pytorch_advice(self):
        # PyPI already serves a CUDA build of torch on Linux and macOS has no
        # CUDA at all, so the PyTorch instructions come first there.
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform):
                warnings, _ = self._run("cpu", (12, 4), platform=platform)
                self.assertEqual(len(warnings), 1)
                self.assertIn("pytorch.org", warnings[0])
                self.assertIn("install.py", warnings[0])

    def test_the_refresh_command_matches_how_speakloop_is_installed(self):
        # The two forms are not interchangeable. An installed tool has no
        # interpreter on PATH that can import speakloop, and a checkout's venv
        # that happens to be active would rewrite a different file; out of a
        # clone the console script may not exist at all.
        with patch.object(detect_hardware.paths, "repo_mode",
                          return_value=True):
            self.assertEqual(detect_hardware._refresh_command(),
                             "`python -m speakloop.detect_hardware`")
        with patch.object(detect_hardware.paths, "repo_mode",
                          return_value=False):
            self.assertEqual(detect_hardware._refresh_command(),
                             "`speakloop --detect-hardware`")

    def test_staleness_is_decided_by_the_file_existing_at_all(self):
        # Not by what it says: the ambiguity comes from detect_device having
        # been allowed to short-circuit, which any existing file permits.
        missing = Path(__file__).with_name("no-such-hardware-config.json")
        with patch.object(detect_hardware, "OUTPUT_FILE", missing):
            self.assertFalse(detect_hardware._stored_device_may_be_stale())
        with patch.object(detect_hardware, "OUTPUT_FILE", Path(__file__)):
            self.assertTrue(detect_hardware._stored_device_may_be_stale())

    def test_a_machine_without_an_nvidia_driver_is_not_warned(self):
        # The overwhelmingly common case: no card, so the CPU is correct.
        warnings, _ = self._run("cpu", None)
        self.assertEqual(warnings, [])

    def test_a_gpu_run_is_silent(self):
        warnings, _ = self._run("cuda", (12, 4))
        self.assertEqual(warnings, [])

    def test_nvidia_smi_is_not_consulted_when_torch_uses_the_gpu(self):
        # Ordering, not cosmetics: this is what keeps the check free at every
        # startup on a machine that is already fine.
        _, detect = self._run("cuda", (12, 4))
        detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
