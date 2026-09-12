# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Speech recognition: faster-whisper on the language of the active profile."""

import numpy as np
# torch before faster_whisper, and the order must stay: on Windows the CUDA
# build of torch puts its cuBLAS and cuDNN libraries on the DLL search path when
# it is imported, and ctranslate2 (the engine of faster-whisper) needs them to
# run on CUDA.
import torch  # noqa: F401
from faster_whisper import WhisperModel

from speakloop import config


COMPUTE_TYPE = "float16" if config.STT_DEVICE == "cuda" else "int8"

# Technical transcription configuration. The audio rate is config.AUDIO_SAMPLE_RATE
# (16 kHz), which the Whisper architecture requires and the recorder delivers.
WHISPER_VAD_MIN_SPEECH_MS = 250   # Shortest duration considered as valid spoken word segments
WHISPER_VAD_MIN_SILENCE_MS = 500  # Silence gap thickness required before triggering split boundaries
WHISPER_VAD_SPEECH_PAD_MS = 300   # Padding attached around text fragments to avoid chopping words


class STTManager:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Instantiates the Whisper AI engine into memory."""
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.STT_DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=config.WHISPER_CPU_THREADS,
            num_workers=1,
        )

    def warm_up(self):
        """Runs a mock inference pass to eliminate initial latency."""
        if self.model is None:
            raise RuntimeError("STT model not loaded. Call load_model() first.")
        dummy_audio = np.zeros(config.AUDIO_SAMPLE_RATE, dtype=np.float32)
        list(self.model.transcribe(dummy_audio,
                                   language=config.WHISPER_LANGUAGE,
                                   beam_size=config.WHISPER_BEAM_SIZE,
                                   vad_filter=True)[0])

    def transcribe(self, audio: np.ndarray) -> str:
        """Passes the audio waveform data into Whisper for text extraction.

        The language is fixed to the one of the active profile
        (config.WHISPER_LANGUAGE) and never detected: the learner practices that
        language, and automatic detection on a short take of a beginner's speech
        lands on the wrong language often enough to break the lesson.
        """
        if self.model is None:
            raise RuntimeError("STT model not loaded. Call load_model() first.")

        segments, _info = self.model.transcribe(
            audio,
            language=config.WHISPER_LANGUAGE,
            task="transcribe",
            beam_size=config.WHISPER_BEAM_SIZE,
            vad_filter=True,
            vad_parameters={
                "min_speech_duration_ms": WHISPER_VAD_MIN_SPEECH_MS,
                "min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS,
                "speech_pad_ms": WHISPER_VAD_SPEECH_PAD_MS,
            },
            no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
            initial_prompt=config.WHISPER_INITIAL_PROMPT,
        )

        text = " ".join(segment.text.strip() for segment in segments).strip()
        return self._clean_transcript(text)

    def _clean_transcript(self, text: str) -> str:
        """Removes trailing structural whitespace and capitalizes the starting character."""
        if not text:
            return ""
        text = " ".join(text.split())
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text
