import numpy as np
from faster_whisper import WhisperModel
import config

class STTManager:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Instantiates the Whisper AI engine into memory."""
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.DEVICE,
            compute_type=config.COMPUTE_TYPE,
            cpu_threads=4,
            num_workers=1,
        )

    def warm_up(self):
        """Runs a mock inference pass to eliminate initial latency."""
        assert self.model is not None
        dummy_audio = np.zeros(config.WHISPER_SAMPLE_RATE, dtype=np.float32)
        list(self.model.transcribe(dummy_audio, language=config.TARGET_LANG_CODE, beam_size=1, vad_filter=True)[0])

    def transcribe(self, audio: np.ndarray) -> str:
        """Passes the audio waveform data into Whisper for text extraction."""
        assert self.model is not None

        segments, _info = self.model.transcribe(
            audio,
            language=config.TARGET_LANG_CODE,
            task="transcribe",
            beam_size=1,
            vad_filter=True,
            vad_parameters={
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 300,
            },
            no_speech_threshold=0.45,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
            initial_prompt=(
                f"This is a conversation with a {config.TARGET_LANGUAGE} tutor. "
                f"The speaker is practicing simple phrases."
            ),
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
		