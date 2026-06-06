"""
real_stt.py

Whisper-small speech-to-text helpers for the real-robot LILAC utterance demo.
"""

from __future__ import annotations

import numpy as np


WHISPER_SMALL_MODEL = "openai/whisper-small"
DEFAULT_STT_SAMPLE_RATE = 16000


class WhisperSmallSTT:
    """
    Small wrapper around the Hugging Face ASR pipeline.
    """

    def __init__(
            self,
            model_name=WHISPER_SMALL_MODEL,
            device="auto",
            language=None,
            task="transcribe",
            verbose=True,
        ):
        self.model_name = str(model_name)
        self.device = device
        self.language = language
        self.task = task
        self.verbose = bool(verbose)
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self

        from transformers import pipeline

        device_arg = self._resolve_device(self.device)
        try:
            self.pipe = pipeline(
                task   = "automatic-speech-recognition",
                model  = self.model_name,
                device = device_arg,
            )
        except Exception:
            if device_arg == -1:
                raise
            if self.verbose:
                print("[STT] failed on device %s; falling back to CPU." % str(device_arg))
            self.pipe = pipeline(
                task   = "automatic-speech-recognition",
                model  = self.model_name,
                device = -1,
            )

        if self.verbose:
            print("[STT] loaded %s" % self.model_name)
        return self

    @staticmethod
    def _resolve_device(device):
        if device != "auto":
            if str(device).lower() == "cpu":
                return -1
            return device

        import torch

        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return -1

    def record_audio(
            self,
            duration_sec=4.0,
            sample_rate=DEFAULT_STT_SAMPLE_RATE,
            input_device=None,
        ):
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(
                "sounddevice is required for Mac microphone input. "
                "Install it in ri_motion_v5_env before running the STT notebook."
            ) from exc

        n_samples = int(float(duration_sec) * int(sample_rate))
        if self.verbose:
            print("[STT] recording %.2f sec @ %d Hz" % (float(duration_sec), int(sample_rate)))
        audio = sd.rec(
            frames     = n_samples,
            samplerate = int(sample_rate),
            channels   = 1,
            dtype      = "float32",
            device     = input_device,
        )
        sd.wait()
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def transcribe_audio(self, audio, sample_rate=DEFAULT_STT_SAMPLE_RATE):
        self.load()

        generate_kwargs = {}
        if self.language:
            generate_kwargs["language"] = self.language
        if self.task:
            generate_kwargs["task"] = self.task

        payload = {
            "array": np.asarray(audio, dtype=np.float32).reshape(-1),
            "sampling_rate": int(sample_rate),
        }
        kwargs = {}
        if generate_kwargs:
            kwargs["generate_kwargs"] = generate_kwargs
        out = self.pipe(payload, **kwargs)
        text = out.get("text", "") if isinstance(out, dict) else str(out)
        return str(text).strip(), out

    def listen_and_transcribe(
            self,
            duration_sec=4.0,
            sample_rate=DEFAULT_STT_SAMPLE_RATE,
            input_device=None,
        ):
        audio = self.record_audio(
            duration_sec = duration_sec,
            sample_rate  = sample_rate,
            input_device = input_device,
        )
        return self.transcribe_audio(audio=audio, sample_rate=sample_rate)
