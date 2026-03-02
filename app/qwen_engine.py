import os
import re
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import tempfile
import subprocess

import torch
import numpy as np
import soundfile as sf


MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
HF_CACHE_DIR = Path(os.getenv("HF_CACHE_DIR", "/cache/huggingface"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/out"))

MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1500"))

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

_voice_clone_prompts: Dict[str, Any] = {}
_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel

            print(f"Loading Whisper model: {WHISPER_MODEL}")
            _whisper_model = WhisperModel(
                WHISPER_MODEL, device="auto", compute_type="int8"
            )
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Whisper model: {e}")
            _whisper_model = False
    return _whisper_model


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio using Whisper to get ref_text for ICL mode."""
    model = _get_whisper_model()
    if not model:
        return ""

    try:
        segments, info = model.transcribe(str(audio_path), language="en")
        transcript = " ".join([segment.text for segment in segments])
        print(f"Transcribed: {transcript[:100]}...")
        return transcript.strip()
    except Exception as e:
        print(f"Warning: Transcription failed: {e}")
        return ""


_pyannote_pipeline = None


def _get_pyannote_pipeline():
    global _pyannote_pipeline
    if _pyannote_pipeline is None:
        try:
            from pyannote.audio import Pipeline

            print("Loading pyannote speaker diarization pipeline...")
            _pyannote_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=None
            )
            print("Pyannote pipeline loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load pyannote pipeline: {e}")
            _pyannote_pipeline = False
    return _pyannote_pipeline


def transcribe_with_timestamps(
    audio_path: Path,
    language: Optional[str] = None,
    timestamps_granularity: str = "word",
    diarize: bool = False,
) -> Dict[str, Any]:
    """
    Transcribe audio with word-level timestamps and optionally diarization.

    Returns:
        {
            "text": "transcribed text",
            "language_code": "en",
            "language_probability": 0.99,
            "words": [{"text": "...", "start": 0.0, "end": 0.5, "type": "word", "logprob": -0.5, ...}]
        }
    """
    import torch

    model = _get_whisper_model()
    if not model:
        return {
            "text": "",
            "language_code": "en",
            "language_probability": 0.0,
            "words": [],
        }

    try:
        # Run transcription
        segments, info = model.transcribe(str(audio_path), language=language or "en")

        language_code = info.language or "en"
        language_prob = info.language_probability or 0.0

        words = []
        full_text_parts = []

        for segment in segments:
            if timestamps_granularity == "word" and segment.words:
                for word in segment.words:
                    word_data = {
                        "text": word.word,
                        "start": word.start,
                        "end": word.end,
                        "type": "word",
                        "logprob": word.probability
                        if hasattr(word, "probability")
                        else -0.5,
                    }
                    words.append(word_data)
                    full_text_parts.append(word.word)
            else:
                # No word-level timestamps, just use segment
                if segment.text:
                    words.append(
                        {
                            "text": segment.text.strip(),
                            "start": segment.start,
                            "end": segment.end,
                            "type": "word",
                            "logprob": -0.5,
                        }
                    )
                    full_text_parts.append(segment.text)

        # Run diarization if requested
        speaker_labels = {}
        if diarize:
            pipeline = _get_pyannote_pipeline()
            if pipeline:
                try:
                    # Run on CPU since we're already using GPU for whisper if available
                    diarization = pipeline(audio_path, min_speakers=1, max_speakers=10)

                    # Create a mapping of time -> speaker
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        for word in words:
                            if word["start"] >= turn.start and word["end"] <= turn.end:
                                word["speaker"] = speaker
                                speaker_labels[speaker] = True
                except Exception as e:
                    print(f"Warning: Diarization failed: {e}")

        text = " ".join(full_text_parts)

        return {
            "text": text,
            "language_code": language_code,
            "language_probability": language_prob,
            "words": words,
        }

    except Exception as e:
        print(f"Warning: Transcription with timestamps failed: {e}")
        return {
            "text": "",
            "language_code": "en",
            "language_probability": 0.0,
            "words": [],
        }


class QwenEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = None
        self.design_model = None
        self.base_model_loaded = False
        self.design_model_loaded = False
        self._lock = asyncio.Lock()

    def _get_model_cache_path(self, model_name: str) -> Path:
        safe_name = model_name.replace("/", "_")
        return MODELS_DIR / safe_name

    def _resolve_revision(self, env_var: str, default: str) -> str:
        rev = os.getenv(env_var, "")
        return rev if rev else default

    async def load_base_model(self):
        async with self._lock:
            if self.base_model_loaded:
                return

            revision = self._resolve_revision("HF_REV_BASE", "main")

            print(f"Loading Qwen3-TTS Base model from {MODEL_BASE}")

            loop = asyncio.get_event_loop()

            def _load():
                from qwen_tts import Qwen3TTSModel

                self.base_model = Qwen3TTSModel.from_pretrained(
                    MODEL_BASE,
                    device_map=self.device,
                    dtype=torch.bfloat16,
                    cache_dir=str(HF_CACHE_DIR),
                )

            await loop.run_in_executor(None, _load)
            self.base_model_loaded = True
            print("Base model loaded successfully")

    async def load_design_model(self):
        async with self._lock:
            if self.design_model_loaded:
                return

            print(f"Loading Qwen3-TTS VoiceDesign model from {MODEL_DESIGN}")

            loop = asyncio.get_event_loop()

            def _load():
                from qwen_tts import Qwen3TTSModel

                self.design_model = Qwen3TTSModel.from_pretrained(
                    MODEL_DESIGN,
                    device_map=self.device,
                    dtype=torch.bfloat16,
                    cache_dir=str(HF_CACHE_DIR),
                )

            await loop.run_in_executor(None, _load)
            self.design_model_loaded = True
            print("Design model loaded successfully")

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks for TTS generation.
        Uses sentence-based splitting for natural flow.
        """
        text = text.strip()
        if not text:
            return []

        # Split on sentence endings
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= MAX_CHUNK_CHARS:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(sentence) > MAX_CHUNK_CHARS:
                    words = sentence.split()
                    current_chunk = ""
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= MAX_CHUNK_CHARS:
                            current_chunk += " " + word if current_chunk else word
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _build_voice_clone_prompt(
        self, prompt_audio: Path, ref_text: str = None
    ) -> Any:
        """Build and cache voice clone prompt for a given anchor audio."""
        voice_key = str(prompt_audio)

        if voice_key in _voice_clone_prompts:
            print(f"Using cached voice clone prompt for {prompt_audio.name}")
            return _voice_clone_prompts[voice_key]

        if self.base_model is None:
            raise RuntimeError("Base model not loaded")

        prompt_wave, prompt_sr = sf.read(str(prompt_audio), dtype="float32")
        if len(prompt_wave.shape) > 1:
            prompt_wave = prompt_wave.mean(axis=1)

        if not ref_text:
            ref_text = transcribe_audio(prompt_audio)
            if not ref_text:
                print("Warning: No ref_text available, using x_vector_only_mode=True")
                return None

        print(f"Building voice clone prompt with ref_text: {ref_text[:50]}...")

        prompt = self.base_model.create_voice_clone_prompt(
            ref_audio=(prompt_wave, prompt_sr),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

        _voice_clone_prompts[voice_key] = prompt
        print(f"Cached voice clone prompt for {prompt_audio.name}")

        return prompt

    def _generate_audio(
        self,
        text: str,
        prompt_audio: Optional[Path] = None,
        ref_text: Optional[str] = None,
        voice_prompt: Any = None,
    ) -> tuple:
        if self.base_model is None:
            raise RuntimeError("Base model not loaded")

        generation_kwargs = {
            "temperature": 0.8,
            "top_p": 0.95,
            "do_sample": True,
        }

        if voice_prompt is not None:
            wavs, sr = self.base_model.generate_voice_clone(
                text=text,
                voice_clone_prompt=voice_prompt,
                language="English",
                **generation_kwargs,
            )
        elif prompt_audio and prompt_audio.exists():
            prompt_wave, prompt_sr = sf.read(str(prompt_audio), dtype="float32")
            if len(prompt_wave.shape) > 1:
                prompt_wave = prompt_wave.mean(axis=1)

            if ref_text:
                wavs, sr = self.base_model.generate_voice_clone(
                    text=text,
                    ref_audio=(prompt_wave, prompt_sr),
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                    language="English",
                    **generation_kwargs,
                )
            else:
                wavs, sr = self.base_model.generate_voice_clone(
                    text=text,
                    ref_audio=(prompt_wave, prompt_sr),
                    ref_text=text,
                    x_vector_only_mode=True,
                    language="English",
                    **generation_kwargs,
                )
        else:
            wavs, sr = self.base_model.generate_voice_clone(
                text=text,
                language="English",
                **generation_kwargs,
            )

        return wavs[0], sr

    def _generate_design(self, prompt: str, sample_text: str) -> tuple:
        if self.design_model is None:
            raise RuntimeError("Design model not loaded")

        generation_kwargs = {
            "temperature": 0.8,
            "top_p": 0.95,
            "do_sample": True,
        }

        wavs, sr = self.design_model.generate_voice_design(
            text=sample_text,
            instruct=prompt,
            language="English",
            **generation_kwargs,
        )

        return wavs[0], sr

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        voice_store,
        voice_settings: Optional[Dict[str, Any]] = None,
        output_format: str = "mp3_44100_128",
    ) -> tuple[Path, Dict[str, Any]]:
        await self.load_base_model()

        anchor_path = voice_store.get_anchor_wav_path(voice_id)

        voice_metadata = voice_store.get_voice(voice_id)
        ref_text = voice_metadata.get("metadata", {}).get("ref_text")

        voice_prompt = None
        if anchor_path.exists():
            loop = asyncio.get_event_loop()

            def build_prompt():
                return self._build_voice_clone_prompt(anchor_path, ref_text)

            voice_prompt = await loop.run_in_executor(None, build_prompt)

        chunks = self._split_text_into_chunks(text)

        if not chunks:
            raise ValueError("No text to synthesize")

        audio_chunks = []

        loop = asyncio.get_event_loop()

        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i + 1}/{len(chunks)}: {chunk[:50]}...")

            def _gen():
                return self._generate_audio(chunk, anchor_path, ref_text, voice_prompt)

            audio, sr = await loop.run_in_executor(None, _gen)
            audio_chunks.append((audio, sr))

        if len(audio_chunks) > 1:
            target_sr = audio_chunks[0][1]

            from scipy import signal

            resampled_chunks = []
            for i, (audio, sr) in enumerate(audio_chunks):
                if sr != target_sr:
                    num_samples = int(len(audio) * target_sr / sr)
                    audio = signal.resample(audio, num_samples)
                if i > 0:
                    silence = np.zeros(int(target_sr * 0.3), dtype=np.float32)
                    audio = np.concatenate([silence, audio])
                resampled_chunks.append(audio)

            audio = np.concatenate(resampled_chunks)
            sr = target_sr
        else:
            audio = audio_chunks[0][0]
            sr = audio_chunks[0][1]

        job_id = str(uuid.uuid4())
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = OUT_DIR / f"{job_id}.wav"
        sf.write(str(audio_path), audio, sr)

        metadata = {
            "job_id": job_id,
            "voice_id": voice_id,
            "text": text,
            "chunks": chunks,
            "voice_settings": voice_settings or {},
            "output_format": output_format,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        job_path = OUT_DIR / "jobs" / f"{job_id}.json"
        job_path.parent.mkdir(parents=True, exist_ok=True)
        with open(job_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return audio_path, metadata

    async def design_voice(
        self, name: str, prompt: str, sample_text: Optional[str] = None
    ) -> tuple:
        await self.load_design_model()

        if not sample_text:
            sample_text = (
                "Hello, this is a test of the text to speech system. "
                "The quick brown fox jumps over the lazy dog. "
                "She sells seashells by the seashore. "
                "How much wood would a woodchuck chuck if a woodchuck could chuck wood."
            )

        loop = asyncio.get_event_loop()

        def _generate():
            return self._generate_design(prompt, sample_text)

        audio, sr = await loop.run_in_executor(None, _generate)

        return (audio, sr), {"prompt": prompt, "sample_text": sample_text, "name": name}


qwen_engine = QwenEngine()
