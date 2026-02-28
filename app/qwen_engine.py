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

MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "700"))


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
        text = text.strip()
        if not text:
            return []

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

    def _generate_audio(self, text: str, prompt_audio: Optional[Path] = None) -> tuple:
        if self.base_model is None:
            raise RuntimeError("Base model not loaded")

        if prompt_audio and prompt_audio.exists():
            prompt_wave, prompt_sr = sf.read(str(prompt_audio), dtype="float32")
            if len(prompt_wave.shape) > 1:
                prompt_wave = prompt_wave.mean(axis=1)
            wavs, sr = self.base_model.generate(
                text=text,
                reference_audio=prompt_wave,
                language="English",
            )
        else:
            wavs, sr = self.base_model.generate(
                text=text,
                language="English",
            )

        return wavs[0], sr

    def _generate_design(self, prompt: str, sample_text: str) -> tuple:
        if self.design_model is None:
            raise RuntimeError("Design model not loaded")

        wavs, sr = self.design_model.generate_custom_voice(
            text=sample_text,
            language="English",
            instruct=prompt,
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

        chunks = self._split_text_into_chunks(text)

        if not chunks:
            raise ValueError("No text to synthesize")

        audio_chunks = []

        loop = asyncio.get_event_loop()

        for i, chunk in enumerate(chunks):
            print(f"Generating chunk {i + 1}/{len(chunks)}: {chunk[:50]}...")

            def _gen():
                return self._generate_audio(chunk, anchor_path)

            audio, sr = await loop.run_in_executor(None, _gen)
            audio_chunks.append((audio, sr))

        if len(audio_chunks) > 1:
            silence = np.zeros(int(audio_chunks[0][1] * 0.3), dtype=np.float32)
            audio = np.concatenate(
                [audio_chunks[0][0]]
                + [silence + chunk[0] for chunk in audio_chunks[1:]]
            )
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
