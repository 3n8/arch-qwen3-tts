import os
import io
import json
import asyncio
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import Response, StreamingResponse, JSONResponse
import soundfile as sf

from app.auth import verify_api_key
from app.voice_store import voice_store
from app.qwen_engine import qwen_engine
from app.preprocess import preprocess_audio, download_file
from app.audio_formats import encode_audio, get_content_type


app = FastAPI(
    title="Qwen3-TTS API",
    description="Self-hosted Qwen3-TTS service - ElevenLabs-compatible API",
    version="1.0.0",
)

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


class TTSRequest(BaseModel):
    text: str
    model_id: Optional[str] = None
    voice_settings: Optional[Dict[str, Any]] = None
    output_format: str = "mp3_44100_128"


class DesignVoiceRequest(BaseModel):
    name: str
    prompt: str
    sample_text: Optional[str] = None
    labels: Optional[Dict[str, str]] = None


class BatchTTSRequest(BaseModel):
    texts: List[str]
    output_format: str = "mp3_44100_128"
    voice_settings: Optional[Dict[str, Any]] = None


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/voices")
async def list_voices(_: str = Depends(verify_api_key)):
    voices = voice_store.list_voices()
    return {"voices": voices}


@app.get("/v1/voices/{voice_id}")
async def get_voice(voice_id: str, _: str = Depends(verify_api_key)):
    voice = voice_store.get_voice(voice_id)
    return voice


@app.post("/v1/text-to-speech/{voice_id}")
async def text_to_speech(
    voice_id: str,
    request: TTSRequest,
    _: str = Depends(verify_api_key),
):
    voice_settings = request.voice_settings or {}

    async with semaphore:
        audio_path, job_metadata = await qwen_engine.synthesize(
            text=request.text,
            voice_id=voice_id,
            voice_store=voice_store,
            voice_settings=voice_settings,
            output_format=request.output_format,
        )

    encoded_path, content_type = encode_audio(audio_path, request.output_format)

    audio_data = encoded_path.read_bytes()
    encoded_path.unlink()
    audio_path.unlink()

    return Response(
        content=audio_data,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{encoded_path.suffix[1:]}"
        },
    )


@app.post("/v1/text-to-speech/{voice_id}/stream")
async def text_to_speech_stream(
    voice_id: str,
    request: TTSRequest,
    _: str = Depends(verify_api_key),
):
    voice_settings = request.voice_settings or {}

    async with semaphore:
        audio_path, job_metadata = await qwen_engine.synthesize(
            text=request.text,
            voice_id=voice_id,
            voice_store=voice_store,
            voice_settings=voice_settings,
            output_format=request.output_format,
        )

    encoded_path, content_type = encode_audio(audio_path, request.output_format)

    audio_data = encoded_path.read_bytes()
    encoded_path.unlink()
    audio_path.unlink()

    return Response(content=audio_data, media_type=content_type)


@app.get("/v1/models")
async def list_models(_: str = Depends(verify_api_key)):
    return {
        "models": [
            {
                "model_id": "qwen3-tts-1.7b-base",
                "name": "Qwen3-TTS 12Hz 1.7B Base",
                "can_be_cloned": True,
                "max_characters": 4096,
                "description": "Qwen3-TTS Base model for voice cloning",
            },
            {
                "model_id": "qwen3-tts-1.7b-voicedesign",
                "name": "Qwen3-TTS 12Hz 1.7B VoiceDesign",
                "can_be_cloned": False,
                "max_characters": 4096,
                "description": "Qwen3-TTS VoiceDesign model for AI voice generation",
            },
        ]
    }


@app.post("/v1/voices/add")
async def add_voice(
    name: str = Form(...),
    labels: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    _: str = Depends(verify_api_key),
):
    if not file and not url:
        raise HTTPException(
            status_code=400, detail="Either 'file' or 'url' must be provided"
        )

    parsed_labels = {}
    if labels:
        try:
            parsed_labels = json.loads(labels)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid labels JSON")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        if file:
            input_path = temp_path / file.filename
            content = await file.read()
            input_path.write_bytes(content)
        elif url:
            input_path = temp_path / "input_audio"
            await download_file(url, input_path)

        anchor_path = await preprocess_audio(
            input_path, enable_vad=True, temp_dir=temp_path
        )

        voice = voice_store.create_voice(
            name=name,
            anchor_wav_path=anchor_path,
            labels=parsed_labels,
            category="cloned",
            method="clone",
            source=url if url else "upload",
        )

        return voice


@app.post("/v1/voices/design")
async def design_voice(
    request: DesignVoiceRequest,
    _: str = Depends(verify_api_key),
):
    async with semaphore:
        (audio, sr), design_info = await qwen_engine.design_voice(
            name=request.name,
            prompt=request.prompt,
            sample_text=request.sample_text,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        anchor_path = temp_path / "anchor.wav"
        sf.write(str(anchor_path), audio, sr)

        voice = voice_store.create_voice(
            name=request.name,
            anchor_wav_path=anchor_path,
            labels=request.labels or {},
            category="designed",
            method="design",
            prompts={
                "prompt": request.prompt,
                "sample_text": design_info.get("sample_text"),
            },
        )

        return voice


@app.post("/v1/text-to-speech/{voice_id}/batch")
async def batch_tts(
    voice_id: str,
    request: BatchTTSRequest,
    _: str = Depends(verify_api_key),
):
    voice_settings = request.voice_settings or {}

    output_paths = []

    async with semaphore:
        for i, text in enumerate(request.texts):
            audio_path, job_metadata = await qwen_engine.synthesize(
                text=text,
                voice_id=voice_id,
                voice_store=voice_store,
                voice_settings=voice_settings,
                output_format=request.output_format,
            )
            encoded_path, _ = encode_audio(audio_path, request.output_format)
            output_paths.append(encoded_path)
            audio_path.unlink()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in output_paths:
            zf.write(path, path.name)
            path.unlink()

    zip_buffer.seek(0)

    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=batch_output.zip"},
    )


@app.get("/v1/user/subscription")
async def get_subscription(_: str = Depends(verify_api_key)):
    return {
        "tier": "self_hosted",
        "character_count": 999999999,
        "character_limit": 999999999,
        "can_extend_character_limit": False,
        "allowed_to_extend_character_limit": False,
        "next_character_count_reset_unix": None,
        "voice_limit": 999999,
        "max_voice_add_edits": 999999,
        "available_models": ["qwen3-tts-1.7b-base", "qwen3-tts-1.7b-voicedesign"],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "3004"))
    uvicorn.run(app, host="0.0.0.0", port=port)
