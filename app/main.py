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
from app.qwen_engine import qwen_engine, transcribe_with_timestamps, transcribe_audio
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


@app.delete("/v1/voices/{voice_id}")
async def delete_voice(voice_id: str, _: str = Depends(verify_api_key)):
    """Delete a voice by ID.
    
    Returns:
        JSON response with voice_id, name and deletion status
    """
    result = voice_store.delete_voice(voice_id)
    return result


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

        from app.qwen_engine import transcribe_audio

        ref_text = transcribe_audio(anchor_path)

        voice = voice_store.create_voice(
            name=name,
            anchor_wav_path=anchor_path,
            labels=parsed_labels,
            category="cloned",
            method="clone",
            source=url if url else "upload",
            ref_text=ref_text,
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


@app.post("/v1/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    language_code: Optional[str] = Form(None),
    timestamps_granularity: Optional[str] = Form("word"),
    diarize: Optional[bool] = Form(False),
    _: str = Depends(verify_api_key),
):
    """ElevenLabs-compatible speech-to-text transcription."""

    # Save uploaded file to temp
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audio_path = temp_path / file.filename

        # Write uploaded content
        content = await file.read()
        audio_path.write_bytes(content)

        # Run transcription
        result = transcribe_with_timestamps(
            audio_path=audio_path,
            language=language_code,
            timestamps_granularity=timestamps_granularity or "word",
            diarize=diarize or False,
        )

    return result


@app.post("/v1/voices/clone")
async def clone_voice_from_youtube(
    youtube_url: str = Form(...),
    name: str = Form("cloned_voice"),
    text: Optional[str] = Form(None),
    _: str = Depends(verify_api_key),
):
    """
    Clone a voice from a YouTube video in a single request.

    This endpoint:
    1. Downloads audio (first 60s) from YouTube
    2. Downloads subtitles from YouTube
    3. Extracts text from subtitles
    4. Creates a voice from the audio
    5. Generates TTS from the subtitle text (or custom text)
    6. Saves to /out/{name}.mp3

    Returns:
        JSON with success status, file path, voice_id, and transcript
    """
    import subprocess
    import re

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    temp_prefix = f"/tmp/{safe_name}"

    try:
        # Step 1: Download audio (first 60 seconds)
        print(f"[clone] Downloading audio from {youtube_url}")
        result = subprocess.run(
            [
                "yt-dlp",
                "-x",
                "--audio-format",
                "wav",
                "--download-sections",
                "*0-60",
                "-o",
                f"{temp_prefix}.wav",
                youtube_url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return {"error": "Failed to download audio", "details": result.stderr}

        # Step 2: Download subtitles
        print(f"[clone] Downloading subtitles from {youtube_url}")
        result = subprocess.run(
            [
                "yt-dlp",
                "--write-auto-subs",
                "--sub-lang",
                "en",
                "--convert-subs",
                "srt",
                "-o",
                temp_prefix,
                youtube_url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Step 3: Prepare audio for voice creation
        print(f"[clone] Preparing audio")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                f"{temp_prefix}.wav",
                "-t",
                "60",
                "-ar",
                "16000",
                "-ac",
                "1",
                f"{temp_prefix}_60s.wav",
            ],
            capture_output=True,
            timeout=60,
        )

        audio_path = Path(f"{temp_prefix}_60s.wav")
        if not audio_path.exists():
            return {"error": "Failed to prepare audio file"}

        # Step 4: Create voice
        print(f"[clone] Creating voice '{name}'")
        voice = voice_store.create_voice(
            name=name,
            anchor_wav_path=audio_path,
            labels={},
            category="cloned",
            method="clone",
        )
        voice_id = voice["voice_id"]

        # Step 5: Extract text from subtitles or use provided text
        if text:
            tts_text = text
        else:
            # Try to extract from subtitles
            srt_path = Path(f"{temp_prefix}.en.srt")
            if srt_path.exists():
                tts_text = extract_text_from_srt(srt_path)
            else:
                # Fallback: transcribe audio
                print(f"[clone] No subtitles found, using Whisper transcription")
                tts_text = transcribe_audio(audio_path)

        if not tts_text:
            return {
                "error": "No text available for TTS. Provide 'text' parameter or ensure YouTube has subtitles."
            }

        # Step 6: Generate TTS
        print(f"[clone] Generating TTS")
        audio_path_out, _ = await qwen_engine.synthesize(
            text=tts_text,
            voice_id=voice_id,
            voice_store=voice_store,
            voice_settings={},
            output_format="mp3_44100_128",
        )

        # Move to output
        output_path = Path(f"/out/{safe_name}_qwen3_v1.mp3")
        audio_path_out.rename(output_path)

        return {
            "success": True,
            "voice_id": voice_id,
            "voice_name": name,
            "file": str(output_path),
            "text_used": tts_text[:500] + "..." if len(tts_text) > 500 else tts_text,
        }

    except Exception as e:
        return {"error": str(e)}


def extract_text_from_srt(srt_path: Path) -> str:
    """Extract text from SRT subtitle file within first 60 seconds."""
    import re

    with open(srt_path, "r") as f:
        content = f.read()

    texts = []
    for block in content.split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            timecode = lines[1]
            match = re.match(
                r"(\d+):(\d+):(\d+),(\d+)", timecode.split("-->")[0].strip()
            )
            if match:
                hours, mins, secs, ms = (
                    int(match.group(1)),
                    int(match.group(2)),
                    int(match.group(3)),
                    int(match.group(4)),
                )
                total_secs = hours * 3600 + mins * 60 + secs + ms / 1000
                if total_secs <= 60:
                    text = " ".join(lines[2:]).strip()
                    if text:
                        texts.append(text)

    # Deduplicate
    seen = set()
    unique_texts = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique_texts.append(t)

    return " ".join(unique_texts)


def parse_youtube_timestamp(url: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse YouTube URL for start timestamp and optional duration.

    Returns: (start_seconds, end_seconds)
    - ?t=1028 or ?t=17m8s -> start at 1028 seconds
    - &t=1028 -> start at 1028 seconds
    - ?t=1028&dur=300 -> start at 1028, end at 1328
    """
    import re

    start_sec = None
    end_sec = None

    # Match ?t=1028 or &t=1028
    match = re.search(r"[?&]t=(\d+)", url)
    if match:
        start_sec = int(match.group(1))

    # Match ?t=17m8s or &t=17m8s
    match = re.search(r"[?&]t=(\d+)m(\d+)s", url)
    if match:
        start_sec = int(match.group(1)) * 60 + int(match.group(2))

    # Match ?dur=300 (duration from start)
    match = re.search(r"[?&]dur=(\d+)", url)
    if match:
        duration = int(match.group(1))
        if start_sec is not None:
            end_sec = start_sec + duration
        else:
            end_sec = duration

    return start_sec, end_sec


@app.post("/v1/voices/add-from-url")
async def add_voice_from_url(
    url: str = Form(...),
    name: str = Form("cloned_voice"),
    duration: Optional[int] = Form(None),
    _: str = Depends(verify_api_key),
):
    """
    Download audio from a URL (YouTube or direct) and create a voice clone.

    Supports YouTube timestamps:
    - https://youtu.be/VIDEO_ID?t=1028 (start at 17:08)
    - https://youtu.be/VIDEO_ID?t=5m30s (start at 5:30)
    - https://youtu.be/VIDEO_ID?t=100&dur=300 (start at 100s, duration 300s)

    Parameters:
    - url: Audio URL or YouTube URL (with optional timestamp)
    - name: Voice name
    - duration: Optional duration in seconds (default: 60 for YouTube)
    """
    import subprocess
    import re

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    temp_prefix = f"/tmp/{safe_name}"

    try:
        # Check if YouTube URL
        is_youtube = "youtube.com" in url or "youtu.be" in url

        if is_youtube:
            # Parse timestamp
            start_sec, end_sec = parse_youtube_timestamp(url)

            # Determine duration
            if duration:
                end_sec = (start_sec or 0) + duration

            # Build yt-dlp command
            cmd = ["yt-dlp", "-x", "--audio-format", "wav", "-o", f"{temp_prefix}.wav"]

            if start_sec is not None and end_sec:
                # Download specific section
                cmd.extend(["--download-sections", f"*{start_sec}-{end_sec}"])
            elif start_sec is not None:
                # Download from timestamp to end (or 60s default)
                dur = duration or 60
                cmd.extend(["--download-sections", f"*{start_sec}-{start_sec + dur}"])
            elif duration:
                # Download first N seconds
                cmd.extend(["--download-sections", f"*0-{duration}"])
            else:
                # Default: first 60 seconds
                cmd.extend(["--download-sections", "*0-60"])

            cmd.append(url)

            print(f"[add-from-url] Downloading from YouTube: {cmd}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                return {
                    "error": "Failed to download from YouTube",
                    "details": result.stderr,
                }
        else:
            # Direct URL
            result = subprocess.run(
                ["curl", "-L", "-o", f"{temp_prefix}.wav", url],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                return {"error": "Failed to download file", "details": result.stderr}

        # Prepare audio (convert to 16kHz mono)
        prepare_result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                f"{temp_prefix}.wav",
                "-ar",
                "16000",
                "-ac",
                "1",
                f"{temp_prefix}_16k.wav",
            ],
            capture_output=True,
            timeout=60,
        )

        audio_path = Path(f"{temp_prefix}_16k.wav")
        if not audio_path.exists():
            # Try original file
            audio_path = Path(f"{temp_prefix}.wav")
            if not audio_path.exists():
                return {"error": "Audio file not found after download"}

        # Create voice
        print(f"[add-from-url] Creating voice '{name}'")
        voice = voice_store.create_voice(
            name=name,
            anchor_wav_path=audio_path,
            labels={},
            category="cloned",
            method="clone",
        )

        return {
            "success": True,
            "voice_id": voice["voice_id"],
            "voice_name": name,
            "message": "Voice created successfully",
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/v1/voices/clone-multispeaker")
async def clone_multispeaker(
    youtube_url: str = Form(...),
    name_prefix: str = Form("speaker"),
    duration: int = Form(1200),
    _: str = Depends(verify_api_key),
):
    """
    Clone multiple voices from a YouTube video using speaker diarization.

    This endpoint:
    1. Downloads audio from YouTube (first N seconds, default 20 minutes)
    2. Runs speech-to-text with diarization to identify speakers
    3. Extracts audio per speaker
    4. Creates separate voice clones for each speaker
    5. Generates TTS samples for each voice

    Parameters:
    - youtube_url: YouTube video URL
    - name_prefix: Prefix for voice names (default: "speaker")
    - duration: How much of the video to process in seconds (default: 1200 = 20 minutes)

    Returns:
    - List of voices created with speaker IDs and file paths
    """
    import subprocess
    import re

    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", name_prefix)
    temp_prefix = f"/tmp/{safe_prefix}"

    try:
        # Step 1: Download audio
        print(f"[clone-multispeaker] Downloading {duration}s from YouTube")
        result = subprocess.run(
            [
                "yt-dlp",
                "-x",
                "--audio-format",
                "wav",
                "--download-sections",
                f"*0-{duration}",
                "-o",
                f"{temp_prefix}.wav",
                youtube_url,
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            return {"error": "Failed to download audio", "details": result.stderr}

        audio_path = Path(f"{temp_prefix}.wav")
        if not audio_path.exists():
            return {"error": "Audio file not found"}

        # Step 2: Run transcription with diarization
        print(f"[clone-multispeaker] Running speech-to-text with diarization")
        diarization_result = transcribe_with_timestamps(
            audio_path=audio_path,
            language="en",
            timestamps_granularity="word",
            diarize=True,
        )

        # Group words by speaker
        speaker_segments = {}
        for word in diarization_result.get("words", []):
            speaker = word.get("speaker", "SPEAKER_00")
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(word)

        if not speaker_segments:
            return {"error": "No speakers identified in audio"}

        print(
            f"[clone-multispeaker] Found {len(speaker_segments)} speakers: {list(speaker_segments.keys())}"
        )

        voices_created = []

        # Step 3-5: For each speaker, extract audio, create voice, generate TTS
        for i, (speaker_id, words) in enumerate(sorted(speaker_segments.items())):
            if not words:
                continue

            # Calculate time range for this speaker
            start_time = words[0]["start"]
            end_time = words[-1]["end"]

            # Add some padding
            start_time = max(0, start_time - 1)
            end_time = end_time + 1

            speaker_name = f"{safe_prefix}_{speaker_id.replace('SPEAKER_', '')}"
            speaker_audio_path = f"{temp_prefix}_{speaker_id}.wav"

            # Extract audio for this speaker
            print(f"[clone-multispeaker] Extracting audio for {speaker_id}")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ss",
                    str(start_time),
                    "-to",
                    str(end_time),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    speaker_audio_path,
                ],
                capture_output=True,
                timeout=60,
            )

            speaker_wav = Path(speaker_audio_path)
            if not speaker_wav.exists():
                print(
                    f"[clone-multispeaker] Warning: No audio extracted for {speaker_id}"
                )
                continue

            # Create voice
            print(f"[clone-multispeaker] Creating voice for {speaker_id}")
            voice = voice_store.create_voice(
                name=speaker_name,
                anchor_wav_path=speaker_wav,
                labels={"speaker": speaker_id},
                category="cloned",
                method="clone",
            )

            voice_id = voice["voice_id"]

            # Generate TTS - combine all text from this speaker
            speaker_text = " ".join([w["text"] for w in words if w.get("text")])

            # Repeat text to get ~30 seconds of audio
            text_needed = 30  # seconds
            chars_per_sec = (
                len(speaker_text) / (end_time - start_time)
                if end_time > start_time
                else 10
            )
            repeats = max(1, int(text_needed * chars_per_sec / len(speaker_text)))
            tts_text = " ".join([speaker_text] * repeats)

            print(f"[clone-multispeaker] Generating TTS for {speaker_id}")
            audio_path_out, _ = await qwen_engine.synthesize(
                text=tts_text,
                voice_id=voice_id,
                voice_store=voice_store,
                voice_settings={},
                output_format="mp3_44100_128",
            )

            # Save to output
            output_path = Path(f"/out/{speaker_name}_qwen3_v1.mp3")
            audio_path_out.rename(output_path)

            voices_created.append(
                {
                    "speaker_id": speaker_id,
                    "voice_id": voice_id,
                    "voice_name": speaker_name,
                    "file": str(output_path),
                    "text_used": speaker_text[:200] + "..."
                    if len(speaker_text) > 200
                    else speaker_text,
                }
            )

        # Cleanup temp files
        try:
            audio_path.unlink()
            for f in Path("/tmp").glob(f"{safe_prefix}_SPEAKER_*.wav"):
                f.unlink()
        except:
            pass

        return {
            "success": True,
            "num_speakers": len(voices_created),
            "voices": voices_created,
        }

    except Exception as e:
        return {"error": str(e)}


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
