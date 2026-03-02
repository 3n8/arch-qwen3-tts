# Qwen3-TTS

Self-hosted Qwen3-TTS service built on [arch-base-image](https://github.com/3n8/arch-base-image), intended as a practical drop-in replacement for ElevenLabs TTS (ComfyUI nodes + typical web apps).

## Features

- **Best Audio Quality**: Uses Qwen3-TTS 12Hz 1.7B Base and VoiceDesign models
- **Voice Persistence**: Anchor.wav pattern ensures consistent voice identity across restarts
- **ElevenLabs-Compatible API**: Works with common ElevenLabs clients
- **GPU-Accelerated**: AMD GPU with ROCm 6.4 support
- **Media Preprocessing**: FFmpeg + VAD trimming for clean voice clones
- **Arch Linux Base**: Built on [arch-base-image](https://github.com/3n8/arch-base-image) for minimal image size
- **Speech-to-Text**: Transcription with speaker diarization support
- **YouTube Integration**: Download audio and subtitles directly from YouTube

## Tools Included

| Tool | Purpose |
|------|---------|
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Download audio/video from YouTube and other sites |
| [ffmpeg](https://ffmpeg.org) | Audio/video processing, format conversion |
| [Qwen3-TTS](https://github.com/Qwen/Qwen3-TTS) | Text-to-speech generation with voice cloning |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Speech-to-text transcription (fallback) |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker diarization (who spoke when) |

## Requirements

- Host: Linux with ROCm 6.4 support
- GPU: AMD GPU with ROCm 6.4 (RX 7000 series, MI300X, etc.)
- Docker + Docker Compose

## First-Time Setup

Create the required directories on the host with correct ownership:

```bash
sudo mkdir -p /opt/appdata/qwen3-tts/{config,models,voices,out,hf_cache}
sudo chown -R 1050:1050 /opt/appdata/qwen3-tts
```

Replace `1050:1050` with your actual PUID:PGID.

## Running

Pull and run the container:

```bash
docker compose up -d
```

Or create a `docker-compose.yml`:

```yaml
services:
  arch-qwen3-tts:
    image: ghcr.io/3n8/arch-qwen3-tts:latest
    container_name: qwen3-tts
    restart: unless-stopped
    user: "${PUID}:${PGID}"
    environment:
      - TZ=Europe/Oslo
      - PORT=3004
      - TTS_API_KEY=sk-your-api-key-here
    volumes:
      - /opt/appdata/qwen3-tts/config:/config
      - /opt/appdata/qwen3-tts/models:/models
      - /opt/appdata/qwen3-tts/voices:/voices
      - /opt/appdata/qwen3-tts/out:/out
      - /opt/appdata/qwen3-tts/hf_cache:/.cache/huggingface
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    ipc: host
    group_add:
      - video
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3004/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

Then start the container:

```bash
docker compose up -d
```

### AMD GPU Troubleshooting

If you encounter HIP/GPU errors with your AMD GPU, you may need to set:

```yaml
- HSA_OVERRIDE_GFX_VERSION=11.0.0
```

View logs with:

```bash
docker logs qwen3-tts
```

## Quick Guide (One Command Voice Cloning)

Clone a voice from a YouTube video and generate speech in a single command.

### Clone Voice from YouTube

```bash
curl -s -X POST "http://localhost:3004/v1/voices/clone" \
  -H "x-tts-api-key: YOUR_API_KEY" \
  -F "youtube_url=YOUTUBE_URL" \
  -F "name=voice_name"
```

**Parameters:**
- `youtube_url` (required): YouTube video URL
- `name` (optional): Name for the voice (default: "cloned_voice")
- `text` (optional): Custom text for TTS. If not provided, uses subtitle text from YouTube

**What it does:**
1. Downloads first 60 seconds of audio from YouTube
2. Downloads subtitles from YouTube
3. Creates a voice clone from the audio
4. Generates TTS using the subtitle text (or custom text if provided)
5. Saves to `/out/voice_name_qwen3_v1.mp3`

**Example:**
```bash
curl -s -X POST "http://localhost:3004/v1/voices/clone" \
  -H "x-tts-api-key: VJcCJrs46L7bdzKrSgsbYlrgZ" \
  -F "youtube_url=https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID" \
  -F "name=my_new_voice"
```

**Response:**
```json
{
  "success": true,
  "voice_id": "abc123-def456...",
  "voice_name": "my_new_voice",
  "file": "/out/my_new_voice_qwen3_v1.mp3",
  "text_used": "Hello world this is a test..."
}
```

## Testing with curl

Base URL: `http://localhost:3004`
API Key: Set via `x-tts-api-key` header

### Health Check

```bash
curl -s http://localhost:3004/healthz
```

### List Voices

```bash
curl -s -H "x-tts-api-key: your-secret-api-key" \
  http://localhost:3004/v1/voices
```

### Create Voice from Audio File

```bash
curl -s -X POST "http://localhost:3004/v1/voices/add" \
  -H "x-tts-api-key: your-secret-api-key" \
  -F "name=My Voice" \
  -F 'labels={"gender": "male"}' \
  -F "file=@/path/to/audio.wav"
```

### Create Voice from URL

```bash
curl -s -X POST "http://localhost:3004/v1/voices/add" \
  -H "x-tts-api-key: your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "Voice from URL", "url": "https://example.com/speech.wav"}'
```

### Design New Voice (AI-Generated)

```bash
curl -s -X POST "http://localhost:3004/v1/voices/design" \
  -H "x-tts-api-key: your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AI Voice",
    "prompt": "A friendly male voice",
    "sample_text": "Hello, this is a test of the text to speech system."
  }'
```

### Synthesize Speech (MP3)

```bash
curl -s -X POST "http://localhost:3004/v1/text-to-speech/voice_id_here" \
  -H "x-tts-api-key: your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "output_format": "mp3_44100_128"}' \
  -o speech.mp3
```

### Synthesize Speech (WAV)

```bash
curl -s -X POST "http://localhost:3004/v1/text-to-speech/voice_id_here" \
  -H "x-tts-api-key: your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "output_format": "wav"}' \
  -o speech.wav
```

### Batch TTS

```bash
curl -s -X POST "http://localhost:3004/v1/text-to-speech/voice_id_here/batch" \
  -H "x-tts-api-key: your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["First sentence.", "Second sentence."],
    "output_format": "mp3_44100_128"
  }' \
  -o batch_output.zip
```

### Speech-to-Text Transcription

Transcribe audio with word-level timestamps and optional speaker diarization.

```bash
curl -s -X POST "http://localhost:3004/v1/speech-to-text" \
  -H "x-tts-api-key: your-secret-api-key" \
  -F "file=@/path/to/audio.wav" \
  -F "timestamps_granularity=word" \
  -F "diarize=false"
```

**Parameters:**
- `file` (required): Audio file to transcribe
- `language_code` (optional): ISO language code (e.g., "en"). Auto-detected if not provided
- `timestamps_granularity` (optional): "word" (default) or "character"
- `diarize` (optional): true/false - Add speaker labels to each word

**Response:**
```json
{
  "text": "Transcribed text here...",
  "language_code": "en",
  "language_probability": 0.99,
  "words": [
    {
      "text": "Hello",
      "start": 0.0,
      "end": 0.5,
      "type": "word",
      "logprob": -0.5,
      "speaker": "SPEAKER_00"  // only when diarize=true
    }
  ]
}
```

### Get Subscription (ElevenLabs Compat)

```bash
curl -s -H "x-tts-api-key: your-secret-api-key" \
  http://localhost:3004/v1/user/subscription
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3004 | Service port |
| `TZ` | UTC | Timezone |
| `TTS_API_KEY` | (required) | API key for authentication |
| `HIP_VISIBLE_DEVICES` | 0 | GPU device ID |
| `HSA_OVERRIDE_GFX_VERSION` | (empty) | Set to 11.0.0 if AMD GPU not detected |
| `MAX_CONCURRENCY` | 1 | Max concurrent synthesis requests |
| `MAX_CHUNK_CHARS` | 700 | Max characters per text chunk |
| `HF_HOME` | /.cache/huggingface | HuggingFace cache location |
| `HF_CACHE_DIR` | /.cache/huggingface | HuggingFace cache location |
| `HF_REV_BASE` | (empty) | HuggingFace revision for Base model |
| `HF_REV_DESIGN` | (empty) | HuggingFace revision for Design model |

## Running as Non-Root User

This container **MUST** run as a non-root user. The `user:` directive is set in docker-compose:

```yaml
user: "${PUID}:${PGID}"
```

### Limitations when running as non-root:

1. **Pre-created directories**: Host directories MUST be created with correct ownership before first run (see First-Time Setup)
2. **Port binding**: Cannot bind to privileged ports (<1024)
3. **GPU access**: May need additional capabilities or group_add for GPU access

If volumes aren't pre-owned, the app will log warnings but attempt to continue.

## Output Formats

Supported output formats:
- `mp3_44100_128` (default)
- `mp3_44100_64`
- `mp3_22050_32`
- `wav`
- `pcm_16000`
- `pcm_24000`
- `ogg_vorbis`
- `flac`

## Voice Storage

Voices are stored at `/voices/{voice_id}/`:
- `anchor.wav` - Canonical mono 16kHz PCM S16LE
- `metadata.json` - Voice metadata

## Models

- **Qwen/Qwen3-TTS-12Hz-1.7B-Base**: Production generation (cloning from anchor.wav)
- **Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign**: AI voice generation (design endpoint)

Models are cached at `/models` and HuggingFace cache at `/.cache/huggingface`.

## Creating High-Quality Voice Clones

This guide explains how to create the best possible voice clones using Qwen3-TTS with In-Context Learning (ICL) mode.

### Why ICL Mode Matters

The system now uses **ICL (In-Context Learning) mode** which:
- Preserves the original speaker's **speaking pace/rhythm**
- Maintains **prosody and intonation**
- Keeps **natural pauses**
- Retains **emotional tone**

Without ICL, voices often speak too fast compared to the original.

### Full Guide

```bash
# Download audio from YouTube
yt-dlp -x --audio-format wav "VIDEO_URL" -o /tmp/voice_name.wav

# Trim to 2-5 minutes (longer = better quality)
ffmpeg -y -i /tmp/voice_name.wav -t 300 -ar 16000 -ac 1 /tmp/voice_trimmed.wav

# Copy to hel and create voice
docker cp /tmp/voice_trimmed.wav hel:/tmp/
ssh hel 'docker exec qwen3-tts curl -s -X POST "http://localhost:3004/v1/voices/add" \
  -H "x-tts-api-key: YOUR_API_KEY" \
  -F "name=Voice Name" \
  -F "file=@/tmp/voice_trimmed.wav"'

# Generate speech
ssh hel 'docker exec qwen3-tts curl -s -X POST "http://localhost:3004/v1/text-to-speech/Voice_ID" \
  -H "x-tts-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Hello there good boy I have been waiting for you\", \"output_format\": \"mp3_44100_128\"}" \
  -o /out/output.mp3'
```

### Quick YouTube Clone Workflow (60-second samples)

This is the streamlined workflow for creating voice clones from YouTube videos with 60-second samples.

**Important: We use YouTube subtitles as the primary text source, NOT Whisper transcription.** This is faster and more accurate for YouTube content.

#### Step 1: Download Audio (first 60 seconds)

```bash
yt-dlp -x --audio-format wav --download-sections "*0-60" -o /tmp/{voice_name}.wav {youtube_url}
```

Example:
```bash
yt-dlp -x --audio-format wav --download-sections "*0-60" -o /tmp/voice_name.wav YOUTUBE_URL
```

#### Step 2: Download Subtitles

**Always prefer subtitles over Whisper transcription** - they're faster and more accurate for YouTube content.

```bash
yt-dlp --write-auto-subs --sub-lang en --convert-subs srt -o /tmp/{voice_name} {youtube_url}
```

#### Step 3: Prepare Audio

Trim to 60 seconds, 16kHz mono for voice cloning:

```bash
ffmpeg -y -i /tmp/{voice_name}.wav -t 60 -ar 16000 -ac 1 /tmp/{voice_name}_60s.wav
```

#### Step 4: Extract Text from Subtitles

Use Python to extract text from the first 60 seconds of subtitles:

```python
import re

with open("/tmp/{voice_name}.en.srt", "r") as f:
    content = f.read()

texts = []
for block in content.split("\n\n"):
    lines = block.strip().split("\n")
    if len(lines) >= 3:
        timecode = lines[1]
        match = re.match(r"(\d+):(\d+):(\d+),(\d+)", timecode.split("-->")[0].strip())
        if match:
            hours, mins, secs, ms = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            total_secs = hours * 3600 + mins * 60 + secs + ms/1000
            if total_secs <= 60:
                text = " ".join(lines[2:]).strip()
                if text:
                    texts.append(text)

# Deduplicate while preserving order
seen = set()
unique_texts = []
for t in texts:
    if t not in seen:
        seen.add(t)
        unique_texts.append(t)

result = " ".join(unique_texts)
print(result)
```

#### Step 5: Create Voice

```bash
curl -s -X POST "http://localhost:3004/v1/voices/add" \
  -H "x-tts-api-key: YOUR_API_KEY" \
  -F "name={voice_name}" \
  -F "file=@/tmp/{voice_name}_60s.wav"
```

Save the returned `voice_id` for the next step.

#### Step 6: Generate TTS

Using Python (recommended for longer text):

```python
import requests

text = "Your extracted subtitle text here"
voice_id = "voice_id_from_step_5"
url = f"http://localhost:3004/v1/text-to-speech/{voice_id}"
headers = {
    "x-tts-api-key": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
data = {
    "text": text,
    "output_format": "mp3_44100_128"
}

resp = requests.post(url, headers=headers, json=data)
with open("/out/{voice_name}_qwen3_v1.mp3", "wb") as f:
    f.write(resp.content)
```

**Important Notes:**
- If the API returns 500 error, the text is too long. Split into shorter chunks and concatenate results.
- For best results, use 30-60 seconds of clean audio for voice cloning.
- Check output duration - if under 30s, repeat the text to reach desired length.

### Detailed Guide (For Future Reference)

#### Step 1: Download Audio from YouTube

```bash
# Full video audio
yt-dlp -x --audio-format wav "https://www.youtube.com/watch?v=VIDEO_ID" -o /tmp/voice_name.wav

# From specific timestamp (if provided)
yt-dlp -x --audio-format wav "https://youtu.be/VIDEO_ID?t=278" -o /tmp/voice_name.wav
```

#### Step 2: Trim to Best Segment

- **Recommended duration**: 2-5 minutes
- **Use the most expressive/characteristic parts** of the audio
- **Avoid** background music, silences, or noisy sections

```bash
# Trim to 2 minutes (from start)
ffmpeg -y -i input.wav -t 120 -ar 16000 -ac 1 output.wav

# Trim from specific time
ffmpeg -y -i input.wav -ss 00:05:00 -t 120 -ar 16000 -ac 1 output.wav
```

#### Step 3: Create Voice

The API automatically transcribes audio using Whisper and uses ICL mode:

```bash
curl -s -X POST "http://localhost:3004/v1/voices/add" \
  -H "x-tts-api-key: YOUR_API_KEY" \
  -F "name=Voice Name" \
  -F "file=@/path/to/audio.wav"
```

Returns:
```json
{
  "voice_id": "abc123...",
  "name": "Voice Name",
  "category": "cloned"
}
```

#### Step 4: Generate Speech

```bash
# MP3 output
curl -s -X POST "http://localhost:3004/v1/text-to-speech/Voice_ID" \
  -H "x-tts-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello there, good boy. I have been waiting for you.",
    "output_format": "mp3_44100_128"
  }' -o output.mp3

# WAV output
curl -s -X POST "http://localhost:3004/v1/text-to-speech/Voice_ID" \
  -H "x-tts-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "output_format": "wav"}' -o output.wav
```

### Troubleshooting: Voice Still Speaks Too Fast

1. **Use longer reference audio** - More audio = better pacing capture
2. **Use clearer audio** - Avoid background music/noise
3. **Post-process with ffmpeg** (last resort):
```bash
# Slow down by 15%
ffmpeg -i input.mp3 -filter:a "atempo=0.85" output.mp3
```

### Voice ID Reference (Current Voices)

| Name | Voice ID | Notes |
|------|----------|-------|
| stuffie | 94ac1e95-6139-4666-b393-b27ceeafc32b | ICL mode |
| simone_asmr | 98976448-84c1-4427-95cf-78afd32165bf | ICL mode |
| yumi_mommy_asmr_1 | 42997ffa-b772-452c-bc97-69ec9184ec69 | ICL mode |
| yumi_mommy_asmr_2 | 3ffaec3c-46f9-46df-976a-0084eb17290c | ICL mode |

### API Key

Use: `VJcCJrs46L7bdzKrSgsbYlrgZ`

## License

MIT
