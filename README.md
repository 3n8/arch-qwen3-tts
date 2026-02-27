# Qwen3-TTS

Self-hosted Qwen3-TTS service intended as a practical drop-in replacement for ElevenLabs TTS (ComfyUI nodes + typical web apps).

## Features

- **Best Audio Quality**: Uses Qwen3-TTS 12Hz 1.7B Base and VoiceDesign models
- **Voice Persistence**: Anchor.wav pattern ensures consistent voice identity across restarts
- **ElevenLabs-Compatible API**: Works with common ElevenLabs clients
- **GPU-Accelerated**: AMD GPU with ROCm 6.4 support
- **Media Preprocessing**: FFmpeg + VAD trimming for clean voice clones

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

Create a `docker-compose.yml` file:

```yaml
services:
  qwen3-tts:
    build: .
    image: qwen3-tts:latest
    container_name: qwen3-tts
    restart: unless-stopped
    user: "${PUID}:${PGID}"
    environment:
      - TZ=${TZ:-Europe/Oslo}
      - PORT=${PORT:-3004}
      - TTS_API_KEY=${TTS_API_KEY:?TTS_API_KEY is required}
      - HIP_VISIBLE_DEVICES=0
      - HSA_OVERRIDE_GFX_VERSION=
      - MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
      - MAX_CHUNK_CHARS=${MAX_CHUNK_CHARS:-700}
      - HF_REV_BASE=
      - HF_REV_DESIGN=
    volumes:
      - /opt/appdata/qwen3-tts/config:/config
      - /opt/appdata/qwen3-tts/models:/models
      - /opt/appdata/qwen3-tts/voices:/voices
      - /opt/appdata/qwen3-tts/out:/out
      - /opt/appdata/qwen3-tts/hf_cache:/root/.cache/huggingface
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    ipc: host
    group_add:
      - video
      - render
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3004/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

Create a `.env` file:

```bash
PUID=1050
PGID=1050
TZ=Europe/Oslo
PORT=3004
TTS_API_KEY=your-secret-api-key
```

Then start the container:

```bash
docker compose up -d
```

### AMD GPU Troubleshooting

If you encounter HIP/GPU errors with your AMD GPU, add this to the environment section:

```yaml
- HSA_OVERRIDE_GFX_VERSION=11.0.0
```

For model pinning, you can also set HuggingFace revisions:

```yaml
- HF_REV_BASE=<commit-sha>
- HF_REV_DESIGN=<commit-sha>
```

View logs with:

```bash
docker logs qwen3-tts
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

Models are cached at `/models` and HuggingFace cache at `/root/.cache/huggingface`.

## License

MIT
