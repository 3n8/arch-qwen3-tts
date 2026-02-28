ARG ROCM_PYTORCH_TAG=rocm6.4.2_ubuntu24.04_py3.12_pytorch_release_2.6.0
FROM rocm/pytorch:${ROCM_PYTORCH_TAG}

LABEL maintainer="3n8"
LABEL org.opencontainers.image.source="https://github.com/3n8/qwen3-tts"

ARG APPNAME=qwen3-tts
ARG TARGETARCH=amd64

ENV HOME=/home/nobody \
    TERM=xterm \
    LANG=en_GB.UTF-8 \
    PATH=/usr/local/bin/system/scripts/docker:/usr/local/bin/run:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HIP_VISIBLE_DEVICES=0 \
    HSA_OVERRIDE_GFX_VERSION= \
    MAX_CONCURRENCY=1 \
    MAX_CHUNK_CHARS=700 \
    PORT=3004 \
    VOICES_DIR=/voices \
    MODELS_DIR=/models \
    OUT_DIR=/out \
    HF_CACHE_DIR=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        jq \
        tzdata \
        moreutils \
        supervisor \
        dumb-init \
        ffmpeg \
        libsndfile1 \
        sox \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash nobody || true

COPY build/common/root/install.sh /tmp/install.sh
COPY build/common/root/supervisord.conf /etc/supervisord.conf
COPY build/common/root/qwen3-tts.conf /etc/supervisor/conf.d/qwen3-tts.conf
COPY build/common/root/init.sh /usr/bin/init.sh
COPY build/common/root/utils.sh /usr/local/bin/system/scripts/docker/utils.sh

RUN chmod +x /tmp/install.sh && /tmp/install.sh; rm -f /tmp/install.sh; \
    chmod +x /usr/bin/init.sh && \
    chmod +x /usr/local/bin/system/scripts/docker/utils.sh && \
    mkdir -p /config /config/run /models /voices /out /root/.cache/huggingface && \
    chmod -R 777 /config /models /voices /out

RUN echo "export TARGETARCH=${TARGETARCH}" >> /etc/image-build-info

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY app /app
RUN chmod -R 755 /app

RUN echo "export APPNAME=${APPNAME}" >> /etc/image-build-info

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["/usr/bin/init.sh"]
