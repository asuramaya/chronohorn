# chronohorn-runner: minimal training image with deps pre-installed.
# Source code stays mutable via hostPath mount (/data/chronohorn/python,
# /data/decepticons/src) so rsync-based code updates work without rebuild.
#
# Build on each k3s node (or once + import) then reference by tag in Job spec
# with imagePullPolicy: Never.

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps. gcc only for any wheel fallback build; most deps have wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
      && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Python deps — only-binary avoids source builds (fast, no network hangs).
RUN pip install --only-binary=:all: \
      sentencepiece \
      huggingface_hub

WORKDIR /data
