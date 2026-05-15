# syntax=docker/dockerfile:1.7

ARG IMAGE=rocm/pytorch
FROM ${IMAGE}

WORKDIR /workspace/pfm1-detector-model

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python - <<'PY'
from pathlib import Path
import subprocess
import sys
import tomllib

toml = tomllib.loads(Path('pyproject.toml').read_text())
deps = [
    dep for dep in toml['project']['dependencies']
]
if deps:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *deps])
PY

COPY src ./src

ENV PYTHONPATH=/workspace/pfm1-detector-model/src \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    TF_CPP_MIN_LOG_LEVEL=1

CMD ["python", "src/training/yolo_train.py"]

