#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="pfm1-detector"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch}"

mkdir -p "$ROOT_DIR/models"

docker build \
  -t "$IMAGE_NAME" \
  --build-arg IMAGE="$BASE_IMAGE" \
  "$ROOT_DIR"

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --mount type=bind,src="$ROOT_DIR/data",dst=/workspace/pfm1-detector-model/data,readonly \
  --mount type=bind,src="$ROOT_DIR/models",dst=/workspace/pfm1-detector-model/models \
  "$IMAGE_NAME"
