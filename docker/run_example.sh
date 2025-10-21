#!/usr/bin/env bash
set -euo pipefail

# Adjust these to your layout
SCRIPT_DIR="."
CODE_DIR="."   # contains run.py and requirements.txt
DATA_DIR="/home/dbl@grazper.net/data"   # your datasets / outputs
IMAGE="mvgenmaster:cuda"

# Build the image (rebuild whenever requirements.txt changes)
# docker build -t "$IMAGE" -f "${SCRIPT_DIR}/docker/Dockerfile" "${SCRIPT_DIR}"

# Run with GPU access; mount code and data; execute your script with placeholder args
docker run --rm -it \
  --gpus all \
  --user $(id -u):$(id -g) \
  -v "${CODE_DIR}":/app \
  -v "${DATA_DIR}":/data \
  --workdir /app \
  "$IMAGE" \
  python run_mvgen_pipeline.py #--input /data/input --output /data/output --foo bar

# For CPU-only testing, just drop the --gpus all flag.
