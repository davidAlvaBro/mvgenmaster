# CUDA runtime base (adjust tag to match your framework's CUDA expectation)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Keep apt non-interactive; unbuffer Python logs
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# System deps + Python 3.10 (no --no-install-recommends as requested)
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-venv \
    ffmpeg libsm6 libxext6 libglm-dev \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Make 'python' and 'pip' point to Python 3.10 tools
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Fixed internal paths for where you'll mount things at runtime
ENV CODE_DIR=/app
ENV DATA_DIR=/data
WORKDIR ${CODE_DIR}

# --- deps baked into the image ---
# Copy only requirements.txt into the image and install now (best layer caching)
COPY ./requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip && pip install -r /opt/app/requirements.txt

# Default to a shell; your run script will pass "python run.py ..."
CMD ["bash"]
