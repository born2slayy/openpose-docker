# Base image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# Install system packages
RUN apt-get update && apt-get install -y \
    git sudo wget build-essential ffmpeg gdebi libopencv-dev cmake \
    python3 python3-pip python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip3 install --no-cache-dir numpy opencv-python-headless \
    fastapi uvicorn[standard] python-multipart

# Clone OpenPose
WORKDIR /usr/local
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose

# Install OpenPose dependencies
RUN cd /usr/local/openpose && bash ./scripts/ubuntu/install_deps.sh

# Copy pretrained models (from your host machine)
COPY ./models /usr/local/openpose/models

# Build OpenPose with Python API
RUN mkdir -p /usr/local/openpose/build && \
    cd /usr/local/openpose/build && \
    cmake .. \
        -DBUILD_PYTHON=ON \
        -DDOWNLOAD_BODY_25_MODEL=OFF \
        -DDOWNLOAD_HAND_MODEL=OFF \
        -DDOWNLOAD_FACE_MODEL=OFF \
        -DPYTHON_EXECUTABLE=/usr/bin/python3 && \
    make -j$(nproc) && \
    make install

# Set Python path to OpenPose wrapper
ENV PYTHONPATH=/usr/local/openpose/build/python:${PYTHONPATH}

# Make openpose.bin accessible
RUN ln -s /usr/local/openpose/build/examples/openpose/openpose.bin /usr/local/bin/openpose.bin

# Working directory for FastAPI app
WORKDIR /workspace

CMD ["bash"]

