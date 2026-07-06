FROM python:3.10-slim

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Tk and X11 libraries for GUI
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    tk \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    openssh-client \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir ruff

# Build the C++ pinocchio substep extension (graceful no-op if headers absent)
COPY experiments/experiments_02_thru_09/pin_ext.cpp \
     experiments/experiments_02_thru_09/build_ext.sh \
     /tmp/pin_ext_build/
RUN bash /tmp/pin_ext_build/build_ext.sh && \
    cp /tmp/pin_ext_build/pin_ext*.so /usr/local/lib/python3.10/site-packages/ || \
    echo "WARNING: pin_ext build failed — falling back to pure Python"

# Set matplotlib backend to TkAgg for interactive plots
ENV MPLBACKEND=TkAgg
