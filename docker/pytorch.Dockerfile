FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Install necessary build tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install non-Torch dependencies
COPY requirements.txt .
RUN grep -v "pyspark" requirements.txt > req_pytorch.txt \
    && pip install --no-cache-dir Pillow \
    && pip install --no-cache-dir --default-timeout=1000 -r req_pytorch.txt

# Copy source code
COPY src /app/src
COPY tests /app/tests
