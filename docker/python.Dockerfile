FROM python:3.10-slim

# Install necessary build tools and libaries
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src

# Copy requirements and install
COPY requirements.txt .
RUN grep -v -E "torch|torchvision|Pillow" requirements.txt > req_core.txt && pip install -r req_core.txt

# Copy source code
COPY src /app/src
COPY tests /app/tests

# Default command (overridden in compose)
CMD ["python", "--version"]
