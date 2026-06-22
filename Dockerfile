# Dockerfile for Sidekick

# Use Python 3.12 slim image
FROM python:3.12-slim

# Install uv (ultra-fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for building Python packages, including PyAV requirements
# The libav* packages provide the pkg-config files needed by PyAV to build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    pkg-config \
    cython3 \
    build-essential \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY sidekick/ ./sidekick/
COPY .env.example /app/.env

# Install dependencies using uv
RUN uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt && uv pip install pytest pytest-xdist
RUN . .venv/bin/activate && playwright install chromium

# Create sandbox directory
RUN mkdir -p sandbox

# Expose port for Gradio
EXPOSE 7860

# Command to run the app
CMD [".venv/bin/python", "sidekick/app.py"]