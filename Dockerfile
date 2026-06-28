# Dockerfile for Sidekick

# Use Python 3.12 slim image
FROM python:3.12-slim

# Install uv and system dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Force Playwright to look for browsers in a specific path
ENV PLAYWRIGHT_BROWSERS_PATH=/app/.cache/ms-playwright

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv using uv sync
# We use --frozen to respect the lockfile exactly
RUN uv sync --frozen --no-dev --no-install-project

# Install Playwright browsers and their system dependencies
# Note: --with-deps requires root to install system packages
RUN .venv/bin/playwright install --with-deps chromium

# Copy project files
COPY sidekick/ ./sidekick/
COPY .env.example .env

# Create data and sandbox directories with wide permissions for container mobility
RUN mkdir -p /app/data /app/sandbox && chmod 777 /app/data /app/sandbox

# Expose port for Gradio
EXPOSE 7860

# Environment defaults for Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
# Path to SQLite database if needed
ENV SQLITE_DB_PATH="/app/data/sidekick.db"

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -f http://localhost:7860/ || exit 1

# Command to run the app using the virtualenv python
CMD ["/app/.venv/bin/python", "sidekick/app.py"]
