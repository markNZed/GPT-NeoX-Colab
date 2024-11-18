# Use a Python base image with Debian for compatibility
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PDM_USE_PEP582=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    pigz \
    pv \
    screen \
    libopenmpi-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && apt-get install -y sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Prepare workspace and set permissions
RUN mkdir -p /content /workspace && \
    chown -R $USERNAME:$USERNAME /content /workspace

# Install Poetry as root
RUN pip install --upgrade pip && pip install poetry

# Switch to non-root user
USER $USERNAME
WORKDIR /workspace

# Set up environment for the user
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Install dependencies using Poetry
COPY --chown=$USERNAME:$USERNAME README.md pyproject.toml poetry.lock /workspace/

# Copy the source code
COPY --chown=$USERNAME:$USERNAME src/ /workspace/src/

RUN --mount=type=cache,target=/workspace/.cache,uid=$USER_UID,gid=$USER_GID \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --verbose
