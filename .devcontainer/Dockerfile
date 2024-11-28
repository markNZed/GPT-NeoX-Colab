# Use an NVIDIA CUDA base image with Ubuntu 22.04 for GPU support
# Using 22.04 so we have Pytohn 3.10 (24.04 has 3.12)
FROM nvidia/cuda:12.6.2-base-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
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
    libopenmpi-dev \
    sudo \
    python3 \
    python3-venv \
    python3-dev

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-6

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

## Create a non-root user
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

# Switch to non-root user
USER $USERNAME
WORKDIR /workspace

# Set up environment for the user
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -

COPY --chown=$USERNAME:$USERNAME README.md pyproject.toml poetry.lock /workspace/
COPY --chown=$USERNAME:$USERNAME src/ /workspace/src/

RUN poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --verbose

ENV SHELL=/bin/bash

CMD ["/bin/bash"]
