# Use an NVIDIA CUDA base image with Ubuntu 24.04 for GPU support
FROM nvidia/cuda:12.6.2-base-ubuntu24.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PDM_USE_PEP582=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    build-essential \
    curl \
    git \
    wget \
    vim \
    pigz \
    pv \
    screen \
    libopenmpi-dev \
    sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000

RUN usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Prepare workspace and set permissions
RUN mkdir -p /content /workspace && \
    chown -R $USERNAME:$USERNAME /content /workspace

# Install Poetry for the non-root user
USER $USERNAME
WORKDIR /workspace

RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Copy and set up the project
COPY --chown=$USERNAME:$USERNAME README.md pyproject.toml poetry.lock /workspace/
COPY --chown=$USERNAME:$USERNAME src/ /workspace/src/


# Install dependencies using Poetry with caching enabled
#RUN --mount=type=cache,target=/home/$USERNAME/.cache/pypoetry \
#    poetry config virtualenvs.in-project true && \
#    poetry install --no-interaction --verbose


# Default command
CMD ["bash"]
