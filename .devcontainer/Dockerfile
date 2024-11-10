# Use a Python base image with Debian for compatibility
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    vim \
    pigz \
    pv \
    screen 
    
# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

RUN pip install \
    jupyterlab \
    lightning \
    dagshub \
    torchvision \
    hydra-core \
    python-dotenv \
    transformers \
    tensorboard \
    boto3 \
    torch \
    dvc[s3] \
    mlflow[extras]

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && apt-get install -y sudo && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

RUN mkdir /content && \
    chown -R $USERNAME:$USERNAME /content

USER $USERNAME

# Set the default working directory
WORKDIR /workspace
