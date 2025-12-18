FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    apt-transport-https \
    gnupg \
    lsb-release \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.28/deb/Release.key | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg \
    && echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.28/deb/ /' | tee /etc/apt/sources.list.d/kubernetes.list \
    && apt-get update \
    && apt-get install -y kubectl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application files
COPY init.sh /app/
COPY bench.py /app/
COPY benchmarks/ /app/benchmarks/
COPY template/ /app/template/

# Make init.sh executable
RUN chmod +x /app/init.sh

# Set environment variables for containerized execution
ENV RUNNING_IN_DOCKER=true
ENV PYTHONUNBUFFERED=1
# HF_HOME will be set dynamically based on data_dir mount
ENV HF_HOME=/data

# Create directories for output and data
# The /data directory will be mounted from host for persistent dataset storage
RUN mkdir -p /app/runs /data

# Entry point
ENTRYPOINT ["/app/init.sh"]


