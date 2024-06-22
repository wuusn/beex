# Use the official Ubuntu 22.04 base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    python3-openslide \
    sudo \
    vim

# sudo without password
RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set the PATH to include conda
ENV PATH=/opt/conda/bin:$PATH

# Create a non-root user
RUN useradd -m -s /bin/bash dockeruser
USER dockeruser
WORKDIR /home/dockeruser

# Create the conda environment
RUN conda create -n bee python=3.11

SHELL [ "conda", "run", "-n", "bee", "/bin/bash", "-c" ]

RUN conda install -n bee -c ejolly -c conda-forge -c defaults pymer4 -y

RUN conda install pip -y

# Copy the project code to the container
COPY --chown=dockeruser:dockeruser . .

COPY --chown=dockeruser:dockeruser requirements.txt .

RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple