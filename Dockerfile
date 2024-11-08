# check more: https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install -y --no-install-recommends \
    git curl vim rsync htop

RUN curl -o ~/miniconda.sh -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya && /opt/conda/bin/conda init bash

RUN curl -o ~/mamba.sh -LO https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && \
    chmod +x ~/mamba.sh && \
    ~/mamba.sh -b -p /opt/mambaforge && \
    rm ~/mamba.sh && /opt/mambaforge/bin/mamba init bash

# install zsh and oh-my-zsh
RUN apt install -y wget git zsh tmux vim g++
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell -p git \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting
    
RUN printf "y\ny\ny\n\n" | bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kin-Zhang/Kin-Zhang/main/scripts/setup_ohmyzsh.sh)"
RUN /opt/conda/bin/conda init zsh && /opt/mambaforge/bin/mamba init zsh

# change to conda env
ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/mambaforge/bin:$PATH

RUN mkdir -p /home/kin/workspace && cd /home/kin/workspace && git clone --recursive https://github.com/KTH-RPL/DeFlow.git
WORKDIR /home/kin/workspace/DeFlow
RUN apt-get update && apt-get install libgl1 -y
# need read the gpu device info to compile the cuda extension
RUN cd /home/kin/workspace/DeFlow && /opt/mambaforge/bin/mamba env create -f environment.yaml
RUN cd /home/kin/workspace/DeFlow/mmcv && export MMCV_WITH_OPS=1 && export FORCE_CUDA=1 && /opt/mambaforge/envs/deflow/bin/pip install -e .


