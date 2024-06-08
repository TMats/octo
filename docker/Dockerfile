# Based on https://github.com/naruya/dl_remote/blob/master/Dockerfile
# [1] https://github.com/robbyrussell/oh-my-zsh
# [2] https://github.com/pyenv/pyenv/wiki/common-build-problems

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        pkg-config \
        unzip

# zsh (from [1]) ----------------
RUN apt-get update && apt-get install -y \
    wget git zsh
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN sed -i 's/# DISABLE_AUTO_UPDATE="true"/DISABLE_AUTO_UPDATE="true"/g' ~/.zshrc

# pyenv (from [2]) ----------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN curl https://pyenv.run | zsh && \
    echo '' >> /root/.zshrc && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.zshrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.zshrc && \
    echo 'eval "$(pyenv init --path)"' >> /root/.zshrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /root/.zshrc
RUN source /root/.zshrc && \
    pyenv install 3.10.13 && \
    pyenv global 3.10.13 && \
    pip install -U pip && \
    pip install setuptools

# OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev

# X window ----------------
RUN apt-get update && apt-get install -y \
    xvfb x11vnc python-opengl icewm
RUN echo 'alias vnc="export DISPLAY=:0; Xvfb :0 -screen 0 1400x900x24 &; x11vnc -display :0 -forever -noxdamage > /dev/null 2>&1 &; icewm-session &"' >> /root/.zshrc

# utils ----------------
RUN apt-get update && apt-get install -y \
    vim \
    libsdl2-2.0-0 \
    cmake \ 
    xvfb \
    xorg-dev \
    freeglut3-dev \
    libglu1-mesa-dev \
    x11-apps \
    ffmpeg

# install python packages ----------------
COPY requirements.txt /root/
RUN source /root/.zshrc && \
    pip install -r /root/requirements.txt

# deep ----------------
RUN source /root/.zshrc && \
    pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
RUN source /root/.zshrc && \
    pip install --upgrade "jax[cuda11_local]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# gsutil ----------------
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# finalize image  ----------------
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN source ~/.zshrc

WORKDIR /root
CMD ["zsh"]
