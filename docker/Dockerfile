# Based on https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
RUN apt-get update && apt-get install -y --no-install-recommends \
     build-essential \
     cmake \
     git \
     curl \
     vim \
     screen \
     tmux \
     byobu \
     wget \
     unzip \
     ca-certificates \
     libjpeg-dev \
     libpng-dev \
     libgtk2.0-dev \
     libopencv-dev \
     libgl1-mesa-glx \
     bash-completion

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=3.6 && \
     /opt/conda/bin/conda install -y pytorch=1.2.0 cudatoolkit=10.0 torchvision=0.4.0 -c pytorch && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
WORKDIR /workspace
RUN chmod -R a+w /workspace