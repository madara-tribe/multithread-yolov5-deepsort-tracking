FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHON_VERSION 3.8.1
WORKDIR /root
RUN apt-get update && apt-get upgrade -y
RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev
RUN apt-get install liblzma-dev

RUN apt install -y  --no-install-recommends make cmake gcc git g++ unzip wget build-essential zlib1g-dev libffi-dev libssl-dev && \
    apt clean && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar zxf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make altinstall



RUN ln -s /usr/local/bin/python3.8 /bin/python3 && \
    ln -s /usr/local/bin/pip3.8 /bin/pip3



# Install python dependencies
COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip3 install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt
RUN pip3 install coremltools gsutil wandb>=0.12.2
RUN pip3 install torch torchvision torchaudio numpy
RUN pip3 install easydict
# RUN pip install --no-cache torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Create working directory
WORKDIR /work
# Set environment variables
ENV HOME=/work
