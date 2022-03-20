ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

#更换国内源文件
RUN apt-get update\
    && apt install ca-certificates
COPY ./sources.list /etc/apt/sources.list
RUN apt-get update \
    && echo "#!/bin/sh\nexit 0" > /usr/sbin/policy-rc.d \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y openssh-server \
    && rm -rf /var/lib/apt/lists/*

# #设置用于ssh连接的root密码
# RUN echo 'root:kongge2022'|chpasswd
# RUN mkdir -p /var/run/sshd

# #允许root连接
# RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# # 开放 22 端口
# EXPOSE 22
# CMD ["/usr/sbin/sshd", "-D"]

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install "mmcv-full>=1.3.17, <=1.5.0" -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN pip install "mmdet>=2.19.0, <=3.0.0"
RUN pip install "mmsegmentation>=0.20.0, <=1.0.0" 

# # Install MMDetection3D
# RUN conda clean --all
# ENV FORCE_CUDA="1"
# RUN pip install "mmdet3d==v1.0.0rc0"

WORKDIR /mmdetection3d