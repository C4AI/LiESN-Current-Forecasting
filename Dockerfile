FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update --fix-missing && \
    apt-get upgrade -y

RUN apt-get install bash \
                    sudo \
                    git \
                    python3.8 \
                    python3-pip -y


ENV G_ID=1000 \
    U_ID=1000 \
    U_NAME=user01 \
    PASS=pass

ENV PATH="/home/$U_NAME/.local/bin:${PATH}"

RUN addgroup --gid $G_ID $U_NAME
RUN adduser --uid $U_ID --ingroup $U_NAME --shell /bin/bash --disabled-password --gecos "" $U_NAME
RUN usermod -aG 100 $U_NAME
RUN usermod -aG sudo $U_NAME
RUN echo "$U_NAME:$PASS" | chpasswd
RUN python3.8 -m pip install --upgrade pip setuptools

USER $U_NAME
WORKDIR /usr/src/config/

COPY requirements.txt /usr/src/code/requirements.txt
WORKDIR /usr/src/code/
RUN python3.8 -m pip install -r requirements.txt


