FROM loopbackkr/pytorch:1.11.0-cuda11.3-cudnn8

WORKDIR /workspace

COPY ./HD_BET ./HD_BET
COPY ./setup.* ./
COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -e .