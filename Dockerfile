FROM loopbackkr/pytorch:1.11.0-cuda11.3-cudnn8

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
COPY setup.* .
RUN pip install --no-cache-dir -e .

COPY HD_BET ./HD_BET