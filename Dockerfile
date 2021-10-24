FROM python:3.8-slim-buster

RUN apt-get update \
    && apt install -y \
    wget \
    git \
    libopenmpi-dev

RUN git clone https://github.com/BobrG//guided-diffusion \
    && cd guided-diffusion \
    && pip3 install . \
    && pip3 install -r requirements.txt
