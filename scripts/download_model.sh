#!/bin/bash


NAME_MODEL="${1:-128x128_classifier.pt}"
URL_MODEL="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/${NAME_MODEL}"
PATH_MODEL="../models/${MODEL_NAME}"

curl $MODEL_URL --create-dirs -o $PATH_MODEL