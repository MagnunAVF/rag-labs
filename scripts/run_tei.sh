#!/bin/bash

model=mixedbread-ai/mxbai-embed-large-v1
volume=$HOME/.cache/huggingface

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN is not set. Set it with: export HF_TOKEN='your_token'"
fi

docker run --name rag_tei \
    --gpus all \
    -p 8082:80 \
    -v $volume:/data \
    --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
    --pull always ghcr.io/huggingface/text-embeddings-inference:latest \
    --model-id $model \
    --max-client-batch-size 16
