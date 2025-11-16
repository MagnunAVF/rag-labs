#!/bin/bash

MODEL="microsoft/Phi-3-mini-128k-instruct"
PORT="${PORT:-8000}"

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN is not set. Set it with: export HF_TOKEN='your_token'"
fi

docker run --gpus all \
	--name rag_vllm \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
	--env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}" \
	-p 8000:8000 \
	--ipc=host \
	vllm/vllm-openai:latest \
	--model "${MODEL}" \
	--host 0.0.0.0 \
	--port 8000 \
  --gpu-memory-utilization 0.8 \
	--trust-remote-code \
  --max-model-len 16384
