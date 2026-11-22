#!/bin/bash
set -euo pipefail

# 使用 sglang 服务独自推理

# python -m sglang.launch_server \
#     --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-8B-Instruct \
#     --tp 1 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --attention-backend triton \
#     --page-size 1 \
#     --log-level debug

# export CUDA_VISIBLE_DEVICES=0
# python -m sglang.launch_server \
#     --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#     --tp 1 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --attention-backend triton \
#     --page-size 1 \
#     --disable-cuda-graph

# export CUDA_VISIBLE_DEVICES=0
# python -m sglang.launch_server \
#     --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#     --tp 1 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --attention-backend triton \
#     --mm-attention-backend triton_attn \
#     --page-size 1
#     # --disable-cuda-graph

export CUDA_VISIBLE_DEVICES=0
python -m sglang.launch_server \
    --model-path /home_18T/liujg/Hugging_Face/mllmTeam/Qwen3-VL-2B-Instruct-W8A8-INT8 \
    --tp 1 \
    --host 127.0.0.1 \
    --port 30000 \
    --attention-backend triton \
    --mm-attention-backend triton_attn \
    --page-size 1
    # --disable-cuda-graph

# export CUDA_VISIBLE_DEVICES=0
# python -m sglang.launch_server \
#     --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#     --tp 1 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --attention-backend triton \
#     --page-size 1 \
#     # --disable-cuda-graph

# export CUDA_VISIBLE_DEVICES=0
# python -m sglang.launch_server \
#     --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B-GPTQ-Int8 \
#     --tp 1 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --attention-backend triton \
#     --page-size 1 \
#     # --disable-cuda-graph

# export CUDA_VISIBLE_DEVICES=0,1
# python -m sglang.launch_server \
#     --model-path /home_16T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-14B \
#     --tp 2 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --attention-backend triton \
#     --page-size 1 
