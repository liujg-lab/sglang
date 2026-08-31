export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# python -m sglang.launch_server \
#     --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#     --tp 1 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --speculative-algorithm SPECTRE \
#     --spectre-role target \
#     --speculative-num-steps 3 \
#     --speculative-eagle-topk 1 \
#     --speculative-num-draft-tokens 4 \
#     --spectre-zmq-addr 127.0.0.1 \
#     --spectre-zmq-port 30009 \
#     --attention-backend triton \
#     --disable-cuda-graph \
#     --page-size 1 \
#     --log-level debug \
#     --skip-server-warmup

# python -m sglang.launch_server \
#     --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#     --tp 1 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --speculative-algorithm SPECTRE \
#     --spectre-role target \
#     --speculative-num-steps 3 \
#     --speculative-eagle-topk 1 \
#     --speculative-num-draft-tokens 4 \
#     --spectre-zmq-addr 127.0.0.1 \
#     --spectre-zmq-port 30009 \
#     --attention-backend triton \
#     --page-size 1 \
#     --log-level debug

# python -m sglang.launch_server \
#     --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#     --tp 1 \
#     --host 127.0.0.1 \
#     --port 30000 \
#     --speculative-algorithm SPECTRE \
#     --spectre-role target \
#     --speculative-num-steps 3 \
#     --speculative-eagle-topk 1 \
#     --speculative-num-draft-tokens 4 \
#     --spectre-zmq-addr 127.0.0.1 \
#     --spectre-zmq-port 30009 \
#     --attention-backend triton \
#     --page-size 1 \
#     --log-level debug \

# # Target
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role target \
#   --speculative-num-steps 4 \
#   --speculative-eagle-topk 1 \
#   --speculative-num-draft-tokens 5 \
#   --standalone-remote-addr 127.0.0.1 \
#   --standalone-remote-port 30019 \
#   --page-size 1 \
#   --port 30000


# Target
python -m sglang.launch_server \
  --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
  --speculative-algorithm STANDALONE_REMOTE \
  --standalone-remote-role target \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 2 \
  --speculative-num-draft-tokens 8 \
  --standalone-remote-addr 127.0.0.1 \
  --standalone-remote-port 30019 \
  --page-size 1 \
  --port 30000