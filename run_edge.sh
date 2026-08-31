export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#   --tp 1 \
#   --host 127.0.0.1 \
#   --port 30008 \
#   --speculative-algorithm SPECTRE \
#   --spectre-role draft \
#   --spectre-zmq-addr 127.0.0.1 \
#   --spectre-zmq-port 30009 \
#   --disable-cuda-graph \
#   --page-size 1 \
#   --skip-server-warmup \
#   --log-level debug \
#   --skip-server-warmup

# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#   --tp 1 \
#   --host 127.0.0.1 \
#   --port 30008 \
#   --speculative-algorithm SPECTRE \
#   --spectre-role draft \
#   --spectre-zmq-addr 127.0.0.1 \
#   --spectre-zmq-port 30009 \
#   --disable-cuda-graph \
#   --page-size 1 \
#   --log-level debug

# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#   --tp 1 \
#   --host 127.0.0.1 \
#   --port 30008 \
#   --speculative-algorithm SPECTRE \
#   --spectre-role draft \
#   --spectre-zmq-addr 172.22.14.162 \
#   --spectre-zmq-port 30009 \
#   --page-size 1 \
#   --log-level debug

# Draft (same or another host; do not --skip-tokenizer-init)
python -m sglang.launch_server \
  --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
  --port 30008 \
  --speculative-algorithm STANDALONE_REMOTE \
  --standalone-remote-role draft \
  --speculative-num-steps 4 \
  --speculative-eagle-topk 2 \
  --speculative-num-draft-tokens 8 \
  --context-length 32768 \
  --standalone-remote-addr 127.0.0.1 \
  --standalone-remote-port 30019 \
  --page-size 1