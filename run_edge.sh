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

python -m sglang.launch_server \
  --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
  --tp 1 \
  --host 127.0.0.1 \
  --port 30008 \
  --speculative-algorithm SPECTRE \
  --spectre-role draft \
  --spectre-zmq-addr 127.0.0.1 \
  --spectre-zmq-port 30009 \
  --disable-cuda-graph \
  --page-size 1 \
  --log-level debug

# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#   --tp 1 \
#   --host 127.0.0.1 \
#   --port 30008 \
#   --speculative-algorithm SPECTRE \
#   --spectre-role draft \
#   --spectre-zmq-addr 127.0.0.1 \
#   --spectre-zmq-port 30009 \
#   --page-size 1 \
#   --log-level debug