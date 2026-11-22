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
#     --spectre-zmq-addr 172.22.14.164 \
#     --spectre-zmq-port 30009 \
#     --attention-backend triton \
#     --mem-fraction-static 0.55 \
#     --page-size 1 \
#     --log-level debug \



# Target
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role target \
#   --speculative-num-steps 4 \
#   --speculative-eagle-topk 2 \
#   --speculative-num-draft-tokens 8 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --mem-fraction-static 0.45 \
#   --page-size 1 \
#   --attention-backend triton \
#   --port 30000 \
#   --speculative-verify-mode rpd \
#   --speculative-rpd-tau 0.2


  python -m sglang.launch_server \
  --model-path /home_18T/liujg/Hugging_Face/mllmTeam/Qwen3-VL-2B-Instruct-W8A8-INT8 \
  --speculative-algorithm TOLL_TREE_REMOTE \
  --toll-tree-role target \
  --toll-tree-policy fixed \
  --toll-tree-fixed-width 8 \
  --toll-tree-fixed-tau 0.1 \
  --toll-tree-addr 127.0.0.1 \
  --toll-tree-port 30019 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --disable-overlap-schedule \
  --max-running-requests 1 \
  --skip-server-warmup \
  --port 30000
