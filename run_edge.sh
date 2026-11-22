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
#   --spectre-zmq-addr 172.22.14.162 \
#   --spectre-zmq-port 30009 \
#   --disable-cuda-graph \
#   --page-size 1

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

# # Draft (same or another host; do not --skip-tokenizer-init)
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 4 \
#   --speculative-eagle-topk 2 \
#   --speculative-num-draft-tokens 8 \
#   --context-length 32768 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1

# # Draft (same or another host; do not --skip-tokenizer-init)
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 4 \
#   --speculative-eagle-topk 2 \
#   --speculative-num-draft-tokens 6 \
#   --context-length 32768 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1

# # Draft (same or another host; do not --skip-tokenizer-init)
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 6 \
#   --speculative-eagle-topk 4 \
#   --speculative-num-draft-tokens 20 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1 \
#   --max-total-tokens 8192 \
#   --cuda-graph-bs 1 2 4


# Draft (same or another host; do not --skip-tokenizer-init)
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3-VL-2B-Instruct-FP8 \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 6 \
#   --speculative-eagle-topk 4 \
#   --speculative-num-draft-tokens 20 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1 \
#   --max-total-tokens 8192 \
#   --cuda-graph-bs 1 2 4

# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/mllmTeam/Qwen3-VL-2B-Instruct-W8A8-INT8 \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 6 \
#   --speculative-eagle-topk 4 \
#   --speculative-num-draft-tokens 20 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1 \
#   --max-total-tokens 8192 \
#   --cuda-graph-bs 1 2 4 \
#   --standalone-remote-draft-vision-compress visionzip \
#   --standalone-remote-draft-vision-keep-ratio 0.25 \
#   --standalone-remote-draft-vision-dominant-ratio 0.85 \
#   --standalone-remote-draft-vision-score-layer -2


# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/h2oai/Qwen3-VL-2B-Instruct-GPTQ-Int4 \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 6 \
#   --speculative-eagle-topk 4 \
#   --speculative-num-draft-tokens 20 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1 \
#   --max-total-tokens 8192 \
#   --cuda-graph-bs 1 2 4

# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 6 \
#   --speculative-eagle-topk 4 \
#   --speculative-num-draft-tokens 20 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1 \
#   --max-total-tokens 8192 \
#   --cuda-graph-bs 1 2 4

# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 7 \
#   --speculative-eagle-topk 4 \
#   --speculative-num-draft-tokens 20 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1 \
#   --max-total-tokens 8192 \
#   --cuda-graph-bs 1 2 4

# Draft (same or another host; do not --skip-tokenizer-init)
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 4 \
#   --speculative-eagle-topk 2 \
#   --speculative-num-draft-tokens 8 \
#   --context-length 32768 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1

# # Draft (same or another host; do not --skip-tokenizer-init)
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#   --port 30008 \
#   --speculative-algorithm STANDALONE_REMOTE \
#   --standalone-remote-role draft \
#   --speculative-num-steps 5 \
#   --speculative-eagle-topk 3 \
#   --speculative-num-draft-tokens 10 \
#   --context-length 32768 \
#   --standalone-remote-addr 172.22.14.164 \
#   --standalone-remote-port 30019 \
#   --attention-backend triton \
#   --page-size 1 \
#   --cuda-graph-bs 1 2 4
#   # --disable-cuda-graph

# Draft (same or another host; do not --skip-tokenizer-init)
# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/Qwen/Qwen3/Qwen3-0.6B \
#   --tp 1 \
#   --port 30008 \
#   --host 127.0.0.1 \
#   --speculative-algorithm SPECTRE \
#   --spectre-role draft \
#   --spectre-zmq-addr 172.22.14.162 \
#   --spectre-zmq-port 30009 \
#   --page-size 1 \
#   --attention-backend triton

# python -m sglang.launch_server \
#   --model-path /home_18T/liujg/Hugging_Face/mllmTeam/Qwen3-VL-2B-Instruct-W8A8-INT8 \
#   --speculative-algorithm OPT_TREE_REMOTE \
#   --opt-tree-role draft \
#   --opt-tree-width 19 \
#   --opt-tree-tau 0.2 \
#   --opt-tree-max-depth 10 \
#   --opt-tree-addr 172.22.14.164 \
#   --opt-tree-port 30019 \
#   --attention-backend triton \
#   --page-size 1 \
#   --port 30008

python3 test/manual/toll_fake_draft.py \
  --addr 127.0.0.1 \
  --port 30019 \
  --mode gold \
  --baseline /home/liujg/workspace/sglang/test_sglang_liujg/result/RSTeller_spectre_serving_results.20261106T015652.487775Z-fc03e358.jsonl \
  --depth 3 --verify-bucket 8
