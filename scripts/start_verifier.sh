set -euo pipefail

export CUDA_VISIBLE_DEVICES="0"
export VLLM_USE_V1="0"
# Đổi XFORMERS thành TRITON_ATTN
export VLLM_ATTENTION_BACKEND="TRITON_ATTN"

MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
SERVED_MODEL_NAME="Llama-3.2-3B-Instruct"
PORT="8080"

vllm serve "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "127.0.0.1" \
    --port "$PORT" \
    --dtype "half" \
    --max-model-len "8192" \
    --gpu-memory-utilization "0.4" \
    --generation-config "vllm"