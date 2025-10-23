#!/bin/bash

# Convenience script to launch vLLM server with specified model
# Usage: ./scripts/launch_vllm.sh <model_name> [port] [endpoint_name] [vllm_flags]
#
# Examples:
#   ./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct
#   ./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct 8001
#   ./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct 8000 my-model
#   ./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8002 qwen3-optimized "--data-parallel-size 4 --max-num-seqs 512"
#
# Optimization presets (for small models < 7B):
#   High Throughput: "--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
#   Balanced:        "--data-parallel-size 2 --tensor-parallel-size 2 --max-num-seqs 256"
#   Low Latency:     "--tensor-parallel-size 4 --max-num-seqs 128"

set -e

# Check if model name is provided
if [ -z "$1" ]; then
    echo "Error: Model name is required"
    echo ""
    echo "Usage: $0 <model_name> [port] [endpoint_name] [vllm_flags]"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen2.5-3B-Instruct"
    echo "  $0 Qwen/Qwen2.5-3B-Instruct 8001"
    echo "  $0 Qwen/Qwen2.5-3B-Instruct 8000 my-model"
    echo "  $0 Qwen/Qwen3-VL-2B-Instruct 8002 optimized \"--data-parallel-size 4 --max-num-seqs 512\""
    echo ""
    echo "Optimization presets (for small models < 7B):"
    echo "  High Throughput:"
    echo "    \"--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95\""
    echo ""
    echo "  Balanced (hybrid parallelism):"
    echo "    \"--data-parallel-size 2 --tensor-parallel-size 2 --max-num-seqs 256\""
    echo ""
    echo "  Default (tensor parallelism):"
    echo "    \"--tensor-parallel-size 4 --enforce-eager\""
    echo ""
    echo "Common models:"
    echo "  - Qwen/Qwen2.5-3B-Instruct"
    echo "  - Qwen/Qwen3-VL-8B-Instruct"
    echo "  - google/gemma-2-9b-it"
    echo "  - meta-llama/Llama-3.1-8B-Instruct"
    echo "  - willcb/DeepSeek-R1-Distill-Qwen-1.5B"
    exit 1
fi

MODEL_NAME="$1"
PORT="${2:-8000}"
ENDPOINT_NAME="${3:-local-vllm}"
VLLM_FLAGS="${4:-}"

echo "=========================================="
echo "Launching vLLM Server"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Endpoint name: $ENDPOINT_NAME"
if [ -n "$VLLM_FLAGS" ]; then
    echo "vLLM flags: $VLLM_FLAGS"
else
    echo "vLLM flags: (using defaults)"
fi
echo ""

# Update endpoints.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Updating endpoints.py..."
python3 "$SCRIPT_DIR/update_endpoint.py" "$MODEL_NAME" --port "$PORT" --name "$ENDPOINT_NAME"

echo ""

# Ensure pbs_results directory exists
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
mkdir -p "$REPO_ROOT/pbs_results"

echo "Submitting PBS job..."

# Submit the job with the model name, port, and vllm flags as variables
if [ -n "$VLLM_FLAGS" ]; then
    JOB_ID=$(qsub -v MODEL="$MODEL_NAME",PORT="$PORT",VLLM_FLAGS="$VLLM_FLAGS" "$SCRIPT_DIR/start_vllm.sh")
else
    JOB_ID=$(qsub -v MODEL="$MODEL_NAME",PORT="$PORT" "$SCRIPT_DIR/start_vllm.sh")
fi

# Create log filename prefix for this server
MODEL_SAFE=$(echo "$MODEL_NAME" | sed 's/\//_/g')
LOG_PREFIX="vllm_${MODEL_SAFE}_port${PORT}"

echo "Job submitted: $JOB_ID"
echo ""
echo "Monitor the job with:"
echo "  qstat $JOB_ID"
echo ""
echo "View logs with:"
echo "  tail -f pbs_results/${LOG_PREFIX}_realtime.log"
echo "  tail -f pbs_results/${LOG_PREFIX}_realtime.err"
echo "  tail -f pbs_results/${LOG_PREFIX}_pbs.out"
echo ""
echo "Once the server is running, use it with:"
echo "  uv run vf-eval <environment> -m $ENDPOINT_NAME"
echo ""

