#!/bin/bash

# Convenience script to launch vLLM server with specified model
# Usage: ./scripts/launch_vllm.sh <model_name> [port] [endpoint_name]
#
# Examples:
#   ./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct
#   ./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct 8001
#   ./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct 8000 my-model

set -e

# Check if model name is provided
if [ -z "$1" ]; then
    echo "Error: Model name is required"
    echo ""
    echo "Usage: $0 <model_name> [port] [endpoint_name]"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen2.5-3B-Instruct"
    echo "  $0 Qwen/Qwen2.5-3B-Instruct 8001"
    echo "  $0 Qwen/Qwen2.5-3B-Instruct 8000 my-model"
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

echo "=========================================="
echo "Launching vLLM Server"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Endpoint name: $ENDPOINT_NAME"
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

# Submit the job with the model name and port as variables
JOB_ID=$(qsub -v MODEL="$MODEL_NAME",PORT="$PORT" "$SCRIPT_DIR/start_vllm.sh")

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

