#!/bin/bash
#PBS -l ncpus=32
#PBS -l ngpus=4 
#PBS -l mem=120gb
#PBS -l walltime=48:00:00
#PBS -N vllm-server
#PBS -M pxm5426@psu.edu 
#PBS -m bea

# Change to the directory where the job was submitted from
cd $PBS_O_WORKDIR

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Current directory: $(pwd)"
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate the uv virtual environment for verifiers
echo "Activating uv virtual environment..."
source .venv/bin/activate

# Set CUDA_HOME for FlashInfer JIT compilation
export CUDA_HOME=/scratch/pxm5426/apps/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc location: $(which nvcc)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
echo ""

# Set CUDA devices - handle PBS GPU UUID to index mapping
echo "Original CUDA_VISIBLE_DEVICES from PBS: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Check if we have UUIDs (contain '-' or ':')
    if [[ "$CUDA_VISIBLE_DEVICES" == *"-"* ]] || [[ "$CUDA_VISIBLE_DEVICES" == *":"* ]]; then
        echo "Detected GPU UUIDs, mapping to integer indices..."
        
        # Get GPU mapping from nvidia-smi
        mapfile -t gpu_info < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
        
        # Build associative array for UUID -> index mapping
        declare -A uuid_to_index
        for line in "${gpu_info[@]}"; do
            index=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
            uuid=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
            uuid_to_index["$uuid"]="$index"
        done
        
        # Map PBS UUIDs to indices
        IFS=',' read -ra uuids <<< "$CUDA_VISIBLE_DEVICES"
        mapped_indices=()
        for uuid in "${uuids[@]}"; do
            uuid=$(echo "$uuid" | tr -d ' ')
            if [ -n "${uuid_to_index[$uuid]}" ]; then
                mapped_indices+=("${uuid_to_index[$uuid]}")
            else
                echo "ERROR: Could not map GPU UUID: $uuid"
                exit 1
            fi
        done
        
        export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${mapped_indices[*]}")
        echo "Mapped to indices: $CUDA_VISIBLE_DEVICES"
    else
        echo "GPU indices already set: $CUDA_VISIBLE_DEVICES"
    fi
else
    # PBS didn't set CUDA_VISIBLE_DEVICES, manually set for 4 GPUs
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "No PBS GPU assignment, using default: $CUDA_VISIBLE_DEVICES"
fi

echo "Final CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set model configuration from PBS variable or use default
# Pass model via: qsub -v MODEL="model/name" start_vllm.sh
# Or use default if not specified
DEFAULT_MODEL="Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME="${MODEL:-$DEFAULT_MODEL}"

# vLLM server configuration
# Can also pass PORT via: qsub -v MODEL="model/name",PORT=8001 start_vllm.sh
PORT="${PORT:-8000}"
HOST=0.0.0.0

# Create unique log filenames based on model and port
# Convert model name to safe filename (replace / with _)
MODEL_SAFE=$(echo "$MODEL_NAME" | sed 's/\//_/g')
LOG_PREFIX="vllm_${MODEL_SAFE}_port${PORT}"
PBS_OUT_LOG="pbs_results/${LOG_PREFIX}_pbs.out"
PBS_ERR_LOG="pbs_results/${LOG_PREFIX}_pbs.err"
REALTIME_LOG="pbs_results/${LOG_PREFIX}_realtime.log"
REALTIME_ERR="pbs_results/${LOG_PREFIX}_realtime.err"

# Redirect PBS stdout and stderr to unique files
exec > "$PBS_OUT_LOG" 2> "$PBS_ERR_LOG"

echo "Model from PBS variable: ${MODEL:-not set, using default}"
echo "Port from PBS variable: ${PORT:-not set, using default}"
echo "Log files:"
echo "  PBS output: $PBS_OUT_LOG"
echo "  PBS errors: $PBS_ERR_LOG"
echo "  Server log: $REALTIME_LOG"
echo "  Server errors: $REALTIME_ERR"

echo "=========================================="
echo "vLLM Server Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  Tensor Parallel Size: 4"
echo "=========================================="
echo ""

# Start vLLM server with tensor parallelism across 4 GPUs
# Redirect output in real-time to avoid PBS buffering
echo "Starting vLLM server with 4 GPUs (tensor parallel)..."
echo "Command: vf-vllm --model $MODEL_NAME --tensor-parallel-size 4 --host $HOST --port $PORT --enforce-eager --disable-log-requests"
echo ""

# Use stdbuf to disable buffering and redirect to log files
stdbuf -o0 -e0 vf-vllm \
    --model "$MODEL_NAME" \
    --tensor-parallel-size 4 \
    --host "$HOST" \
    --port "$PORT" \
    --enforce-eager \
    --disable-log-requests \
    > "$REALTIME_LOG" 2> "$REALTIME_ERR"

echo ""
echo "vLLM server stopped at: $(date)"
echo "Job finished."

