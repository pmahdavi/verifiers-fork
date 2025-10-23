# vLLM Server Setup with PBS

This directory contains PBS scripts to start a vLLM inference server on the cluster. The default configuration uses **4 GPUs** with tensor parallelism for efficient model serving.

## Files

1. **start_vllm.sh** - PBS script that starts the vLLM server
2. **launch_vllm.sh** - Convenience script to launch server with any model
3. **update_endpoint.py** - Python script to update `configs/endpoints.py`
4. **test_vllm_connection.py** - Test script to verify server is running

## Quick Start

### Method 1: Using the Launch Script (Recommended)

The easiest way to start a vLLM server with any model:

```bash
# From the repo root directory
./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct

# Or with custom port and endpoint name:
./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct 8001 my-model
```

This script will:
1. Update `configs/endpoints.py` automatically
2. Submit the PBS job with your model
3. Print instructions for monitoring and using the server

Common models:
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`
- `willcb/DeepSeek-R1-Distill-Qwen-1.5B`

### Method 2: Manual PBS Submission

If you prefer manual control:

```bash
# 1. Create output directory
mkdir -p pbs_results

# 2. Submit with model name as variable
qsub -v MODEL="Qwen/Qwen2.5-3B-Instruct" scripts/start_vllm.sh

# Or with custom port:
qsub -v MODEL="Qwen/Qwen2.5-3B-Instruct",PORT=8001 scripts/start_vllm.sh

# 3. Update endpoints.py manually
python scripts/update_endpoint.py Qwen/Qwen2.5-3B-Instruct --port 8000
```

### 4. Check Job Status

```bash
qstat -u $USER
```

### 5. Monitor the Server

```bash
# Check the PBS job output (job info, setup, etc.)
tail -f pbs_results/vllm_server.out

# Check the vLLM server output (main server logs)
tail -f pbs_results/vllm_realtime.log

# Check for errors
tail -f pbs_results/vllm_realtime.err
```

### 6. Test the Connection

Once the server is running (check `pbs_results/vllm_realtime.log` for "Application startup complete"):

```bash
# Activate the environment first
source .venv/bin/activate

# Test connection (from repo root)
python scripts/test_vllm_connection.py
```

### 7. Use with vf-eval

Once the server is running, you can use it with any verifiers environment:

```bash
# Make sure you're in the verifiers directory and environment is activated
source .venv/bin/activate

# Run evaluation with your local vLLM server
uv run vf-eval your-environment -m local-vllm -n 5 -r 3
```

## Important Notes

### Resource Requirements

The script requests:
- **32 CPUs** - Adjust based on model size
- **4 GPUs** - For tensor parallelism (models are sharded across GPUs)
- **120GB RAM** - Adjust based on model size
- **48 hours** walltime - Adjust as needed

### Model Selection

Different models have different memory requirements. With **4 GPUs** and tensor parallelism, the model weights are distributed across all GPUs:

- 3B models: ~2GB per GPU (~8GB total)
- 7-8B models: ~4-5GB per GPU (~16-20GB total)  
- 13B models: ~8-10GB per GPU (~30-40GB total)
- 30B+ models: ~10-15GB per GPU (~40-60GB total)

For larger models, you may need quantization even with 4 GPUs. For smaller models (3-7B), consider reducing to 1-2 GPUs for better efficiency.

### Port Configuration

Default port is `8000`. If you need to run multiple vLLM servers:

1. Change `PORT=8000` to a different port in the PBS script
2. Update the corresponding endpoint in `configs/endpoints.py`

### Endpoints Configuration

If you use `launch_vllm.sh`, the endpoint is automatically updated in `configs/endpoints.py`. For manual updates:

```bash
# Update the local-vllm endpoint
python scripts/update_endpoint.py Qwen/Qwen2.5-3B-Instruct

# Or create a custom endpoint on a different port
python scripts/update_endpoint.py Qwen/Qwen2.5-3B-Instruct --port 8001 --name my-model
```

The endpoint configuration will look like:

```python
"local-vllm": {
    "model": "Qwen/Qwen2.5-3B-Instruct",  # Matches your vLLM server
    "url": "http://0.0.0.0:8000/v1",      # Port matches your server
    "key": "EMPTY",
},
```

## Troubleshooting

### Job won't start
- Check queue availability: `qstat -q`
- Check resource limits: reduce cpus/mem/walltime

### Server crashes
- Check `pbs_results/vllm_realtime.err` for errors
- Check `pbs_results/vllm_realtime.log` for server output
- Model may be too large for available GPU memory (4 GPUs total)
- Try a smaller model or enable quantization

### Can't connect to server
- Server needs time to load the model (check `pbs_results/vllm_realtime.log`)
- Look for "Application startup complete" or "Uvicorn running" in the logs
- Make sure you're testing from the same node (or use proper hostname)
- Check firewall/network settings

### Out of Memory
Edit the PBS script and add quantization:
```bash
vf-vllm \
    --model "$MODEL_NAME" \
    --quantization awq \  # Add this line
    --host "$HOST" \
    --port "$PORT" \
    --enforce-eager \
    --disable-log-requests
```

## Stopping the Server

```bash
# Find your job ID
qstat -u $USER

# Delete the job
qdel <JOB_ID>
```

## Advanced Usage

### Adjusting GPU Count

The default configuration uses **4 GPUs** with tensor parallelism. To use a different number:

Edit the PBS script:
```bash
#PBS -l ngpus=2  # Request 2 GPUs instead of 4

# In the script body, update the tensor parallel size:
vf-vllm \
    --model "$MODEL_NAME" \
    --tensor-parallel-size 2 \  # Match the number of GPUs
    ...

# Note: The CUDA_VISIBLE_DEVICES mapping will automatically handle 2 GPUs
```

For single GPU (no tensor parallelism):
```bash
#PBS -l ngpus=1

# Remove the --tensor-parallel-size flag entirely
vf-vllm \
    --model "$MODEL_NAME" \
    ...
```

### Enable Tool Calling

For models that support tool use:
```bash
vf-vllm \
    --model "$MODEL_NAME" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \  # or 'mistral', 'llama', etc.
    ...
```

