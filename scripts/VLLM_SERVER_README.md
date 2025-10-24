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
# Basic usage (uses default tensor parallelism)
./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct

# With custom port and endpoint name:
./scripts/launch_vllm.sh Qwen/Qwen2.5-3B-Instruct 8001 my-model

# With optimization flags for 4-5× better performance:
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8002 qwen3-optimized \
  "--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
```

**Usage**: `./scripts/launch_vllm.sh <model> [port] [endpoint_name] [vllm_flags]`

This script will:
1. Update `configs/endpoints.py` automatically
2. Submit the PBS job with your model and configuration
3. Print instructions for monitoring and using the server

Common models:
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`
- `willcb/DeepSeek-R1-Distill-Qwen-1.5B`

**💡 Pro Tip**: See [`docs/vllm_optimization/VLLM_OPTIMIZATION_PRESETS.md`](../docs/vllm_optimization/VLLM_OPTIMIZATION_PRESETS.md) for recommended configurations that can give you **4-5× better throughput**!

### Method 2: Manual PBS Submission

If you prefer manual control:

```bash
# 1. Create output directory
mkdir -p pbs_results

# 2. Submit with model name as variable
qsub -v MODEL="Qwen/Qwen2.5-3B-Instruct" scripts/start_vllm.sh

# Or with custom port:
qsub -v MODEL="Qwen/Qwen2.5-3B-Instruct",PORT=8001 scripts/start_vllm.sh

# Or with custom vLLM flags:
qsub -v MODEL="Qwen/Qwen2.5-3B-Instruct",PORT=8001,VLLM_FLAGS="--data-parallel-size 4 --max-num-seqs 512" scripts/start_vllm.sh

# 3. Update endpoints.py manually
python scripts/update_endpoint.py Qwen/Qwen2.5-3B-Instruct --port 8000
```

### 4. Check Job Status

```bash
qstat -u $USER
```

### 5. Monitor the Server

Each server creates unique log files based on the model name and port:

```bash
# View logs (filenames shown after job submission)
# Format: vllm_{model}_{port}_{type}.log

# Example for Qwen/Qwen2.5-3B-Instruct on port 8000:
tail -f pbs_results/vllm_Qwen_Qwen2.5-3B-Instruct_port8000_realtime.log
tail -f pbs_results/vllm_Qwen_Qwen2.5-3B-Instruct_port8000_realtime.err
tail -f pbs_results/vllm_Qwen_Qwen2.5-3B-Instruct_port8000_pbs.out
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

## Running Multiple vLLM Servers Simultaneously

### ⚠️ Important: GPU Resource Contention

**Each vLLM server uses ALL GPUs** specified in its configuration. You **CANNOT** run multiple servers with overlapping GPUs.

#### Example: Why This Fails
```bash
# ❌ BAD: Both servers try to use ALL 4 GPUs
Server 1: --data-parallel-size 4  (uses GPUs 0,1,2,3)
Server 2: --data-parallel-size 4  (uses GPUs 0,1,2,3)
# Result: OOM errors, crashes, conflicts
```

### ✅ Option 1: Run Servers Sequentially (Recommended)

For evaluating multiple models, run them **one at a time**:

```bash
# 1. Run evaluation with Model A
tmux new -s eval-model-a
uv run vf-eval env -m model-a -n -1 -r 4 -s -v
# Ctrl+b, d to detach

# 2. After Model A completes, run Model B
tmux new -s eval-model-b
uv run vf-eval env -m model-b -n -1 -r 4 -s -v
# Ctrl+b, d to detach
```

**Why this is better**:
- Each server gets full GPU resources (optimal performance)
- No memory contention
- Easier to monitor and debug
- Better total throughput

### ✅ Option 2: Partition GPUs Across Servers

If you **must** run simultaneously, partition the GPUs:

```bash
# Server 1: 2B model on GPUs 0,1 (data parallel)
CUDA_VISIBLE_DEVICES=0,1 ./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B 8010 model-2b \
  "--data-parallel-size 2 --max-num-seqs 32"

# Server 2: 4B model on GPU 2 (single GPU)
CUDA_VISIBLE_DEVICES=2 ./scripts/launch_vllm.sh Qwen/Qwen3-VL-4B 8011 model-4b \
  "--max-num-seqs 8"

# Server 3: 8B model on GPU 3 (single GPU)  
CUDA_VISIBLE_DEVICES=3 ./scripts/launch_vllm.sh Qwen/Qwen3-VL-8B 8012 model-8b \
  "--max-num-seqs 4"
```

**Trade-offs**:
- ✅ Can evaluate multiple models simultaneously
- ❌ Each server has fewer GPUs (lower throughput per model)
- ❌ More complex configuration
- ❌ Harder to optimize each server individually

### Resource Planning Table

| # of Models | Strategy | Config Example | Total Throughput |
|-------------|----------|----------------|------------------|
| **1 model** | Sequential | DP=4 on 4 GPUs | **100%** ✅ Best |
| **2 models** | Partition | DP=2 on 2 GPUs each | **~70%** |
| **4 models** | Partition | 1 GPU each | **~40%** ❌ Slow |

**Recommendation**: Unless you have a specific need for concurrent serving, **always run sequentially** for best performance.

### Using Tmux for Sequential Evaluations

Create separate tmux sessions for monitoring:

```bash
# Session 1: 2B model evaluation
tmux new -s eval-2b
cd /scratch/pxm5426/repos/verifiers-fork
uv run vf-eval inoi -m qwen3-2b-32k -t 32768 -n -1 -r 4 -a '{"use_think": false}' -s -v
# Ctrl+b, d to detach

# Check status later
tmux a -t eval-2b

# After 2B completes, run 4B
tmux new -s eval-4b
uv run vf-eval inoi -m qwen3-4b-32k -t 32768 -n -1 -r 4 -a '{"use_think": false}' -s -v
# Ctrl+b, d
```

**Monitor all sessions**:
```bash
tmux ls  # List all sessions
tmux a -t eval-2b  # Attach to specific session
```

---

## Stopping the Server

```bash
# Find your job ID
qstat -u $USER

# Delete the job
qdel <JOB_ID>

# If running multiple servers, stop specific one:
qdel <JOB_ID_FOR_SPECIFIC_SERVER>
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

## Performance Optimization

The default configuration uses tensor parallelism, which is **not optimal for small models** (< 7B parameters). You can achieve **4-5× better throughput** by using the right configuration.

### Quick Wins for Small Models (2-7B)

**Default (slow)**:
```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8000 default
# Uses tensor parallelism: ~412 tokens/s generation throughput
```

**Optimized (4-5× faster)**:
```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8001 optimized \
  "--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
# Uses data parallelism: ~1600-2000 tokens/s generation throughput
```

### Why the Difference?

| Configuration | Tensor Parallel (TP=4) | Data Parallel (DP=4) |
|---------------|------------------------|----------------------|
| **How it works** | Splits 2B model across 4 GPUs | Runs full 2B model on each GPU |
| **Model per GPU** | 0.5B params (1GB) | 2B params (4GB) |
| **Communication** | Heavy (every layer) | None (independent) |
| **Throughput** | 1× baseline | **4× baseline** |
| **Best for** | Large models (20B+) | Small models (< 7B) |

### Optimization Presets

See **[`docs/vllm_optimization/VLLM_OPTIMIZATION_PRESETS.md`](../docs/vllm_optimization/VLLM_OPTIMIZATION_PRESETS.md)** for detailed presets and decision trees.

**Quick reference**:

#### Small Models (< 7B) - Use Data Parallelism
```bash
# High Throughput (recommended)
"--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"

# Balanced
"--data-parallel-size 2 --tensor-parallel-size 2 --max-num-seqs 256"
```

#### Medium Models (7-20B) - Use Hybrid
```bash
# Best for 8B-20B models
"--tensor-parallel-size 2 --data-parallel-size 2 --max-num-seqs 256 --max-num-batched-tokens 8192"
```

#### Large Models (20B+) - Use Tensor Parallelism
```bash
# Required for 70B models
"--tensor-parallel-size 4 --max-num-seqs 128 --max-num-batched-tokens 8192"
```

### Key Parameters Explained

- **`--data-parallel-size N`**: Run N independent model replicas (N× throughput, no communication overhead)
- **`--tensor-parallel-size N`**: Split model across N GPUs (for models too large for single GPU)
- **`--max-num-seqs N`**: Maximum concurrent requests (higher = more throughput, more memory)
- **`--max-num-batched-tokens N`**: Prefill chunk size (larger = faster prefill, more memory per batch)
- **`--gpu-memory-utilization X`**: Fraction of GPU memory for KV cache (0.95 = use 95%)
- Remove **`--enforce-eager`**: Enables CUDA graphs (1.5-2× faster decode)

### Monitoring and Tuning

After starting your server, monitor the metrics:

```bash
tail -f pbs_results/vllm_*_realtime.log | grep "Engine 000"
```

**Key metric: KV cache usage**
- < 30%: Increase `--max-num-seqs` (you can batch more requests!)
- 30-70%: Optimal
- > 80%: Decrease `--max-num-seqs` (risk of OOM)

**Example tuning workflow**:
```bash
# Start with conservative batch size
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8000 test \
  "--data-parallel-size 4 --max-num-seqs 128"

# Check KV cache usage in logs
# If usage is 15%, you can handle 4× more requests!

# Increase batch size
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8001 test \
  "--data-parallel-size 4 --max-num-seqs 512"
```

### Additional Resources

- **[Chunked Prefill Tutorial](../docs/vllm_optimization/CHUNKED_PREFILL_TUTORIAL.md)**: Deep dive into how vLLM processes long contexts
- **[Optimization Presets](../docs/vllm_optimization/VLLM_OPTIMIZATION_PRESETS.md)**: Comprehensive guide with specific configurations for different use cases
- **[Performance Analysis Guide](../docs/vllm_optimization/PERFORMANCE_OPTIMIZATION_GUIDE.md)**: Detailed analysis of your current configuration and optimization opportunities

