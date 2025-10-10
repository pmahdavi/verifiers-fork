# Modal Deployment for Verifiers

Run Verifiers training on [Modal](https://modal.com) serverless GPU infrastructure with a single command.

📚 **New to Modal?** Check out the [comprehensive tutorial](TUTORIAL.md) for step-by-step guidance!

## Quick Start

```bash
# Run with default settings (Wordle GRPO training)
modal run modal/deploy.py

# Run custom command
modal run modal/deploy.py --command "your command here"

# Run GSM8K training (recommended for first-time users)
MODAL_GPU_CONFIG="A100-80GB:2" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests & sleep 30 && CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py" \
  --experiment-name "my-first-run"
```

## Prerequisites

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Set up Modal account**:
   ```bash
   modal setup
   ```

3. **Create Modal secrets** (required for W&B and HuggingFace):
   ```bash
   # Create wandb secret
   modal secret create wandb WANDB_API_KEY=your_wandb_key_here

   # Create huggingface secret
   modal secret create huggingface HF_TOKEN=your_hf_token_here
   ```

   Or create them via the web UI:
   - https://modal.com/secrets

## Usage Examples

### Basic GRPO Training (Wordle)

```bash
# Run with defaults (Wordle, 8 GPUs: 2 trainer + 6 inference)
modal run modal/deploy.py
```

### Custom Training Command

The key is to provide your full training command as a string. The script will run it on Modal with all GPUs and dependencies available.

```bash
# Example: Wordle training with explicit GPU allocation
modal run modal/deploy.py --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-1.7B-Wordle --data-parallel-size 6 --enforce-eager --disable-log-requests & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 1.7B"
```

### Different Environments

```bash
# GSM8K training
modal run modal/deploy.py --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-1.7B-Instruct --data-parallel-size 6 --enforce-eager & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py --size 1.7B"

# Math Python training
modal run modal/deploy.py --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-4B-Instruct --data-parallel-size 6 --enforce-eager & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_math_python.py --size 4B"
```

### Custom Experiment Name

```bash
modal run modal/deploy.py \
  --command "your training command" \
  --experiment-name "my-custom-experiment"
```

### Different GPU Types

Set the `MODAL_GPU_CONFIG` environment variable to change GPU configuration:

```bash
# Use 8 H100 GPUs (default)
MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py

# Use 8 A100-80GB GPUs
MODAL_GPU_CONFIG="A100-80GB:8" modal run modal/deploy.py

# Use 4 A100-80GB GPUs (if you need fewer)
MODAL_GPU_CONFIG="A100-80GB:4" modal run modal/deploy.py --command "your adjusted command for 4 GPUs"
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--command` | str | (wordle training) | Command to run on Modal |
| `--experiment-name` | str | auto-generated | Name for this experiment |
| `--download-results` | bool | True | Show download instructions |

## GPU Configuration

Set the `MODAL_GPU_CONFIG` environment variable to change GPU allocation:

```bash
# Examples
MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py
MODAL_GPU_CONFIG="A100-80GB:8" modal run modal/deploy.py
MODAL_GPU_CONFIG="L40S:4" modal run modal/deploy.py

# Default if not set: H100:8
modal run modal/deploy.py
```

**Important**: Make sure your `--command` allocates GPUs correctly based on the total available. For example, if you request 8 GPUs and want 6 for inference and 2 for training:
- Inference: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`
- Training: `CUDA_VISIBLE_DEVICES=6,7`

## Outputs

All outputs are saved to Modal volumes:

- **Logs**: Training logs and W&B tracking
- **Checkpoints**: Model checkpoints at specified intervals
- **Outputs**: Any other training artifacts

### Downloading Results

```bash
# Download entire experiment
modal volume get verifiers-outputs <experiment-name> ./outputs/<experiment-name>

# List experiments
modal volume ls verifiers-outputs

# List files in an experiment
modal volume ls verifiers-outputs/<experiment-name>
```

## Monitoring

```bash
# View real-time logs
modal app logs

# Monitor GPU usage
modal app stats

# View on W&B (if enabled)
# Go to: https://wandb.ai/<your-username>/<your-project>
```

## Architecture

The deployment uses Modal 1.0 API with:

- **Base Image**: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
- **Package Manager**: `uv` for fast dependency installation
- **Secrets**: Automatic injection of W&B and HuggingFace tokens
- **Volumes**: Persistent storage for outputs and cache
- **Code Mounting**: Your local `verifiers/`, `configs/`, `examples/`, and `environments/` are mounted at runtime for fast iteration

### Image Build Process

1. Install system dependencies (git, curl, build tools)
2. Install `uv` package manager
3. Copy and install Python dependencies from `pyproject.toml`
4. Install `verifiers[train]` with flash-attn
5. Set up virtual environment at `/app/.venv`
6. Mount local source code at runtime (for fast iteration)

## Typical GRPO Training Setup

Verifiers GRPO training typically uses:
- **Inference Server**: vLLM with data parallelism across multiple GPUs
- **Training**: Accelerate/DeepSpeed across 1-2 GPUs

Example command structure:
```bash
# Start vLLM in background
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm \
  --model <model-name> \
  --data-parallel-size 6 \
  --enforce-eager \
  --disable-log-requests &

# Wait for vLLM to start
sleep 30

# Start training in foreground
CUDA_VISIBLE_DEVICES=6,7 accelerate launch \
  --num-processes 2 \
  --config-file configs/zero3.yaml \
  <training-script> \
  --size <model-size>
```

## Cost Estimation

Modal GPU pricing (as of 2025):

| GPU Type | Cost/GPU/Hour | Example: 4 GPUs | Example: 8 GPUs |
|----------|---------------|-----------------|-----------------|
| T4 | $0.59 | $2.36/hour | $4.72/hour |
| L4 | $0.80 | $3.20/hour | $6.40/hour |
| A10 | $1.10 | $4.40/hour | $8.80/hour |
| L40S | $1.95 | $7.80/hour | $15.60/hour |
| A100-40GB | $2.10 | $8.40/hour | $16.80/hour |
| A100-80GB | $2.50 | $10.00/hour | $20.00/hour |
| H100 | $3.95 | $15.80/hour | $31.60/hour |
| H200 | $4.54 | $18.16/hour | $36.32/hour |
| B200 | $6.25 | $25.00/hour | $50.00/hour |

## Troubleshooting

### Build Issues

If you get dependency errors:
```bash
# Make sure dependencies are up to date locally
cd /path/to/verifiers
uv sync --extra train

# Then deploy
modal run modal/deploy.py
```

### Secret Issues

If secrets are missing:
```bash
# List existing secrets
modal secret list

# Recreate if needed
modal secret create wandb WANDB_API_KEY=$WANDB_API_KEY
modal secret create huggingface HF_TOKEN=$HF_TOKEN
```

### Out of Memory

If you hit OOM errors:
- Reduce batch size in your training script
- Allocate more GPUs: `MODAL_GPU_CONFIG="H100:16" modal run modal/deploy.py`
- Use larger GPU types: `MODAL_GPU_CONFIG="A100-80GB:8" modal run modal/deploy.py`

### vLLM Server Not Starting

If training fails because vLLM isn't ready:
- Increase the `sleep` time between starting vLLM and training (e.g., `sleep 60`)
- Check vLLM logs via `modal app logs`

### GPU Allocation Errors

Make sure your command's `CUDA_VISIBLE_DEVICES` matches the total GPUs in `MODAL_GPU_CONFIG`:
```bash
# Wrong: requesting 8 GPUs but only 4 available
MODAL_GPU_CONFIG="H100:4" modal run modal/deploy.py --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ..."

# Correct: 8 GPUs available, 6 for inference, 2 for training
MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm ... & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch ..."
```

## Development Workflow

1. Make changes to your code locally in `verifiers/`, `examples/`, etc.
2. Run deployment - your changes are automatically synced
3. No need to rebuild the image unless you change dependencies in `pyproject.toml`

**Fast iteration**: Source code is mounted at runtime, so changes to code don't require image rebuilds!

## Resources

- [Modal Documentation](https://modal.com/docs)
- [Verifiers Documentation](https://verifiers.readthedocs.io/)
- [Modal Pricing](https://modal.com/pricing)
- [Verifiers Examples](../examples/grpo/)
