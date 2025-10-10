# Modal Training Tutorial for Verifiers

This tutorial will teach you how to run your Verifiers GRPO training on Modal's serverless GPU infrastructure, providing a scalable and cost-effective way to train your models.

## Table of Contents

1. [Why Use Modal?](#why-use-modal)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Understanding the Setup](#understanding-the-setup)
5. [Running Different Environments](#running-different-environments)
6. [GPU Configuration](#gpu-configuration)
7. [Monitoring Your Training](#monitoring-your-training)
8. [Cost Management](#cost-management)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Why Use Modal?

Modal provides several advantages for training Verifiers models:

- **No Infrastructure Management**: No need to set up or maintain GPU servers
- **Serverless**: Pay only for what you use, with per-second billing
- **Scalable**: Easy access to various GPU types (T4, L4, A100, H100, etc.)
- **Fast Iteration**: Your local code is mounted at runtime, so changes are reflected immediately
- **Persistent Storage**: Automatic handling of model checkpoints and outputs

## Prerequisites

### 1. Install Modal CLI

```bash
pip install modal
```

### 2. Set Up Modal Account

```bash
modal setup
```

This will open your browser to authenticate with Modal.

### 3. Create Required Secrets

Verifiers training typically needs W&B for logging and HuggingFace for model downloads:

```bash
# Create W&B secret
modal secret create wandb WANDB_API_KEY=your_wandb_key_here

# Create HuggingFace secret
modal secret create huggingface HF_TOKEN=your_hf_token_here
```

Or create them via the Modal web UI: https://modal.com/secrets

### 4. Verify Your Setup

```bash
# Check that secrets exist
modal secret list
```

You should see `wandb` and `huggingface` in the list.

## Quick Start

The simplest way to run a training job:

```bash
# From the verifiers repository root
cd /path/to/verifiers

# Run GSM8K training with 2 A100-80GB GPUs
MODAL_GPU_CONFIG="A100-80GB:2" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests & sleep 30 && CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py" \
  --experiment-name "my-gsm8k-run"
```

**Important Flags**:
- `--detach`: Keeps training running even if you disconnect (recommended for long runs)
- `--experiment-name`: Names your experiment for easy identification

## Understanding the Setup

### Architecture

Verifiers GRPO training uses a **dual-process architecture**:

```
┌─────────────────────────────────────────────────────┐
│                  Modal Container                     │
│                                                      │
│  ┌────────────────┐         ┌──────────────────┐   │
│  │   GPU 0        │         │     GPU 1        │   │
│  │                │         │                  │   │
│  │  vLLM Server   │◄────────┤  GRPO Trainer    │   │
│  │  (Inference)   │  HTTP   │  (DeepSpeed)     │   │
│  └────────────────┘         └──────────────────┘   │
│         │                            │              │
│         │                            ▼              │
│         │                    Weight Updates         │
│         └─────────────────────────────────────────►│
└─────────────────────────────────────────────────────┘
```

**Process Flow**:
1. vLLM server loads model and serves inference requests on GPU 0
2. GRPO trainer on GPU 1 requests completions from vLLM
3. Trainer computes rewards and updates model weights
4. Updated weights are synced back to vLLM for next iteration

### The Command Structure

A typical training command has three parts:

```bash
# Part 1: Start vLLM inference server (background)
CUDA_VISIBLE_DEVICES=0 vf-vllm --model MODEL_NAME --enforce-eager --disable-log-requests &

# Part 2: Wait for vLLM to initialize
sleep 30 &&

# Part 3: Start GRPO training (foreground)
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
  --config-file configs/zero3.yaml examples/grpo/TRAINING_SCRIPT.py
```

**Key Points**:
- `&` runs vLLM in background
- `sleep 30` gives vLLM time to load the model
- `&&` ensures training only starts if previous commands succeed
- Training runs in foreground so Modal captures its output

## Running Different Environments

All training examples are in `examples/grpo/`. Here's how to run each one:

### GSM8K (Math Reasoning)

**Model**: Qwen3-0.6B
**GPUs Needed**: 2 (1 inference + 1 training)
**Difficulty**: ⭐ Easy

```bash
MODAL_GPU_CONFIG="A100-80GB:2" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B --enforce-eager --disable-log-requests & sleep 30 && CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py" \
  --experiment-name "gsm8k-experiment"
```

### Wordle

**Model**: Qwen3-1.7B or Qwen3-4B
**GPUs Needed**: 8 (6 inference + 2 training)
**Difficulty**: ⭐⭐ Medium

```bash
# 1.7B model
MODAL_GPU_CONFIG="H100:8" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-1.7B-Wordle --data-parallel-size 6 --enforce-eager --disable-log-requests & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 1.7B" \
  --experiment-name "wordle-1.7b"

# 4B model
MODAL_GPU_CONFIG="H100:8" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-4B-Wordle --data-parallel-size 6 --enforce-eager --disable-log-requests & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 4B" \
  --experiment-name "wordle-4b"
```

### Math Python (Code Generation)

**Model**: Qwen2.5-1.7B or 4B
**GPUs Needed**: 8
**Difficulty**: ⭐⭐⭐ Hard

```bash
MODAL_GPU_CONFIG="H100:8" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-1.7B-Instruct --data-parallel-size 6 --enforce-eager & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_math_python.py --size 1.7B" \
  --experiment-name "math-python-1.7b"
```

### Reverse Text

**Model**: Qwen2.5-1.7B-Instruct
**GPUs Needed**: 8
**Difficulty**: ⭐ Easy (good for testing)

```bash
MODAL_GPU_CONFIG="H100:8" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-1.7B-Instruct --data-parallel-size 6 --enforce-eager & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_reverse_text.py" \
  --experiment-name "reverse-text"
```

### Custom Environments

To run your own environment that's not in the default build:

```bash
MODAL_GPU_CONFIG="H100:8" modal run --detach modal/deploy.py \
  --command "uv pip install -e environments/your_env && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model YOUR_MODEL --data-parallel-size 6 --enforce-eager & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_your_env.py" \
  --experiment-name "your-env-experiment"
```

**Note**: The following environments are pre-installed:
- `gsm8k`
- `math_python`
- `wordle`
- `reverse_text`
- `continuation_quality`

## GPU Configuration

### Available GPU Types

Modal offers various GPU types. Choose based on your model size and budget:

| GPU Type | Memory | Cost/Hour | Best For |
|----------|--------|-----------|----------|
| T4 | 16GB | $0.59 | Small models (<1B params) |
| L4 | 24GB | $0.80 | Small models, testing |
| A10G | 24GB | $1.10 | Medium models (~1-3B) |
| A100-40GB | 40GB | $2.10 | Medium-large models |
| A100-80GB | 80GB | $2.50 | Large models (7B+) |
| H100 | 80GB | $3.95 | Fast training, large models |

### Setting GPU Count

Use the `MODAL_GPU_CONFIG` environment variable:

```bash
# 2 GPUs (minimum for GRPO)
MODAL_GPU_CONFIG="A100-80GB:2" modal run modal/deploy.py ...

# 4 GPUs
MODAL_GPU_CONFIG="H100:4" modal run modal/deploy.py ...

# 8 GPUs (typical for larger models)
MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py ...
```

### GPU Allocation Guidelines

**For 2 GPU Setup** (Small Models: 0.6B-1B):
```bash
# 1 inference + 1 training
CUDA_VISIBLE_DEVICES=0 vf-vllm ... &
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 ...
```

**For 8 GPU Setup** (Medium Models: 1.7B-4B):
```bash
# 6 inference + 2 training (recommended)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --data-parallel-size 6 ... &
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 ...
```

**For 16 GPU Setup** (Large Models: 7B+):
```bash
# 12 inference + 4 training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 vf-vllm --data-parallel-size 12 ... &
CUDA_VISIBLE_DEVICES=12,13,14,15 accelerate launch --num-processes 4 ...
```

## Monitoring Your Training

### View Live Logs

Modal provides a web UI to monitor your training:

1. When you run `modal run`, you'll see a URL like:
   ```
   View run at https://modal.com/apps/USERNAME/main/ap-XXXXX
   ```

2. Open this URL in your browser to see:
   - Real-time logs
   - GPU utilization
   - Memory usage
   - Running time and cost

### Command-Line Monitoring

```bash
# View logs of most recent app
modal app logs

# Monitor GPU usage
modal app stats

# List all running apps
modal app list
```

### W&B Integration

Your training metrics are automatically logged to Weights & Biases:

1. Go to https://wandb.ai
2. Find your project (default: `verifiers-grpo` or as configured in training script)
3. View metrics, loss curves, and sample generations

## Cost Management

### Estimating Costs

**Formula**: `Cost = GPUs × Cost per GPU × Hours`

**Example Calculations**:

```
GSM8K (0.6B model, 2× A100-80GB, 1 hour):
$2.50/GPU/hr × 2 GPUs × 1 hr = $5.00

Wordle (1.7B model, 8× H100, 2 hours):
$3.95/GPU/hr × 8 GPUs × 2 hrs = $63.20

Math Python (4B model, 8× H100, 4 hours):
$3.95/GPU/hr × 8 GPUs × 4 hrs = $126.40
```

### Cost Optimization Tips

1. **Start Small**: Test with small models and fewer steps first
   ```python
   # In your training script, reduce max_steps for testing
   training_args.max_steps = 10  # Instead of 200
   ```

2. **Use Cheaper GPUs**: For testing, use L4 or A10G instead of H100
   ```bash
   MODAL_GPU_CONFIG="L4:2" modal run modal/deploy.py ...
   ```

3. **Stop When Done**: Monitor training and stop if performance plateaus
   ```bash
   # You can stop a detached run from Modal's web UI
   # Or find the app ID and stop it:
   modal app stop ap-XXXXX
   ```

4. **Use Detached Mode**: Avoid keeping your laptop running
   ```bash
   modal run --detach ...  # Always use --detach for long jobs
   ```

### Viewing Usage and Costs

Check your Modal dashboard: https://modal.com/usage

## Troubleshooting

### Issue: "Environment not found"

**Error**: `ModuleNotFoundError: No module named 'your_env'`

**Solution**: Install the environment before training:
```bash
--command "uv pip install -e environments/your_env && CUDA_VISIBLE_DEVICES=..."
```

### Issue: "OOM (Out of Memory)"

**Error**: Training crashes with CUDA out of memory

**Solutions**:
1. Reduce batch size in training script:
   ```python
   training_args.per_device_train_batch_size = 4  # Reduce from 8
   ```

2. Use more GPUs:
   ```bash
   MODAL_GPU_CONFIG="H100:16" modal run modal/deploy.py ...
   ```

3. Use GPUs with more memory:
   ```bash
   MODAL_GPU_CONFIG="A100-80GB:8" modal run modal/deploy.py ...
   ```

### Issue: "vLLM not ready"

**Error**: Training starts before vLLM is ready, causing connection errors

**Solution**: Increase sleep time:
```bash
--command "... & sleep 60 && CUDA_VISIBLE_DEVICES=..."  # Increase from 30 to 60
```

### Issue: "Image build too slow"

**Symptom**: First run takes 5-10 minutes to build

**This is normal** for the first run! Modal caches the built image, so subsequent runs will be much faster (30 seconds).

**Tips**:
- The image only rebuilds if you change dependencies in `pyproject.toml`
- Code changes (in `verifiers/`, `examples/`, `environments/`) don't trigger rebuilds

### Issue: "GPU allocation mismatch"

**Error**: `CUDA_VISIBLE_DEVICES` references GPUs that don't exist

**Solution**: Make sure total GPUs match what's requested:
```bash
# Wrong: Requesting only 4 GPUs but using GPU indices up to 7
MODAL_GPU_CONFIG="H100:4" modal run modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm ... & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 ..."

# Correct: Request 8 GPUs
MODAL_GPU_CONFIG="H100:8" modal run modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm ... & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 ..."
```

## Advanced Usage

### Downloading Results

After training completes, download your outputs:

```bash
# List available experiments
modal volume ls verifiers-outputs

# Download specific experiment
modal volume get verifiers-outputs my-gsm8k-run ./local-outputs/

# Download just checkpoints
modal volume get verifiers-outputs my-gsm8k-run/checkpoints ./checkpoints/
```

### Resuming Training

To resume from a checkpoint:

1. First, ensure your checkpoint is saved in the Modal volume
2. Modify your training command to include `--resume-from-checkpoint`:

```bash
MODAL_GPU_CONFIG="H100:8" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model MODEL & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py --resume-from-checkpoint /outputs/my-experiment/checkpoint-100" \
  --experiment-name "my-experiment-resumed"
```

### Custom Training Scripts

To run your own training script:

1. Create your script in `examples/grpo/train_my_task.py`
2. Use it in the command:

```bash
MODAL_GPU_CONFIG="H100:8" modal run --detach modal/deploy.py \
  --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model YOUR_MODEL --data-parallel-size 6 --enforce-eager & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_my_task.py" \
  --experiment-name "my-task"
```

### Multi-Experiment Workflows

Run multiple experiments programmatically:

```bash
#!/bin/bash
# run_experiments.sh

for lr in 1e-6 1e-5 1e-4; do
  for model in "0.6B" "1.7B"; do
    MODAL_GPU_CONFIG="A100-80GB:8" modal run --detach modal/deploy.py \
      --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-${model} --data-parallel-size 6 --enforce-eager & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py --learning-rate ${lr}" \
      --experiment-name "gsm8k-${model}-lr${lr}"

    # Add delay between submissions to avoid overwhelming Modal
    sleep 10
  done
done
```

### Using Different Accelerate Configs

Modal deployment includes the default `configs/zero3.yaml` DeepSpeed config. To use a different config:

1. Create your config file: `configs/my_custom_config.yaml`
2. Reference it in your command:

```bash
--command "... accelerate launch --config-file configs/my_custom_config.yaml ..."
```

## Best Practices

### 1. Always Use Detached Mode for Long Runs

```bash
# Good
modal run --detach modal/deploy.py ...

# Bad (for long runs)
modal run modal/deploy.py ...
```

### 2. Name Your Experiments Descriptively

```bash
# Good
--experiment-name "gsm8k-qwen3-0.6b-lr1e5-2025-01-10"

# Less useful
--experiment-name "test"
```

### 3. Start with Short Runs

Reduce `max_steps` in your training script first to verify everything works:

```python
training_args.max_steps = 10  # Test first
# training_args.max_steps = 200  # Full training
```

### 4. Monitor Early

Check the Modal web UI within the first 5 minutes to catch issues early.

### 5. Keep Dependencies Updated

Periodically update your local verifiers installation:

```bash
cd /path/to/verifiers
uv sync --extra train
```

Then redeploy - Modal will rebuild the image with updated dependencies.

## Summary

You now know how to:
- ✅ Set up Modal for Verifiers training
- ✅ Run any example from `examples/grpo/`
- ✅ Configure GPUs for different model sizes
- ✅ Monitor training and manage costs
- ✅ Debug common issues
- ✅ Download results and resume training

## Next Steps

1. **Try a small experiment**: Start with GSM8K on 2 A100s
2. **Scale up**: Once comfortable, try larger models and more GPUs
3. **Customize**: Create your own environments and training scripts
4. **Share**: Publish your trained models to HuggingFace!

## Getting Help

- **Modal Docs**: https://modal.com/docs
- **Verifiers Docs**: https://verifiers.readthedocs.io/
- **Modal Community**: https://modal.com/slack
- **Issues**: https://github.com/PrimeIntellect-ai/verifiers/issues

Happy training! 🚀