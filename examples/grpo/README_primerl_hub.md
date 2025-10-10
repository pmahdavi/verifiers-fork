# GRPO Training with PrimeRL Hub Environments

This guide explains how to use environments from the [PrimeRL Hub](https://app.primeintellect.ai/dashboard/environments) for GRPO (Group Relative Policy Optimization) training.

## Overview

The PrimeRL Hub is a community platform for discovering and sharing RL environments. This integration allows you to:
- Use any environment from the hub for GRPO training
- Leverage pre-built environments for math, coding, reasoning, and more
- Fine-tune language models with reinforcement learning

## Prerequisites

1. Install the required tools:
```bash
# Install uv for package management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install verifiers with training support
uv add 'verifiers[train]' && uv pip install flash-attn --no-build-isolation

# Install the prime CLI
uv tool install prime
```

2. Authenticate with Prime Intellect:
```bash
prime login
```

## Quick Start

### 1. Browse Available Environments

Visit the [Environments Hub](https://app.primeintellect.ai/dashboard/environments) to explore available environments.

### 2. Install an Environment

```bash
# Install a specific environment
prime env install primeintellect/math-python

# Install with version pinning
prime env install primeintellect/math-python@1.0.0

# List installed environments
prime env list
```

### 3. Run GRPO Training

Start the inference server (requires GPUs 0-5):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model Qwen/Qwen2.5-1.5B-Instruct \
    --data-parallel-size 6 --enforce-eager --disable-log-requests
```

In another terminal, start training (requires GPUs 6-7):
```bash
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_primerl_hub.py \
    --env-id math-python --model Qwen/Qwen2.5-1.5B-Instruct
```

## Key Hyperparameters

### Core Training Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--num-generations` | Number of rollouts per prompt | 8 | Must divide effective batch size |
| `--batch-size` | Per-device batch size | 8 | Adjust based on GPU memory |
| `--gradient-accumulation-steps` | Gradient accumulation | 4 | Increases effective batch size |
| `--max-steps` | Maximum training steps | 500 | Total optimization steps |
| `--eval-steps` | Evaluation frequency | 20 | Run eval every N steps |

### Generation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--max-tokens` | Max tokens per turn | 1024 | Environment-dependent |
| `--max-seq-len` | Max total sequence length | 4096 | Must fit in memory |
| `--temperature` | Sampling temperature | 1.0 | 0.1-2.0 typical |
| `--top-p` | Nucleus sampling | 1.0 | 0.1-1.0 |
| `--top-k` | Top-k sampling | None | Optional integer |

### GRPO Algorithm Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--beta` | KL coefficient | 0.001 | 0.0 disables reference model |
| `--epsilon` | PPO-style clipping | 0.2 | Controls policy update size |
| `--learning-rate` | Learning rate | 1e-6 | Typically 1e-7 to 1e-5 |
| `--max-concurrent` | Max concurrent env requests | 1024 | Lower for resource-heavy envs |

## Environment-Specific Examples

### Math Environments
```bash
prime env install primeintellect/math-python

python examples/grpo/train_primerl_hub.py \
    --env-id math-python \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-generations 16 \
    --max-tokens 2048 \
    --beta 0.1
```

### Coding Environments
```bash
prime env install primeintellect/code-debug

python examples/grpo/train_primerl_hub.py \
    --env-id code-debug \
    --model deepseek-ai/deepseek-coder-6.7b-instruct \
    --temperature 0.8 \
    --max-seq-len 8192
```

### Reasoning Environments
```bash
prime env install primeintellect/arc-1d

python examples/grpo/train_primerl_hub.py \
    --env-id arc-1d \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --batch-size 2 \
    --max-tokens 4096 \
    --beta 0.0
```

### Tool-Use Environments
```bash
prime env install primeintellect/tool-test

python examples/grpo/train_primerl_hub.py \
    --env-id tool-test \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-concurrent 256 \
    --num-generations 16
```

## Advanced Usage

### Custom Environment Arguments
```bash
python examples/grpo/train_primerl_hub.py \
    --env-id math-python \
    --env-args num_train_examples=5000 num_eval_examples=100 use_think=true
```

### LoRA Fine-tuning
```bash
python examples/grpo/train_primerl_hub.py \
    --env-id wordle \
    --model meta-llama/Llama-3.2-8B-Instruct \
    --use-lora \
    --batch-size 16
```

### Multi-GPU Configurations

For 4 GPUs (2 inference, 2 training):
```bash
# Inference
CUDA_VISIBLE_DEVICES=0,1 vf-vllm --model your-model --data-parallel-size 2

# Training
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_primerl_hub.py ...
```

For 8 GPUs (6 inference, 2 training):
```bash
# Inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model your-model --data-parallel-size 6

# Training
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_primerl_hub.py ...
```

## Understanding num_generations

The `num_generations` parameter controls how many completions are sampled per prompt. This is crucial for GRPO as it needs multiple samples to compute advantages.

**Important constraints:**
- Must be at least 2 (GRPO requires multiple samples)
- Must evenly divide the effective batch size
- Effective batch size = `per_device_train_batch_size × num_processes × gradient_accumulation_steps`

**Example:**
- Batch size: 8, Processes: 2, Accumulation: 4
- Effective batch size: 8 × 2 × 4 = 64
- Valid num_generations: 2, 4, 8, 16, 32, 64

## Monitoring Training

Training logs include:
- Rewards (mean, std, min, max)
- KL divergence from reference policy
- Generation metrics (tokens/sec, response lengths)
- Environment-specific metrics

Use Weights & Biases for detailed monitoring:
```bash
wandb login
# Training will automatically log to W&B
```

## Troubleshooting

### Environment Not Found
```
Error: Could not import 'math-python' environment
```
Solution: Install the environment first with `prime env install owner/env-name`

### Invalid num_generations
```
Error: Effective batch size (64) must be divisible by num_generations (10)
```
Solution: Choose a num_generations that divides the effective batch size (e.g., 8, 16, 32)

### Out of Memory
- Reduce `batch_size` or `max_seq_len`
- Enable gradient checkpointing (default for models >1B)
- Use fewer inference GPUs and more training GPUs
- Enable LoRA with `--use-lora`

### Slow Generation
- Increase `max_concurrent` for async environments
- Use more GPUs for inference
- Reduce `max_tokens` if appropriate

## Creating Custom Environments

To create your own environment for the hub:

1. Initialize a new environment:
```bash
vf-init my-custom-env
```

2. Implement the environment (see verifiers documentation)

3. Publish to the hub:
```bash
cd environments/my-custom-env
prime env push
```

## References

- [Verifiers Documentation](https://verifiers.readthedocs.io/)
- [PrimeRL Hub](https://app.primeintellect.ai/dashboard/environments)
- [GRPO Paper](https://arxiv.org/abs/2410.09585)
- [Prime Intellect Discord](https://discord.gg/primeintellect)