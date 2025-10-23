# vLLM Optimization Presets

Quick reference for common vLLM configurations optimized for different use cases.

## Usage

```bash
./scripts/launch_vllm.sh <model> <port> <endpoint_name> "<vllm_flags>"
```

---

## Presets for Small Models (< 7B parameters)

### 🚀 High Throughput (Recommended for 2-7B models)
**Best for**: Batch processing, high request volume, maximum throughput

```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8000 qwen3-optimized \
  "--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
```

**What it does**:
- Each GPU runs the full model independently (4× throughput)
- Handles 512+ concurrent requests
- Larger prefill chunks (8192 tokens)
- Uses 95% of GPU memory for KV cache

**Expected performance**: 4-5× faster than default tensor parallelism

---

### ⚖️ Balanced (Hybrid Parallelism)
**Best for**: Medium request volume, moderate prompt lengths

```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8001 qwen3-balanced \
  "--data-parallel-size 2 --tensor-parallel-size 2 --max-num-seqs 256 --max-num-batched-tokens 4096"
```

**What it does**:
- 2 data parallel replicas, each using 2 GPUs with tensor parallelism
- Handles 256+ concurrent requests
- Balanced memory per replica

**Expected performance**: 2-3× faster than default

---

### ⚡ Low Latency (Fast Response)
**Best for**: Real-time applications, single-user scenarios, minimum latency

```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8002 qwen3-lowlat \
  "--tensor-parallel-size 4 --max-num-seqs 64 --max-num-batched-tokens 2048"
```

**What it does**:
- Uses tensor parallelism for fastest single-request processing
- Smaller batches (64 requests)
- Standard chunk size (2048 tokens)

**Expected performance**: Lowest per-request latency, lower total throughput

---

### 🔬 Debug/Development
**Best for**: Testing, debugging, development

```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8003 qwen3-debug \
  "--tensor-parallel-size 4 --enforce-eager --max-num-seqs 32"
```

**What it does**:
- Eager mode for better error messages
- Small batch size for faster iteration
- Tensor parallelism for compatibility

**Expected performance**: Slower, but easier to debug

---

## Presets for Medium Models (7B - 20B parameters)

### 🚀 High Throughput
```bash
./scripts/launch_vllm.sh meta-llama/Llama-3.1-8B-Instruct 8010 llama-optimized \
  "--tensor-parallel-size 2 --data-parallel-size 2 --max-num-seqs 256 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
```

**Rationale**: 8B model needs ~16GB, so use TP=2 (8GB/GPU) then DP=2 for throughput

---

### ⚖️ Balanced
```bash
./scripts/launch_vllm.sh meta-llama/Llama-3.1-8B-Instruct 8011 llama-balanced \
  "--tensor-parallel-size 2 --max-num-seqs 128 --max-num-batched-tokens 4096"
```

**Rationale**: Simple TP=2 for memory efficiency, moderate batch size

---

## Presets for Large Models (20B - 70B parameters)

### 🚀 High Throughput
```bash
./scripts/launch_vllm.sh meta-llama/Llama-3.1-70B-Instruct 8020 llama70b-optimized \
  "--tensor-parallel-size 4 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
```

**Rationale**: 70B model needs ~140GB, so must use TP=4 (35GB/GPU)

---

### ⚡ Low Latency
```bash
./scripts/launch_vllm.sh meta-llama/Llama-3.1-70B-Instruct 8021 llama70b-lowlat \
  "--tensor-parallel-size 4 --max-num-seqs 32 --max-num-batched-tokens 2048"
```

**Rationale**: TP=4 required, smaller batches for lower latency

---

## Special Configurations

### 💾 Memory-Constrained
**When**: GPU memory is limited or want to maximize batch size

```bash
"--data-parallel-size 4 --quantization awq --max-num-seqs 1024 --gpu-memory-utilization 0.9"
```

**What it does**: Uses AWQ quantization (2× less memory), very high batch size

---

### 🎯 Very Long Context (> 100K tokens)
**When**: Processing extremely long documents

```bash
"--data-parallel-size 2 --max-num-batched-tokens 4096 --max-num-seqs 32 --gpu-memory-utilization 0.95 --enable-chunked-prefill"
```

**What it does**: Smaller chunks to manage memory, fewer concurrent requests

---

### 🔄 Speculative Decoding (Experimental)
**When**: Want 2-3× faster generation for suitable tasks

```bash
"--data-parallel-size 4 --speculative-model <draft-model> --num-speculative-tokens 5 --max-num-seqs 256"
```

**What it does**: Uses smaller draft model to predict tokens, main model verifies

---

## Quick Decision Tree

```
Is your model < 7B parameters?
├─ YES: Use data parallelism (--data-parallel-size 4)
│   └─ High volume? Use --max-num-seqs 512+
│   └─ Low latency? Use --max-num-seqs 64
│
└─ NO: Model is 7B+ parameters
    ├─ Model 7B-20B?
    │   └─ Use TP=2, DP=2 (hybrid)
    │
    └─ Model 20B-70B?
        └─ Use TP=4, DP=1
```

---

## Parameter Reference

### Parallelism Strategy
- `--tensor-parallel-size N`: Split model across N GPUs (for large models)
- `--data-parallel-size N`: Run N independent copies (for throughput)

**Rule of thumb**:
- Small models (< 7B): Use DP only
- Medium models (7-20B): Use TP=2, DP=remaining
- Large models (20B+): Use TP=4 or TP=8

### Batch Size
- `--max-num-seqs N`: Maximum concurrent requests
  - Small: 32-64 (low latency)
  - Medium: 128-256 (balanced)
  - Large: 512-1024 (high throughput)

### Prefill Chunking
- `--max-num-batched-tokens N`: Tokens per prefill chunk
  - Small: 2048 (default, good for short prompts)
  - Medium: 4096-8192 (faster prefill)
  - Large: 16384+ (for very long contexts)

### Memory
- `--gpu-memory-utilization X`: Fraction of GPU memory for KV cache
  - Conservative: 0.85 (safe, leaves headroom)
  - Balanced: 0.90 (default)
  - Aggressive: 0.95 (maximum KV cache)

### Performance
- Remove `--enforce-eager`: Enable CUDA graphs (1.5-2× faster decode)
- `--enable-chunked-prefill`: Enable for context > 32K tokens
- `--enable-prefix-caching`: Enable for shared prompt prefixes (default: auto)

---

## Example Workflows

### Workflow 1: Testing Different Configurations

```bash
# Start with default
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8000 qwen-default

# Benchmark
uv run vf-eval your-env -m qwen-default -n 100 -v

# Try optimized
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8001 qwen-optimized \
  "--data-parallel-size 4 --max-num-seqs 512"

# Benchmark again
uv run vf-eval your-env -m qwen-optimized -n 100 -v

# Compare results!
```

### Workflow 2: Gradually Increasing Batch Size

```bash
# Start conservative
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8000 test \
  "--data-parallel-size 4 --max-num-seqs 128"

# Monitor KV cache usage
tail -f pbs_results/vllm_*_port8000_realtime.log | grep "KV cache usage"

# If usage < 30%, increase batch size
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8001 test \
  "--data-parallel-size 4 --max-num-seqs 256"

# Keep increasing until KV cache is 60-80%
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8002 test \
  "--data-parallel-size 4 --max-num-seqs 512"
```

### Workflow 3: A/B Testing Parallelism Strategies

```bash
# Server A: Tensor Parallelism (default)
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8000 qwen-tp

# Server B: Data Parallelism (optimized)
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8001 qwen-dp \
  "--data-parallel-size 4 --max-num-seqs 512"

# Run same workload on both
time uv run vf-eval your-env -m qwen-tp -n 100
time uv run vf-eval your-env -m qwen-dp -n 100

# Compare throughput in logs
grep "generation throughput" pbs_results/vllm_*_port8000_realtime.log
grep "generation throughput" pbs_results/vllm_*_port8001_realtime.log
```

---

## Troubleshooting

### Problem: OOM (Out of Memory)

**Solutions** (try in order):
1. Reduce `--max-num-seqs` by 50%
2. Reduce `--gpu-memory-utilization` to 0.85
3. Reduce `--max-num-batched-tokens` to 2048
4. Add quantization: `--quantization awq`

### Problem: Lower throughput than expected

**Check**:
1. Is CUDA graphs enabled? (no `--enforce-eager`)
2. Is batch size high enough? (check KV cache usage)
3. Are you using data parallelism for small models?

### Problem: High latency per request

**Solutions**:
1. Reduce `--max-num-seqs` (less batching = lower latency)
2. Use tensor parallelism for faster single-request processing
3. Reduce `--max-num-batched-tokens` to 2048

---

## Performance Monitoring

### Key Metrics to Watch

```bash
# Watch real-time server stats
tail -f pbs_results/vllm_*_realtime.log | grep "Engine 000"
```

**Important metrics**:
- **Prompt throughput**: Prefill speed (tokens/s)
- **Generation throughput**: Decode speed (tokens/s) 
- **Running requests**: Current batch size
- **KV cache usage**: Memory utilization
  - < 30%: Increase batch size
  - 30-70%: Optimal
  - > 80%: Risk of OOM, decrease batch size

### Calculating Optimal Batch Size

```
Target KV cache usage: 60%
Current usage: 1.6%
Current batch size: 8

Optimal batch size ≈ 8 × (60 / 1.6) ≈ 300

→ Try --max-num-seqs 256 or 512
```

---

## Summary

**For 2-7B models**: Use data parallelism for 4× throughput
```bash
"--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
```

**For 7-20B models**: Use hybrid parallelism
```bash
"--tensor-parallel-size 2 --data-parallel-size 2 --max-num-seqs 256 --max-num-batched-tokens 8192"
```

**For 20B+ models**: Use tensor parallelism only
```bash
"--tensor-parallel-size 4 --max-num-seqs 128 --max-num-batched-tokens 8192"
```

**Always**: Remove `--enforce-eager` for production (enables CUDA graphs)

