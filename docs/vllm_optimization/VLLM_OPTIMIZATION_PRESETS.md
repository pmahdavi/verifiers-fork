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

### 🖼️ Vision-Language Models (VL Models)
**When**: Using multimodal models like Qwen3-VL, LLaVA, etc.

```bash
# For 2B VL models (higher memory than text-only!)
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Thinking 8011 qwen3-vl \
  "--data-parallel-size 4 --max-num-seqs 32 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.95 --max-model-len 40960"
```

**Important notes**:
- **VL models use significantly more KV cache** than text-only models (observed: ~4× difference)
- **Always check "Maximum concurrency" in logs** - don't rely on text-only estimates
- Vision encoders add substantial memory overhead (~4.4GB per request vs ~1-1.5GB for text-only)
- **Reduce `max_num_seqs` significantly** compared to text-only models

**Observed example (Qwen3-VL-2B-Thinking)**:
- **Available KV cache**: 37.15 GB per GPU
- **Maximum concurrency**: 8.49 requests per GPU @ 40K context
- **Memory per request**: ~4.4 GB (vs ~1-1.5 GB for text-only 2B models)
- **Total capacity**: 8.49 × 4 GPUs = ~34 concurrent requests (use `max_num_seqs=32`)

### 🧠 Reasoning Models (Thinking/CoT Models)
**When**: Using models like DeepSeek-R1, Qwen-Thinking, etc.

```bash
# For Qwen3-Thinking models
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Thinking 8011 qwen3-thinking \
  "--data-parallel-size 4 --max-num-seqs 32 --reasoning-parser qwen3 --max-model-len 40960"
```

**Important notes**:
- **Always add `--reasoning-parser <format>`** to parse native reasoning format
  - `qwen3` for Qwen Thinking models
  - `deepseek-r1` for DeepSeek R1 models
- **Do NOT use `<think>` XML tags** in prompts - vLLM handles native format
- **Generate long sequences** - reasoning models often need 10K-50K tokens
- **Set higher `max_model_len`** to accommodate reasoning + answer

**Client-side usage**:
```bash
# Use use_think: false because vLLM handles reasoning parsing
uv run vf-eval your-env -m qwen3-thinking -t 32768 -a '{"use_think": false}'
```

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

## 🔍 How to Choose max_num_seqs (CRITICAL!)

**⚠️ Don't guess - use vLLM's reported capacity!**

### The Right Way:

1. **Start your server with a conservative `max_num_seqs`** (e.g., 32)
2. **Check the startup logs** for this line:
   ```
   Maximum concurrency for 40,960 tokens per request: 8.49x
   ```
3. **Calculate your actual capacity**:
   ```
   Per-GPU capacity = 8.49 (from logs)
   Total capacity = 8.49 × num_gpus (e.g., 8.49 × 4 = ~34)
   ```
4. **Set `max_num_seqs` to match** (or slightly below) this capacity

### Real Example:

```bash
# Start with conservative setting
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Thinking 8011 test \
  "--data-parallel-size 4 --max-num-seqs 32 --max-model-len 40960"

# Check logs:
# INFO: Maximum concurrency for 40,960 tokens per request: 8.49x
# → 8.49 × 4 GPUs = ~34 total capacity

# ✅ Your setting (32) matches capacity → GOOD!
# ❌ If you set 96, you'd be 3× over capacity → BAD!
```

### Why Theoretical Calculations Fail:

**Common mistake**: Estimating KV cache per request
```python
# ❌ WRONG - theoretical calculation
model_size_gb = 4
available_memory = 48 - 4 = 44
kv_per_request_estimate = 1.4  # Guess based on model size
capacity_estimate = 44 / 1.4 = ~31 requests  # WRONG!

# ✅ RIGHT - actual from vLLM
# Maximum concurrency: 8.49x per GPU
# → Only ~8 requests per GPU, not 31!
```

**Why the difference?**
- Vision encoders use extra memory
- Activation memory not accounted for
- CUDA graph memory overhead
- Safety margins in vLLM's calculation

**🎯 Trust vLLM's "Maximum concurrency" metric, not manual calculations!**

---

## ⚠️ Common Misconceptions

Before diving into configurations, here are critical misconceptions that can lead to poor performance:

### ❌ Myth: "Throughput should remain constant during generation"

**Reality**: Throughput **decreases significantly** during long generation (20K+ tokens).

**Why**: O(n²) attention complexity means each new token takes progressively longer.

**Example**:
- Start: 800 tok/s (generating tokens 0-5K)
- Middle: 650 tok/s (generating tokens 10-20K)
- End: 500 tok/s (generating tokens 25-32K)
- **~40% degradation is NORMAL!**

**What to do**: Factor this into time estimates. See [LONG_CONTEXT_GENERATION_GUIDE.md](LONG_CONTEXT_GENERATION_GUIDE.md) for details.

### ❌ Myth: "VL models have same KV cache needs as text-only models"

**Reality**: Vision-Language models use **3-4× more KV cache** per request!

**Example** (2B parameter models):
- Text-only Qwen3-2B: ~1.5 GB per request @ 40K context
- VL Qwen3-VL-2B: ~4.4 GB per request @ 40K context
- **3× difference!**

**Impact**: Can only batch 1/3 as many VL requests compared to text-only.

**What to do**: Always check "Maximum concurrency" in logs, don't estimate!

### ❌ Myth: "I can calculate max_num_seqs myself"

**Reality**: Manual calculations **fail badly** for VL models, long contexts, or with CUDA graphs.

**Why manual calculations fail**:
```python
# ❌ WRONG approach
model_size = 4 GB
available_memory = 48 - 4 = 44 GB
estimated_kv_per_request = 1.5 GB  # Rough estimate
capacity = 44 / 1.5 = ~29 requests  # WRONG!

# ✅ RIGHT approach from vLLM logs
# "Maximum concurrency: 5.63x per GPU"
actual_capacity = 5.63 × 4 GPUs = ~22 requests  # Correct!
```

**What to do**: **Always trust vLLM's "Maximum concurrency" metric!**

### ❌ Myth: "More GPUs = always faster"

**Reality**: For small models (< 7B), more GPUs with tensor parallelism = **slower**!

**Why**:
- TP=4 for 2B model: Each GPU holds 0.5B params, 95% idle, constant communication
- DP=4 for 2B model: Each GPU holds 2B params, 100% utilized, zero communication

**What to do**: Use data parallelism for models < 7B parameters.

### ❌ Myth: "Setting max_model_len higher = better"

**Reality**: Over-allocating context length **wastes capacity** and hurts performance.

**Example**:
```bash
# ❌ BAD: Model supports 256K but you only need 40K
--max-model-len 262144
# Result: vLLM reserves KV cache for 256K, fewer concurrent requests

# ✅ GOOD: Set to actual need
--max-model-len 40960  # 4K prompt + 36K generation
# Result: Optimal KV cache allocation, more concurrent requests
```

**What to do**: Set `max_model_len` to realistic maximum, not model's theoretical limit.

### ❌ Myth: "reasoning-parser and use_think: true work together"

**Reality**: These are **mutually exclusive** - using both causes parsing to fail!

**Correct usage**:
```bash
# vLLM server (handles native reasoning format)
--reasoning-parser qwen3

# vf-eval client (don't double-parse!)
-a '{"use_think": false}'  # Let vLLM handle it
```

**What to do**: If using `--reasoning-parser` on server, set `use_think: false` in client.

See [LONG_CONTEXT_GENERATION_GUIDE.md](LONG_CONTEXT_GENERATION_GUIDE.md) for more details on these topics.

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

### Problem: Server starts but requests are slow/queued

**Diagnosis**: You set `max_num_seqs` too high for your actual KV cache capacity

**Solution**:
```bash
# 1. Check your logs for this line during startup:
grep "Maximum concurrency" pbs_results/vllm_*_realtime.log
# Output: Maximum concurrency for 40,960 tokens per request: 8.49x

# 2. Calculate correct max_num_seqs:
# Per-GPU capacity × num_gpus = total capacity
# 8.49 × 4 = ~34 (use max_num_seqs=32)

# 3. Restart with correct setting
./scripts/launch_vllm.sh YourModel 8011 fixed \
  "--data-parallel-size 4 --max-num-seqs 32"
```

**Why this happens**:
- Setting `max_num_seqs=96` when capacity is only 34 causes 3× overcapacity
- Requests queue up waiting for KV cache space
- Throughput is throttled by memory, not compute

### Problem: Lower throughput than expected

**Check**:
1. Is CUDA graphs enabled? (no `--enforce-eager`)
2. Is batch size high enough? (check KV cache usage)
3. Are you using data parallelism for small models?
4. **Did you check "Maximum concurrency" in logs?** ← Most common issue!

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

