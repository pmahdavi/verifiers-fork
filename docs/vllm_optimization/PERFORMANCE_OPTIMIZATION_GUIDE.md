# Performance Optimization Guide for vLLM Server

## Executive Summary

**Current Configuration Analysis:**
- ✅ **Strengths**: Good prefix caching (72%), chunked prefill enabled
- ⚠️ **Critical Issues**: Over-parallelized small model, CUDA graphs disabled, underutilized batch capacity
- 🎯 **Potential Speedup**: **3-5× faster inference** with optimized configuration

---

## Current Configuration (BASELINE)

```bash
# From your logs:
Model: Qwen/Qwen3-VL-2B-Instruct (2B params = ~4GB)
GPUs: 4 × CUDA (tensor parallel)
Configuration:
  --tensor-parallel-size 4
  --data-parallel-size 1
  --enforce-eager            ⚠️ MAJOR BOTTLENECK
  gpu_memory_utilization: 0.9
  max_num_batched_tokens: 2048
  max_num_seqs: (auto, likely ~256)

Performance Metrics:
  Prompt throughput: 78.5 tokens/s
  Generation throughput: 412.2 tokens/s
  Concurrent requests: 8
  KV cache usage: 1.6% ⚠️ MASSIVE UNDERUTILIZATION
```

---

## Critical Performance Issues

### 🔴 Issue #1: Extreme Over-Parallelization

**Problem**: Using 4 GPUs with tensor parallelism for a **2B parameter model** is like using a sledgehammer to crack a nut.

**Why this is bad**:
```
Model size: 2B params × 2 bytes (bf16) = 4GB
Per GPU with TP=4: 4GB ÷ 4 = 1GB per GPU

Each GPU is:
  - 95% idle
  - Wasting time on inter-GPU communication (NCCL)
  - Adding latency for every single token
```

**Evidence from logs**:
```
Line 45-48: Custom allreduce is disabled because it's not supported 
            on more than two PCIe-only GPUs
```
This means inter-GPU communication is going over **slow PCIe**, not fast NVLink!

**Performance Impact**: 
- Every attention layer requires 4-way communication
- For 24 layers × 8 attention heads = **192 communications per forward pass**
- Each communication adds ~0.5-2ms latency over PCIe
- Total overhead: **96-384ms per forward pass** (!)

### 🔴 Issue #2: CUDA Graphs Disabled

**Problem**: `--enforce-eager` disables CUDA graph optimization.

**Why this is bad**:
```
Eager Mode (current):
  [Python] → [Launch Kernel 1] → [Python] → [Launch Kernel 2] → ...
  Python overhead: ~50-100μs per kernel
  For decode (30 kernels/token): 1.5-3ms overhead per token

CUDA Graphs (optimized):
  [Python] → [Execute Entire Graph] 
  Python overhead: ~50-100μs TOTAL
  For decode: ~0.05-0.1ms overhead per token
```

**Performance Impact**: 
- Decode throughput reduced by **30-50%**
- Your 412 tokens/s could be **600-800+ tokens/s**

**From logs (line 8)**:
```
INFO: Cudagraph is disabled under eager mode
```

### 🔴 Issue #3: Severely Underutilized Batch Capacity

**Problem**: KV cache usage at 1.6% means you're wasting 98.4% of available memory.

**Current**:
```
8 concurrent requests
1.6% KV cache usage
412 tokens/s total throughput
51.5 tokens/s per request
```

**Optimized** (with higher batch size):
```
80 concurrent requests (10× more)
16% KV cache usage (still plenty of room)
3000+ tokens/s total throughput
37.5 tokens/s per request (slight decrease per request, but 10× more requests!)
```

---

## Recommended Optimizations (Priority Order)

### 🥇 Priority 1: Fix Parallelization Strategy

#### Option A: Pure Data Parallelism (RECOMMENDED for 2B model)

**Command**:
```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8004 qwen3-vl-optimized
```

**Then edit `scripts/start_vllm.sh` to:**
```bash
# Line 132-139, replace with:
stdbuf -o0 -e0 vf-vllm \
    --model "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --host "$HOST" \
    --port "$PORT" \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.95 \
    > "$REALTIME_LOG" 2> "$REALTIME_ERR"
```

**Benefits**:
- **4× throughput**: Each GPU serves independent requests
- **Zero communication overhead**: No inter-GPU NCCL calls during inference
- **Perfect for small models**: 2B model fits entirely on each GPU

**Expected Performance**:
```
Prompt throughput: 150+ tokens/s (was 78.5)
Generation throughput: 800+ tokens/s (was 412)
Speedup: ~2× per-request, 4× total throughput
```

#### Option B: Hybrid Parallelism (for very high batch sizes)

**Command**:
```bash
stdbuf -o0 -e0 vf-vllm \
    --model "$MODEL_NAME" \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --host "$HOST" \
    --port "$PORT" \
    --max-num-seqs 1024 \
    --gpu-memory-utilization 0.95 \
    > "$REALTIME_LOG" 2> "$REALTIME_ERR"
```

**Benefits**:
- **2× throughput** from data parallelism
- Slightly more memory per DP replica (TP=2 means 2GB per replica)
- Can handle even larger batches

**When to use**: If you need to batch 200+ concurrent requests

### 🥈 Priority 2: Enable CUDA Graphs

**Change**: Remove `--enforce-eager` flag

**Updated command**:
```bash
stdbuf -o0 -e0 vf-vllm \
    --model "$MODEL_NAME" \
    --data-parallel-size 4 \
    --host "$HOST" \
    --port "$PORT" \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.95 \
    > "$REALTIME_LOG" 2> "$REALTIME_ERR"
```

**Benefits**:
- **1.5-2× faster decode** (generation phase)
- Lower latency per token
- Better GPU utilization

**Why it's safe**:
- CUDA graphs are stable in vLLM v0.11.0
- Only needed `--enforce-eager` for debugging or very specific edge cases
- Your model (Qwen3-VL-2B) is well-supported

### 🥉 Priority 3: Increase Batch Size

**Change**: Explicitly set higher `--max-num-seqs`

**Command**:
```bash
--max-num-seqs 512   # or 1024 for even higher throughput
```

**Benefits**:
- Utilize that 98% unused KV cache!
- Handle 50-100× more concurrent requests
- Massively increase total throughput

**How to determine optimal value**:
```bash
# Start conservative:
--max-num-seqs 256

# Monitor KV cache usage, if < 30%, increase:
--max-num-seqs 512

# Keep increasing until KV cache reaches 70-80%:
--max-num-seqs 1024
```

### 🏅 Priority 4: Optimize Memory Settings

**Change**: Increase GPU memory utilization for more KV cache

**Command**:
```bash
--gpu-memory-utilization 0.95   # was 0.9
```

**Benefits**:
- 5% more memory for KV cache
- Can batch more requests
- Still leaves safety margin for memory spikes

### 🎖️ Priority 5: Optimize Chunked Prefill

**Change**: Increase chunk size for faster prefill

**Command**:
```bash
--max-num-batched-tokens 8192   # was 2048
```

**Benefits**:
- Fewer prefill iterations for long prompts
- Faster time-to-first-token (TTFT)
- Better GPU utilization during prefill

**Trade-off**: Slightly less flexible batching (more memory per chunk)

---

## Complete Optimized Configuration

### RECOMMENDED: Update `scripts/start_vllm.sh`

Replace lines 132-139 with:

```bash
# Use stdbuf to disable buffering and redirect to log files
stdbuf -o0 -e0 vf-vllm \
    --model "$MODEL_NAME" \
    --data-parallel-size 4 \
    --host "$HOST" \
    --port "$PORT" \
    --max-num-seqs 512 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    > "$REALTIME_LOG" 2> "$REALTIME_ERR"
```

**Changes**:
- ✅ Removed `--tensor-parallel-size 4` (defaults to 1)
- ✅ Added `--data-parallel-size 4` (4× throughput!)
- ✅ Removed `--enforce-eager` (enable CUDA graphs)
- ✅ Added `--max-num-seqs 512` (64× more batching)
- ✅ Added `--max-num-batched-tokens 8192` (4× faster prefill)
- ✅ Added `--gpu-memory-utilization 0.95` (more KV cache)
- ✅ Removed `--disable-log-requests` (was deprecated anyway)

---

## Expected Performance Improvements

### Baseline (Current):
```
Configuration:
  - TP=4, DP=1, eager mode
  - max_num_seqs: ~256 (auto)
  - 8 concurrent requests

Metrics:
  - Prompt throughput: 78.5 tokens/s
  - Generation throughput: 412 tokens/s
  - Per-request latency: ~51.5 tokens/s
  - KV cache usage: 1.6%
```

### Optimized (Recommended):
```
Configuration:
  - TP=1, DP=4, CUDA graphs
  - max_num_seqs: 512
  - 80+ concurrent requests

Expected Metrics:
  - Prompt throughput: 200-250 tokens/s (3.2× faster)
  - Generation throughput: 1600-2000 tokens/s (4-5× faster)
  - Per-request latency: ~40-60 tokens/s (similar or better)
  - KV cache usage: 15-30% (still room for more!)
```

### Performance Breakdown:

| Optimization | Speedup | Why |
|-------------|---------|-----|
| **Data Parallelism** (TP=4→DP=4) | **4×** | Each GPU serves independent requests, zero communication overhead |
| **CUDA Graphs** (remove eager) | **1.5-2×** | Eliminates Python overhead, faster kernel launches |
| **Higher Batch Size** (8→80 reqs) | **10× throughput** | Amortizes fixed costs across more requests |
| **Chunked Prefill Tuning** | **1.3-1.5×** | Fewer iterations for long prompts |

**Combined Expected Speedup**: **4× to 5× faster total throughput**

---

## Testing & Validation

### Step 1: Benchmark Current Configuration

```bash
# Run baseline benchmark
time uv run vf-eval your-environment -m local-vllm -n 100 -r 1 -v

# Note the timing:
# Example output: "Completed in 120.5 seconds"
```

### Step 2: Apply Optimizations

```bash
# Stop current server
qdel <job_id>

# Update scripts/start_vllm.sh with optimized config (see above)

# Launch optimized server
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8005 qwen3-vl-optimized
```

### Step 3: Benchmark Optimized Configuration

```bash
# Wait for server to be ready (~2 min)
# Check with: curl http://0.0.0.0:8005/v1/models

# Update endpoint to port 8005
python scripts/update_endpoint.py Qwen/Qwen3-VL-2B-Instruct 8005 qwen3-vl-optimized

# Run same benchmark
time uv run vf-eval your-environment -m qwen3-vl-optimized -n 100 -r 1 -v

# Compare timing:
# Expected: ~30-40 seconds (3-4× faster)
```

### Step 4: Monitor Server Metrics

```bash
# Watch server logs for performance stats
tail -f pbs_results/vllm_Qwen_Qwen3-VL-2B-Instruct_port8005_realtime.log | grep "Engine 000"

# Look for:
# - Higher generation throughput (should be 1600-2000 tokens/s)
# - More concurrent requests (should handle 50-100+)
# - Higher KV cache usage (should be 15-30%)
```

---

## Advanced Optimizations (Optional)

### Option 1: Async Scheduling (Experimental)

```bash
--async-scheduling
```

**Benefits**: Overlaps scheduling with computation
**Risk**: Experimental feature, may have bugs

### Option 2: Speculative Decoding (for even faster generation)

```bash
--speculative-model <smaller-draft-model>
--num-speculative-tokens 5
```

**Benefits**: 2-3× faster generation for suitable tasks
**Requirement**: Need a compatible draft model

### Option 3: Quantization (if memory becomes an issue)

```bash
--quantization awq    # or fp8 for newer GPUs
```

**Benefits**: 2× less memory, can fit larger batches
**Trade-off**: Slight quality degradation (~1-2%)

---

## Common Pitfalls to Avoid

### ❌ Don't: Use tensor parallelism for small models

**Bad**:
```bash
--tensor-parallel-size 4   # for 2B model
```

**Good**:
```bash
--data-parallel-size 4     # for 2B model
```

**Rule of thumb**:
- Model < 7B params: Use DP (data parallelism)
- Model 7B-20B: Use TP=2, DP=remaining GPUs
- Model 20B-70B: Use TP=4, DP=remaining GPUs
- Model > 70B: Use TP=8 (requires NVLink!)

### ❌ Don't: Enable eager mode unless debugging

**Bad**:
```bash
--enforce-eager
```

**Good**: Just remove it, CUDA graphs are stable

### ❌ Don't: Leave batch size at default

**Bad**: Not specifying `--max-num-seqs`

**Good**:
```bash
--max-num-seqs 512   # or higher based on workload
```

### ❌ Don't: Forget to monitor KV cache usage

**Bad**: Setting high batch size without checking logs

**Good**: Monitor and adjust:
```bash
tail -f logs | grep "KV cache usage"
# Aim for 50-70% usage for optimal throughput
```

---

## Summary: Quick Start Optimized Config

### For Immediate 4-5× Speedup:

1. **Edit `scripts/start_vllm.sh` lines 132-139:**

```bash
stdbuf -o0 -e0 vf-vllm \
    --model "$MODEL_NAME" \
    --data-parallel-size 4 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --host "$HOST" \
    --port "$PORT" \
    > "$REALTIME_LOG" 2> "$REALTIME_ERR"
```

2. **Launch optimized server:**

```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8005 qwen3-vl-optimized
```

3. **Monitor and validate:**

```bash
# Check it started correctly
tail -f pbs_results/vllm_Qwen_Qwen3-VL-2B-Instruct_port8005_realtime.log

# Look for:
# - "data_parallel_size=4" in startup logs
# - Higher throughput in Engine metrics
# - No errors about NCCL/communication
```

4. **Run benchmarks and compare!**

---

## Troubleshooting

### Issue: OOM (Out of Memory) errors

**Solution**: Reduce batch size or memory utilization
```bash
--max-num-seqs 256
--gpu-memory-utilization 0.85
```

### Issue: "NCCL initialization failed"

**Solution**: Check GPU visibility and clean up
```bash
# In PBS script, ensure correct GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Clean up stale processes
pkill -9 -u $USER vllm
```

### Issue: Slower than expected

**Diagnosis**:
```bash
# Check if CUDA graphs actually enabled
grep "Cudagraph" pbs_results/vllm_*_realtime.log

# Check data parallelism configured
grep "data_parallel_size=4" pbs_results/vllm_*_realtime.log

# Check actual throughput
grep "Engine 000.*throughput" pbs_results/vllm_*_realtime.log
```

---

## Next Steps

1. ✅ Apply the recommended optimized configuration
2. ✅ Benchmark and compare performance
3. ✅ Monitor KV cache usage and adjust `--max-num-seqs` 
4. ✅ Iterate on chunk size based on your prompt lengths
5. ✅ Consider quantization if you need even higher throughput

**Questions?** Check the logs and compare metrics before/after optimization!

