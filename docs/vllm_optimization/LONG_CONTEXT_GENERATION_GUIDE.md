# Long-Context Generation Performance Guide

## Overview

When generating very long sequences (10K-50K tokens), you'll observe **performance characteristics that differ significantly** from short-generation scenarios. This guide explains what to expect, why it happens, and how to optimize for long-context workloads.

---

## Expected Performance Behavior

### Normal Throughput Degradation

**Key Insight**: Throughput **decreases over time** during long generation. This is **NORMAL and EXPECTED**, not a bug!

#### Real-World Example

From a 4B VL model generating 32K tokens:

| Time Elapsed | Throughput/Engine | Total Throughput | KV Cache | Sequence Length |
|--------------|-------------------|------------------|----------|-----------------|
| **Start** (0-2 min) | ~350-450 tok/s | ~1,600 tok/s | 2-5% | 0-4K tokens |
| **Mid** (5-10 min) | ~250-300 tok/s | ~1,000 tok/s | 15-30% | 8-16K tokens |
| **Late** (15-20 min) | ~160-170 tok/s | ~650 tok/s | 50-60% | 24-32K tokens |

**Degradation**: From 1,600 tok/s → 650 tok/s (**~60% slower** at the end)

### Why This Happens

#### 1. **Quadratic Attention Complexity**
```python
# Attention computation grows with sequence length
Attention Cost = O(seq_len²)

At 1K tokens:  1,000² = 1,000,000 operations
At 32K tokens: 32,000² = 1,024,000,000 operations  # 1024× more!
```

#### 2. **Growing KV Cache Access**
```
Each new token must attend to ALL previous tokens:
- Token 1: Attends to 0 previous tokens (instant)
- Token 1,000: Attends to 999 previous tokens
- Token 32,000: Attends to 31,999 previous tokens
```

**Memory bandwidth becomes the bottleneck** - reading 32K KV pairs from GPU memory takes time!

#### 3. **KV Cache Memory Pressure**
```
Early generation: 5% KV cache → GPU has plenty of bandwidth
Late generation: 60% KV cache → More memory traffic, slower access
```

---

## Performance Expectations by Generation Length

### Short Generation (< 1K tokens)
```
✅ Stable throughput throughout generation
✅ Minimal KV cache growth
✅ Consistent per-token latency
✅ High batch capacity

Throughput: ~1,500-2,000 tok/s (stable)
Time estimate: ~1 second for 1K tokens
```

### Medium Generation (1K-10K tokens)
```
⚠️ Slight throughput degradation (10-20%)
⚠️ Moderate KV cache growth
⚠️ Per-token latency increases slightly

Throughput: Starts ~1,500 tok/s, ends ~1,200 tok/s
Time estimate: ~7-10 seconds for 10K tokens
```

### Long Generation (10K-30K tokens)
```
⚠️ Significant throughput degradation (40-60%)
⚠️ Substantial KV cache growth
⚠️ Per-token latency increases noticeably
⚠️ Reduced concurrent request capacity

Throughput: Starts ~1,200 tok/s, ends ~600-700 tok/s
Time estimate: ~40-50 seconds for 30K tokens
```

### Very Long Generation (30K+ tokens)
```
⚠️ Severe throughput degradation (60-80%)
⚠️ KV cache near capacity
⚠️ Risk of OOM if batching too many requests
⚠️ May need to reduce max_num_seqs

Throughput: Can drop to 300-500 tok/s
Time estimate: ~2-3 minutes for 50K tokens
```

---

## Optimization Strategies

### 1. Set Realistic `max_model_len`

**Don't over-allocate context**:

```bash
# ❌ BAD: Model supports 256K but you only need 40K
--max-model-len 262144
# Result: Wastes KV cache capacity, allows over-capacity requests

# ✅ GOOD: Set to actual need + safety margin
--max-model-len 40960  # For 4K prompt + 32K generation
# Result: vLLM can accurately calculate capacity
```

**Why this matters**:
- vLLM calculates "Maximum concurrency" based on `max_model_len`
- Setting it too high → incorrect capacity estimates → performance issues

### 2. Adjust Batch Size for Long Contexts

**Different `max_num_seqs` for different context lengths**:

```bash
# For short contexts (< 4K total)
--max-num-seqs 512  # Can batch many requests

# For medium contexts (4K-20K total)
--max-num-seqs 128  # Moderate batching

# For long contexts (20K-50K total)
--max-num-seqs 32   # Conservative batching
```

**Validation**: Check "Maximum concurrency" in startup logs!

```
INFO: Maximum concurrency for 40,960 tokens per request: 5.63x
→ Per-GPU capacity: 5.63 requests
→ Total capacity (4 GPUs): 5.63 × 4 = ~22 requests
→ Set max_num_seqs=24 or lower ✅
```

### 3. Monitor Throughput Trends

**Track throughput during evaluation**:

```bash
# Watch generation throughput over time
tail -f pbs_results/vllm_*_realtime.log | grep "Avg generation throughput"

# Look for:
# - Initial throughput (first 2-3 minutes)
# - Steady-state throughput (after 10+ minutes)
# - Degradation rate
```

**Healthy patterns**:
- ✅ Gradual decrease (linear or log-linear)
- ✅ Stabilizes after initial drop
- ✅ No sudden crashes or spikes

**Problem patterns**:
- ❌ Sudden drops to near-zero (OOM or queue buildup)
- ❌ Erratic fluctuations (over-capacity)
- ❌ Throughput never recovers (memory leak?)

### 4. Use Chunked Prefill for Long Prompts

**For prompts > 4K tokens**:

```bash
# Enable chunked prefill with appropriate chunk size
--enable-chunked-prefill \
--max-num-batched-tokens 16384  # Larger chunks for long contexts
```

**Why**: Larger chunks = fewer iterations = faster prefill

**Trade-off**: More memory per chunk, but worth it for long prompts

### 5. Consider Model Size vs. Context Length Trade-off

**For very long contexts, smaller models may be faster**:

| Model | Params | Context | Throughput @ 32K | Total Time |
|-------|--------|---------|------------------|------------|
| **2B** | 2B | 40K | ~800 tok/s | **~40 sec** ✅ |
| **4B** | 4B | 40K | ~650 tok/s | **~50 sec** |
| **8B** | 8B | 40K | ~475 tok/s | **~67 sec** |

**Insight**: 2B model is **68% faster** than 8B for 32K generation despite lower quality per-token.

**Decision**: Choose based on:
- Quality needs (8B > 4B > 2B)
- Latency needs (2B > 4B > 8B)
- Throughput needs (2B > 4B > 8B for long contexts)

---

## Common Misconceptions

### ❌ Myth 1: "Throughput should be constant"

**Reality**: Throughput **must** decrease as sequence length grows due to O(n²) attention.

**What to expect**: 40-60% degradation for 30K+ token generation is normal.

### ❌ Myth 2: "More GPUs = always faster"

**Reality**: For long contexts, memory bandwidth matters more than compute.

**Example**:
- 4B model on 1 GPU: May be faster than
- 4B model on 4 GPUs (TP=4): Due to inter-GPU communication overhead

### ❌ Myth 3: "VL models and text-only models have same KV cache needs"

**Reality**: VL models use **3-4× more KV cache** per request!

**Example** (Qwen3-VL-2B vs Qwen3-2B):
- Text-only 2B: ~1.5 GB KV cache per request @ 40K
- VL 2B: ~4.4 GB KV cache per request @ 40K

**Impact**: 3× fewer concurrent requests for VL models!

### ❌ Myth 4: "I can set max_num_seqs based on calculation"

**Reality**: **Always trust vLLM's "Maximum concurrency" metric**, not manual estimates.

**Why manual calculations fail**:
- Vision encoders add hidden memory overhead
- Activation memory not accounted for
- CUDA graph memory
- Safety margins

---

## Best Practices for Long-Context Workloads

### 1. Test with Representative Workload

```bash
# Don't optimize for average case - test worst case!

# ❌ BAD: Test with 5K token generation
uv run vf-eval env -m model -t 5000 -n 10

# ✅ GOOD: Test with maximum expected generation
uv run vf-eval env -m model -t 32768 -n 10

# Then optimize based on actual observed throughput
```

### 2. Start Conservative, Then Scale Up

```bash
# Step 1: Conservative configuration
./scripts/launch_vllm.sh MODEL PORT NAME \
  "--data-parallel-size 4 --max-num-seqs 16 --max-model-len 40960"

# Step 2: Monitor during actual workload
tail -f pbs_results/vllm_*_realtime.log | grep "KV cache usage"

# Step 3: If KV cache < 50%, increase batch size
./scripts/launch_vllm.sh MODEL PORT NAME \
  "--data-parallel-size 4 --max-num-seqs 24 --max-model-len 40960"
```

### 3. Plan Time Estimates Realistically

**For 908 rollouts with 32K generation each**:

```python
# Calculation:
concurrent_capacity = 24  # From "Maximum concurrency" logs
avg_throughput = 650  # tokens/s (accounting for degradation)
tokens_per_rollout = 32_000

time_per_rollout = tokens_per_rollout / avg_throughput
# = 32,000 / 650 ≈ 49 seconds

num_batches = 908 / concurrent_capacity
# = 908 / 24 ≈ 38 batches

total_time = num_batches * time_per_rollout
# = 38 × 49 ≈ 1,862 seconds ≈ 31 minutes

# Add 20% overhead for prefill, queueing:
estimated_time = 31 min × 1.2 ≈ 37 minutes
```

### 4. Document Your Observations

**Keep a performance log**:

```bash
# After each major evaluation, record:
echo "Model: Qwen3-VL-4B" >> performance_log.txt
echo "Config: DP=4, max_seqs=24, context=40K" >> performance_log.txt
echo "Initial throughput: 800 tok/s" >> performance_log.txt
echo "Steady-state: 650 tok/s" >> performance_log.txt
echo "Degradation: 18.75%" >> performance_log.txt
echo "Total time: 35 minutes for 908 rollouts" >> performance_log.txt
echo "---" >> performance_log.txt
```

**Use this to**:
- Compare configurations
- Validate estimates for future runs
- Identify performance regressions

---

## Troubleshooting Long-Context Issues

### Issue: Throughput drops to near-zero after 10-15 minutes

**Symptoms**:
```
Initial: 800 tok/s → After 15 min: 50 tok/s or lower
```

**Likely causes**:
1. **OOM (Out of Memory)**: KV cache exceeded capacity
2. **Over-capacity batching**: Set `max_num_seqs` too high
3. **Memory leak**: Rare, but possible

**Solutions**:
```bash
# 1. Check for OOM errors
grep "OutOfMemoryError\|CUDA out of memory" pbs_results/vllm_*_realtime.err

# 2. Check "Maximum concurrency" vs your setting
grep "Maximum concurrency" pbs_results/vllm_*_realtime.log
# If your max_num_seqs > (concurrency × num_gpus), reduce it!

# 3. Restart with lower batch size
--max-num-seqs 16  # Reduce by 50%
```

### Issue: Individual requests taking much longer than expected

**Symptoms**:
```
Expected: 32K tokens in ~50 seconds
Actual: 32K tokens in 3-5 minutes
```

**Likely causes**:
1. **Queue buildup**: Too many concurrent requests
2. **CPU bottleneck**: Insufficient CPU cores
3. **Network bottleneck**: Client-server communication

**Solutions**:
```bash
# 1. Check waiting requests
tail -f pbs_results/vllm_*_realtime.log | grep "Waiting"
# If "Waiting: 10+" frequently → reduce max_num_seqs

# 2. Check CPU utilization
top -u $USER
# If CPUs at 100% → request more CPUs in PBS script

# 3. Check if streaming
# Disable streaming for batch workloads:
sampling_args={"stream": False}
```

### Issue: KV cache usage reaches 100%

**Symptoms**:
```
GPU KV cache usage: 98%, 99%, 100%
Requests queueing indefinitely
```

**Immediate fix**:
```bash
# 1. Stop new requests
# 2. Wait for current batch to complete
# 3. Restart with lower max_num_seqs

--max-num-seqs 16  # Reduce by 50%
```

**Long-term fix**:
```bash
# Calculate correct batch size from logs:
# "Maximum concurrency: X.XX" → use (X × num_gpus × 0.9)

# Example: Max concurrency 5.63 × 4 GPUs = 22.5
--max-num-seqs 20  # Use 90% of capacity for safety
```

---

## Summary: Quick Reference

### For Long-Context Generation (20K-50K tokens):

✅ **DO**:
- Set `max_model_len` to actual need, not model maximum
- Use vLLM's "Maximum concurrency" metric for batch size
- Expect 40-60% throughput degradation
- Monitor throughput trends during generation
- Start with conservative `max_num_seqs`
- Account for degradation in time estimates

❌ **DON'T**:
- Assume constant throughput
- Set `max_num_seqs` based on manual calculations
- Ignore "Maximum concurrency" warnings
- Batch as many requests as for short contexts
- Expect same performance as short generation
- Over-allocate context length unnecessarily

### Performance Targets:

| Context Length | Expected Degradation | Typical Throughput |
|----------------|---------------------|-------------------|
| **< 4K** | 0-10% | 1,500-2,000 tok/s |
| **4K-10K** | 10-20% | 1,200-1,500 tok/s |
| **10K-20K** | 20-40% | 800-1,200 tok/s |
| **20K-30K** | 40-60% | 600-800 tok/s |
| **30K-50K** | 60-80% | 400-600 tok/s |

*(For 4B model with data parallelism on 4× A6000 GPUs)*

---

## Related Resources

- [VLLM_OPTIMIZATION_PRESETS.md](VLLM_OPTIMIZATION_PRESETS.md) - Optimized configurations
- [CHUNKED_PREFILL_TUTORIAL.md](CHUNKED_PREFILL_TUTORIAL.md) - Understanding prefill behavior
- [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md) - General optimization guide

