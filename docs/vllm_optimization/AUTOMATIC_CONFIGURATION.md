# vLLM Automatic Configuration: What's Available

## Summary

**Short Answer**: vLLM does **NOT** have automatic detection of optimal parallelism strategies (tensor-parallel-size vs data-parallel-size). You must manually specify these based on your model size and use case.

**However**, vLLM provides:
- Sensible defaults for other parameters
- Auto-detection of some model properties
- Tools like GuideLLM for benchmarking configurations

---

## Our Default Setup (Still Available!)

Yes, we still have the default setup! When you don't provide custom flags:

```bash
# This uses the default configuration
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8000 default
```

**Default behavior** (from `scripts/start_vllm.sh`):
```bash
# If no VLLM_FLAGS provided:
--tensor-parallel-size 4 --enforce-eager
```

This default is **conservative and compatible** but not optimized for small models.

---

## What vLLM Auto-Detects

### ✅ Automatically Detected by vLLM:

1. **Model Context Length** (`max_model_len`)
   - Derived from model config if not specified
   - Example: Qwen3-VL-2B-Instruct → 262,144 tokens

2. **Data Type** (`dtype`)
   - Auto-selects based on model:
     - FP32/FP16 models → FP16
     - BF16 models → BF16

3. **Chunked Prefill**
   - Auto-enabled for models with context > 32K tokens
   - Sets default `max_num_batched_tokens=2048`

4. **CUDA Graphs**
   - Auto-enabled unless `--enforce-eager` is set
   - Automatically determines `max_seq_len_to_capture=8192`

5. **Quantization Support**
   - Auto-detects if model has pre-quantization (FP8, AWQ, etc.)

### ❌ NOT Automatically Detected:

1. **Tensor Parallel Size** - You must specify based on model size
2. **Data Parallel Size** - You must specify based on throughput needs
3. **Batch Size (`max_num_seqs`)** - Defaults to conservative value
4. **GPU Memory Utilization** - Defaults to 0.9 (90%)
5. **Optimal Chunk Size** - Defaults to 2048 tokens

---

## Why No Auto-Detection for Parallelism?

### The Challenge:

Choosing between tensor parallelism (TP) and data parallelism (DP) depends on:

1. **Model Size** - Does it fit on a single GPU?
2. **Request Volume** - Do you need high throughput?
3. **Hardware** - Do you have NVLink or PCIe?
4. **Latency Requirements** - Single-request speed vs total throughput?

**These are business/workload decisions, not technical constraints!**

### vLLM's Philosophy:

vLLM provides **powerful primitives** but expects users to understand their workload:
- Small model + high volume → Use DP
- Large model (must split) → Use TP
- Medium model + very high volume → Use TP + DP hybrid

**This is by design** - automatic selection would require profiling your specific workload, which vLLM doesn't do.

---

## Tools for Configuration Tuning

### GuideLLM (Benchmarking Tool)

vLLM community provides **GuideLLM** for evaluating different configurations:

```bash
# Install GuideLLM
pip install guidellm

# Benchmark different configurations
guidellm --model Qwen/Qwen3-VL-2B-Instruct \
         --backend vllm \
         --output results.json
```

**What GuideLLM does**:
- Runs benchmark workloads
- Measures throughput, latency, and resource usage
- Compares different configurations
- Generates performance reports

**What it doesn't do**:
- Automatically apply optimal configuration
- Tell you which parallelism strategy to use
- Configure your deployment

**GitHub**: https://github.com/vllm-project/guidellm

### Our Approach: Documented Best Practices

Instead of automatic detection, we provide:

1. **Decision Trees** ([VLLM_OPTIMIZATION_PRESETS.md](VLLM_OPTIMIZATION_PRESETS.md))
   ```
   Is your model < 7B parameters?
   ├─ YES: Use --data-parallel-size 4
   └─ NO: Use --tensor-parallel-size based on model size
   ```

2. **Ready-to-Use Presets**
   ```bash
   # Small models (< 7B) - High Throughput
   "--data-parallel-size 4 --max-num-seqs 512"
   
   # Medium models (7-20B) - Hybrid
   "--tensor-parallel-size 2 --data-parallel-size 2"
   
   # Large models (20B+) - Tensor Parallel
   "--tensor-parallel-size 4 --max-num-seqs 128"
   ```

3. **Monitoring Guidance**
   - Watch KV cache usage
   - Adjust batch size iteratively
   - Benchmark before/after changes

---

## What Other Systems Do

### Comparative Approaches:

1. **Ray Serve (with vLLM)**
   - Still requires manual parallelism specification
   - Auto-scales number of replicas (not parallelism strategy)

2. **TensorRT-LLM**
   - Requires explicit engine build with parallelism strategy
   - No runtime auto-detection

3. **DeepSpeed Inference**
   - Auto-infers TP size if model doesn't fit on single GPU
   - **But**: Only for memory constraints, not performance optimization

4. **SGLang (vLLM alternative)**
   - Similar to vLLM - manual configuration required
   - Provides different scheduling policies

**Conclusion**: Most LLM serving systems require manual parallelism configuration because the optimal choice depends on business requirements, not just technical constraints.

---

## Our Flexible System vs Auto-Detection

### Why Our Approach is Better:

1. **Explicit Over Implicit**
   ```bash
   # You know exactly what you're getting
   ./scripts/launch_vllm.sh MODEL PORT NAME \
     "--data-parallel-size 4 --max-num-seqs 512"
   ```

2. **Reproducible**
   - Configuration is in your command, not hidden heuristics
   - Easy to version control and document

3. **Debuggable**
   - When something goes wrong, you know what settings are active
   - No "magic" decisions to debug

4. **Educational**
   - Forces you to understand your workload
   - Makes trade-offs explicit

### When Auto-Detection Would Help:

1. **Beginner Users** - Don't know where to start
   - **Our solution**: Provide documented presets
   
2. **Rapidly Changing Workloads** - Need dynamic adjustment
   - **vLLM limitation**: Would require online profiling
   - **Workaround**: Use monitoring + manual adjustment

3. **Production Deployments** - Want optimal settings without manual tuning
   - **Our solution**: Benchmark with GuideLLM, then deploy

---

## Practical Workflow

### For New Users:

1. **Start with a preset**:
   ```bash
   # For small models (< 7B)
   ./scripts/launch_vllm.sh MODEL 8000 baseline \
     "--data-parallel-size 4 --max-num-seqs 512"
   ```

2. **Monitor performance**:
   ```bash
   tail -f pbs_results/vllm_*_realtime.log | grep "Engine 000"
   ```

3. **Check key metrics**:
   - KV cache usage: Should be 30-70%
   - Generation throughput: Higher is better
   - Running requests: Should be close to `max_num_seqs`

4. **Adjust iteratively**:
   - If KV cache < 30%: Increase `--max-num-seqs`
   - If OOM: Decrease `--max-num-seqs` or add `--quantization`
   - If throughput low: Check if using right parallelism strategy

### For Advanced Users:

1. **Benchmark with GuideLLM**:
   ```bash
   guidellm benchmark \
     --model MODEL \
     --backend vllm \
     --config config1.yaml \
     --config config2.yaml \
     --output comparison.json
   ```

2. **Compare configurations**:
   - TP=4 vs DP=4 for your specific workload
   - Different batch sizes
   - Different chunk sizes

3. **Deploy winner**:
   ```bash
   ./scripts/launch_vllm.sh MODEL PORT optimized \
     "<winning configuration>"
   ```

---

## Feature Requests / Future Work

### What Could Be Automated (Future):

1. **Model Size Detection → Parallelism Suggestion**
   ```bash
   # vLLM could warn:
   "Model size: 2B params (4GB). Consider --data-parallel-size 4 
    instead of --tensor-parallel-size 4 for better throughput"
   ```

2. **Runtime Profiling → Batch Size Adjustment**
   ```bash
   # vLLM could adjust dynamically:
   "KV cache usage: 5%. Increasing max_num_seqs from 64 to 128"
   ```

3. **Workload-Based Recommendations**
   ```bash
   # After warmup period:
   "Detected: Low batch utilization, high latency sensitivity.
    Recommendation: Use --tensor-parallel-size 4 for lower latency"
   ```

### Why These Don't Exist Yet:

1. **Complexity** - Runtime profiling adds overhead
2. **Risk** - Dynamic changes can cause OOM or instability
3. **Use Case Diversity** - Hard to build heuristics that work for everyone
4. **Engineering Priority** - Core performance work is higher priority

---

## Recommendations

### For Most Users (Small Models < 7B):

**Use our documented presets** - they're based on solid principles:

```bash
# High Throughput (recommended)
./scripts/launch_vllm.sh MODEL PORT optimized \
  "--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
```

**Why this works**:
- Based on model size → parallelism strategy mapping
- Tested on real workloads
- Documented and reproducible

### For Production Deployments:

1. **Benchmark first**: Use GuideLLM or manual benchmarks
2. **Monitor continuously**: Track KV cache, throughput, latency
3. **Iterate based on data**: Adjust based on real metrics, not guesses
4. **Document your config**: Keep it in version control

### For Research/Experimentation:

1. **Try multiple configs**: Use our presets as starting points
2. **Measure everything**: Log all metrics
3. **Share findings**: Contribute back to the community

---

## Summary

### ✅ What We Have:

1. **Default configuration** (still available)
   - Tensor parallelism across 4 GPUs
   - Conservative and compatible

2. **Flexible custom configuration**
   - Pass any vLLM flags via 4th argument
   - Full control over parallelism, batching, memory

3. **Documented best practices**
   - Decision trees for choosing configuration
   - Ready-to-use presets for common scenarios
   - Performance monitoring guidance

4. **Educational resources**
   - Deep dives into how things work
   - Real examples with explanations
   - Troubleshooting guides

### ❌ What vLLM Doesn't Have (Yet):

1. Automatic detection of optimal parallelism strategy
2. Runtime adjustment of configuration
3. Workload-based recommendations
4. Auto-tuning for specific hardware

### 🎯 Best Approach:

**Use documented presets → Monitor → Adjust based on metrics**

This gives you:
- Fast time-to-production (presets)
- Optimal performance (monitoring)
- Full control (explicit configuration)
- Reproducibility (documented settings)

---

## Related Resources

- [VLLM_OPTIMIZATION_PRESETS.md](VLLM_OPTIMIZATION_PRESETS.md) - Ready-to-use configurations
- [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md) - Detailed analysis
- [CHUNKED_PREFILL_TUTORIAL.md](CHUNKED_PREFILL_TUTORIAL.md) - Deep dive into internals
- [vLLM Documentation](https://docs.vllm.ai/) - Official docs
- [GuideLLM GitHub](https://github.com/vllm-project/guidellm) - Benchmarking tool

