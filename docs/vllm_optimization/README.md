# vLLM Optimization Documentation

This directory contains comprehensive guides for optimizing vLLM inference server performance.

## Quick Start

**New to vLLM optimization?** Start here:
1. Read [WHATS_NEW_VLLM_OPTIMIZATION.md](WHATS_NEW_VLLM_OPTIMIZATION.md) for an overview of changes
2. Try the optimized presets from [VLLM_OPTIMIZATION_PRESETS.md](VLLM_OPTIMIZATION_PRESETS.md)
3. **For long-context generation (10K+ tokens)**: Read [LONG_CONTEXT_GENERATION_GUIDE.md](LONG_CONTEXT_GENERATION_GUIDE.md) first!
4. Deep dive into [CHUNKED_PREFILL_TUTORIAL.md](CHUNKED_PREFILL_TUTORIAL.md) to understand the internals

**Working with specific model types?**
- Vision-Language models (Qwen3-VL, LLaVA): See VL Models section in [VLLM_OPTIMIZATION_PRESETS.md](VLLM_OPTIMIZATION_PRESETS.md#-vision-language-models-vl-models)
- Reasoning models (Qwen-Thinking, DeepSeek-R1): See Reasoning Models section in [VLLM_OPTIMIZATION_PRESETS.md](VLLM_OPTIMIZATION_PRESETS.md#-reasoning-models-thinkingcot-models)
- Multiple models simultaneously: See "Running Multiple Servers" in [VLLM_SERVER_README.md](../../scripts/VLLM_SERVER_README.md#running-multiple-vllm-servers-simultaneously)

## Documentation Files

### 🤖 [AUTOMATIC_CONFIGURATION.md](AUTOMATIC_CONFIGURATION.md)
**Does vLLM have automatic configuration detection?**

Explains what vLLM auto-detects, what it doesn't, and why our documented presets approach is better.

**Key Topics:**
- What vLLM automatically detects (and doesn't)
- Why no auto-detection for parallelism strategies
- Tools like GuideLLM for benchmarking
- Our default setup (still available!)
- Practical workflows for configuration tuning

### 📖 [WHATS_NEW_VLLM_OPTIMIZATION.md](WHATS_NEW_VLLM_OPTIMIZATION.md)
**Overview of the flexible vLLM configuration system**

Summary of changes, usage examples, and quick reference for getting started with optimized configurations.

**Key Topics:**
- Flexible command-line configuration
- Performance comparison (4-5× speedup)
- Quick start examples
- Troubleshooting guide

### 🎯 [VLLM_OPTIMIZATION_PRESETS.md](VLLM_OPTIMIZATION_PRESETS.md)
**Ready-to-use optimization configurations**

Copy-paste configurations for different model sizes and use cases.

**Key Topics:**
- Presets for small (< 7B), medium (7-20B), and large (20B+) models
- High throughput, balanced, and low-latency configurations
- Decision trees for choosing the right config
- Performance monitoring guidelines
- **Special sections**: Vision-Language models, Reasoning models

### 🔬 [CHUNKED_PREFILL_TUTORIAL.md](CHUNKED_PREFILL_TUTORIAL.md)
**Deep dive into chunked prefill mechanism**

Technical explanation of how vLLM processes long contexts efficiently.

**Key Topics:**
- Two-phase LLM inference (prefill vs decode)
- Why chunked prefill is necessary
- PagedAttention and FlashAttention
- Performance analysis with real server logs
- Configuration and tuning parameters

### 📊 [PERFORMANCE_OPTIMIZATION_GUIDE.md](PERFORMANCE_OPTIMIZATION_GUIDE.md)
**Detailed performance analysis and recommendations**

Comprehensive guide analyzing common performance issues and providing specific fixes.

**Key Topics:**
- Critical performance bottlenecks (over-parallelization, CUDA graphs, batching)
- Recommended optimizations by priority
- Expected performance improvements
- Benchmarking methodology
- Advanced optimizations

### 📏 [LONG_CONTEXT_GENERATION_GUIDE.md](LONG_CONTEXT_GENERATION_GUIDE.md) **NEW!**
**Performance behavior for long-context generation (10K-50K tokens)**

Essential guide for understanding and optimizing very long generation workloads.

**Key Topics:**
- **Normal performance degradation** during long generation (40-60% slower is expected!)
- Why throughput decreases over time (O(n²) attention complexity)
- Performance expectations by generation length
- Optimization strategies for long contexts
- **Common misconceptions** debunked
- Time estimation formulas for planning evaluations
- Troubleshooting long-context specific issues

## Quick Examples

### Default Configuration (Baseline)
```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8000 default
```

### Optimized Configuration (4-5× Faster)
```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8001 optimized \
  "--data-parallel-size 4 --max-num-seqs 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.95"
```

### Custom Configuration
```bash
./scripts/launch_vllm.sh Qwen/Qwen3-VL-2B-Instruct 8002 custom \
  "--data-parallel-size 2 --tensor-parallel-size 2 --max-num-seqs 256"
```

## Key Performance Gains

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Generation Throughput | 412 tokens/s | 1600-2000 tokens/s | **4-5×** |
| Prompt Throughput | 78.5 tokens/s | 200-250 tokens/s | **3×** |
| Concurrent Requests | 8 | 80-100+ | **10×** |
| GPU Utilization | ~25% | ~90%+ | **4×** |

## Common Use Cases

### For 2-7B Models
Use **data parallelism** for maximum throughput:
```bash
"--data-parallel-size 4 --max-num-seqs 512"
```

### For 7-20B Models
Use **hybrid parallelism** (TP + DP):
```bash
"--tensor-parallel-size 2 --data-parallel-size 2 --max-num-seqs 256"
```

### For 20B+ Models
Use **tensor parallelism** (model too large for single GPU):
```bash
"--tensor-parallel-size 4 --max-num-seqs 128"
```

## Monitoring Performance

Watch real-time metrics:
```bash
tail -f pbs_results/vllm_*_realtime.log | grep "Engine 000"
```

**Key metrics to watch:**
- **Prompt throughput**: Prefill speed (tokens/s)
- **Generation throughput**: Decode speed (tokens/s)
- **Running requests**: Current batch size
- **KV cache usage**: Should be 30-70% for optimal performance
- **Prefix cache hit rate**: Higher is better (60%+ is excellent)

## Related Documentation

- [vLLM Server Setup Guide](../../scripts/VLLM_SERVER_README.md) - Main README for vLLM server
- [Launch Script Usage](../../scripts/launch_vllm.sh) - Script for launching vLLM servers
- [Verifiers Documentation](../source/index.md) - Main project documentation

## Contributing

When adding new optimization guides:
1. Follow the existing documentation structure
2. Include practical examples with real commands
3. Provide performance benchmarks where applicable
4. Update this README with links to new guides

