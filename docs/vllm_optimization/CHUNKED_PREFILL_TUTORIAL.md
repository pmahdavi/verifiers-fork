# Deep Dive: Chunked Prefill in vLLM

## Table of Contents
1. [Introduction](#introduction)
2. [Background: LLM Inference Phases](#background-llm-inference-phases)
3. [The Problem: Long Context Processing](#the-problem-long-context-processing)
4. [The Solution: Chunked Prefill](#the-solution-chunked-prefill)
5. [Technical Details](#technical-details)
6. [Performance Analysis](#performance-analysis)
7. [Configuration and Tuning](#configuration-and-tuning)
8. [Understanding Your Server Logs](#understanding-your-server-logs)
9. [Advanced Considerations](#advanced-considerations)

---

## Introduction

**Chunked prefill** is a critical optimization technique in vLLM that enables efficient processing of long input sequences. When you see this in your server logs:

```
INFO 10-23 10:51:24 [scheduler.py:205] Chunked prefill is enabled with max_num_batched_tokens=2048.
```

It means vLLM is intelligently dividing long prompts into smaller, more manageable pieces. This tutorial will explain **why** this is necessary, **how** it works, and **what** it means for your inference performance.

---

## Background: LLM Inference Phases

To understand chunked prefill, you first need to understand the **two-phase nature of LLM inference**:

### Phase 1: Prefill (Prompt Processing)
**What happens**: The model processes all input tokens at once to build the initial KV cache.

**Characteristics**:
- **Compute-bound**: Heavy matrix multiplications (Q·K^T) across all tokens
- **Parallel processing**: All input tokens can be processed simultaneously
- **Complexity**: O(n²) for self-attention where n = number of input tokens
- **Memory**: Allocates KV cache blocks for the entire context

**Example**: If you send a 10,000 token prompt, the prefill phase processes all 10,000 tokens in a single forward pass through the model.

### Phase 2: Decode (Token Generation)
**What happens**: The model generates one new token at a time autoregressively.

**Characteristics**:
- **Memory-bound**: Primarily accessing stored KV cache
- **Sequential**: Each token depends on the previous ones
- **Complexity**: O(n) per token, where n = current context length
- **Throughput**: Limited by memory bandwidth, not computation

**Example**: After prefill, the model generates "Hello" → "world" → "!" one token at a time.

---

## The Problem: Long Context Processing

### Why Long Sequences Are Challenging

When processing very long sequences (e.g., 32K+ tokens), the **prefill phase** faces several bottlenecks:

#### 1. **Memory Explosion**
- Self-attention requires computing an **attention matrix** of size `(seq_len × seq_len)`
- For 32K tokens: `32,768 × 32,768 = 1,073,741,824` attention scores
- With bfloat16 (2 bytes): ~2GB **just for attention scores**
- Add in activations, gradients, KV cache → easily exceeds GPU memory

#### 2. **Batching Inefficiency**
- GPUs are designed for **batch processing**
- Long prompts monopolize GPU memory → fewer requests in a batch
- Example:
  - **Without chunking**: Process 2 prompts of 30K tokens each = 60K tokens
  - **With chunking**: Process 30 prompts of 2K tokens each = 60K tokens
  - Same total work, but batching 30 smaller requests improves GPU utilization

#### 3. **Latency Spikes**
- User waits for **entire prefill** to complete before seeing any output
- 30K token prefill might take 10+ seconds
- Poor user experience (especially for streaming)

#### 4. **Out-of-Memory (OOM) Errors**
- Single massive attention matrix can exceed available VRAM
- Leads to crashes or request failures

---

## The Solution: Chunked Prefill

### Core Concept

Instead of processing a 30,000-token prompt in one massive forward pass:

```
Traditional:
[Token 1...30,000] → Single Prefill → Start Decoding
```

Chunked prefill splits it into smaller chunks:

```
Chunked (chunk_size=2048):
[Token 1...2,048]       → Prefill Chunk 1 → Update KV Cache
[Token 2,049...4,096]   → Prefill Chunk 2 → Update KV Cache
[Token 4,097...6,144]   → Prefill Chunk 3 → Update KV Cache
...
[Token 28,673...30,000] → Prefill Chunk 15 → Start Decoding
```

### Key Benefits

1. **Controlled Memory Usage**: Each chunk uses predictable, bounded memory
2. **Better Batching**: Mix prefill chunks with decode requests in the same batch
3. **Improved GPU Utilization**: Smaller, more uniform workloads
4. **Reduced Latency**: Start generating tokens sooner (with continuous batching)
5. **No OOM Crashes**: Memory usage stays within safe limits

---

## Technical Details

### How Attention Works with Chunked Prefill

Let's walk through a concrete example with a 5,000-token prompt and chunk_size=2,048:

#### **Chunk 1: Tokens 1-2,048**
```python
# Process first chunk
Q1, K1, V1 = compute_qkv(tokens[0:2048])      # Shape: [2048, hidden_dim]
attention_1 = softmax(Q1 @ K1.T) @ V1         # Attend only within chunk
kv_cache.store(K1, V1)                        # Store for future chunks
```

**Attention matrix**: `2,048 × 2,048` ✅ Fits in memory

#### **Chunk 2: Tokens 2,049-4,096**
```python
# Process second chunk
Q2, K2, V2 = compute_qkv(tokens[2048:4096])   # Shape: [2048, hidden_dim]

# Now we need to attend to BOTH previous tokens AND current chunk
K_all = concat(K1, K2)                        # Shape: [4096, hidden_dim]
V_all = concat(V1, V2)                        # Shape: [4096, hidden_dim]
attention_2 = softmax(Q2 @ K_all.T) @ V_all   # Attend to all 4096 tokens
kv_cache.store(K2, V2)                        # Store new chunk
```

**Attention matrix**: `2,048 × 4,096` ✅ Still manageable

#### **Chunk 3: Tokens 4,097-5,000**
```python
Q3, K3, V3 = compute_qkv(tokens[4096:5000])   # Shape: [904, hidden_dim]

# Attend to all previous tokens
K_all = concat(K1, K2, K3)                    # Shape: [5000, hidden_dim]
V_all = concat(V1, V2, V3)                    # Shape: [5000, hidden_dim]
attention_3 = softmax(Q3 @ K_all.T) @ V_all   # Attend to all 5000 tokens
kv_cache.store(K3, V3)
```

**Attention matrix**: `904 × 5,000` ✅ Controlled size

### Critical Insight: Why Computation Time Stays Constant

**Theoretical expectation**: Later chunks should take longer because they attend to more tokens.
- Chunk 1: Attends to 2,048 tokens
- Chunk 2: Attends to 4,096 tokens  
- Chunk 3: Attends to 5,000 tokens

**Actual observation**: All chunks take approximately the same time!

#### Why?

1. **Optimized Memory Access (PagedAttention)**
   - vLLM's PagedAttention kernel stores KV cache in **contiguous memory blocks**
   - Fetching historical KV pairs is extremely fast (GPU memory bandwidth >> compute)
   - Memory access latency is minimal compared to computation

2. **Hardware-Level Parallelism**
   - Modern GPUs have **tensor cores** optimized for large matrix multiplications
   - Computing `2048×4096` matrix multiplication is nearly as fast as `2048×2048`
   - GPU saturates compute units regardless of matrix size (within reasonable bounds)

3. **FlashAttention Optimization**
   - Your logs show: `Using Flash Attention backend on V1 engine`
   - FlashAttention uses **kernel fusion** and **tiling** to minimize memory I/O
   - Computation is so fast that memory access dominates, and PagedAttention optimizes that

---

## Performance Analysis

### Real-World Metrics from Your Server

Looking at your log line 313:
```
INFO 10-23 11:10:56 [loggers.py:127] Engine 000: 
  Avg prompt throughput: 78.5 tokens/s
  Avg generation throughput: 412.2 tokens/s
  Running: 8 reqs
  Waiting: 0 reqs
  GPU KV cache usage: 1.6%
  Prefix cache hit rate: 72.1%
```

#### Key Observations:

1. **Prompt Throughput: 78.5 tokens/s**
   - This is the **prefill** speed (processing input tokens)
   - With chunked prefill, this stays consistent regardless of prompt length
   - Without chunking, long prompts would severely reduce this number

2. **Generation Throughput: 412.2 tokens/s**
   - This is the **decode** speed (generating output tokens)
   - **5.2× faster** than prefill! This is expected:
     - Decode is memory-bound (just reading KV cache)
     - Prefill is compute-bound (computing full attention matrices)

3. **GPU KV Cache Usage: 1.6%**
   - Very low! You have **massive headroom** for more requests
   - With tensor-parallel-size=4, you're using 4 GPUs
   - Each GPU is only using 1.6% of its KV cache capacity

4. **Prefix Cache Hit Rate: 72.1%**
   - **Critical metric**: 72.1% of prefill work is being **reused** from cache!
   - This means many requests share common prompt prefixes
   - Example: System prompts, few-shot examples, etc.
   - **Huge speedup**: Those tokens don't need to be recomputed

### Throughput Breakdown

When generating (line 314-396), you see:
```
Avg generation throughput: 315.2 tokens/s (running 8 requests)
```

**Per-request throughput**: `315.2 / 8 ≈ 39.4 tokens/s per request`

This is excellent! Each request is generating ~39 tokens/second while batched with 7 others.

---

## Configuration and Tuning

### Key Parameters

#### 1. `max_num_batched_tokens` (Default: 2048 for long-context models)

**What it controls**: Maximum tokens processed in a **single scheduler iteration**.

**From your logs (line 7)**:
```
Chunked prefill is enabled with max_num_batched_tokens=2048
```

**How it works**:
- If a prompt has 10,000 tokens → split into chunks of 2,048
- First iteration: Process tokens 0-2,047
- Second iteration: Process tokens 2,048-4,095
- Continue until all prefill chunks are done

**Tuning guide**:
```bash
# Smaller chunks (better batching, lower memory, higher overhead)
vf-vllm --model MODEL --max-num-batched-tokens 512

# Larger chunks (less overhead, higher memory, worse batching)
vf-vllm --model MODEL --max-num-batched-tokens 8192

# Disable chunking (only for short contexts!)
vf-vllm --model MODEL --enable-chunked-prefill false
```

**When to adjust**:
- **Increase** if you have excess GPU memory and want faster prefill
- **Decrease** if you're getting OOM errors or want better batching

#### 2. `enable_chunked_prefill` (Default: Auto-enabled for context > 32K)

**From vLLM source**:
```python
if max_model_len > 32768 and enable_chunked_prefill is None:
    enable_chunked_prefill = True  # Auto-enable
```

Your model (Qwen3-VL-2B-Instruct) has `max_model_len=262,144` (line 6), so chunked prefill was **automatically enabled**.

#### 3. `enable_prefix_caching` (Default: True)

**From your logs (line 11)**:
```
enable_prefix_caching=True
```

**How it works with chunked prefill**:
1. First request processes prompt: `[System prompt (1000 tokens)] + [User query (500 tokens)]`
2. Chunks are cached: `Chunk 1: tokens 0-1000` gets cached
3. Second request with same system prompt: **Cache hit!** Skip computing chunk 1
4. Result: 72.1% cache hit rate (from your logs)

**Impact**:
- **Without prefix caching**: Every request recomputes all chunks
- **With prefix caching**: Shared chunks computed only once
- Your 72.1% hit rate means **72% of your prefill work is free**!

---

## Understanding Your Server Logs

Let's decode the key sections from your server startup:

### 1. **Server Configuration (Lines 1-7)**
```
INFO 10-23 10:51:24 [model.py:1510] Using max model len 262144
INFO 10-23 10:51:24 [scheduler.py:205] Chunked prefill is enabled with max_num_batched_tokens=2048
```

**What this means**:
- Your model supports up to **262,144 tokens** (256K context!)
- Because 262K >> 32K threshold → chunked prefill auto-enabled
- Chunk size set to 2,048 tokens

### 2. **Tensor Parallelism Setup (Lines 74-77)**
```
INFO 10-23 10:51:44 [parallel_state.py:1208] rank 0 in world size 4 is assigned as DP rank 0, PP rank 0, TP rank 0
INFO 10-23 10:51:44 [parallel_state.py:1208] rank 1 in world size 4 is assigned as DP rank 0, PP rank 0, TP rank 1
INFO 10-23 10:51:44 [parallel_state.py:1208] rank 2 in world size 4 is assigned as DP rank 0, PP rank 0, TP rank 2
INFO 10-23 10:51:44 [parallel_state.py:1208] rank 3 in world size 4 is assigned as DP rank 0, PP rank 0, TP rank 3
```

**Decoding**:
- **World size 4**: You're using 4 GPUs
- **TP rank 0-3**: Tensor Parallelism across 4 GPUs (each GPU holds 1/4 of the model)
- **DP rank 0**: No Data Parallelism (all GPUs serve the same requests)
- **PP rank 0**: No Pipeline Parallelism (model isn't split across layers)

**How this affects chunked prefill**:
- Each chunk is processed across all 4 GPUs in parallel
- GPU 0 computes 1/4 of attention heads, GPU 1 computes next 1/4, etc.
- Final result combined via **NCCL all-reduce** (lines 33-40)

### 3. **Flash Attention (Lines 88-90)**
```
INFO 10-23 10:51:52 [cuda.py:366] Using Flash Attention backend on V1 engine.
```

**Why this matters for chunked prefill**:
- Flash Attention optimizes memory I/O during attention computation
- Makes chunked prefill **even more efficient**
- Reduces the overhead of processing multiple chunks

### 4. **Runtime Metrics (Line 313)**
```
Engine 000: 
  Avg prompt throughput: 78.5 tokens/s       ← Prefill speed
  Avg generation throughput: 412.2 tokens/s  ← Decode speed
  Running: 8 reqs                            ← Concurrent requests
  GPU KV cache usage: 1.6%                   ← Memory utilization
  Prefix cache hit rate: 72.1%               ← Cache efficiency
```

**Interpreting**:
- **8 concurrent requests**: vLLM is batching 8 requests together
- **Low KV cache usage**: You can handle **many more requests** (1.6% is tiny!)
- **72% cache hit**: Most prefill work is being skipped due to shared prefixes

---

## Advanced Considerations

### 1. **Chunked Prefill + Continuous Batching**

vLLM uses **continuous batching** to mix prefill and decode:

```
Iteration 1:
  - Request A: Prefill chunk 1 (tokens 0-2047)
  - Request B: Prefill chunk 3 (tokens 4096-6143)
  - Request C: Decode token 15
  - Request D: Decode token 42

Iteration 2:
  - Request A: Prefill chunk 2 (tokens 2048-4095)
  - Request B: Prefill chunk 4 (tokens 6144-8191)
  - Request C: Decode token 16
  - Request D: Decode token 43
  - Request E: Prefill chunk 1 (new request!)
```

**Benefits**:
- No GPU idle time between requests
- Amortizes prefill cost across multiple iterations
- Maintains high decode throughput

### 2. **When to Disable Chunked Prefill**

From vLLM documentation, chunked prefill might **hurt performance** when:

1. **Very short prompts** (< 512 tokens on average)
   - Overhead of chunking exceeds benefits
   - Better to process in one shot

2. **Low prefix cache hit rate** (< 20%)
   - If prompts don't share prefixes, caching doesn't help
   - Chunking overhead provides no benefit

3. **Extremely high throughput requirements**
   - Chunking adds slight latency per request
   - For latency-critical apps, might disable

**How to check if chunking helps**:
```bash
# Run with chunking (default)
vf-eval your-env -m local-vllm -n 100 --time-limit 300

# Run without chunking
vf-vllm --model MODEL --enable-chunked-prefill false --port 8003
# Update endpoint to use port 8003
vf-eval your-env -m local-vllm -n 100 --time-limit 300

# Compare throughput metrics
```

### 3. **Memory Analysis**

**Without chunked prefill** (30K token prompt):
```
Attention matrix: 30,000 × 30,000 = 900M elements
Memory (bf16):    900M × 2 bytes = 1.8 GB
Plus activations: ~3-4 GB total
```

**With chunked prefill** (chunk_size=2048):
```
Largest attention matrix: 2,048 × 30,000 = 61M elements
Memory (bf16):            61M × 2 bytes = 122 MB
Plus activations:         ~200-300 MB total
```

**Reduction**: ~15× less memory per chunk!

### 4. **Prefix Caching Deep Dive**

Your **72.1% cache hit rate** is excellent. Here's what's happening:

**Example scenario**:
```python
# Request 1
prompt = "You are a helpful AI assistant.\n\nQuestion: What is 2+2?"
# Prefill: Chunks 1-3, all computed, all cached

# Request 2 (shares system prompt)
prompt = "You are a helpful AI assistant.\n\nQuestion: What is the capital of France?"
# Prefill: Chunk 1 (cache hit! ✅), Chunk 2 (cache hit! ✅), Chunk 3 (new, compute)

# Cache hit rate: 2/3 chunks = 66.7%
```

**Across many requests**: Average 72.1% of chunks are reused!

### 5. **Performance Optimization Checklist**

Based on your logs, here's how to optimize:

```bash
# ✅ Already optimal:
- Chunked prefill enabled (auto)
- Prefix caching enabled (72% hit rate!)
- Flash Attention enabled
- Tensor parallelism (4 GPUs)

# 🔧 Potential improvements:
1. Increase batch size (KV cache usage only 1.6%!)
   → You can handle 50-100x more concurrent requests
   
2. Tune max_num_batched_tokens based on workload:
   - If avg prompt < 1000 tokens → increase to 4096
   - If getting OOM → decrease to 1024
   
3. Monitor prefix cache hit rate:
   - If drops below 30% → consider disabling prefix caching
   - If stays high → perfect!

4. Consider data parallelism if you have idle GPUs:
   vf-vllm --model MODEL --data-parallel-size 2 --tensor-parallel-size 2
   → Doubles throughput for high request volume
```

---

## Summary

**Chunked prefill** is a sophisticated optimization that:

1. **Splits long prompts** into manageable chunks (default 2048 tokens)
2. **Enables batching** of prefill and decode requests together
3. **Reduces memory usage** by 10-15× for long contexts
4. **Works seamlessly** with prefix caching (72% hit rate in your case!)
5. **Maintains consistent performance** thanks to PagedAttention and FlashAttention

**Your server is well-configured**:
- ✅ Auto-enabled for 256K context model
- ✅ Optimized chunk size (2048)
- ✅ Excellent cache hit rate (72.1%)
- ✅ Low memory usage (1.6% KV cache)
- ✅ High throughput (412 tokens/s generation)

**Next steps to learn more**:
1. Experiment with different chunk sizes (`--max-num-batched-tokens`)
2. Monitor how cache hit rate changes with different prompts
3. Load test with more concurrent requests (you have 98% KV cache headroom!)
4. Profile with vLLM's built-in metrics to understand bottlenecks

---

## References

- vLLM Blog: [Chunked Prefill on AMD](https://blog.vllm.ai/2024/10/23/vllm-serving-amd.html)
- vLLM Discussions: [Chunked Prefill Performance](https://discuss.vllm.ai/t/computation-time-remain-consistent-across-chunks-in-chunked-prefill-despite-linearly-growing-attention-complexity/744)
- PagedAttention Paper: [Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
- FlashAttention: [Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

