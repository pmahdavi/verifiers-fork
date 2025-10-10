# Deep Dive: Understanding train_gsm8k.py and GRPO Implementation

## Executive Summary

This report provides a comprehensive analysis of how to run and understand the GRPO (Group Relative Policy Optimization) training pipeline for the GSM8K math problem-solving task. The implementation showcases sophisticated software design patterns including async batch generation, modular environment architecture, and efficient GPU orchestration.

---

## Table of Contents

1. [How to Run](#how-to-run)
2. [Architecture Overview](#architecture-overview)
3. [Software Design Patterns](#software-design-patterns)
4. [GRPO Algorithm Implementation](#grpo-algorithm-implementation)
5. [Key Design Decisions](#key-design-decisions)
6. [Code Flow Analysis](#code-flow-analysis)

---

## How to Run

### Prerequisites

```bash
# Install the verifiers package with training dependencies
uv add 'verifiers[train]' && uv pip install flash-attn --no-build-isolation

# Install the GSM8K environment
vf-install gsm8k
```

### Quick Evaluation

Before training, you can evaluate an existing model:

```bash
# Quick evaluation with default settings
vf-eval gsm8k -m gpt-4.1-mini

# With custom parameters
vf-eval gsm8k -m model_name -n 100 -r 3
```

### Training Setup

The training requires **two separate processes** running concurrently:

**Shell 1: vLLM Inference Server**
```bash
# Start vLLM server for inference (policy rollouts)
CUDA_VISIBLE_DEVICES=0 vf-vllm --model willcb/Qwen3-0.6B \
    --enforce-eager --disable-log-requests
```

**Shell 2: Training Process**
```bash
# Start GRPO training with DeepSpeed ZeRO-3
CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 \
    --config-file configs/zero3.yaml examples/grpo/train_gsm8k.py
```

### Configuration Files

**configs/zero3.yaml**: DeepSpeed ZeRO-3 configuration
- Stage 3 parameter sharding across GPUs
- BF16 mixed precision training
- Optimized for 2-16 GPU setups

**configs/endpoints.py**: Model endpoint configurations
- Maps model names to inference endpoints
- Supports OpenAI, OpenRouter, local vLLM, etc.

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GRPO Training System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌─────────────────┐              │
│  │   Training   │         │  vLLM Inference │              │
│  │   Process    │◄───────►│     Server      │              │
│  │  (GPU 1)     │  HTTP   │    (GPU 0)      │              │
│  └──────────────┘         └─────────────────┘              │
│         │                                                     │
│         │ DeepSpeed ZeRO-3                                   │
│         ▼                                                     │
│  ┌──────────────────────────────────────────┐               │
│  │        GRPOTrainer                        │               │
│  │  ┌────────────────────────────────────┐  │               │
│  │  │  AsyncBatchGenerator               │  │               │
│  │  │  - Async rollout generation        │  │               │
│  │  │  - Overlaps training & inference   │  │               │
│  │  │  - Pipeline ahead batches          │  │               │
│  │  └────────────────────────────────────┘  │               │
│  │                                            │               │
│  │  ┌────────────────────────────────────┐  │               │
│  │  │  Policy Model (LoRA)               │  │               │
│  │  │  - Gradient updates                │  │               │
│  │  │  - Weight sync to vLLM             │  │               │
│  │  └────────────────────────────────────┘  │               │
│  │                                            │               │
│  │  ┌────────────────────────────────────┐  │               │
│  │  │  Reference Model (frozen)          │  │               │
│  │  │  - KL divergence computation       │  │               │
│  │  └────────────────────────────────────┘  │               │
│  └──────────────────────────────────────────┘               │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────────┐               │
│  │        Environment                        │               │
│  │  ┌────────────────────────────────────┐  │               │
│  │  │  GSM8K SingleTurnEnv               │  │               │
│  │  │  - Dataset management              │  │               │
│  │  │  - Rollout orchestration           │  │               │
│  │  │  - Reward computation              │  │               │
│  │  └────────────────────────────────────┘  │               │
│  │                                            │               │
│  │  ┌────────────────────────────────────┐  │               │
│  │  │  Rubric (Reward Functions)         │  │               │
│  │  │  - correct_answer_reward_func      │  │               │
│  │  │  - format_reward_func (ThinkParser)│  │               │
│  │  └────────────────────────────────────┘  │               │
│  └──────────────────────────────────────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

```
train_gsm8k.py (entry point, 47 lines)
    │
    ├── vf.load_environment("gsm8k") → GSM8K Environment
    │   ├── Dataset: gsm8k train/test splits
    │   ├── Parser: ThinkParser with extract_boxed_answer
    │   ├── Rubric: [correctness (weight=1.0), format (weight=0.0)]
    │   └── SingleTurnEnv wrapper
    │
    ├── vf.get_model_and_tokenizer(model_name) → Model & Tokenizer
    │
    ├── vf.grpo_defaults(run_name) → GRPOConfig
    │   └── Training hyperparameters
    │
    ├── vf.lora_defaults() → LoRA PeftConfig
    │   └── LoRA adapter configuration
    │
    └── vf.GRPOTrainer
        ├── model: Policy model with LoRA
        ├── ref_model: Reference model (frozen)
        ├── env: Environment instance
        ├── async_generator: AsyncBatchGenerator
        └── vllm_client: Weight synchronization
```

---

## Software Design Patterns

### 1. **Environment Abstraction Pattern**

The codebase uses a hierarchical environment design:

```python
Environment (base)
    ↓
MultiTurnEnv (abstract multi-turn logic)
    ↓
SingleTurnEnv (single response per prompt)
    ↓
GSM8K Environment (domain-specific implementation)
```

**Key Benefits:**
- **Modularity**: Swap datasets/tasks without changing trainer code
- **Reusability**: Same trainer works for math, code, reasoning, tools
- **Extensibility**: Create new environments by implementing `is_completed()` and `env_response()`

**Implementation (environments/gsm8k/gsm8k.py:9-44)**:
```python
def load_environment(use_think=True, system_prompt=BOXED_SYSTEM_PROMPT, ...):
    dataset = load_example_dataset("gsm8k", split="train")

    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

    def correct_answer_reward_func(parser, completion, answer, **kwargs):
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    rubric = vf.Rubric(
        parser=parser,
        funcs=[correct_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0]  # Only correctness matters for reward
    )

    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, ...)
```

### 2. **Async Producer-Consumer Pattern**

The `AsyncBatchGenerator` implements a sophisticated async pipeline:

```python
# Pattern: Overlapping computation and I/O
┌─────────────────────────────────────────────────────────────┐
│                    Timeline                                  │
├─────────────────────────────────────────────────────────────┤
│ Step 0: Generate batch 0 (async)                            │
│ Step 1: ├─ Train on batch 0                                 │
│         └─ Generate batch 1 (async, parallel)               │
│ Step 2: ├─ Train on batch 1                                 │
│         └─ Generate batch 2 (async, parallel)               │
│         ...                                                  │
└─────────────────────────────────────────────────────────────┘
```

**Implementation (verifiers/trainers/async_batch_generator.py:43-89)**:
- **Worker Thread**: Runs async event loop for OpenAI client
- **Request Queue**: Batches to generate
- **Result Queue**: Completed batches
- **Synchronization**: Thread-safe access with locks

**Key Design Decision**: Using threads + asyncio instead of pure asyncio
- Reason: Trainer runs in main thread; async generation in worker thread
- Benefit: Non-blocking rollout generation during training

### 3. **Strategy Pattern for Reward Functions**

The `Rubric` class encapsulates multiple reward functions:

```python
class Rubric:
    funcs: List[Callable]  # Reward function strategies
    weights: List[float]   # Weighted combination

    def score(self, **kwargs):
        scores = [func(**kwargs) for func in self.funcs]
        return sum(w * s for w, s in zip(self.weights, scores))
```

**Benefits:**
- Composable rewards (correctness + format + style + ...)
- Easy A/B testing (adjust weights)
- Non-reward metrics (weight=0.0) for logging only

### 4. **Template Method Pattern**

The `Trainer` class hierarchy uses template method:

```python
Trainer (HuggingFace)
    ↓
GRPOTrainer
    ├── compute_loss() [overridden]
    ├── _prepare_inputs() [overridden]
    ├── get_train_dataloader() [overridden]
    └── evaluate() [overridden]
```

**Benefit**: Inherit HuggingFace training infrastructure (logging, checkpointing, evaluation) while customizing core RL logic.

### 5. **Dependency Injection Pattern**

The trainer receives all dependencies through constructor:

```python
GRPOTrainer(
    model=model,              # Injected policy model
    env=vf_env,               # Injected environment
    args=training_args,       # Injected configuration
    processing_class=tokenizer,  # Injected tokenizer
    peft_config=lora_config   # Injected adapter config
)
```

**Benefits:**
- Testability: Easy to mock dependencies
- Flexibility: Swap implementations without code changes
- Configuration: Centralized in one place (grpo_defaults)

---

## GRPO Algorithm Implementation

### Algorithm Overview

GRPO (Group Relative Policy Optimization) is an on-policy RL algorithm similar to PPO but optimized for LLMs:

**Core Idea**:
1. Generate multiple completions per prompt (group)
2. Compute rewards for each completion
3. Normalize rewards within each group (advantage estimation)
4. Update policy to maximize high-reward completions
5. Use clipping to prevent large policy updates

### Mathematical Foundation

**Objective Function**:
```
L(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)] - β * KL(π_θ || π_ref)

where:
- r(θ) = π_θ(a|s) / π_old(a|s)  (probability ratio)
- A = R - mean(R_group)          (advantage)
- ε = clipping parameter (0.2)
- β = KL coefficient (0.001)
```

**Advantage Computation** (verifiers/trainers/grpo_trainer.py:1195-1212):
```python
def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
    # rewards shape: (batch_size,) where batch_size = num_prompts * num_generations

    # Reshape to (num_prompts, num_generations)
    mean_grouped = rewards.view(-1, self.num_generations).mean(dim=1)
    std_grouped = rewards.view(-1, self.num_generations).std(dim=1)

    # Broadcast back to (batch_size,)
    mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
    std_grouped = std_grouped.repeat_interleave(self.num_generations, dim=0)

    # Compute advantages (centered at 0 within each group)
    advantages = rewards - mean_grouped

    if self.scale_rewards:
        advantages = advantages / (std_grouped + 1e-4)  # Optional normalization

    return advantages
```

**Loss Computation** (verifiers/trainers/grpo_trainer.py:1214-1322):
```python
def compute_loss(self, model, inputs, ...):
    # 1. Get current policy log probabilities
    per_token_logps = self._get_per_token_logps(
        model, input_ids, attention_mask, logits_to_keep
    )

    # 2. Get old policy log probabilities (from rollout time)
    old_per_token_logps = inputs["old_per_token_logps"]

    # 3. Compute probability ratio
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)

    # 4. Compute clipped ratio
    coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

    # 5. Compute two loss candidates
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)

    # 6. Take minimum (pessimistic bound)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # 7. Add KL divergence penalty
    if self.beta != 0.0:
        ref_per_token_logps = self._get_per_token_logps(
            self.ref_model, input_ids, attention_mask, logits_to_keep
        )
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )
        per_token_loss = per_token_loss + self.beta * per_token_kl

    # 8. Aggregate token losses
    if self.loss_type == "dr_grpo":
        # Dr. GRPO: normalize by global constant (eliminates length bias)
        loss = (per_token_loss * completion_mask).sum() / (
            per_token_loss.size(0) * self.max_seq_len
        )

    return loss
```

### Key Algorithm Features

#### 1. **Group-Based Advantage Estimation**

Instead of comparing to a learned value function (like PPO), GRPO compares completions within their generation group:

```python
# Example: 2 prompts, 3 generations each
rewards = [0.8, 1.0, 0.6,  # Prompt 1 completions
           0.4, 0.5, 0.3]  # Prompt 2 completions

# Within-group normalization
advantages = [0.0, +0.2, -0.2,  # Prompt 1 (mean=0.8)
              +0.067, +0.167, -0.233]  # Prompt 2 (mean=0.4)
```

**Why this matters**:
- No separate critic network needed
- Reduces variance in advantage estimates
- Naturally handles prompt difficulty variations

#### 2. **Multi-Iteration Updates**

The `num_iterations` parameter (μ in the paper) controls how many gradient steps to take per batch of rollouts:

```python
# Default: num_iterations = 1 (one-step update, like PPO)
# Can set higher for better sample efficiency

# Training loop with num_iterations = 4:
for batch in dataloader:
    rollouts = generate(batch)  # Generate once
    for i in range(4):
        loss = compute_loss(rollouts)  # Reuse rollouts 4 times
        loss.backward()
        optimizer.step()
```

**Tradeoff**: Higher iterations → better sample efficiency but higher risk of over-optimization.

#### 3. **Three Loss Formulations**

The implementation supports three loss types (grpo_config.py:334-348):

**GRPO** (original):
```python
loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).mean()
```
- Problem: Length bias (prefers short completions for positive advantages)

**BNPO** (batch-normalized):
```python
loss = (per_token_loss * mask).sum() / mask.sum()
```
- Normalizes over local batch only
- Slight variance with different batch sizes

**Dr. GRPO** (recommended):
```python
loss = (per_token_loss * mask).sum() / (batch_size * max_seq_len)
```
- Normalizes by global constant
- Eliminates length bias
- **Used by default in this implementation**

#### 4. **Masking Strategies**

Two critical masking decisions:

**Environment Response Masking** (`mask_env_responses=True`):
```python
# In multi-turn environments, don't train on environment responses
# Only train on model generations

Messages:
  [user]        "Solve: 2+2=?"           ← Don't train
  [assistant]  "<think>2+2=4</think>4"   ← Train on this
  [env]        "Correct!"                ← Don't train (masked)
```

**Truncated Completion Masking** (`mask_truncated_completions=True`):
```python
# Don't penalize completions cut off by max_tokens
if completion_reached_max_length and no_eos_token:
    loss_mask[completion] = 0  # Exclude from loss
```

---

## Key Design Decisions

### 1. **Separate Training and Inference Processes**

**Decision**: Run vLLM inference server separately from training process

**Rationale**:
- vLLM uses PagedAttention and optimized kernels for fast inference
- Training needs backward pass capabilities
- Separating allows independent scaling (more inference GPUs vs training GPUs)

**Implementation**:
- Training process: Updates model weights via DeepSpeed
- Inference process: Receives weight updates via NCCL/gRPC
- Communication: HTTP requests for generation, NCCL for weight sync

**Code Location** (grpo_trainer.py:775-862):
```python
def _move_model_to_vllm(self):
    # Synchronize all processes
    self.accelerator.wait_for_everyone()

    # For PEFT, merge adapters before syncing
    if is_peft_model(self.model):
        with gather_if_zero3(list(self.model.parameters())):
            self.model.merge_adapter()
            for name, param in self.model.named_parameters():
                # Update vLLM weights
                if self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(name, param.data)
            self.model.unmerge_adapter()

    # Reset vLLM cache
    if self.accelerator.is_main_process:
        self.vllm_client.reset_prefix_cache()

    self.accelerator.wait_for_everyone()
```

### 2. **Async Batch Generation Pipeline**

**Decision**: Generate future batches while training on current batch

**Rationale**:
- Training involves forward pass, backward pass, optimizer step (~1-2s)
- Generation involves environment rollouts with vLLM (~5-10s)
- **Without async**: Total time = generation_time + training_time
- **With async**: Total time ≈ max(generation_time, training_time)

**Configuration** (grpo_config.py:242-248):
```python
num_batches_ahead: int = 1  # How many batches to pipeline
# 0: Synchronous (simple, slower)
# 1: Generate next batch while training (recommended)
# 2+: More aggressive pipelining (uses more memory)
```

**Implementation** (grpo_trainer.py:961-1193):
```python
def _prepare_inputs(self, inputs):
    if self._step % generate_every == 0:
        # Submit future batches to maintain pipeline
        for batch_id in range(self._next_batch_id, target_batch_id + 1):
            if self.accelerator.is_main_process:
                self.async_generator.submit_batch(request)

        # Retrieve completed batch
        if self.accelerator.is_main_process:
            batch_result = self.async_generator.get_batch(batch_id_to_retrieve)
```

**Benefit**: ~2x throughput improvement for typical configurations.

### 3. **DeepSpeed ZeRO-3 for Memory Efficiency**

**Decision**: Use ZeRO-3 parameter sharding instead of DDP

**Rationale**:
- **ZeRO-1**: Shard optimizer states → 4x memory reduction
- **ZeRO-2**: + Shard gradients → 8x memory reduction
- **ZeRO-3**: + Shard parameters → 16x+ memory reduction

**Tradeoff**:
- More communication overhead
- Slightly slower training
- **But**: Enables training larger models on same hardware

**Configuration** (configs/zero3.yaml):
```yaml
deepspeed_config:
  zero_stage: 3
  zero3_init_flag: true  # Initialize parameters in sharded state
  zero3_save_16bit_model: true  # Save consolidated model
  offload_optimizer_device: none  # Keep on GPU
  offload_param_device: none  # Keep on GPU
```

### 4. **LoRA for Parameter-Efficient Fine-Tuning**

**Decision**: Use LoRA adapters by default

**Rationale**:
- Train only ~0.1-1% of parameters
- Faster training iterations
- Lower memory requirements
- Easy to swap/merge adapters

**Configuration** (trainers/__init__.py:40-46):
```python
def lora_defaults(r=8, alpha=16):
    return LoraConfig(
        r=8,  # Low-rank dimension
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
```

**Adapter Math**:
- Original weight: W ∈ ℝ^(d×k)
- LoRA: W + α/r * (A × B), where A ∈ ℝ^(d×r), B ∈ ℝ^(r×k)
- Parameters: d×k → d×r + r×k
- Example: 4096×4096 matrix: 16M params → 131K params (r=8)

### 5. **Reference Model Strategy**

**Decision**: Use multiple reference model strategies based on configuration

**Options** (grpo_trainer.py:497-513):

1. **No reference model** (β=0.0):
   - Skip KL penalty entirely
   - Faster training
   - Risk: Policy divergence

2. **PEFT adapter disable**:
   - Disable LoRA adapter to get base model
   - Memory efficient (no separate model)
   - Only works with LoRA

3. **Separate reference model**:
   - Full copy of initial model
   - Required for full fine-tuning
   - 2x memory usage

4. **Synced reference model** (`sync_ref_model=True`):
   - Periodically update reference: π_ref ← α·π_θ + (1-α)·π_ref
   - Prevents reference from becoming too stale
   - Inspired by TR-DPO paper

**Configuration** (grpo_config.py:369-390):
```python
sync_ref_model: bool = True
ref_model_mixup_alpha: float = 0.5  # α in update equation
ref_model_sync_steps: int = 100  # Update every N steps
```

### 6. **Custom Sampler for GRPO**

**Decision**: Implement `RepeatSampler` instead of using standard PyTorch samplers

**Purpose**: Ensure identical prompts are distributed across GPUs for group normalization

**Implementation** (grpo_trainer.py:55-159):
```python
class RepeatSampler:
    """
    Generates indices like:
    [0, 0, 1, 1, 2, 2,  # First GPU (3 prompts, 2 generations each)
     0, 0, 1, 1, 2, 2]  # Repeat for gradient accumulation

    Key properties:
    1. Each prompt appears num_generations times consecutively
    2. Batches are repeated num_iterations times for multi-step updates
    3. Same random seed across processes ensures sync
    """
```

**Why this matters**:
- GRPO advantage computation requires grouping completions by prompt
- Each GPU needs all completions for a subset of prompts
- Synchronized sampling ensures correct group formation

---

## Code Flow Analysis

### Training Loop Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                                │
├─────────────────────────────────────────────────────────────────┤
│ train_gsm8k.py                                                   │
│   ├─ Load environment: vf.load_environment("gsm8k")             │
│   ├─ Load model: vf.get_model_and_tokenizer(model_name)         │
│   ├─ Create config: vf.grpo_defaults(run_name)                  │
│   ├─ Create LoRA: vf.lora_defaults()                            │
│   └─ Create trainer: vf.GRPOTrainer(...)                        │
│                                                                   │
│ GRPOTrainer.__init__()                                           │
│   ├─ Apply LoRA to model                                        │
│   ├─ Create reference model                                     │
│   ├─ Initialize vLLM client                                     │
│   ├─ Create AsyncBatchGenerator                                 │
│   ├─ Filter/prepare datasets                                    │
│   └─ Setup metrics & logging                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAINING LOOP (trainer.train())                              │
├─────────────────────────────────────────────────────────────────┤
│ For each training step:                                          │
│                                                                   │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ A. BATCH GENERATION (every gradient_accumulation_steps)  │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ _prepare_inputs()                                         │   │
│ │   ├─ Check if need to generate: _step % generate_every   │   │
│ │   ├─ Sync weights to vLLM: _move_model_to_vllm()         │   │
│ │   │   ├─ Gather DeepSpeed ZeRO-3 parameters             │   │
│ │   │   ├─ Merge LoRA adapters (if using PEFT)            │   │
│ │   │   ├─ Send weights to vLLM via NCCL                  │   │
│ │   │   ├─ Reset vLLM prefix cache                        │   │
│ │   │   └─ Wait for all processes                         │   │
│ │   │                                                       │   │
│ │   ├─ Submit async batches:                               │   │
│ │   │   For batch_id in [current, current+num_ahead]:     │   │
│ │   │     ├─ Gather batch data from all processes         │   │
│ │   │     └─ async_generator.submit_batch(request)        │   │
│ │   │                                                       │   │
│ │   ├─ Retrieve completed batch:                           │   │
│ │   │   └─ batch_result = async_generator.get_batch(id)   │   │
│ │   │       (blocks until ready)                           │   │
│ │   │                                                       │   │
│ │   ├─ Broadcast results to all processes                  │   │
│ │   ├─ Compute advantages from rewards                     │   │
│ │   ├─ Compute old policy logprobs                         │   │
│ │   ├─ Shuffle and split for gradient accumulation        │   │
│ │   └─ Buffer inputs for reuse                            │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ B. ASYNC GENERATION (parallel in worker thread)          │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ AsyncBatchGenerator._generation_worker()                 │   │
│ │   Loop forever:                                           │   │
│ │     ├─ Get request from queue                            │   │
│ │     ├─ _generate_batch_async(request)                    │   │
│ │     │   ├─ env.a_generate() - Call environment           │   │
│ │     │   │   ├─ Create rollout for each prompt           │   │
│ │     │   │   ├─ Send requests to vLLM (async)            │   │
│ │     │   │   ├─ Parse responses                           │   │
│ │     │   │   ├─ Compute rewards via Rubric               │   │
│ │     │   │   └─ Return GenerateOutputs                    │   │
│ │     │   │                                                 │   │
│ │     │   └─ env.process_env_results_vllm()               │   │
│ │     │       ├─ Convert to token IDs                      │   │
│ │     │       ├─ Create attention masks                    │   │
│ │     │       ├─ Apply masking strategies                  │   │
│ │     │       └─ Return ProcessedOutputs                   │   │
│ │     │                                                     │   │
│ │     └─ Put result in queue                               │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ C. LOSS COMPUTATION                                       │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ compute_loss(model, inputs)                               │   │
│ │   ├─ Forward pass: get current policy logprobs           │   │
│ │   ├─ Compute probability ratios                          │   │
│ │   │   ratio = exp(logp_new - logp_old)                  │   │
│ │   ├─ Compute clipped ratios                             │   │
│ │   │   clipped = clamp(ratio, 1-ε, 1+ε)                  │   │
│ │   ├─ Compute GRPO loss                                   │   │
│ │   │   loss = -min(ratio * A, clipped * A)               │   │
│ │   ├─ Add KL penalty (if β > 0)                          │   │
│ │   │   ├─ Reference model forward pass                    │   │
│ │   │   └─ loss += β * KL(π_θ || π_ref)                   │   │
│ │   ├─ Aggregate with loss_type (dr_grpo)                 │   │
│ │   └─ Log metrics (clip ratios, KL, etc.)                │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│ ┌──────────────────────────────────────────────────────────┐   │
│ │ D. BACKWARD & OPTIMIZATION                                │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ Standard PyTorch training step:                           │   │
│ │   ├─ loss.backward()                                     │   │
│ │   ├─ Gradient accumulation (if needed)                   │   │
│ │   ├─ optimizer.step()                                    │   │
│ │   ├─ optimizer.zero_grad()                               │   │
│ │   └─ lr_scheduler.step()                                 │   │
│ └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ 3. EVALUATION (every eval_steps)                                │
├─────────────────────────────────────────────────────────────────┤
│ evaluate()                                                       │
│   ├─ async_generator.evaluate(num_samples=-1)                  │
│   │   └─ env.evaluate() - Run eval dataset                     │
│   ├─ Compute eval metrics                                       │
│   │   ├─ Mean reward                                            │
│   │   ├─ Per-reward-function scores                             │
│   │   └─ Completion statistics                                  │
│   └─ Log to wandb/console                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Detailed Step-by-Step Example

Let's trace one complete training iteration with concrete values:

**Configuration**:
```python
per_device_train_batch_size = 12  # Completions per GPU
num_generations = 12              # Completions per prompt
gradient_accumulation_steps = 8   # Accumulate 8 micro-batches
num_processes = 1                 # Single GPU for training
```

**Derived Values**:
```python
unique_prompts_per_device = 12 / 12 = 1
generation_batch_size = 12 * 1 * 8 = 96 completions
unique_prompts_per_batch = 96 / 12 = 8 prompts
```

**Step 0 (Training Step 0)**:

1. **Check if generation needed**:
   - `_step=0 % (8*1) == 0` → Yes, generate new batch

2. **Sync weights to vLLM**:
   ```python
   _move_model_to_vllm()
   # Main process gathers all parameters and sends to vLLM
   # Takes ~1-2 seconds for 1B parameter model
   ```

3. **Submit batch for generation**:
   ```python
   # Gather prompts from all processes
   all_prompts = [prompt_0, prompt_1, ..., prompt_7]  # 8 unique prompts

   # Submit to async generator
   async_generator.submit_batch(BatchRequest(
       batch_id=0,
       env_inputs={"prompt": all_prompts, "answer": all_answers, ...}
   ))
   ```

4. **Async generation (parallel)**:
   ```python
   # In worker thread:
   env.a_generate(prompts) generates:
   # Prompt 0: [completion_0, completion_1, ..., completion_11]
   # Prompt 1: [completion_0, completion_1, ..., completion_11]
   # ...
   # Prompt 7: [completion_0, completion_1, ..., completion_11]
   # Total: 96 completions

   # Compute rewards:
   rewards = rubric.score(completions)
   # Example: [1.0, 0.0, 1.0, ..., 0.5, 0.0, ...]  # 96 rewards
   ```

5. **Retrieve batch**:
   ```python
   batch_result = async_generator.get_batch(0)  # Blocks until ready
   # Returns: prompts, completions, rewards, token IDs, masks
   ```

6. **Compute advantages**:
   ```python
   rewards = [1.0, 0.0, 1.0, 0.5, 0.8, 0.3, ..., 0.0]  # 96 values

   # Group by prompt (12 completions per prompt)
   grouped = rewards.view(8, 12)
   # [[1.0, 0.0, 1.0, 0.5, 0.8, 0.3, 1.0, 0.2, 0.9, 0.4, 0.6, 0.7],  # Prompt 0
   #  [0.0, 1.0, 0.0, ...],  # Prompt 1
   #  ...]

   # Compute group means
   means = grouped.mean(dim=1)  # [0.625, ...]

   # Advantages = reward - group_mean
   advantages = rewards - means.repeat_interleave(12)
   # [+0.375, -0.625, +0.375, -0.125, +0.175, -0.325, ...]
   ```

7. **Compute old logprobs**:
   ```python
   # Forward pass with current model
   old_logps = model(input_ids, attention_mask).logits
   # Detach and store for later use
   ```

8. **Split for gradient accumulation**:
   ```python
   # Shuffle all 96 completions
   shuffled = shuffle([completions, advantages, logps])

   # Split into 8 micro-batches of 12 completions each
   micro_batches = [
       shuffled[0:12],   # Accumulation step 0
       shuffled[12:24],  # Accumulation step 1
       ...
       shuffled[84:96]   # Accumulation step 7
   ]
   ```

9. **Training substep 0 (first micro-batch)**:
   ```python
   inputs = micro_batches[0]  # 12 completions

   loss = compute_loss(model, inputs)
   # - Forward pass on 12 completions
   # - Compute ratio = exp(logp_new - logp_old)
   # - Compute clipped_ratio = clamp(ratio, 0.8, 1.2)
   # - loss = -min(ratio * advantages, clipped * advantages)
   # - Add KL penalty

   loss.backward()  # Accumulate gradients
   ```

10. **Training substeps 1-7**: Repeat for remaining micro-batches

11. **Optimizer step**:
    ```python
    optimizer.step()      # Update model parameters
    optimizer.zero_grad() # Clear gradients
    ```

**Step 8 (Training Step 8)**:

- `_step=8 % 8 == 0` → Generate new batch (batch_id=1)
- Reuse previous pattern

**Key Insight**: With `num_iterations=1`, we generate new rollouts every 8 training steps. With `num_iterations=4`, we'd reuse the same rollouts for 32 training steps (4 iterations × 8 accumulation steps).

---

## Performance Characteristics

### Memory Usage

**Training Process** (per GPU):
```
Model parameters:         ~1.2 GB (1B model, bf16)
LoRA adapters:           ~10 MB  (r=8)
Optimizer states:        ~2.4 GB (AdamW, bf16)
Gradients:               ~1.2 GB
Activations:             ~2-4 GB (depends on sequence length)
Batch data:              ~500 MB
Reference model:         ~1.2 GB (if not using PEFT)
────────────────────────────────
Total (PEFT):            ~8-10 GB
Total (Full FT):         ~10-12 GB
```

**With DeepSpeed ZeRO-3** (4 GPUs):
```
Parameters:              ~300 MB per GPU (sharded)
Optimizer states:        ~600 MB per GPU (sharded)
Gradients:               ~300 MB per GPU (sharded)
Activations:             ~2-4 GB per GPU (not sharded)
────────────────────────────────
Total per GPU:           ~4-6 GB
```

**Inference Process** (vLLM):
```
Model parameters:        ~1.2 GB
KV cache:                ~4-8 GB (depends on batch size)
────────────────────────────────
Total:                   ~6-10 GB
```

### Throughput Analysis

**Without async generation**:
```
Time per iteration = generation_time + training_time
                   = 10s + 2s
                   = 12s per iteration
```

**With async generation** (`num_batches_ahead=1`):
```
Iteration 0: Generate batch 0 (10s)
Iteration 1: Train on batch 0 (2s) | Generate batch 1 (10s, parallel)
Iteration 2: Train on batch 1 (2s) | Generate batch 2 (10s, parallel)
...

Time per iteration ≈ max(generation_time, training_time)
                   ≈ 10s per iteration
```

**Speedup**: ~1.2x (12s → 10s)

**Bottleneck**: Generation is slower than training, so throughput is generation-limited. Increasing vLLM parallelism (more GPUs, tensor parallelism) would further improve throughput.

### Hyperparameter Scaling

| Hyperparameter | Effect on Memory | Effect on Throughput | Effect on Learning |
|----------------|------------------|----------------------|--------------------|
| `per_device_train_batch_size` | Linear ↑ | Linear ↑ (if fits in memory) | Better gradient estimates |
| `num_generations` | Linear ↑ | Linear ↓ | Better advantage estimates |
| `gradient_accumulation_steps` | Constant | Linear ↓ | Larger effective batch size |
| `max_seq_len` | Quadratic ↑ (attention) | Quadratic ↓ | Allows longer reasoning |
| `num_iterations` | Constant | Constant | Better sample efficiency |
| `num_batches_ahead` | Linear ↑ (buffering) | Sub-linear ↑ | No effect |

---

## Advanced Topics

### 1. **Prefix Caching in vLLM**

The implementation resets vLLM's prefix cache after weight updates:

```python
self.vllm_client.reset_prefix_cache()
```

**Why?** vLLM caches computations for common prompt prefixes. After weight updates, cached KV values are stale.

**Optimization opportunity**: If using a system prompt, could preserve that prefix across weight updates (marginal gain).

### 2. **Handling Distributed Training**

The code carefully orchestrates multi-GPU training:

**Synchronization points** (grpo_trainer.py):
```python
self.accelerator.wait_for_everyone()  # Barrier for all processes
```

**Broadcasts** (main process → all processes):
```python
broadcast_object_list([data], from_process=0)  # Python objects
```

**Gathers** (all processes → main process):
```python
all_data = gather_object(local_data)  # Collect from all GPUs
```

**Key invariant**: All processes must call collective operations in the same order, or deadlock occurs.

### 3. **Dataset Filtering and Truncation**

The trainer automatically filters and truncates the dataset:

**Prompt length filtering** (grpo_trainer.py:416-446):
```python
# Remove prompts longer than max_prompt_length
train_dataset = train_dataset.filter(
    lambda ex: len(tokenizer.encode(ex["prompt"])) <= max_prompt_length
)
```

**Batch-size truncation** (grpo_trainer.py:480-489):
```python
# Ensure dataset size is divisible by global batch size
truncated_size = (dataset_size // global_batch_size) * global_batch_size
train_dataset = train_dataset.select(range(truncated_size))
```

**Why?** Ensures all GPUs have equal work and GRPO groups are complete.

### 4. **Logging and Monitoring**

The implementation tracks extensive metrics:

**Reward metrics**:
- `reward`: Overall reward (weighted sum)
- `reward_std`: Within-group standard deviation
- `rewards/{func_name}`: Individual reward function scores

**Clip ratio metrics**:
- `clip_ratio/low_mean`: Fraction clipped below 1-ε
- `clip_ratio/high_mean`: Fraction clipped above 1+ε
- `clip_ratio/region_mean`: Total fraction clipped

**Completion metrics**:
- `completions/mean_length`: Average completion tokens
- `completions/clipped_ratio`: Fraction truncated by max_tokens
- `completions/mean_terminated_length`: Average for completions with EOS

**KL divergence**: `kl` (if β > 0)

**WandB logging**: Logs full completion tables with prompts, completions, and rewards.

### 5. **Error Handling and Edge Cases**

**Timeout handling** (async_batch_generator.py:140-183):
```python
timeout = timeout or self.generation_timeout
if time.time() - start_time > timeout:
    raise TimeoutError(f"Batch {batch_id} generation timed out")
```

**Empty dataset handling**: Raises clear error if dataset is too small.

**NaN handling**: Uses custom `nanmean`, `nanmin`, `nanmax` for robust statistics.

**Process synchronization**: Extensive use of barriers and broadcasts to prevent deadlocks.

---

## Troubleshooting Guide

### Common Issues

**1. NCCL hangs during vLLM weight sync**
- **Symptom**: Training freezes when syncing weights to vLLM
- **Cause**: Inter-GPU communication issues
- **Solution**: Set `NCCL_P2P_DISABLE=1` or `NCCL_CUMEM_ENABLE=1`

**2. Out of memory (OOM)**
- **Symptom**: CUDA OOM error during training
- **Solutions**:
  - Reduce `per_device_train_batch_size`
  - Reduce `max_seq_len`
  - Enable `gradient_checkpointing=True`
  - Increase `gradient_accumulation_steps`

**3. Slow generation**
- **Symptom**: Generation takes much longer than training
- **Solutions**:
  - Increase vLLM parallelism (more GPUs, `--tensor-parallel-size`)
  - Increase `num_batches_ahead` for more pipelining
  - Reduce `max_tokens` if completions don't need to be long

**4. Reward always 0.0**
- **Symptom**: Training runs but reward stays at 0
- **Causes**:
  - Parser not extracting answers correctly
  - Answer format mismatch between dataset and parser
  - Reward function has bug
- **Debug**: Add print statements in reward function, check `log_completions=True` output

**5. NaN loss**
- **Symptom**: Loss becomes NaN during training
- **Causes**:
  - Learning rate too high
  - Numerical instability in advantage computation
  - KL divergence exploding
- **Solutions**:
  - Reduce `learning_rate`
  - Enable `scale_rewards=True`
  - Reduce `beta` (KL coefficient)

### Debugging Tools

**1. Log completions**:
```python
training_args.log_completions = True
```
This prints/logs sample completions every `logging_steps` to verify correct parsing and rewards.

**2. Reduce scale for testing**:
```python
training_args.max_steps = 10
training_args.per_device_train_batch_size = 4
training_args.num_generations = 4
```

**3. Check dataset**:
```python
vf_env = vf.load_environment("gsm8k", num_eval_examples=10)
print(vf_env.dataset[0])  # Inspect first example
```

**4. Dry run evaluation**:
```bash
vf-eval gsm8k -m gpt-4.1-mini -n 5 -r 1
```

---

## Conclusion

The `train_gsm8k.py` script is a minimal entry point (~50 lines) to a sophisticated RL training system with:

1. **Modular architecture**: Swappable environments, parsers, rubrics
2. **Efficient training**: Async generation, DeepSpeed ZeRO-3, LoRA
3. **Production-ready**: Extensive logging, error handling, distributed training
4. **Research-friendly**: Multiple loss types, configurable clipping, reference model strategies

The key insight is that **complexity is hidden behind clean abstractions**:
- Training script: Configure and run
- Trainer: Orchestrate training loop
- Environment: Define task and rewards
- Async generator: Optimize throughput
- vLLM: Fast inference

This design allows researchers to focus on experimenting with:
- New environments (new `load_environment` functions)
- New reward functions (new `Rubric` definitions)
- New algorithms (subclass `GRPOTrainer`)

Without needing to modify the core infrastructure.

---

## References

- **Paper**: "GRPO: Group Relative Policy Optimization" (DeepSeek-R1)
- **Code**: https://github.com/PrimeIntellect-ai/verifiers
- **Docs**: https://verifiers.readthedocs.io
- **vLLM**: https://docs.vllm.ai
- **DeepSpeed**: https://www.deepspeed.ai/tutorials/zero/
- **LoRA**: "Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

---

**Report Generated**: 2025-10-08
**Version**: Based on verifiers v0.1.5.dev1
**Author**: Analysis of /scratch/pxm5426/repos/verifiers/examples/grpo/train_gsm8k.py
