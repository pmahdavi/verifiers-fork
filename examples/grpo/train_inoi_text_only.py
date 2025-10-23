"""
GRPO Training for INOI (Iranian National Olympiad in Informatics) - Text-Only

This script trains a 1.7B model on text-only INOI problems using GRPO.
Key decisions:
- max_tokens=4096: Forces concise reasoning (longer tokens led to rambling)
- max_prompt_length=2048: INOI problems are ~1000 tokens
- Format reward (0.25): Encourages proper answer formatting
- Full fine-tuning: No LoRA for 1.7B model

Usage:
# vLLM inference server (4 GPUs):
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm \
    --model 'willcb/Qwen3-1.7B' \
    --data-parallel-size 4 \
    --enforce-eager \
    --disable-log-requests \
    --max-model-len 8192

# Training (2 GPUs):
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_inoi_text_only.py
"""

import verifiers as vf
from environments.inoi.inoi import load_environment

# ============ Environment Setup ============
vf_env = load_environment(
    dataset_name="combviz/inoi",
    num_train_examples=-1,        # All text-only examples (~592)
    num_eval_examples=50,          # Eval subset for speed
    use_think=True,                # Enable CoT reasoning
    filter_multimodal=False,       # TEXT-ONLY (critical!)
)

# ============ Modify Rubric to Add Format Reward (0.25 weight) ============
vf_env.rubric.reward_weights = [1.0, 0.25]  # correctness + format

print(f"Training examples: {len(vf_env.get_dataset())}")
eval_dataset = vf_env.get_eval_dataset()
print(f"Eval examples: {len(eval_dataset) if eval_dataset else 0}")
print(f"Reward functions: {vf_env.rubric.get_reward_func_names()}")
print(f"Reward weights: {vf_env.rubric.get_reward_weights()}")

# ============ Model Setup ============
model_name = "willcb/Qwen3-1.7B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "inoi-text-grpo_" + model_name.split("/")[-1].lower()

# ============ Training Arguments ============
training_args = vf.grpo_defaults(run_name=run_name)

# Batch configuration (following Option B: 2, 16, 16)
training_args.per_device_train_batch_size = 2
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 16

# Sequence lengths (balanced for INOI: force concise reasoning)
training_args.max_tokens = 4096          # Max completion length (force conciseness)
training_args.max_seq_len = 8192         # Total: prompt + completion
training_args.max_prompt_length = 2048   # INOI problems are ~1000 tokens, add buffer

# Memory optimization (bare minimum)
training_args.gradient_checkpointing = True

# Training schedule
training_args.max_steps = 500
training_args.eval_strategy = "steps"
training_args.eval_steps = 50
training_args.save_strategy = "steps"
training_args.save_steps = 100

# GRPO hyperparameters (following math_python pattern)
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.1

# ============ Create Trainer ============
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    # No LoRA - full fine-tuning for 1.7B
)

# ============ Train ============
print(f"\nStarting training: {run_name}")
print(f"Effective batch size: {2 * 16 * 16 * 2} generations")
print(f"Total generations: {500 * 2 * 16 * 16 * 2}")

trainer.train()

