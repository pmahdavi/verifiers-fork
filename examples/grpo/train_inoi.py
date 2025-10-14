import verifiers as vf
from environments.inoi.inoi import load_environment

"""
Training script for INOI (Iranian National Olympiad in Informatics) environment.
This environment contains mathematical olympiad problems with multimodal support.

# Load INOI environment directly from environments/inoi/
# No installation required - the environment is already in the repo

# Quick eval (if you want to test before training)
# You can run evaluation on the INOI test set with:
# python environments/inoi/example_usage.py

# Inference server (for vLLM-based inference during training):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model 'willcb/Qwen3-1.7B' \
    --data-parallel-size 6 --enforce-eager --disable-log-requests \
    --max-model-len 16384

# Training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_inoi.py
"""

# Load INOI environment (TEXT-ONLY, no images)
vf_env = load_environment(
    dataset_name="pmahdavi/inoi",
    num_train_examples=-1,  # Use all training examples
    num_eval_examples=100,  # Use 100 examples for evaluation
    use_think=True,  # Enable chain-of-thought reasoning
    filter_multimodal=False,  # Keep only text-only problems (no images)
)

# Model configuration - Using Qwen3 1.7B (same as math-python example)
model_name = "willcb/Qwen3-1.7B"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "inoi-grpo_" + model_name.split("/")[-1].lower()

# GRPO training arguments (based on train_math_python.py settings)
training_args = vf.grpo_defaults(run_name=run_name)

# Batch and generation settings - reduced for long sequences
training_args.per_device_train_batch_size = 2  # Reduced from 8 due to long sequences
training_args.num_generations = 16
training_args.gradient_accumulation_steps = 16  # Increased to maintain effective batch size

# Sequence length settings - using very long sequences for complex olympiad problems
training_args.max_tokens = 8192  # Allow longer generations for detailed solutions
training_args.max_seq_len = 16384  # Extra long context for complex problems

# Memory optimization
training_args.gradient_checkpointing = True  # Enable gradient checkpointing to save memory

# Training schedule
training_args.max_steps = 200
training_args.eval_strategy = "steps"
training_args.eval_steps = 25
training_args.save_strategy = "steps"
training_args.save_steps = 50

# GRPO-specific hyperparameters
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.1

# Create GRPO trainer (full fine-tuning like math-python, no LoRA)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    # peft_config=vf.lora_defaults()  # Commented out - doing full fine-tuning
)

# Start training
trainer.train()
