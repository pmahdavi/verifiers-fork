"""
GRPO Training with PrimeRL Hub Environments

This script demonstrates how to use environments from the PrimeRL Hub
(https://app.primeintellect.ai/dashboard/environments) with GRPO training.

Key Hyperparameters:
- num_generations: Number of rollouts/completions to generate per prompt (default: 8)
- per_device_train_batch_size: Batch size per device 
- gradient_accumulation_steps: Steps to accumulate before update
- max_tokens: Maximum tokens to generate per turn
- max_seq_len: Maximum sequence length (prompt + completion)
- temperature, top_p, top_k: Sampling parameters
- beta: KL regularization coefficient (0.0 disables reference model)
- epsilon: PPO-style clipping parameter
- max_concurrent: Maximum concurrent environment requests

Usage:
1. First install the prime CLI:
   uv tool install prime
   prime login

2. Browse available environments:
   Visit https://app.primeintellect.ai/dashboard/environments

3. Install an environment from the hub:
   prime env install owner/environment-name
   # or with version pinning:
   prime env install owner/environment-name@version

4. Run GRPO training:
   # Inference server (shell 0):
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model your-model-name \
       --data-parallel-size 6 --enforce-eager --disable-log-requests

   # Training (shell 1):
   CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
       --config-file configs/zero3.yaml examples/grpo/train_primerl_hub.py \
       --env-id your-env-id --model your-model-name

Example with specific environments:
   # For a math environment:
   prime env install primeintellect/math-python
   python examples/grpo/train_primerl_hub.py --env-id math-python

   # For a coding environment:
   prime env install primeintellect/code-debug
   python examples/grpo/train_primerl_hub.py --env-id code-debug
"""

import argparse
import logging
from typing import Dict, Any

import verifiers as vf


logger = logging.getLogger(__name__)


def get_default_training_args(env_id: str, model_size: str) -> Dict[str, Any]:
    """Get default training arguments based on environment and model size."""
    
    # Base configuration with all key hyperparameters
    base_config = {
        # Core training parameters
        "per_device_train_batch_size": 8,
        "num_generations": 8,  # Number of rollouts per prompt
        "gradient_accumulation_steps": 4,
        "max_steps": 500,
        "eval_strategy": "steps",
        "eval_steps": 20,
        "save_strategy": "steps", 
        "save_steps": 100,
        
        # Generation parameters
        "max_tokens": 1024,  # Max tokens per turn
        "max_seq_len": 4096,  # Max total sequence length
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        
        # GRPO-specific parameters
        "beta": 0.001,  # KL coefficient (0.0 disables reference model)
        "epsilon": 0.2,  # PPO-style clipping
        "scale_rewards": False,  # Whether to normalize rewards
        "mask_env_responses": True,  # Mask environment responses in loss
        "mask_truncated_completions": True,  # Mask truncated sequences
        "zero_truncated_completions": False,  # Zero reward for truncated
        
        # Optimization parameters
        "learning_rate": 1e-6,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 10,
        "max_grad_norm": 0.01,
        "bf16": True,
        
        # Reference model synchronization
        "sync_ref_model": True,
        "ref_model_mixup_alpha": 0.5,
        "ref_model_sync_steps": 100,
        
        # Async generation parameters
        "max_concurrent": 1024,  # Max concurrent environment requests
        "num_batches_ahead": 1,  # Look-ahead batches for async generation
        "async_generation_timeout": 600.0,
        
        # Logging
        "logging_steps": 1,
        "log_completions": True,
        "report_to": "wandb",
    }
    
    # Environment-specific adjustments
    env_configs = {
        "math": {
            "max_tokens": 2048,
            "max_seq_len": 8192,
            "per_device_train_batch_size": 4,
            "num_generations": 16,  # More rollouts for math problems
            "beta": 0.1,  # Higher KL penalty for math
        },
        "code": {
            "max_tokens": 2048,
            "max_seq_len": 8192,
            "per_device_train_batch_size": 4,
            "num_generations": 12,
            "temperature": 0.8,  # Lower temperature for code
        },
        "reasoning": {
            "max_tokens": 4096,
            "max_seq_len": 16384,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "num_generations": 8,
            "beta": 0.0,  # No KL penalty for complex reasoning
        },
        "tool": {
            "max_tokens": 1024,
            "max_seq_len": 8192,
            "per_device_train_batch_size": 4,
            "num_generations": 16,
            "max_concurrent": 256,  # Lower concurrency for tool environments
        },
        "wordle": {
            "max_tokens": 1024,
            "max_seq_len": 4096,
            "num_generations": 16,
            "beta": 0.0,
            "max_grad_norm": 0.1,
        },
    }
    
    # Model size adjustments
    if "70B" in model_size or "70b" in model_size:
        base_config["per_device_train_batch_size"] = 1
        base_config["gradient_accumulation_steps"] = 32
        base_config["gradient_checkpointing"] = True
        base_config["num_generations"] = 4  # Fewer rollouts for large models
    elif "13B" in model_size or "13b" in model_size:
        base_config["per_device_train_batch_size"] = 2
        base_config["gradient_accumulation_steps"] = 16
        base_config["gradient_checkpointing"] = True
        base_config["num_generations"] = 8
    elif "7B" in model_size or "7b" in model_size:
        base_config["per_device_train_batch_size"] = 4
        base_config["gradient_accumulation_steps"] = 8
        base_config["gradient_checkpointing"] = True
    elif "1.5B" in model_size or "1.7B" in model_size or "1.8B" in model_size:
        base_config["per_device_train_batch_size"] = 8
        base_config["gradient_accumulation_steps"] = 8
        base_config["num_generations"] = 16
    elif "0.5B" in model_size or "0.6B" in model_size:
        base_config["per_device_train_batch_size"] = 12
        base_config["gradient_accumulation_steps"] = 8
        base_config["num_generations"] = 12
    
    # Apply environment-specific config if available
    for env_type, config in env_configs.items():
        if env_type in env_id.lower():
            base_config.update(config)
            break
    
    return base_config


def validate_generation_batch_size(training_args, num_processes: int = 1):
    """Validate that generation batch size is compatible with num_generations."""
    effective_batch_size = (
        training_args.per_device_train_batch_size * 
        num_processes * 
        training_args.gradient_accumulation_steps
    )
    
    if effective_batch_size % training_args.num_generations != 0:
        possible_values = [
            n for n in range(2, effective_batch_size + 1) 
            if effective_batch_size % n == 0
        ]
        raise ValueError(
            f"Effective batch size ({effective_batch_size}) must be divisible by "
            f"num_generations ({training_args.num_generations}). "
            f"Valid values: {possible_values}"
        )


def main(args):
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = vf.get_model_and_tokenizer(args.model)
    
    # Load environment from PrimeRL Hub
    logger.info(f"Loading environment: {args.env_id}")
    
    # Parse environment arguments if provided
    env_kwargs = {}
    if args.env_args:
        for arg in args.env_args:
            key, value = arg.split("=", 1)
            # Try to parse as int/float/bool
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if isinstance(value, str) and value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
            env_kwargs[key] = value
    
    # Add common environment arguments
    if args.num_train_examples:
        env_kwargs["num_train_examples"] = args.num_train_examples
    if args.num_eval_examples:
        env_kwargs["num_eval_examples"] = args.num_eval_examples
    if args.use_think is not None:
        env_kwargs["use_think"] = args.use_think
    
    try:
        vf_env = vf.load_environment(env_id=args.env_id, **env_kwargs)
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        logger.info("Make sure you've installed the environment using:")
        logger.info(f"  prime env install owner/{args.env_id}")
        raise
    
    # Set up training arguments
    run_name = f"{args.env_id}-grpo-{args.model.split('/')[-1]}"
    training_args = vf.grpo_defaults(run_name=run_name)
    
    # Apply default configurations
    default_config = get_default_training_args(args.env_id, args.model)
    for key, value in default_config.items():
        setattr(training_args, key, value)
    
    # Override with command line arguments if provided
    if args.batch_size:
        training_args.per_device_train_batch_size = args.batch_size
    if args.num_generations:
        training_args.num_generations = args.num_generations
    if args.gradient_accumulation_steps:
        training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.max_steps:
        training_args.max_steps = args.max_steps
    if args.eval_steps:
        training_args.eval_steps = args.eval_steps
    if args.learning_rate:
        training_args.learning_rate = args.learning_rate
    if args.temperature is not None:
        training_args.temperature = args.temperature
    if args.top_p is not None:
        training_args.top_p = args.top_p
    if args.top_k is not None:
        training_args.top_k = args.top_k
    if args.beta is not None:
        training_args.beta = args.beta
    if args.epsilon is not None:
        training_args.epsilon = args.epsilon
    if args.max_tokens:
        training_args.max_tokens = args.max_tokens
    if args.max_seq_len:
        training_args.max_seq_len = args.max_seq_len
    if args.max_concurrent:
        training_args.max_concurrent = args.max_concurrent
    
    # Validate configuration
    num_processes = args.num_processes if args.num_processes else 1
    validate_generation_batch_size(training_args, num_processes)
    
    # Log final configuration
    logger.info("=" * 60)
    logger.info("GRPO Training Configuration:")
    logger.info("=" * 60)
    logger.info(f"Environment: {args.env_id}")
    logger.info(f"Model: {args.model}")
    logger.info("-" * 60)
    logger.info("Core Training Parameters:")
    logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"  Number of rollouts (num_generations): {training_args.num_generations}")
    logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * num_processes * training_args.gradient_accumulation_steps}")
    logger.info(f"  Max steps: {training_args.max_steps}")
    logger.info(f"  Eval steps: {training_args.eval_steps}")
    logger.info("-" * 60)
    logger.info("Generation Parameters:")
    logger.info(f"  Max tokens per turn: {training_args.max_tokens}")
    logger.info(f"  Max sequence length: {training_args.max_seq_len}")
    logger.info(f"  Temperature: {training_args.temperature}")
    logger.info(f"  Top-p: {training_args.top_p}")
    logger.info(f"  Top-k: {training_args.top_k}")
    logger.info(f"  Max concurrent requests: {training_args.max_concurrent}")
    logger.info("-" * 60)
    logger.info("GRPO Algorithm Parameters:")
    logger.info(f"  Beta (KL coefficient): {training_args.beta}")
    logger.info(f"  Epsilon (clipping): {training_args.epsilon}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Max grad norm: {training_args.max_grad_norm}")
    logger.info(f"  Mask env responses: {training_args.mask_env_responses}")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        lora_config=vf.lora_defaults() if args.use_lora else None,
    )
    
    # Start training
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # Save final model if requested
    if args.save_model:
        logger.info(f"Saving model to {args.save_model}")
        trainer.save_model(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO training with PrimeRL Hub environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="Environment ID from PrimeRL Hub (e.g., 'math-python', 'code-debug')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path"
    )
    
    # Environment arguments
    parser.add_argument(
        "--env-args",
        nargs="*",
        help="Additional arguments for environment (format: key=value)"
    )
    parser.add_argument(
        "--num-train-examples",
        type=int,
        help="Number of training examples to use"
    )
    parser.add_argument(
        "--num-eval-examples",
        type=int,
        help="Number of evaluation examples to use"
    )
    parser.add_argument(
        "--use-think",
        type=lambda x: x.lower() == "true",
        help="Whether to use thinking/reasoning in the environment"
    )
    
    # Core training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Per-device training batch size"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        help="Number of rollouts/completions per prompt (must divide effective batch size)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Number of steps to accumulate gradients"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        help="Evaluation interval (in steps)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    
    # Generation hyperparameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate per turn"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Maximum total sequence length (prompt + completion)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling"
    )
    
    # GRPO-specific hyperparameters
    parser.add_argument(
        "--beta",
        type=float,
        help="KL regularization coefficient (0.0 disables reference model)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="PPO-style clipping parameter"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        help="Maximum concurrent requests to the environment"
    )
    
    # Other arguments
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    parser.add_argument(
        "--save-model",
        type=str,
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Number of processes (for validation)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    main(args)