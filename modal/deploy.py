#!/usr/bin/env python3
"""
Modal deployment for verifiers.

Run verifiers training on Modal serverless GPU infrastructure with a single command.
"""

import modal
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Define project root to access local files for the image build
project_root = Path(__file__).parent.parent

# Create the Modal app
app = modal.App("verifiers-training")

def build_dependencies():
    """This function is run once during the image build process."""
    import subprocess
    import os
    from pathlib import Path

    # Install core verifiers with training dependencies
    subprocess.run(["uv", "sync", "--extra", "train"], check=True)
    # Install flash-attn without build isolation
    subprocess.run(["uv", "pip", "install", "flash-attn", "--no-build-isolation"], check=True)

    # Install commonly-used environments (to save build time, add more as needed)
    # For a specific env, just pass it in your command: `uv pip install -e environments/your_env && ...`
    common_envs = [
        "gsm8k",
        "math_python",
        "wordle",
        "reverse_text",
        "continuation_quality"
    ]

    env_dir = Path("/app/environments")
    if env_dir.exists():
        for env_name in common_envs:
            env_path = env_dir / env_name
            if env_path.exists() and (env_path / "pyproject.toml").exists():
                print(f"Installing environment: {env_name}")
                try:
                    subprocess.run(["uv", "pip", "install", "-e", str(env_path)], check=True)
                    print(f"✓ Successfully installed {env_name}")
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Failed to install {env_name}: {e}")
                    print(f"   Skipping and continuing...")

# Define persistent volumes for outputs and caches
outputs_volume = modal.Volume.from_name("verifiers-outputs", create_if_missing=True)
cache_volume = modal.Volume.from_name("verifiers-cache", create_if_missing=True)

# Build the container image with all dependencies
verifiers_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    # Set CUDA environment
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
    })
    # Install system dependencies
    .apt_install([
        "git",
        "curl",
        "build-essential",
        "sudo",
        "vim",
        "htop",
        "tmux",
        "nvtop",
        "openssh-client",
    ])
    # Install uv package manager
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR=/usr/local/bin sh",
    )
    # Set uv environment variables
    .env({
        "PATH": "/usr/local/bin:$PATH",
        "UV_PYTHON_INSTALL_DIR": "/usr/local/share/uv/python",
        "UV_CACHE_DIR": "/usr/local/share/uv/cache",
        "UV_COMPILE_BYTECODE": "1",
        "UV_LINK_MODE": "copy",
    })
    # Set up the app directory and install dependencies
    .workdir("/app")
    # Copy only files needed for dependency installation
    .add_local_file(project_root / "pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file(project_root / "README.md", "/app/README.md", copy=True)
    .add_local_dir(project_root / "verifiers", "/app/verifiers", copy=True)  # Needed for version detection
    .add_local_dir(project_root / "environments", "/app/environments", copy=True)  # Needed to install environments
    .run_function(build_dependencies)
    # Set the virtual environment's Python as the default
    .env({"PATH": "/app/.venv/bin:$PATH"})
    # Set runtime environment variables
    .env({
        "HF_HUB_CACHE": "/cache/huggingface",
        "TORCH_HOME": "/cache/torch",
        "WANDB_DIR": "/outputs/wandb",
    })
    # Mount source code at runtime for fast iteration (overwrites the copied verifiers/)
    .add_local_dir(project_root / "verifiers", "/app/verifiers")
    .add_local_dir(project_root / "configs", "/app/configs")
    .add_local_dir(project_root / "examples", "/app/examples")
    .add_local_dir(project_root / "environments", "/app/environments")
)

# Get GPU configuration from environment variable
# Default to H100:8 if not specified (verifiers typically needs more GPUs)
DEFAULT_GPU_CONFIG = os.environ.get("MODAL_GPU_CONFIG", "H100:8")

@app.function(
    image=verifiers_image,
    gpu=DEFAULT_GPU_CONFIG,
    cpu=16.0,
    memory=65536,
    volumes={
        "/outputs": outputs_volume,
        "/cache": cache_volume,
    },
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    timeout=86400,
    enable_memory_snapshot=True,
)
def run_command(
    command: str,
    experiment_name: str,
):
    """
    Run an arbitrary command on Modal.

    This function runs any command in the container with all dependencies installed.
    The code is mounted from your local machine to ensure it's always up-to-date.

    Args:
        command: The command to run (e.g., "CUDA_VISIBLE_DEVICES=0,1 accelerate launch ...")
        experiment_name: Name for this experiment (used for output directory)
    """
    import subprocess
    import os
    import shlex

    print("Running with local code mounted from your machine.")

    # Setup output directory
    output_dir = f"/outputs/{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Set environment variables
    env = os.environ.copy()
    env["WANDB_DIR"] = output_dir
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "dummy")  # vLLM doesn't need real key

    # Parse the command string - use shell=True to handle complex commands
    print("="*60)
    print("Starting command on Modal")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Experiment: {experiment_name}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Command: {command}")
    print("="*60)

    # Run the command with shell=True to handle pipes, redirects, etc.
    result = subprocess.run(command, shell=True, env=env, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(1)

    print("\n" + "="*60)
    print("Command completed successfully!")
    print("="*60)

    # List output files
    print("\nOutput files:")
    for root, dirs, files in os.walk(output_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if not file.startswith('.'):
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                print(f"  {rel_path}")

    return f"Command completed! Results saved to {output_dir}"


@app.local_entrypoint()
def main(
    command: Optional[str] = None,
    experiment_name: Optional[str] = None,
    download_results: bool = True,
):
    """
    Deploy a command on Modal.

    Args:
        command: Command to run (e.g., "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model ... & ...")
                 If not provided, uses default wordle training example
        experiment_name: Name for this experiment (auto-generated if not provided)
        download_results: Whether to show download instructions after completion

    GPU Configuration:
        Set the MODAL_GPU_CONFIG environment variable to change GPU allocation.
        Default: H100:8

        MODAL_GPU_CONFIG="A100-80GB:8" modal run modal/deploy.py

    Examples:
        # Run with default command (wordle training example)
        modal run modal/deploy.py

        # Run custom training command
        modal run modal/deploy.py --command "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-1.7B-Wordle --data-parallel-size 6 --enforce-eager --disable-log-requests & sleep 30 && CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 1.7B"

        # Run with different GPU configuration
        MODAL_GPU_CONFIG="A100-80GB:8" modal run modal/deploy.py --command "your command here"

        # Custom experiment name
        modal run modal/deploy.py --command "..." --experiment-name "my-experiment"
    """
    import time

    # Default command if none provided
    if command is None:
        command = (
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm "
            "--model willcb/Qwen3-1.7B-Wordle "
            "--data-parallel-size 6 "
            "--enforce-eager "
            "--disable-log-requests & "
            "sleep 30 && "
            "CUDA_VISIBLE_DEVICES=6,7 accelerate launch "
            "--num-processes 2 "
            "--config-file configs/zero3.yaml "
            "examples/grpo/train_wordle.py "
            "--size 1.7B"
        )

    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = int(time.time())
        # Try to extract a meaningful name from the command
        if "wordle" in command.lower():
            prefix = "wordle"
        elif "math" in command.lower():
            prefix = "math"
        elif "gsm8k" in command.lower():
            prefix = "gsm8k"
        elif "train" in command.lower():
            prefix = "train"
        else:
            prefix = "experiment"
        experiment_name = f"{prefix}-{timestamp}"

    # Parse the actual GPU config being used
    actual_gpu_config = DEFAULT_GPU_CONFIG

    # Try to parse GPU type and count from config
    if ":" in actual_gpu_config:
        gpu_type_actual, gpu_count_str = actual_gpu_config.rsplit(":", 1)
        try:
            gpu_count_actual = int(gpu_count_str)
        except ValueError:
            gpu_type_actual = actual_gpu_config
            gpu_count_actual = 1
    else:
        gpu_type_actual = actual_gpu_config
        gpu_count_actual = 1

    print(f"\n🚀 Modal Deployment: Verifiers")
    print(f"="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Configuration:")
    print(f"  - Command: {command}")
    print(f"  - GPU config: {actual_gpu_config}")

    # Cost estimation (Modal pricing as of 2025)
    cost_per_gpu = {
        "T4": 0.59,
        "L4": 0.80,
        "A10": 1.10,
        "L40S": 1.95,
        "A100-40GB": 2.10,
        "A100-80GB": 2.50,
        "H100": 3.95,
        "H200": 4.54,
        "B200": 6.25,
    }
    gpu_cost = next((v for k, v in cost_per_gpu.items() if gpu_type_actual.startswith(k)), 2.10)

    print(f"  - Estimated cost: ~${gpu_count_actual * gpu_cost:.2f}/hour ({gpu_type_actual} pricing)")

    print(f"\n✅  This script runs your local code on Modal.")
    print(f"  - Your project directory is mounted into the container.")
    print(f"  - Changes to your code will be reflected on the next run.")

    print(f"="*60)

    # Run command
    print("\n📦 Starting command on Modal...")
    print("(This may take a few minutes to build the container on first run)")
    if os.environ.get("MODAL_GPU_CONFIG"):
        print(f"✅ Using custom GPU config from MODAL_GPU_CONFIG: {actual_gpu_config}")
    else:
        print(f"ℹ️  Using default GPU config: {actual_gpu_config}")
        print(f"   To change, set: MODAL_GPU_CONFIG='A100-80GB:8' modal run modal/deploy.py")

    # Run the command
    result = run_command.remote(
        command=command,
        experiment_name=experiment_name,
    )

    print(result)

    # Download results
    if download_results:
        print(f"\n📥 To download results, use the Modal CLI:")
        print(f"  modal volume get verifiers-outputs {experiment_name} ./outputs/{experiment_name}")
        print(f"\nOr download individual files programmatically using volume.read_file_into_fileobj()")

    print("\n✅ Training session complete!")

    # Print useful commands
    print("\n📝 Useful commands:")
    print(f"  # View logs")
    print(f"  modal app logs")
    print(f"  ")
    print(f"  # Monitor GPU usage")
    print(f"  modal app stats")
    print(f"  ")
    print(f"  # Download results")
    print(f"  modal volume get verifiers-outputs {experiment_name} ./outputs/{experiment_name}")
    print(f"  ")
    print(f"  # List all experiments in volume")
    print(f"  modal volume ls verifiers-outputs")
    print(f"  ")
    print(f"  # List files in this experiment")
    print(f"  modal volume ls verifiers-outputs/{experiment_name}")


if __name__ == "__main__":
    main()
