#!/usr/bin/env python3
"""
Modal vLLM Endpoint Deployment

Deploy a vLLM server as a persistent Modal web endpoint for running evaluations.
This allows you to run vf-eval locally while using Modal GPUs for inference.

Usage:
    # Deploy with default model (Qwen3-VL-32B-Thinking)
    modal deploy modal/vllm_endpoint.py
    
    # Deploy with custom model
    modal deploy modal/vllm_endpoint.py --model "Qwen/Qwen2.5-7B-Instruct"
    
    # Deploy with custom GPU config
    MODAL_GPU_CONFIG="A100-80GB:4" modal deploy modal/vllm_endpoint.py
    
    # Stop the endpoint
    modal app stop vllm-endpoint

After deployment, you'll get a URL like:
    https://your-username--vllm-endpoint-web.modal.run

Add this to configs/endpoints.py and use it with vf-eval.
"""

import modal
import os

# Create the Modal app
app = modal.App("vllm-endpoint")

# Define the vLLM image with all dependencies
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install([
        "git",
        "curl",
        "build-essential",
    ])
    .pip_install([
        "vllm",  # Latest version with Qwen3-VL support  
    ])
)

# Get GPU configuration from environment variable
# 32B VL model needs substantial GPU memory - default to 4x A100-80GB
DEFAULT_GPU_CONFIG = os.environ.get("MODAL_GPU_CONFIG", "A100-80GB:4")

# Persistent cache volume for model downloads
cache_volume = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Default model
DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Thinking"


@app.function(
    image=vllm_image,
    gpu=DEFAULT_GPU_CONFIG,
    cpu=8.0,
    memory=32768,  # 32GB RAM
    volumes={
        "/cache": cache_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
    scaledown_window=3600,  # Keep alive for 1 hour after last request (avoid cold starts)
    timeout=86400,  # 24 hours max runtime
    allow_concurrent_inputs=1000,
    min_containers=1,  # Keep 1 container always warm (avoids cold start entirely)
)
@modal.web_server(8000, startup_timeout=1800)  # 30 min for model loading (generous for 32B VL model)
def web():
    """
    Start vLLM's OpenAI-compatible server.
    
    This runs vLLM's native server which provides OpenAI API compatibility.
    """
    import subprocess
    import time
    
    # Parse tensor parallel size from GPU config
    try:
        tensor_parallel_size = int(DEFAULT_GPU_CONFIG.split(":")[-1])
    except (ValueError, IndexError):
        tensor_parallel_size = 1
    
    print("🚀 Starting vLLM OpenAI server")
    print(f"   Model: {DEFAULT_MODEL}")
    print(f"   Tensor parallel size: {tensor_parallel_size}")
    
    # Start vLLM OpenAI server as a subprocess in the background
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", DEFAULT_MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "8192",
        "--enforce-eager",  # Skip torch.compile for faster startup (~2 min vs ~9 min)
        "--enable-prefix-caching",
        "--download-dir", "/cache/huggingface",
        "--trust-remote-code",
    ]
    
    print(f"Starting vLLM: {' '.join(cmd)}")
    # Start vLLM as a background process
    process = subprocess.Popen(cmd)
    
    # Wait for the process to complete (keeps Modal heartbeat alive)
    # Modal will route traffic to port 8000 once it detects the server is listening
    try:
        while True:
            time.sleep(60)  # Sleep in chunks so we can handle signals
            if process.poll() is not None:  # Process has exited
                raise RuntimeError(f"vLLM server exited with code {process.returncode}")
    except KeyboardInterrupt:
        print("Shutting down vLLM server...")
        process.terminate()
        process.wait()
        raise


@app.local_entrypoint()
def main(model: str = DEFAULT_MODEL):
    """
    Deploy the vLLM endpoint to Modal.
    
    Args:
        model: HuggingFace model identifier
        
    Examples:
        modal deploy modal/vllm_endpoint.py
        modal deploy modal/vllm_endpoint.py --model "Qwen/Qwen2.5-7B-Instruct"
        MODAL_GPU_CONFIG="A100-80GB:2" modal deploy modal/vllm_endpoint.py
    """
    print("\n🚀 Deploying vLLM Endpoint")
    print("="*60)
    print(f"Model: {model}")
    print(f"GPU Config: {DEFAULT_GPU_CONFIG}")
    print("="*60)
    
    # Parse GPU config for cost estimation
    if ":" in DEFAULT_GPU_CONFIG:
        gpu_type, gpu_count_str = DEFAULT_GPU_CONFIG.rsplit(":", 1)
        try:
            gpu_count = int(gpu_count_str)
        except ValueError:
            gpu_type = DEFAULT_GPU_CONFIG
            gpu_count = 1
    else:
        gpu_type = DEFAULT_GPU_CONFIG
        gpu_count = 1
    
    # Cost estimation
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
    gpu_cost = next((v for k, v in cost_per_gpu.items() if gpu_type.startswith(k)), 2.10)
    
    print("\n💰 Estimated Cost:")
    print(f"  - Base: ~${gpu_count * gpu_cost:.2f}/hour (while serving requests)")
    print(f"  - Idle: ~${gpu_count * gpu_cost * 0.1:.2f}/hour (5 min idle timeout)")
    print("  - Note: Container scales to zero when idle > 5 minutes")
    
    print("\n📝 After deployment, you'll receive a URL like:")
    print("  https://your-username--vllm-endpoint-web.modal.run")
    
    print("\n🔧 Add to configs/endpoints.py:")
    print('  "qwen-vl-modal": {')
    print(f'      "model": "{model}",')
    print('      "url": "https://YOUR-USERNAME--vllm-endpoint-web.modal.run/v1",')
    print('      "key": "EMPTY",')
    print('  }')
    
    print("\n📊 Then run evaluations locally:")
    print('  uv run vf-eval inoi -m qwen-vl-modal -n -1 -r 4 -a \'{"use_think": false}\' -s -v')
    
    print("\n⚠️  To stop the endpoint:")
    print("  modal app stop vllm-endpoint")
    
    print("\n✅ Deployment instructions complete!")
    print("="*60)


if __name__ == "__main__":
    main()
