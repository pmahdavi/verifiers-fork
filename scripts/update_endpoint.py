#!/usr/bin/env python3
"""
Update endpoints.py configuration for a vLLM server.

Usage:
    python scripts/update_endpoint.py <model_name> [--port PORT] [--name ENDPOINT_NAME]

Examples:
    python scripts/update_endpoint.py Qwen/Qwen2.5-3B-Instruct
    python scripts/update_endpoint.py Qwen/Qwen2.5-3B-Instruct --port 8001
    python scripts/update_endpoint.py Qwen/Qwen2.5-3B-Instruct --name my-model --port 8002
"""

import argparse
import re
import sys
from pathlib import Path


def update_endpoints(model_name: str, port: int = 8000, endpoint_name: str = "local-vllm"):
    """Update or add an endpoint in endpoints.py."""
    
    # Find endpoints.py
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    endpoints_file = repo_root / "configs" / "endpoints.py"
    
    if not endpoints_file.exists():
        print(f"Error: {endpoints_file} not found", file=sys.stderr)
        return False
    
    # Read current content
    content = endpoints_file.read_text()
    
    # Check if endpoint exists
    endpoint_exists = f'"{endpoint_name}":' in content
    
    if endpoint_exists:
        # Update existing endpoint - update model name
        # More flexible pattern that handles different whitespace/formatting
        pattern = rf'("{endpoint_name}":\s*\{{[^}}]*"model":\s*")[^"]*(")'
        content = re.sub(pattern, rf'\g<1>{model_name}\g<2>', content, flags=re.DOTALL)
        
        # Update URL/port
        url_pattern = rf'("{endpoint_name}":\s*\{{[^}}]*"url":\s*"http://[^:]*:)\d+(/v1")'
        content = re.sub(url_pattern, rf'\g<1>{port}\g<2>', content, flags=re.DOTALL)
        
        print(f"✓ Updated endpoint '{endpoint_name}':")
    else:
        # Add new endpoint before the closing brace
        new_endpoint = f'''    "{endpoint_name}": {{
        "model": "{model_name}",
        "url": "http://0.0.0.0:{port}/v1",
        "key": "EMPTY",
    }},
}}
'''
        # Remove trailing brace and whitespace, add new endpoint
        content = content.rstrip().rstrip('}').rstrip() + '\n' + new_endpoint
        
        print(f"✓ Added new endpoint '{endpoint_name}':")
    
    # Write back
    endpoints_file.write_text(content)
    
    print(f"  Model: {model_name}")
    print(f"  URL: http://0.0.0.0:{port}/v1")
    print(f"\nYou can now use this endpoint with:")
    print(f"  uv run vf-eval <environment> -m {endpoint_name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update endpoints.py configuration for a vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Qwen/Qwen2.5-3B-Instruct
  %(prog)s Qwen/Qwen2.5-3B-Instruct --port 8001
  %(prog)s Qwen/Qwen2.5-3B-Instruct --name my-model --port 8002

Common models:
  - Qwen/Qwen2.5-3B-Instruct
  - Qwen/Qwen3-VL-8B-Instruct
  - google/gemma-2-9b-it
  - meta-llama/Llama-3.1-8B-Instruct
  - willcb/DeepSeek-R1-Distill-Qwen-1.5B
        """
    )
    
    parser.add_argument("model", help="Model name (e.g., Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--name", default="local-vllm", help="Endpoint name (default: local-vllm)")
    
    args = parser.parse_args()
    
    success = update_endpoints(args.model, args.port, args.name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

