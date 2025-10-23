#!/usr/bin/env python3
"""
Quick test script to verify vLLM server is running and accessible.
Usage: python test_vllm_connection.py [host] [port]
"""
import sys
from openai import OpenAI

def test_connection(host="0.0.0.0", port=8000):
    url = f"http://{host}:{port}/v1"
    
    print(f"Testing connection to vLLM server at {url}")
    print("-" * 60)
    
    try:
        client = OpenAI(
            base_url=url,
            api_key="EMPTY"
        )
        
        # Test with a simple completion
        print("Sending test request...")
        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",  # Must match the model loaded in vLLM
            messages=[
                {"role": "user", "content": "Say 'Hello, I am working!' and nothing else."}
            ],
            max_tokens=20,
            temperature=0.0
        )
        
        print("\n✅ Connection successful!")
        print(f"Model: {response.model}")
        print(f"Response: {response.choices[0].message.content}")
        print("-" * 60)
        return True
        
    except Exception as e:
        print("\n❌ Connection failed!")
        print(f"Error: {str(e)}")
        print("-" * 60)
        print("\nTroubleshooting:")
        print("1. Is the vLLM server running? (Check PBS job status: qstat)")
        print("2. Are you on the correct node? (Check PBS output file)")
        print("3. Is the port correct? (Default: 8000)")
        return False

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    test_connection(host, port)

