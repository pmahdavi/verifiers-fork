#!/bin/bash
# Test script for PrimeRL Hub GRPO training

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}PrimeRL Hub GRPO Training Test Script${NC}"
echo "======================================"

# Check if prime CLI is installed
if ! command -v prime &> /dev/null; then
    echo -e "${RED}Error: prime CLI not found${NC}"
    echo "Please install with: uv tool install prime"
    exit 1
fi

# Function to test environment loading
test_env() {
    local env_id=$1
    local model=${2:-"Qwen/Qwen2.5-0.5B-Instruct"}  # Small model for testing
    
    echo -e "\n${YELLOW}Testing environment: $env_id${NC}"
    
    # Dry run to test configuration
    python examples/grpo/train_primerl_hub.py \
        --env-id "$env_id" \
        --model "$model" \
        --max-steps 2 \
        --eval-steps 1 \
        --num-generations 2 \
        --batch-size 2 \
        --gradient-accumulation-steps 1 \
        --dry-run 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Environment $env_id loaded successfully${NC}"
    else
        echo -e "${RED}✗ Failed to load environment $env_id${NC}"
    fi
}

# Test hyperparameter validation
test_hyperparams() {
    echo -e "\n${YELLOW}Testing hyperparameter validation${NC}"
    
    # Test invalid num_generations
    echo "Testing invalid num_generations (should fail)..."
    python -c "
import sys
sys.path.insert(0, '.')
from examples.grpo.train_primerl_hub import validate_generation_batch_size
from types import SimpleNamespace

args = SimpleNamespace(
    per_device_train_batch_size=8,
    num_generations=7,  # Invalid: doesn't divide 16
    gradient_accumulation_steps=2
)
try:
    validate_generation_batch_size(args, num_processes=1)
    print('FAIL: Should have raised ValueError')
except ValueError as e:
    print('PASS: Correctly rejected invalid num_generations')
    print(f'  Error: {e}')
"
}

# Test environment argument parsing
test_env_args() {
    echo -e "\n${YELLOW}Testing environment argument parsing${NC}"
    
    python -c "
import sys
sys.path.insert(0, '.')

# Test parsing different types
test_args = [
    'num_examples=100',
    'temperature=0.8',
    'use_think=true',
    'use_tools=false'
]

for arg in test_args:
    key, value = arg.split('=', 1)
    # Try to parse as int/float/bool
    try:
        parsed = int(value)
        print(f'✓ {arg} -> int({parsed})')
    except ValueError:
        try:
            parsed = float(value)
            print(f'✓ {arg} -> float({parsed})')
        except ValueError:
            if value.lower() in ['true', 'false']:
                parsed = value.lower() == 'true'
                print(f'✓ {arg} -> bool({parsed})')
            else:
                print(f'✓ {arg} -> str({value})')
"
}

# Show example commands
show_examples() {
    echo -e "\n${YELLOW}Example Commands:${NC}"
    echo ""
    
    cat << 'EOF'
# 1. Basic training with math environment
prime env install primeintellect/math-python
python examples/grpo/train_primerl_hub.py --env-id math-python

# 2. Custom hyperparameters
python examples/grpo/train_primerl_hub.py \
    --env-id wordle \
    --num-generations 16 \
    --batch-size 4 \
    --max-tokens 512 \
    --temperature 0.8

# 3. With environment arguments
python examples/grpo/train_primerl_hub.py \
    --env-id gsm8k \
    --env-args num_train_examples=1000 num_eval_examples=50

# 4. LoRA fine-tuning
python examples/grpo/train_primerl_hub.py \
    --env-id code-debug \
    --use-lora \
    --model meta-llama/Llama-3.2-3B-Instruct

# 5. Large model with optimized settings
python examples/grpo/train_primerl_hub.py \
    --env-id math-python \
    --model meta-llama/Llama-3.2-70B-Instruct \
    --batch-size 1 \
    --gradient-accumulation-steps 32 \
    --num-generations 4
EOF
}

# Main test flow
echo -e "\n${GREEN}Running tests...${NC}"

# Run tests
test_hyperparams
test_env_args

echo -e "\n${YELLOW}Note: Full environment loading tests require:${NC}"
echo "1. Prime CLI authentication (prime login)"
echo "2. Installed environments (prime env install owner/name)"
echo "3. Running vLLM server"

show_examples

echo -e "\n${GREEN}Test script completed!${NC}"