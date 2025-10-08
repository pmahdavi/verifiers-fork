#!/usr/bin/env python3
"""Test script for INOI environment."""

import sys
import json
from inoi import load_environment


def test_basic_loading():
    """Test basic environment loading."""
    print("Testing basic environment loading...")
    try:
        env = load_environment(
            num_train_examples=5,
            num_eval_examples=2,
            reviewed_only=True
        )
        print(f"✓ Environment loaded successfully")
        print(f"  - Training examples: {len(env.dataset)}")
        print(f"  - Evaluation examples: {len(env.eval_dataset)}")
        return True
    except Exception as e:
        print(f"✗ Failed to load environment: {e}")
        return False


def test_multiple_choice():
    """Test loading only multiple choice questions."""
    print("\nTesting multiple choice questions...")
    try:
        env = load_environment(
            answer_type="Multiple_Choice",
            num_train_examples=3,
            num_eval_examples=1
        )
        print(f"✓ Multiple choice environment loaded")
        
        # Check first example
        if len(env.dataset) > 0:
            example = env.dataset[0]
            print(f"  - Example prompt type: {type(example['prompt'])}")
            print(f"  - Example answer: {example['answer']}")
            print(f"  - Answer type: {example.get('answer_type', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Failed to load MC environment: {e}")
        return False


def test_yes_no():
    """Test loading only yes/no questions."""
    print("\nTesting yes/no questions...")
    try:
        env = load_environment(
            answer_type="Yes/No",
            num_train_examples=3,
            num_eval_examples=1
        )
        print(f"✓ Yes/No environment loaded")
        print(f"  - Training examples: {len(env.dataset)}")
        return True
    except Exception as e:
        print(f"✗ Failed to load Yes/No environment: {e}")
        return False


def test_exam_filtering():
    """Test filtering by exam directory."""
    print("\nTesting exam directory filtering...")
    try:
        env = load_environment(
            exam_directory="First Round\\10",
            num_train_examples=10
        )
        print(f"✓ Filtered environment loaded")
        print(f"  - Examples from First Round\\10: {len(env.dataset)}")
        return True
    except Exception as e:
        print(f"✗ Failed to filter by exam: {e}")
        return False


def test_prompt_format():
    """Test the prompt formatting."""
    print("\nTesting prompt format...")
    try:
        env = load_environment(num_train_examples=1)
        if len(env.dataset) > 0:
            example = env.dataset[0]
            prompt = example['prompt']
            
            print(f"✓ Prompt structure:")
            print(f"  - Type: {type(prompt)}")
            print(f"  - Length: {len(prompt)}")
            if isinstance(prompt, list) and len(prompt) > 0:
                print(f"  - First message role: {prompt[0].get('role', 'N/A')}")
                content = prompt[0].get('content', [])
                print(f"  - Content items: {len(content) if isinstance(content, list) else 'Not a list'}")
                
                # Check for multimodal content
                has_text = any(item.get('type') == 'text' for item in content if isinstance(item, dict))
                has_images = any(item.get('type') == 'image_url' for item in content if isinstance(item, dict))
                print(f"  - Has text: {has_text}")
                print(f"  - Has images: {has_images}")
        return True
    except Exception as e:
        print(f"✗ Failed to test prompt format: {e}")
        return False


def main():
    """Run all tests."""
    print("=== INOI Environment Test Suite ===\n")
    
    tests = [
        test_basic_loading,
        test_multiple_choice,
        test_yes_no,
        test_exam_filtering,
        test_prompt_format
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("All tests passed! ✓")
        return 0
    else:
        print(f"Some tests failed. ({len(tests) - passed} failures)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
