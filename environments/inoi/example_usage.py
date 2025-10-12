#!/usr/bin/env python3
"""
Example usage of the INOI environment.

Demonstrates:
- Loading from HuggingFace datasets
- Text-only vs. multimodal filtering
- Running evaluations with OpenAI models
- Generating datasets of model responses
"""

import os
from openai import OpenAI

from environments.inoi.inoi import load_environment


def basic_example():
    """Basic example: Load and explore the environment."""
    print("=" * 60)
    print("BASIC EXAMPLE: Loading INOI Environment")
    print("=" * 60)

    # Load environment from HuggingFace
    env = load_environment(
        dataset_name="pxm5426/inoi-dataset",
        num_train_examples=10,
        num_eval_examples=5,
        use_think=True,  # Enable chain-of-thought reasoning
    )

    print(f"\nEnvironment loaded successfully!")
    print(f"Training examples: {len(env.get_dataset())}")
    print(f"Evaluation examples: {len(env.get_eval_dataset())}")

    # Show a sample problem
    dataset = env.get_dataset()
    if len(dataset) > 0:
        print("\n--- Sample Problem ---")
        example = dataset[0]
        prompt = example['prompt']

        if isinstance(prompt, list) and len(prompt) > 0:
            content = prompt[0].get('content', '')

            # Handle both text-only and multimodal content
            if isinstance(content, str):
                # Text-only
                print(content[:500] + "..." if len(content) > 500 else content)
            elif isinstance(content, list):
                # Multimodal
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        print(text[:500] + "..." if len(text) > 500 else text)
                        break

                # Check for images
                image_count = sum(1 for item in content
                                if isinstance(item, dict) and item.get('type') == 'image_url')
                if image_count > 0:
                    print(f"\n[Problem contains {image_count} image(s)]")

        print(f"\nCorrect Answer: {example['answer']}")
        info = example.get('info', {})
        print(f"Problem Type: {info.get('answer_type', 'Unknown')}")
        print(f"Exam: {info.get('exam_directory', 'Unknown')}")


def multimodal_filtering_example():
    """Example: Filter for text-only or multimodal problems."""
    print("\n" + "=" * 60)
    print("MULTIMODAL FILTERING EXAMPLE")
    print("=" * 60)

    # Load text-only problems
    env_text = load_environment(
        dataset_name="pxm5426/inoi-dataset",
        filter_multimodal=False,  # Keep only text-only
        num_train_examples=100,
    )
    print(f"Text-only examples: {len(env_text.get_dataset())}")

    # Load multimodal problems
    env_multimodal = load_environment(
        dataset_name="pxm5426/inoi-dataset",
        filter_multimodal=True,  # Keep only multimodal
        num_train_examples=100,
    )
    print(f"Multimodal examples: {len(env_multimodal.get_dataset())}")


def evaluation_example():
    """Example: Run model evaluation (requires OpenAI API key)."""
    print("\n" + "=" * 60)
    print("EVALUATION EXAMPLE")
    print("=" * 60)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping evaluation example (OPENAI_API_KEY not set)")
        print("Set OPENAI_API_KEY environment variable to run this example")
        return

    # Load environment
    env = load_environment(
        dataset_name="pxm5426/inoi-dataset",
        num_eval_examples=5,
        use_think=True,
    )

    # Initialize client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Run evaluation
    print("\nRunning evaluation on 3 examples with gpt-4o-mini...")
    results = env.evaluate(
        client=client,
        model="gpt-4o-mini",
        num_examples=3,
        score_rollouts=True,
    )

    print(f"\nEvaluation Results:")
    print(f"  Examples evaluated: {len(results.reward)}")
    print(f"  Average reward: {sum(results.reward) / len(results.reward):.2%}")

    # Show first example
    if len(results.completion) > 0:
        print(f"\nFirst Example:")
        print(f"  Reward: {results.reward[0]}")
        completion = results.completion[0]
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get('content', '')
            print(f"  Completion: {content[:200]}...")


def dataset_generation_example():
    """Example: Generate a dataset of model responses."""
    print("\n" + "=" * 60)
    print("DATASET GENERATION EXAMPLE")
    print("=" * 60)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping dataset generation (OPENAI_API_KEY not set)")
        return

    # Load environment
    env = load_environment(
        dataset_name="pxm5426/inoi-dataset",
        num_eval_examples=10,
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Generate completions with multiple rollouts per example
    print("\nGenerating 2 completions for each of 5 problems...")
    results = env.evaluate(
        client=client,
        model="gpt-4o-mini",
        num_examples=5,
        rollouts_per_example=2,
    )

    # Create dataset
    dataset = env.make_dataset(
        results,
        rollouts_per_example=2,
        state_columns=[],
    )

    print(f"\nGenerated dataset:")
    print(f"  Total examples: {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Average reward: {sum(dataset['reward']) / len(dataset['reward']):.2%}")

    # Optionally push to HuggingFace Hub
    # dataset.push_to_hub("your-username/inoi-gpt4-mini-results")
    print("\nTo upload to HuggingFace Hub, uncomment the last line")


def main():
    """Run all examples."""
    print("\nINOI Environment Examples\n")

    # Run examples
    try:
        basic_example()
    except Exception as e:
        print(f"Basic example failed: {e}")

    try:
        multimodal_filtering_example()
    except Exception as e:
        print(f"Multimodal filtering example failed: {e}")

    try:
        evaluation_example()
    except Exception as e:
        print(f"Evaluation example failed: {e}")

    try:
        dataset_generation_example()
    except Exception as e:
        print(f"Dataset generation example failed: {e}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
