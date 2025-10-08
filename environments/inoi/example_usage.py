#!/usr/bin/env python3
"""Example usage of the INOI environment with MongoDB data."""

import inoi
from verifiers.utils import set_seed


def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    print("Loading INOI Environment from MongoDB...")
    
    # Load environment with specific configuration
    env = inoi.load_environment(
        connection_string="mongodb://localhost:27017/",  # MongoDB connection
        exam_directory=None,  # Load from all exam directories
        answer_type="Multiple_Choice",  # Focus on multiple choice questions
        num_train_examples=100,  # Use 100 training examples
        num_eval_examples=20,   # Use 20 evaluation examples  
        reviewed_only=True,     # Only use reviewed problems
        use_think=True,         # Enable chain-of-thought reasoning
    )
    
    print(f"\nEnvironment loaded successfully!")
    print(f"Training examples: {len(env.dataset)}")
    print(f"Evaluation examples: {len(env.eval_dataset)}")
    
    # Show a sample problem
    if len(env.dataset) > 0:
        print("\n--- Sample Problem ---")
        example = env.dataset[0]
        prompt = example['prompt']
        
        if isinstance(prompt, list) and len(prompt) > 0:
            content = prompt[0].get('content', [])
            
            # Extract text content
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text = item.get('text', '')
                    # Show first 500 characters
                    print(text[:500] + "..." if len(text) > 500 else text)
                    break
            
            # Check for images
            image_count = sum(1 for item in content 
                            if isinstance(item, dict) and item.get('type') == 'image_url')
            if image_count > 0:
                print(f"\n[Problem contains {image_count} image(s)]")
        
        print(f"\nCorrect Answer: {example['answer']}")
        print(f"Problem Type: {example.get('answer_type', 'Unknown')}")
        print(f"Exam: {example.get('exam_directory', 'Unknown')}")
    
    # Example: Train a model (pseudo-code)
    print("\n--- Example Training Loop ---")
    print("# Initialize your model")
    print("model = YourModel()")
    print()
    print("# Training loop")
    print("for epoch in range(num_epochs):")
    print("    for batch in env.dataset:")
    print("        # Forward pass with multimodal prompt")
    print("        output = model(batch['prompt'])")
    print("        ")
    print("        # Parse answer using environment's parser")
    print("        parsed_answer = env.parser.parse_answer(output)")
    print("        ")
    print("        # Calculate reward using environment's rubric")
    print("        reward = env.rubric.score(output, batch['answer'])")
    print("        ")
    print("        # Update model based on reward")
    print("        model.update(reward)")
    
    # Show statistics by exam directory
    print("\n--- Dataset Statistics ---")
    exam_stats = {}
    for item in env.dataset:
        exam = item.get('exam_directory', 'Unknown')
        exam_stats[exam] = exam_stats.get(exam, 0) + 1
    
    print("Problems by exam directory:")
    for exam, count in sorted(exam_stats.items()):
        print(f"  {exam}: {count} problems")


if __name__ == "__main__":
    main()
