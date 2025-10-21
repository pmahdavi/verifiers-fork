# INOI Environment

Iranian National Olympiad in Informatics (INOI) environment for verifiers. Provides access to **1,135 multimodal math reasoning problems** with **100% image coverage**.

## Features

- **HuggingFace Integration**: Loads dataset directly from `combviz/inoi` (no MongoDB required)
- **100% Image Coverage**: 1,228 embedded images (485 problem images + 743 solution images)
- **Bilingual Solutions**: Short English solutions + detailed explanations
- **Multiple Problem Types**: Multiple-choice, yes/no, open-ended, and context-based questions
- **PIL Images**: All images embedded as PIL Image objects
- **Math Verification**: Supports various answer formats with math-verify
- **Comprehensive Metadata**: Problem types, techniques, answers, and more

## Quick Stats

| Metric | Value |
|--------|-------|
| **Total Problems** | 1,135 |
| **Train Split** | 908 (80%) |
| **Test Split** | 227 (20%) |
| **Problem Images** | 485 PNG files |
| **Solution Images** | 743 PNG files |
| **Total Images** | 1,228 (100% coverage) |
| **Problems with Context** | 227 (20%) |
| **Multimodal Problems** | 543 (47.8%) |

## Installation

```bash
cd environments/inoi
uv pip install -e .
```

## Quick Start

```python
from environments.inoi.inoi import load_environment

# Load from HuggingFace
env = load_environment(
    dataset_name="combviz/inoi",
    num_train_examples=100,
    num_eval_examples=20,
    use_think=True  # Enable chain-of-thought reasoning
)

# Access training data
dataset = env.get_dataset()
for example in dataset:
    prompt = example['prompt']      # Multimodal prompt with PIL Images
    answer = example['answer']       # Ground truth answer
    info = example['info']           # Problem metadata
```

## Dataset Structure

Loaded from HuggingFace: [`combviz/inoi`](https://huggingface.co/datasets/combviz/inoi)

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `combiz_0001`) |
| `problem_type` | string | Type: `mc-standalone`, `mc-dependent`, `yes-no`, `second-round`, etc. |
| `problem` | string | Problem statement (context separated by `---`) |
| `images_list` | list[string] | Problem image filenames |
| `images` | list[PIL.Image] | Problem images (embedded) |
| `solution_short` | string | Concise English solution |
| `solution_images_list` | list[string] | Solution image filenames |
| `solution_images` | list[PIL.Image] | Solution images (embedded) |
| `solution` | string | Detailed step-by-step solution |
| `choices` | string | Multiple choice options (if applicable) |
| `correct_option` | int | Correct option index (for MC) |
| `answer_value` | string | Final answer value |
| `answer_type` | string | Type: `Multiple_Choice`, `Yes/No`, etc. |
| `technique_label` | string | Problem-solving technique (JSON) |
| `exam_directory` | string | Source exam (e.g., `First Round\10`) |
| `problem_number` | int | Original problem number |
| `original_problem_id` | string | MongoDB ObjectID |

### Splits

- **Train**: 908 examples (for model training and development)
- **Test**: 227 examples (for final evaluation and benchmarking)

## Usage Examples

### 1. Basic Example: Load and Explore

```python
from environments.inoi.inoi import load_environment

# Load environment
env = load_environment(
    dataset_name="combviz/inoi",
    num_train_examples=10,
    num_eval_examples=5,
    use_think=True
)

print(f"Training examples: {len(env.get_dataset())}")
print(f"Evaluation examples: {len(env.get_eval_dataset())}")

# Show a sample problem
dataset = env.get_dataset()
example = dataset[0]
print(f"Problem: {example['info']['problem'][:200]}...")
print(f"Answer: {example['answer']}")
```

### 2. Multimodal Filtering

```python
# Load only text-only problems
env_text = load_environment(
    dataset_name="combviz/inoi",
    filter_multimodal=False,  # Keep only text-only
    num_train_examples=100
)

# Load only multimodal problems (with images)
env_multimodal = load_environment(
    dataset_name="combviz/inoi",
    filter_multimodal=True,  # Keep only multimodal
    num_train_examples=100
)

print(f"Text-only: {len(env_text.get_dataset())}")
print(f"Multimodal: {len(env_multimodal.get_dataset())}")
```

### 3. Running Evaluations

```python
import os
from openai import OpenAI
from environments.inoi.inoi import load_environment

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("Set OPENAI_API_KEY environment variable")
    exit(1)

# Load environment
env = load_environment(
    dataset_name="combviz/inoi",
    num_eval_examples=10,
    use_think=True
)

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Run evaluation
results = env.evaluate(
    client=client,
    model="gpt-4o-mini",
    num_examples=5,
    score_rollouts=True
)

print(f"Average reward: {sum(results.reward) / len(results.reward):.2%}")
```

### 4. Dataset Generation

```python
from openai import OpenAI
from environments.inoi.inoi import load_environment

env = load_environment(dataset_name="combviz/inoi", num_eval_examples=10)
client = OpenAI()

# Generate completions with multiple rollouts
results = env.evaluate(
    client=client,
    model="gpt-4o-mini",
    num_examples=5,
    rollouts_per_example=2
)

# Create dataset
dataset = env.make_dataset(
    results,
    rollouts_per_example=2,
    state_columns=[]
)

print(f"Generated {len(dataset)} examples")

# Optionally push to HuggingFace Hub
# dataset.push_to_hub("your-username/inoi-gpt4-mini-results")
```

### 5. Accessing Images

```python
from datasets import load_dataset

# Load dataset directly from HuggingFace
dataset = load_dataset("combviz/inoi")
example = dataset['train'][2]  # combiz_0003

# Access problem images
if len(example['images']) > 0:
    problem_img = example['images'][0]
    problem_img.show()  # Display image

# Access solution images
if len(example['solution_images']) > 0:
    solution_img = example['solution_images'][0]
    solution_img.show()  # Display image

# View short solution
print(example['solution_short'])
```

### 6. Filtering by Type

```python
from datasets import load_dataset

dataset = load_dataset("combviz/inoi")
train = dataset['train']

# Get only multiple choice problems
mc_problems = train.filter(lambda x: 'mc' in x['problem_type'])

# Get problems with images (problem or solution)
multimodal = train.filter(
    lambda x: len(x['images']) > 0 or len(x['solution_images']) > 0
)

# Get open-ended problems (second round)
open_ended = train.filter(lambda x: 'second-round' in x['problem_type'])

print(f"Multiple Choice: {len(mc_problems)}")
print(f"Multimodal: {len(multimodal)}")
print(f"Open-Ended: {len(open_ended)}")
```

### 7. Parsing Context

```python
def parse_problem(problem_text):
    """Parse problem into context and question."""
    if '---' in problem_text:
        parts = problem_text.split('---', 1)
        return {
            'context': parts[0].strip(),
            'question': parts[1].strip()
        }
    else:
        return {
            'context': '',
            'question': problem_text.strip()
        }

# Use with dataset
from datasets import load_dataset
dataset = load_dataset("combviz/inoi")

for example in dataset['train']:
    parsed = parse_problem(example['problem'])
    if parsed['context']:
        print(f"Context: {parsed['context'][:100]}...")
        print(f"Question: {parsed['question'][:100]}...")
```

## Problem Types

| Type | Count | Description |
|------|-------|-------------|
| `mc-standalone` | 490 | Multiple choice, standalone |
| `mc-standalone-img` | 292 | Multiple choice with images |
| `mc-dependent` | 100 | Multiple choice with shared context |
| `mc-dependent-img` | 44 | Context-based + images |
| `second-round` | 83 | Open-ended problems |
| `second-round-img` | 32 | Open-ended with images |
| `yes-no` | 58 | Yes/No questions |
| `yes-no-img` | 36 | Yes/No with images |

## Image Coverage

- **100% coverage**: All 1,228 images present and embedded
- **Problem Images**: 485 PNG files
- **Solution Images**: 743 PNG files
- **Format**: All images converted to PNG
- **Naming**: Standardized convention (`fr{round}_p{num}_{seq}.png`)
- **Embedding**: PIL Image objects for immediate use

## Answer Formats

### Multiple Choice
Model should output: `\boxed{N}` where N is the option number (1-5)

### Yes/No
Model should output: `\boxed{Yes}` or `\boxed{No}`

### Open-Ended
Model should output numerical answer in `\boxed{...}` format

## Scripts

### Convert MongoDB to HuggingFace
```bash
python scripts/convert_mongodb_to_hf.py
```

### Upload to HuggingFace
```bash
python scripts/upload_to_hf.py --repo combviz/inoi --private
```

### Convert SVGs to PNG
```bash
python scripts/simple_browser_convert.py
```

## Dataset URL

🔗 **HuggingFace**: https://huggingface.co/datasets/combviz/inoi

## Project Structure

```
environments/inoi/
├── assets/            # 1,228 PNG images (31 MB)
├── outputs/           # Evaluation results (5.2 MB)
├── data/              # Final dataset (41 MB)
├── scripts/           # Utility scripts (3 files)
├── docs/              # Documentation (7 files)
├── inoi.py            # Main environment
├── pyproject.toml     # Package config
└── README.md          # This file
```

See `docs/DIRECTORY_STRUCTURE.md` for detailed structure guide.

## Documentation

All documentation is in the `docs/` directory:

- `docs/DATASET_CARD.md` - Complete dataset documentation (also on HuggingFace)
- `docs/FINAL_DATASET_STATUS.md` - Comprehensive verification report
- `docs/DATABASE_EXPLORATION_SUMMARY.md` - MongoDB structure reference
- `docs/MONGODB_CONVERSION.md` - Conversion process guide
- `docs/SOLUTION_SHORT_FIX_REPORT.md` - Critical bug fix documentation
- `docs/CLEANUP_SUMMARY.md` - Directory cleanup report
- `docs/DIRECTORY_STRUCTURE.md` - Project structure guide

## License

MIT License - See LICENSE file for details.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{mahdavi2025combigraphvis,
title={CombiGraph-Vis: A Multimodal Olympiad Benchmark for Discrete Mathematical Reasoning},
author={Hamed Mahdavi and Pouria Mahdavinia and Alireza Farhadi and Pegah Mohammadipour and Samira Malek and Pedram Mohammadipour and Majid Daliri and Alireza Hashemi and Amir Khasahmadi and Vasant G. Honavar},
year={2025},
url={https://openreview.net/forum?id=WvH8ZVw3m9}
}
```

---

**Status**: ✅ Production Ready  
**Last Updated**: October 2025  
**Dataset Size**: ~36 MB (with embedded images)
