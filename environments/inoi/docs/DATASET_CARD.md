---
license: mit
task_categories:
- question-answering
- multiple-choice
language:
- en
size_categories:
- 1K<n<10K
tags:
- math
- olympiad
- problem-solving
- inoi
- persian
- multimodal
pretty_name: INOI Math Olympiad Problems
---

# INOI Math Olympiad Dataset

## Dataset Description

This dataset contains **1,135 math problems** from the **Iranian National Olympiad in Informatics (INOI)**, spanning multiple competition rounds from 2006-2024. Each problem includes the original problem statement, detailed solution, and associated images.

### Key Features

- 🎯 **1,135 curated problems** with full solutions
- 📊 **Train/Test split**: 908 / 227 examples
- 🖼️ **1,228 embedded images** (100% coverage)
- 📝 **Multiple problem types**: Multiple choice, open-ended, context-based
- ✅ **High-quality solutions** with step-by-step explanations
- 🔢 **Rich metadata**: Problem types, answers, techniques
- 🌐 **Bilingual solutions**: English short solutions and detailed solutions

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Problems** | 1,135 |
| **Train Split** | 908 (80%) |
| **Test Split** | 227 (20%) |
| **Problem Images** | 485 PNG files |
| **Solution Images** | 743 PNG files |
| **Total Images** | 1,228 PNG files (100% coverage) |
| **Avg Problem Length** | ~500 characters |
| **Avg Solution Length** | ~2,800 characters |
| **Solution Short Coverage** | 100% (all problems) |

## Problem Types

The dataset includes diverse problem formats:

| Type | Count | Percentage |
|------|-------|------------|
| **Multiple Choice (Standalone)** | 490 | 43.2% |
| **Multiple Choice (with Images)** | 292 | 25.7% |
| **Multiple Choice (Context-based)** | 100 | 8.8% |
| **Context + Image Problems** | 44 | 3.9% |
| **Second Round Problems** | 83 | 7.3% |
| **Second Round (with Images)** | 32 | 2.8% |
| **Yes/No Questions** | 58 | 5.1% |
| **Yes/No (with Images)** | 36 | 3.2% |

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique problem identifier (e.g., `combiz_0003`) |
| `problem_type` | str | `'original'` or `'synthetic'` |
| `problem` | str | Problem statement (context + problem, separated by `---` if context exists) |
| `images_list` | List[str] | Filenames of problem images |
| `images` | List[PIL.Image] | Embedded problem images |
| `solution_short` | str | Concise English solution (100% coverage) |
| `solution_images_list` | List[str] | Filenames of solution images |
| `solution_images` | List[PIL.Image] | Embedded solution images |
| `solution` | str | Full rewritten solution with detailed explanation |
| `choices` | List[str] | Multiple choice options (if applicable) |
| `correct_option` | str | Correct answer letter (if multiple choice) |
| `answer_value` | str | Expected answer value |
| `answer_type` | str | Type of answer expected |
| `technique_label` | str | Problem-solving technique category |
| `exam_directory` | str | Source exam (e.g., `'First Round\\10'`) |
| `problem_number` | int | Problem number within exam |
| `original_problem_id` | str | MongoDB ObjectId reference |

### Key Features

- **Context Separation**: Problems with context use `---` separator between context and question (227 problems)
- **Image Lists**: Separate `*_list` fields provide filenames for easy reference
- **Image Embedding**: All images embedded as PIL Image objects for immediate display
- **Bilingual Solutions**: Many problems have both English (`solution_short`) and detailed (`solution`) versions

### Data Splits

| Split | Examples | Use Case |
|-------|----------|----------|
| **Train** | 908 | Model training and development |
| **Test** | 227 | Final evaluation and benchmarking |

## Usage

### Basic Loading

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("combviz/inoi")

print(f"Train: {len(dataset['train'])} problems")  # 908
print(f"Test: {len(dataset['test'])} problems")    # 227
```

### Accessing Problems with Images

```python
# Get a problem with images
record = dataset['train'][2]  # combiz_0003

print(f"Problem ID: {record['id']}")
print(f"Problem: {record['problem'][:200]}...")

# Access problem images
print(f"\nProblem images: {record['images_list']}")  # ['fr10_p3_0.png']
if record['images']:
    record['images'][0].show()  # Display problem image

# Access short solution
print(f"\nShort solution: {record['solution_short'][:200]}...")

# Access solution images
print(f"\nSolution images: {record['solution_images_list']}")
for i, img in enumerate(record['solution_images']):
    print(f"Solution image {i}: {img.size}")
    # img.show()  # Uncomment to display

# Access full solution
print(f"\nFull solution: {record['solution'][:200]}...")
```

### Filtering by Problem Type

```python
# Get problems with multiple choice
mc_problems = [r for r in dataset['train'] if r['choices']]
print(f"Multiple choice problems: {len(mc_problems)}")

# Get problems with images
image_problems = [r for r in dataset['train'] if r['images_list']]
print(f"Problems with images: {len(image_problems)}")

# Get problems with context
context_problems = [r for r in dataset['train'] if '---' in r['problem']]
print(f"Problems with context: {len(context_problems)}")
```

### Working with Images

```python
import numpy as np
from PIL import Image

# Access problem with images
record = dataset['train'][2]

# Images are already PIL Image objects
for i, img in enumerate(record['images']):
    print(f"Image {i}:")
    print(f"  Size: {img.size}")
    print(f"  Mode: {img.mode}")
    
    # Convert to numpy if needed
    img_array = np.array(img)
    print(f"  Array shape: {img_array.shape}")
```

### Filter by Exam Round

```python
# Get all First Round 10 problems
fr10_problems = [
    r for r in dataset['train'] 
    if 'First Round' in r['exam_directory'] and '\\10' in r['exam_directory']
]
print(f"First Round 10: {len(fr10_problems)} problems")

# Get all Second Round problems
sr_problems = [
    r for r in dataset['train']
    if 'Second Round' in r['exam_directory']
]
print(f"Second Round: {len(sr_problems)} problems")
```

## Use Cases

### 1. Math Problem Solving
Train models to solve competitive math olympiad problems.

### 2. Solution Generation
Generate detailed step-by-step solutions for math problems.

### 3. Multimodal Reasoning
Develop vision-language models that can interpret diagrams and solve problems.

### 4. Answer Verification
Build verifiers to assess correctness of generated solutions.

### 5. Difficulty Classification
Classify problem difficulty based on olympiad round and type.

### 6. Educational AI
Create tutoring systems that explain solutions interactively.

## Data Collection & Processing

### Source
Problems were collected from the Iranian National Olympiad in Informatics (INOI) archives, spanning competitions from 2006-2024.

### Conversion Pipeline
1. **Extraction**: Problems extracted from MongoDB database
2. **Image Processing**: 
   - SVG to PNG conversion using `cairosvg`
   - Browser-based rendering for malformed SVGs (35 files)
   - Standardized naming convention: `{round}_p{num}_{seq}.png`
3. **Text Processing**: Markdown formatting with image reference updates
4. **Quality Assurance**: Manual verification of solutions and image references
5. **Standardization**: Unified schema and consistent formatting

### Image Coverage
- **100% coverage**: All 1,228 images present and embedded
  - 485 problem images (406 problems have images)
  - 743 solution images (495 problems have solutions)
- **Format**: PNG (all SVGs converted using cairosvg and browser-based rendering)
- **Naming**: Standardized convention (`fr{round}_p{num}_{seq}.png` for problems, `fr{round}_p{num}_sol{seq}.png` for solutions)
- **Embedding**: All images embedded as PIL Image objects for immediate viewing

## Data Quality

### Strengths
✅ Complete problem statements and solutions  
✅ 100% image coverage with all images embedded  
✅ Rich metadata and problem categorization  
✅ Verified answers and explanations  
✅ Diverse problem types and difficulties  
✅ Bilingual solutions (English and detailed versions)  

### Limitations
⚠️ 5 solutions use Persian/Arabic characters as symbolic notation  
⚠️ Problem difficulty not explicitly labeled  
⚠️ Alt text in images shows original filenames (for provenance)  

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

## License

This dataset is released under the MIT License.

---

**Status**: ✅ Production Ready  
**Last Updated**: October 2025  
**Dataset Size**: ~36 MB (with embedded images)  
**Image Coverage**: 100% (1,228/1,228 images)
