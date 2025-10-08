# INOI Environment

This environment provides access to problems from the Iranian National Olympiad in Informatics (INOI) dataset stored in MongoDB. The dataset contains multimodal problems with text and diagrams, supporting both multiple-choice and yes/no question formats.

## Features

- **Multimodal Support**: Handles problems with text and SVG/image diagrams
- **Bilingual**: Problems include both Persian and English versions
- **Question Types**: Supports multiple-choice (5 options) and yes/no questions
- **MongoDB Integration**: Loads data directly from MongoDB
- **Flexible Filtering**: Filter by exam round, question type, or review status

## Installation

```bash
pip install -e .
```

## Usage

```python
import inoi

# Load environment with default settings
env = inoi.load_environment()

# Load with specific filters
env = inoi.load_environment(
    connection_string="mongodb://localhost:27017/",
    exam_directory="First Round\\10",  # Filter by specific exam
    answer_type="Multiple_Choice",      # or "Yes/No"
    num_train_examples=100,             # Limit training examples
    num_eval_examples=20,               # Limit evaluation examples
    reviewed_only=True,                 # Only use reviewed problems
    use_think=True,                     # Use ThinkParser for chain-of-thought
)
```

## Dataset Structure

The INOI collection in MongoDB contains the following fields:

- `problem`: Problem statement in markdown format
- `choices`: Multiple choice options (for MC questions)
- `correct_option`: Correct answer number (1-5) for MC questions
- `answer_value`: Answer value (varies by question type)
- `answer_type`: Either "Multiple_Choice" or "Yes/No"
- `images_list`: References to images used in the problem
- `persian_solution`: Solution in Persian
- `english_solution`: Solution in English
- `english_solution_local_images`: Solution with localized image paths
- `exam_directory`: Exam round identifier (e.g., "First Round\\10")
- `problem_number`: Problem number within the exam
- `is_reviewed`: Boolean indicating if the problem has been reviewed
- `opedia_url`: External reference URL

## Multimodal Content

Problems may contain images referenced in markdown format:
- Local references: `![](img-0.svg)`
- URL references: `![](https://example.com/image.png)`

The environment automatically processes these images and includes them in the multimodal prompt format.

## Example Problem

```markdown
### Question 3. 

The figure below shows 6 cities and the roads between them. The numbers between consecutive cities indicate the distance between them. We want to build a gas station on a road or in one of the cities such that the sum of the distances from different cities to the gas station, which we call Y, is minimized. What is the integer part of Y?

![](img-0.svg)

1. 63
2. 69  
3. 70
4. 76
5. 92
```

## MongoDB Setup

Ensure MongoDB is running and accessible. The default connection string is `mongodb://localhost:27017/`.

The environment expects:
- Database: `inoi`
- Collection: `inoi`

## Statistics

- Total problems: ~1135
- Multiple Choice: ~1041 
- Yes/No: ~94
- Various exam rounds from First Round 6 to 18+

## Notes

- Image handling currently downloads external images on-the-fly. For production use, consider caching images locally.
- The environment uses the standard `\boxed{}` format for answers.
- For multiple-choice questions, the answer should be the option number (1-5).
- For yes/no questions, the answer should be "Yes" or "No".
