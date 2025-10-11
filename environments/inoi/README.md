# INOI Environment

Iranian National Olympiad in Informatics (INOI) environment for verifiers. Provides access to 1135 multimodal math reasoning problems with 99.7% image coverage.

## Features

- **HuggingFace Integration**: Loads dataset directly from HuggingFace Hub (no MongoDB required)
- **Multimodal Support**: 332 problems with images (99.7% coverage: 331/332)
- **Bilingual**: Problems in both Persian and English
- **Question Types**: Multiple-choice (5 options) and yes/no questions
- **PIL Images**: Images automatically decoded as PIL Images via HF Image feature
- **Math Verification**: Uses math-verify for expression equivalence checking

## Installation

```bash
cd environments/inoi
uv pip install -e .
```

## Quick Start

```python
import inoi

# Load from HuggingFace (no MongoDB needed!)
env = inoi.load_environment()

# Access training data
for example in env.train_dataset:
    prompt = example['prompt']      # Multimodal prompt with PIL Images
    answer = example['answer']       # Ground truth answer
    info = example['info']           # Problem metadata

# Access evaluation data
for example in env.eval_dataset:
    # Test your model
    completion = your_model(example['prompt'])
    reward = env.reward_fn(completion, example['answer'], example['info'])
```

## Dataset Structure

Loaded from HuggingFace: `pmahdavi/inoi`

### Fields

- `problem`: Problem statement (markdown)
- `choices`: Multiple choice options (for MC questions)
- `answer_type`: "mc-stand", "mc-dep", or "yes-no"
- `correct_option`: Correct option number (1-5)
- `answer_value`: Ground truth answer
- `images`: List of PIL Images (automatically decoded by HF)
- `images_list`: Original image filenames
- `exam_directory`: Exam identifier (e.g., "First Round/10")
- `problem_number`: Problem number
- `is_reviewed`: Review status

### Splits

- **Train**: 908 examples (267 with images)
- **Test**: 227 examples (64 with images)

## Multimodal Content

Images are provided as PIL Images in the `images` field:

```python
example = env.train_dataset[0]
pil_images = example['images']  # List of PIL.Image.Image objects

# Images are automatically converted to base64 for OpenAI API format
prompt = inoi.format_prompt(example)
# Returns: [{"role": "user", "content": [
#   {"type": "text", "text": "..."},
#   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
# ]}]
```

## Example Problem

```markdown
### Question 3.

The figure below shows 6 cities and the roads between them. The numbers
indicate distances. Where should a gas station be built to minimize the
sum of distances Y from all cities? What is the integer part of Y?

[Image: Graph diagram]

1. 63
2. 69
3. 70
4. 76
5. 92
```

**Expected Answer**: `\boxed{2}` (option number)

## Image Coverage

- **Total multimodal problems**: 332
- **With valid images**: 331 (99.7%)
- **Missing**: 1 image (`First Round/7/56.svg` - not in source data)

All images are pre-converted to PNG format and quality-verified (no all-black/all-white images).

## SVG Conversion

SVG images were converted to PNG using:
- **CairoSVG**: 325 images (78.3%)
- **Browser (Chromium)**: 64 images (16.5%) - used for complex SVGs that cairosvg couldn't render

See `SVG_CONVERSION_STATUS.md` for details on the conversion process and how to reproduce.

## Statistics

- **Total problems**: 1,135
- **Multiple Choice**: 1,041 (91.7%)
- **Yes/No**: 94 (8.3%)
- **Multimodal**: 332 (29.2%)
- **Train/Test split**: 908/227 (80%/20%)

## Answer Formats

### Multiple Choice
Model should output: `\boxed{N}` where N is 1-5

### Yes/No
Model should output: `\boxed{Yes}` or `\boxed{No}`

## Advanced Usage

```python
# Filter by answer type
env = inoi.load_environment(
    num_train_examples=100,
    num_eval_examples=20,
    answer_type="Multiple_Choice",  # or "Yes/No"
    reviewed_only=True,
)

# Use with verifiers
import verifiers as vf

env = inoi.load_environment()
result = vf.sample(
    env,
    model="openai:gpt-4",
    n_train_examples=5,
    n_eval_examples=10,
)
```

## Scripts

### Upload to HuggingFace
```bash
python prepare_and_upload_hf.py
```

### Convert SVGs to PNG
```bash
python simple_browser_convert.py
```

## Dataset URL

https://huggingface.co/datasets/pmahdavi/inoi

## License

See dataset card on HuggingFace for license information.

## Citation

If you use this dataset, please cite the original INOI competition and the dataset creators.
