# MongoDB to HuggingFace Conversion Guide

This document describes how to convert the INOI MongoDB database to HuggingFace format.

## Overview

The `convert_mongodb_to_hf.py` script:
1. Connects to MongoDB and loads `inoi` + `synthetic_data` collections
2. Joins collections via `problem_id` cross-reference
3. Processes and unifies image references
4. Creates HuggingFace dataset compatible with `inoi.py` environment

## Installation

```bash
pip install -r requirements_converter.txt
```

## MongoDB Configuration

Update the connection string in `convert_mongodb_to_hf.py`:

```python
MONGODB_URI = "mongodb://your-host:port/"
# or for MongoDB Atlas:
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/"
```

## Usage

```bash
cd /scratch/pxm5426/repos/verifiers-fork/environments/inoi
python convert_mongodb_to_hf.py
```

## Output Files

The script generates:

1. **`inoi_hf_dataset/`** - HuggingFace dataset directory
   - `train` split: 908 examples (80%)
   - `test` split: 227 examples (20%)

2. **`inoi_dataset_preview.csv`** - CSV preview for inspection

3. **`image_mapping.txt`** - Complete mapping of renamed images

## Dataset Schema

The output dataset has these columns (compatible with existing `inoi.py`):

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `id` | string | Generated | Unique ID: `combiz_0001`, `combiz_0002`, ... |
| `problem` | string | `inoi.problem` + `inoi.context` | Combined problem statement |
| `images_list` | list[string] | Processed | Renamed image filenames |
| `solution` | string | `synthetic_data.rewritten_solution` | Solution text |
| `technique_label` | string | `synthetic_data.solution_technique_data.overall_classification` | Solution technique |
| `problem_type` | string | Derived | `yes-no`, `mc-standalone`, `mc-dependent`, `second-round` (+ `-img`) |
| `choices` | string | `inoi.choices` | Answer choices |
| `correct_option` | int | `inoi.correct_option` | Correct answer number (1-5) |
| `answer_value` | string | `inoi.answer_value` | Answer value |
| `answer_type` | string | `inoi.answer_type` | `Multiple_Choice` or `Yes/No` |
| `exam_directory` | string | `inoi.exam_directory` | Exam identifier |
| `problem_number` | int | `inoi.problem_number` | Problem number |
| `original_problem_id` | string | `inoi._id` | Original MongoDB ObjectId |

## Image Processing

### Unified Naming Convention

Images are renamed from original names to standardized format:

**Format:** `{round_type}{round_num}_p{problem_num}_{sequence}.{ext}`

**Examples:**
```
Original: img-0.svg     → New: fr10_p31_0.svg
Original: q7_1.png      → New: fr15_p7_0.png
Original: 3.png         → New: sr25_p5_0.png
Original: 13.svg        → New: fr29_p18_1.svg
```

Where:
- `fr` = First Round, `sr` = Second Round
- `10` = Round number
- `p31` = Problem 31
- `0` = Image sequence (0, 1, 2, ...)

### HTML to Markdown Conversion

The script automatically converts HTML image tags to Markdown:

```html
<!-- Before -->
<img src="13.svg" style="width:45%;">

<!-- After -->
![](13.svg)
```

### Fixed Problems

The script handles these problematic cases:

1. **HTML img tags** (Problems 18-20, First Round 29) → Converted to Markdown
2. **Duplicate references** → Deduplicated
3. **Missing semicolons** → Properly parsed

## Context + Problem Combination

If a problem has a `context` field, it's combined with the problem:

```
{context}

---

{problem}
```

The `---` separator provides clear visual distinction.

## Problem Type Classification

The `problem_type` field is determined by:

1. **Second Round**: If `exam_directory` contains "Second" → `second-round`
2. **Yes/No**: If `answer_type` == "Yes/No" → `yes-no`
3. **Multiple Choice**:
   - If `choice_dependency_data.label` == "standalone" → `mc-standalone`
   - If `choice_dependency_data.label` == "choice-dependent" → `mc-dependent`
4. **Image Suffix**: Appends `-img` if problem has images

**Result Examples:**
- `yes-no`
- `mc-standalone-img`
- `mc-dependent`
- `second-round-img`

## Cross-Collection Join

The script joins collections using:
```
inoi._id ↔ synthetic_data.problem_id
```

If a problem doesn't have synthetic data, it's skipped with a warning.

## Expected Statistics

For the complete dataset (1,135 problems):

```
Total problems: 1,135
Problems with images: ~404 (35.6%)
Problems with context: ~335 (29.5%)

Problem Type Distribution:
  mc-standalone:       ~650
  mc-standalone-img:   ~250
  mc-dependent:        ~157
  yes-no:              ~40
  second-round:        ~38
```

## Integration with inoi.py

After conversion, the dataset can be used with the existing environment:

```python
from datasets import load_from_disk

# Load converted dataset
dataset = load_from_disk('inoi_hf_dataset')

# Upload to HuggingFace
dataset.push_to_hub('your-username/inoi')

# Use with inoi.py
from environments.inoi import load_environment
env = load_environment(dataset_name='your-username/inoi')
```

## Troubleshooting

### Missing Synthetic Data

If you see warnings about missing synthetic data:
```
Warning: No synthetic data for problem ObjectId(...)
```

This means some problems in `inoi` don't have corresponding entries in `synthetic_data`. These problems are skipped.

### MongoDB Connection Issues

1. Verify the connection string
2. Check network/firewall settings
3. Ensure MongoDB is running
4. Verify credentials if using authentication

### Processing Errors

Check the full error traceback. Common issues:
- Missing required fields
- Malformed image references
- Unexpected data types

## Field Mapping Reference

### From inoi collection:
- `_id` → `original_problem_id`
- `problem` → Part of combined `problem` field
- `context` → Part of combined `problem` field (if exists)
- `choices` → `choices`
- `correct_option` → `correct_option`
- `answer_value` → `answer_value`
- `answer_type` → `answer_type`
- `exam_directory` → `exam_directory`
- `problem_number` → `problem_number`
- `images_list` → Processed → `images_list`

### From synthetic_data collection:
- `rewritten_solution` → `solution`
- `solution_technique_data.overall_classification` → `technique_label`
- `choice_dependency_data.label` → Used in `problem_type` derivation

### Generated fields:
- `id` = `combiz_{sequence:04d}`
- `problem_type` = Derived from multiple fields

## Validation

Before converting, you can validate your MongoDB data structure using MongoDB MCP tools or a validation script.

Expected collections:
- `inoi`: 1,135 documents
- `synthetic_data`: 1,135 documents (matched via `problem_id`)

## Next Steps After Conversion

1. **Review the CSV**: `inoi_dataset_preview.csv`
2. **Check image mapping**: `image_mapping.txt`
3. **Load and test**:
   ```python
   from datasets import load_from_disk
   dataset = load_from_disk('inoi_hf_dataset')
   print(dataset)
   ```
4. **Upload to HuggingFace**:
   ```python
   dataset.push_to_hub('your-username/inoi')
   ```
5. **Update prepare_and_upload_hf.py**: Point it to your new dataset
6. **Add PIL Images**: Run `prepare_and_upload_hf.py` to add images from assets/

## Notes

- The script expects 1,135 total problems
- Images are renamed but NOT moved - you'll need to rename actual image files separately
- The `images_list` field contains the NEW image names
- All image references in the `problem` field are updated to new names
- Compatible with existing `inoi.py` environment code

---

**Last Updated:** Based on MongoDB structure analysis from 2024
