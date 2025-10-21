# INOI MongoDB Database Exploration Summary

**Date:** October 21, 2025  
**Database:** `inoi` (MongoDB Cluster)  
**Total Size:** ~973 MB

This document summarizes the complete exploration and analysis of the INOI MongoDB database, including structure, statistics, and conversion requirements.

---

## Table of Contents

1. [Database Overview](#database-overview)
2. [Collection Structures](#collection-structures)
3. [Field Descriptions](#field-descriptions)
4. [Cross-Collection Relationships](#cross-collection-relationships)
5. [Data Statistics](#data-statistics)
6. [Image Reference Analysis](#image-reference-analysis)
7. [Problematic Cases](#problematic-cases)
8. [Conversion Strategy](#conversion-strategy)

---

## Database Overview

### MongoDB Cluster Structure

**Total Databases:** 5
- `AoPS` - ~487 MB
- **`inoi`** - ~973 MB ⭐ (our focus)
- `olympiad_solvers` - ~1.6 MB
- `admin` - ~380 KB (system)
- `local` - ~4.2 GB (system)

### Collections in `inoi` Database

**3 collections total:**
1. **`inoi`** - Original INOI problems (5.94 MB, 1,135 documents)
2. **`synthetic_data`** - AI-generated solutions & analysis (2.59 GB, 1,135 documents)
3. `mini_courses` - (not explored)

---

## Collection Structures

### 1. `inoi` Collection

**Purpose:** Stores original INOI (Iranian National Olympiad in Informatics) exam problems

**Statistics:**
- **Document Count:** 1,135 problems
- **Storage Size:** 5.94 MB
- **Indexes:** 1 (default `_id_` index)

**Schema (17 fields):**

| Field | Type | Null? | Description |
|-------|------|-------|-------------|
| `_id` | ObjectId | No | Primary key |
| `problem` | String | No | Problem statement (English markdown) |
| `choices` | String | No | Multiple choice options |
| `correct_option` | Number | No | The correct choice number (1-5) |
| `context` | String/Null | Yes | Additional problem context (for follow-up questions) |
| `images_list` | String/Null | Yes | Semicolon-separated list of images |
| `answer_value` | Number/String | No | The actual answer value |
| `answer_type` | String | No | Type: "Multiple_Choice" or "Yes/No" |
| `problem_number` | Number | No | Problem number within exam |
| `exam_directory` | String | No | Exam path (e.g., "First Round\\10") |
| `persian_solution` | String/Null | Yes | Solution in Persian |
| `english_solution` | String | No | Solution in English |
| `english_solution_local_images` | String | No | Solution with local image paths |
| `crawled_persian_markdown` | String | No | Original Persian markdown from crawl |
| `opedia_url` | String | No | Source URL (https://opedia.ir/...) |
| `is_reviewed` | Boolean | No | Review status flag |
| `backups` | Array | No | Backup history of field changes |

**Backups Array Structure:**
Each backup entry contains:
- `field`: String - Name of field that was backed up
- `value`: String/Number/Null - Previous value
- `timestamp`: Date - When backup was created
- `synthetic_data_id`: ObjectId - Reference to related synthetic_data document

---

### 2. `synthetic_data` Collection

**Purpose:** Stores AI-generated solutions and analysis data, including judging results across 11 AI models

**Statistics:**
- **Document Count:** 1,135 documents (1:1 mapping with inoi collection)
- **Storage Size:** 2.59 GB (⚠️ Very large - extensive AI model outputs)
- **Indexes:** 
  - `_id_` (default)
  - `judging_results_1` (on judging_results field)

**Schema (9 top-level fields):**

| Field | Type | Description |
|-------|------|-------------|
| `_id` | ObjectId | Primary key |
| `problem_id` | ObjectId | **Foreign key → `inoi._id`** |
| `workflows` | Object | Workflow execution metadata (5 workflows) |
| `solution_technique_data` | Object | Analysis of solution techniques |
| `choice_dependency_data` | Object | Choice dependency classification |
| `problem_validation_data` | Object | Problem validation results |
| `per_model_outputs` | Object | Generated solutions per model |
| `judging_results` | Object | **Judging results for 11 AI models** |
| `rewritten_solution` | String | Rewritten solution text |

---

## Field Descriptions

### `inoi` Collection Fields (Detailed)

#### Core Problem Fields

**`problem`** (String):
- English problem statement in markdown format
- May contain image references: `![](img-0.svg)`
- Some problems have minimal text if context provides main content
- Average length: ~500-1000 characters

**`context`** (String/Null):
- Present in **385 problems (33.9%)**
- Contains background information or previous problem text
- Used for follow-up questions (e.g., "In the previous question...")
- **CRITICAL:** Images may be referenced here instead of in `problem` field
- Average length: ~235 characters

**`choices`** (String):
- Format: `1. option1; 2. option2; 3. option3; 4. option4; 5. option5`
- Example: `"1. $0$; 2. $1$; 3. $2$; 4. $3$; 5. $4$"`
- Empty for Yes/No questions

**`images_list`** (String/Null):
- Present in **404 problems (35.6%)**
- Format: Semicolon-separated markdown image references
- Example: `"![img-0.svg](img-0.svg); ![img-1.svg](img-1.svg)"`
- May include alt text: `"![Image: description](file.svg)"`
- **WARNING:** Some entries contain HTML: `<img src="file.svg">`

#### Answer Fields

**`correct_option`** (Number):
- For Multiple Choice: 1-5
- For Yes/No: May still have a value

**`answer_value`** (Number/String):
- The actual mathematical answer
- Can be numeric: `11`, `76`, `41`
- Can be expression: `"$2^{256}$"`, `"$\\frac{35}{128}$"`
- For Yes/No: `"Yes"` or `"No"`

**`answer_type`** (String):
- **"Multiple_Choice"**: 1,041 problems (91.7%)
- **"Yes/No"**: 94 problems (8.3%)

#### Metadata Fields

**`exam_directory`** (String):
- Format: `"First Round\\10"` or `"Second Round\\25"`
- **35 unique exams** spanning:
  - First Round: Years 5-34 (30 exams, 1,065 problems)
  - Second Round: Years 24-26, 30, 32 (5 exams, 115 problems)

**`problem_number`** (Number):
- Problem number within that exam (1-60)
- Used for sequencing

**`is_reviewed`** (Boolean):
- **600 problems reviewed (52.9%)**
- **535 problems unreviewed (47.1%)**

**`backups`** (Array):
- Present in **382 problems (33.7%)**
- Total **451 backup entries**
- Most backed up field: `english_solution_local_images` (358 backups)

---

### `synthetic_data` Collection Fields (Detailed)

#### Reference Field

**`problem_id`** (ObjectId):
- **Foreign key reference to `inoi._id`**
- **1:1 relationship** - each inoi problem has exactly one synthetic_data document
- Used for joining collections

#### Solution Field

**`rewritten_solution`** (String):
- AI-rewritten solution text
- Clean, formatted solution
- **This is what we use for HF `solution` field**

#### Analysis Fields

**`solution_technique_data`** (Object):
```javascript
{
  "analysis_entries": [...],  // Array of technique analyses
  "overall_classification": "..."  // Overall technique label
}
```
- `overall_classification` → Used as `technique_label` in HF dataset

**`choice_dependency_data`** (Object):
```javascript
{
  "label": "standalone" | "choice-dependent",
  "confidence": null,
  "reasons": [...]
}
```
- **`label`** field used to determine if MC problem is standalone or choice-dependent
- **978 standalone** (86.2%)
- **157 choice-dependent** (13.8%)

**`problem_validation_data`** (Object):
```javascript
{
  "overall_severity": 1-5,
  "summary_comment": "...",
  "aggregated_findings": [...],
  "is_issue_detected": true/false
}
```
- **710 problems (62.6%)** have detected issues
- Severity distribution:
  - Level 5 (Highest): 389 problems
  - Level 4: 246 problems
  - Level 3: 48 problems
  - Level 2: 25 problems
  - Level 1: 2 problems

**`workflows`** (Object):
Contains 5 workflow types (all 1,135 documents have complete workflows):
- `problem_validation_workflow`
- `choice_dependency_classifier`
- `solver`
- `solution_technique_analyzer`
- `solution_rewriter`

**`per_model_outputs`** (Object):
```javascript
{
  "generated_solutions_list": [...]  // Array of generated solutions
}
```

**`judging_results`** (Object):
Contains results for **11 AI models** with 8 attempts each:

| Rank | Model | Accuracy | Correct/Total |
|------|-------|----------|---------------|
| 🥇 | gpt-5 | 78.1% | 7,033 / 9,005 |
| 🥈 | gemini-2_5-pro | 75.8% | 6,886 / 9,080 |
| 🥉 | gpt-5-mini | 65.4% | 5,941 / 9,080 |
| 4 | gemini-2_5-flash | 63.4% | 5,749 / 9,072 |
| 5 | gpt-5-nano | 58.9% | 4,713 / 8,008 |
| 6 | gemini-2_5-flash-lite | 50.8% | 4,609 / 9,080 |
| 7 | gpt-4o | 27.6% | 2,501 / 9,072 |
| 8 | gemma-3-27b-it | 27.5% | 2,493 / 9,080 |
| 9 | gemma-3-12b-it | 23.2% | 2,104 / 9,073 |
| 10 | gpt-4o-mini | 22.5% | 2,039 / 9,080 |
| 11 | gemma-3-4b-it | 16.2% | 1,465 / 9,068 |

Each model entry structure:
```javascript
{
  "model-name": {
    "is_final_answer_correct": [bool, bool, ...],  // 8 attempts
    "extracted_answers": [
      {
        "kind": "expression" | "numeric",
        "raw_value": "41",
        "standardized_choice_key": "4",
        "normalized_text": null
      },
      // ... 7 more attempts
    ]
  }
}
```

---

## Cross-Collection Relationships

### Primary Relationship

```
inoi._id (ObjectId) ←→ synthetic_data.problem_id (ObjectId)
```

**Cardinality:** 1:1 (each problem has exactly one synthetic data document)

**Example:**
```
inoi._id: ObjectId("689678f87b0414c529b7b5c5")
  ↓ (Problem #5, First Round\10)
synthetic_data.problem_id: ObjectId("689678f87b0414c529b7b5c5")
  ↓ (Links to same problem)
```

### Secondary References

**`backups` array in `inoi` collection:**
- `synthetic_data_id` field references synthetic_data documents
- Tracks which synthetic data generation caused the backup

---

## Data Statistics

### Overall Completeness

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Problems** | 1,135 | 100% |
| **Reviewed** | 600 | 52.9% |
| **With Context** | 385 | 33.9% |
| **With Images** | 404 | 35.6% |
| **With Persian Solution** | 1,000 | 88.1% |
| **With Backups** | 382 | 33.7% |

### Problem Types

| Type | Count | Percentage |
|------|-------|------------|
| **Multiple Choice** | 1,041 | 91.7% |
| **Yes/No** | 94 | 8.3% |

### Exam Coverage

**35 exams total:**

#### First Round (30 exams, 1,065 problems)
- **Years:** 5-34
- **Largest:** Years 7, 8, 9 (60 problems each)
- **Review rates:** Vary from 2.5% (Year 15) to 83% (Year 26)

#### Second Round (5 exams, 115 problems)
- **Years:** 24, 25, 26, 30, 32
- Generally harder than First Round problems

### Top 10 Exams by Size

| Exam | Problem Count | Reviewed | Has Images |
|------|---------------|----------|------------|
| First Round 7 | 60 | 32 (53%) | 23 (38%) |
| First Round 8 | 60 | 32 (53%) | 21 (35%) |
| First Round 9 | 60 | 28 (47%) | 23 (38%) |
| First Round 10 | 44 | 26 (59%) | 11 (25%) |
| First Round 6 | 40 | 19 (48%) | 11 (28%) |
| First Round 11 | 40 | 17 (43%) | 7 (18%) |
| First Round 13 | 40 | 7 (18%) | 14 (35%) |
| First Round 14 | 39 | 17 (44%) | 8 (21%) |
| First Round 16 | 39 | 23 (59%) | 13 (33%) |
| First Round 17 | 39 | 23 (59%) | 19 (49%) |

### Choice Dependency Analysis

From `synthetic_data.choice_dependency_data`:

| Type | Count | Percentage |
|------|-------|------------|
| **Standalone** | 978 | 86.2% |
| **Choice-dependent** | 157 | 13.8% |

**Standalone:** Problem can be solved without looking at choices  
**Choice-dependent:** Solution requires examining the given choices

### Problem Validation Results

From `synthetic_data.problem_validation_data`:

| Status | Count | Percentage |
|--------|-------|------------|
| **Issues detected** | 710 | 62.6% |
| **No issues** | 425 | 37.4% |

**Severity of Issues** (710 problems with issues):

| Severity Level | Count | Percentage |
|---------------|-------|------------|
| 5 (Critical) | 389 | 54.8% |
| 4 (High) | 246 | 34.6% |
| 3 (Medium) | 48 | 6.8% |
| 2 (Low) | 25 | 3.5% |
| 1 (Minor) | 2 | 0.3% |

### Backup History

**Most frequently backed up fields** (451 total backups):

| Field | Backup Count | Percentage |
|-------|-------------|------------|
| `english_solution_local_images` | 358 | 79.4% |
| `problem` | 49 | 10.9% |
| `correct_option` | 24 | 5.3% |
| `answer_value` | 10 | 2.2% |
| `choices` | 10 | 2.2% |

**Insight:** Most corrections are for solution images, followed by problem text fixes.

---

## Image Reference Analysis

### Overall Image Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total problems with images** | 404 | 35.6% |
| **Total image references** | 480 | - |
| **Unique filenames** | ~380 | - |
| **Shared filenames** | ~160 | (used in multiple problems) |

### Image Reference Formats

#### In `problem` Field:

**Standard Markdown** (most common):
```markdown
![](img-0.svg)
![](img-1.png)
```

**With Alt Text** (rare):
```markdown
![Image description](filename.svg)
```

#### In `context` Field:

**Same as problem field** - images can be referenced here too!

**CRITICAL FINDING:** Images may appear in `context`, `problem`, or BOTH. Must check both fields when extracting.

#### In `images_list` Field:

**Single Image:**
```markdown
![img-0.svg](img-0.svg)
```

**Multiple Images** (semicolon-separated):
```markdown
![img-0.svg](img-0.svg); ![img-1.svg](img-1.svg)
```

**With Descriptive Alt Text:**
```markdown
![Image: Three coins in a row labeled A, B, C](59.svg)
```

**HTML Format** (rare - 3 problems):
```html
<img src="13.svg" style="width:45%;">
```

### Image Naming Patterns

| Pattern | Usage | Example | Count |
|---------|-------|---------|-------|
| `img-X.ext` | Most common | `img-0.svg`, `img-1.svg` | ~250 |
| `problemnum.ext` | Some problems | `59.svg`, `54.svg` | ~50 |
| `qXX.ext` | Occasional | `q39.png` | ~30 |
| `QXX.ext` | Occasional | `Q38.svg` | ~10 |
| `exam_qXX.ext` | Descriptive | `18_q16.png`, `olympiad20_q9.svg` | ~40 |
| `solution-X-Y.ext` | In solutions | `solution-3-1.png` | In english_solution_local_images |

### File Extensions

| Extension | Count | Usage |
|-----------|-------|-------|
| `.svg` | ~350 | Vector graphics (diagrams, graphs, geometric figures) |
| `.png` | ~130 | Raster images (screenshots, complex visuals) |

### Regex Extraction Results

**Pattern:** `![...](...)`

When applied to `problem` + `context` combined:

| Images Extracted | Problem Count |
|------------------|---------------|
| **0 images** | 0 (100% coverage!) ✅ |
| **1 image** | 358 |
| **2 images** | 31 |
| **3 images** | 6 |
| **4 images** | 3 |
| **5 images** | 6 |

**Total extracted:** 480 images from 404 problems

**Coverage:** ✅ **100%** when checking both `problem` AND `context` fields

### Image Distribution by Problem

| Image Count | Problem Count |
|-------------|---------------|
| 1 image | 358 (88.6%) |
| 2 images | 31 (7.7%) |
| 3-5 images | 15 (3.7%) |

**Maximum images in one problem:** 6 images (Problems 19-20, First Round 29)

---

## Problematic Cases

### Summary of Issues Found

**Total:** 9 problematic problems (2.2% of image problems)

### Issue Type 1: HTML img Tags (3 problems)

**Affected Problems:**
- First Round 29, Problem 18 (`689678f87b0414c529b7b866`)
- First Round 29, Problem 19 (`689678f87b0414c529b7b867`)
- First Round 29, Problem 20 (`689678f87b0414c529b7b868`)

**Issue:**
```html
<img src="13.svg" style="width:45%;">
<img src="12.svg" style="width:45%;">
```

**Location:** In `context` field (shared context for Mamali DNA problems)

**Solution:** Convert to Markdown `![](13.svg)` during processing

**Why it happened:** HTML used for side-by-side layout styling

---

### Issue Type 2: Missing Image Reference (1 problem)

**Affected Problem:**
- First Round 15, Problem 7 (`689678f87b0414c529b7b68c`)

**Issue:**
- `images_list` contains: `"![Q7_1 Image](q7_1.png);![Q7_2 Image](q7_2.png)"`
- Problem text only references: `![](q7_1.png)`
- Missing: `q7_2.png` is listed but never referenced

**Problem Text:**
```markdown
This cube (from two corners with some rotation) is shown 
in the figures below: How many of the following figures...
![](q7_1.png)
```

**Diagnosis:** Text says "figures below" (plural) but only one image shown. Likely `q7_2.png` should be displayed but markdown reference is missing.

**Solution:** Either add `![](q7_2.png)` or remove from `images_list`

---

### Issue Type 3: Misleading Semicolons in Alt Text (1 problem)

**Affected Problem:**
- First Round 18, Problem 16 (`689678f87b0414c529b7b70b`)

**Issue:**
```markdown
![Image: For 3 boxes, 5 configurations are shown. Four of them are 
distinct compositions: (1) column of 3; (2) column of 2 and 1; 
(3) column of 1 and 2; (4) three columns...](18_q16.png)
```

**Problem:** Semicolons in alt text cause split on `;` to think there are 4 images when there's only 1

**Solution:** Parse more carefully, or use different separator in `images_list`

---

### Issue Type 4: Legitimate Duplicates (4 problems)

**Affected Problems:**
- First Round 19, Problem 7 (`689678f87b0414c529b7b72a`)
- First Round 33, Problem 10 (`689678f87b0414c529b7b8ae`)
- First Round 9, Problem 58 (`689678f97b0414c529b7b9ba`)
- Second Round 25, Problem 5 (`689678f97b0414c529b7b9da`)

**Issue:** Same image appears in both `context` and `problem` fields

**Why it happens:**
- Follow-up questions include previous problem in `context`
- Both reference the same figure
- Example: Problem 7 asks about same figure as Problem 6

**Solution:** ✅ NOT AN ERROR - Deduplication should count image once

---

## Image Reference Extraction Strategy

### Complete Extraction Algorithm

```python
def extract_all_images(problem_doc):
    # Step 1: Get both fields
    context = problem_doc.get('context', '') or ''
    problem = problem_doc.get('problem', '') or ''
    
    # Step 2: Combine with separator
    combined = f"{context}\n\n{problem}" if context else problem
    
    # Step 3: Fix HTML tags
    combined = re.sub(r'<img[^>]+src="([^"]+)"[^>]*>', r'![](\1)', combined)
    
    # Step 4: Extract markdown images
    pattern = r'!\[.*?\]\(([^)]+)\)'
    images = re.findall(pattern, combined)
    
    # Step 5: Deduplicate while preserving order
    unique_images = []
    seen = set()
    for img in images:
        if img not in seen:
            unique_images.append(img)
            seen.add(img)
    
    return unique_images
```

**Coverage:** ✅ **100%** of images extracted successfully

### Why Both Fields Matter

| Scenario | Example | Location |
|----------|---------|----------|
| **Simple problem** | Single question | Images in `problem` only |
| **Follow-up question** | "In the previous question..." | Images in `context` |
| **Multi-part context** | Shared setup for Q18-20 | Images in `context`, specific images in `problem` |
| **Related problems** | Same figure for two problems | Image appears in both `context` and `problem` |

**Key Insight:** 
- 335 problems (82.9% of image problems) have non-empty `context`
- Of those, 166 problems with images have context
- **Must concatenate context + problem before image extraction**

---

## Conversion Strategy

### Unified Image Naming Convention

**Proposed Format:** `{round_type}{round_num}_p{problem_num}_{sequence}.{ext}`

**Examples:**
```
Original             →  New Name
-------------------- →  -----------------------
img-0.svg            →  fr10_p31_0.svg
img-1.svg            →  fr10_p31_1.svg
q7_1.png             →  fr15_p7_0.png
13.svg               →  fr29_p18_1.svg
3.png                →  sr25_p5_0.png
```

**Benefits:**
- ✅ Unique names (includes exam + problem number)
- ✅ Easy sorting and organization
- ✅ Human-readable
- ✅ Preserves file extension
- ✅ Sequential numbering within problem

### Context + Problem Concatenation

**Format:**
```markdown
{context}

---

{problem}
```

**Rules:**
1. If `context` is null or empty: use `problem` only
2. If `context` exists: combine with `---` separator
3. Update image references in BOTH parts
4. Result goes into HF `problem` field (note: not `prompt` - matches existing schema)

### HuggingFace Dataset Schema

**Field Mapping:**

| HF Field | MongoDB Source | Processing |
|----------|----------------|------------|
| `id` | Generated | `combiz_0001`, `combiz_0002`, ... |
| `problem` | `inoi.context` + `inoi.problem` | Combined with `---` separator, images renamed |
| `images_list` | Extracted from text | Renamed to new convention |
| `solution` | `synthetic_data.rewritten_solution` | Direct copy |
| `technique_label` | `synthetic_data.solution_technique_data.overall_classification` | Extract nested field |
| `problem_type` | Derived | Based on exam/answer_type/choice_dependency |
| `choices` | `inoi.choices` | Direct copy |
| `correct_option` | `inoi.correct_option` | Direct copy |
| `answer_value` | `inoi.answer_value` | Convert to string |
| `answer_type` | `inoi.answer_type` | Direct copy |
| `exam_directory` | `inoi.exam_directory` | Direct copy |
| `problem_number` | `inoi.problem_number` | Direct copy |
| `original_problem_id` | `inoi._id` | Convert to string |

### Problem Type Derivation Logic

```python
def determine_problem_type(inoi_doc, synthetic_doc):
    exam_dir = inoi_doc['exam_directory']
    answer_type = inoi_doc['answer_type']
    
    # Check exam directory
    if 'second' in exam_dir.lower():
        base_type = 'second-round'
    
    # Check answer type
    elif answer_type == 'Yes/No':
        base_type = 'yes-no'
    
    elif answer_type == 'Multiple_Choice':
        # Check choice dependency
        label = synthetic_doc['choice_dependency_data']['label']
        
        if label == 'choice-dependent':
            base_type = 'mc-dependent'
        else:  # 'standalone'
            base_type = 'mc-standalone'
    
    # Add image suffix
    if inoi_doc['images_list']:
        base_type += '-img'
    
    return base_type
```

**Resulting Types:**
- `yes-no`
- `yes-no-img`
- `mc-standalone`
- `mc-standalone-img`
- `mc-dependent`
- `mc-dependent-img`
- `second-round`
- `second-round-img`

---

## AI Model Performance Analysis

### Best Performing Models

Based on **~9,000 attempts per model** (8 attempts × 1,135 problems):

**Top 5:**
1. **gpt-5**: 78.1% accuracy
2. **gemini-2_5-pro**: 75.8% accuracy
3. **gpt-5-mini**: 65.4% accuracy
4. **gemini-2_5-flash**: 63.4% accuracy
5. **gpt-5-nano**: 58.9% accuracy

**Notable Observations:**
- GPT-5 series outperforms other models
- Gemini 2.5 Pro competitive with GPT-5 variants
- **GPT-4o surprisingly low** at 27.6% (possible configuration issue)
- Gemma models struggle with these problems (16-27% accuracy)

### Problem Difficulty by Exam

**Hardest Exams** (% of problems no model solved perfectly):

| Exam | Difficulty Score | Perfect Solves |
|------|-----------------|----------------|
| Second Round 32 | 60% | 8/20 (40%) |
| Second Round 24 | 48% | 13/25 (52%) |
| First Round 29 | 48% | 13/25 (52%) |
| First Round 12 | 44% | 19/34 (56%) |

**Easiest Exams:**

| Exam | Difficulty Score | Perfect Solves |
|------|-----------------|----------------|
| First Round 21 | 6.7% | 28/30 (93%) |
| First Round 33 | 6.7% | 14/15 (93%) |
| First Round 6 | 7.5% | 37/40 (93%) |

**Trend:** Second Round problems are significantly harder than First Round.

---

## Common Pitfalls & Important Notes

### 1. Always Check Both `problem` AND `context` Fields

❌ **Wrong:**
```python
images = extract_images(doc['problem'])
```

✅ **Correct:**
```python
combined = (doc.get('context', '') or '') + '\n\n' + doc['problem']
images = extract_images(combined)
```

**Why:** 166 problems with images have them in `context` field

---

### 2. Handle HTML img Tags

**Found in:** Problems 18-20, First Round 29

❌ **Won't match regex:**
```html
<img src="13.svg" style="width:45%;">
```

✅ **Convert first:**
```python
text = re.sub(r'<img[^>]+src="([^"]+)"[^>]*>', r'![](\1)', text)
```

---

### 3. Cross-Reference Collections Correctly

❌ **Wrong field name:**
```python
synthetic_doc = synthetic_collection.find_one({'_id': problem_id})
```

✅ **Correct:**
```python
synthetic_doc = synthetic_collection.find_one({'problem_id': problem_id})
```

**Why:** `synthetic_data` uses `problem_id` field, not `_id`

---

### 4. Handle Null/Empty Context

❌ **Causes "null" in output:**
```python
combined = f"{doc['context']}\n\n{doc['problem']}"
```

✅ **Safe handling:**
```python
context = (doc.get('context') or '').strip()
problem = (doc.get('problem') or '').strip()
combined = f"{context}\n\n---\n\n{problem}" if context else problem
```

---

### 5. Deduplicate Image References

**Why:** Same image may appear in both `context` and `problem`

✅ **Deduplicate:**
```python
unique_images = []
seen = set()
for img in all_images:
    if img not in seen:
        unique_images.append(img)
        seen.add(img)
```

---

### 6. Problem Type Edge Cases

**Second Round Problems:**
- Even if Multiple Choice, should be `second-round` (not `mc-standalone`)
- Second Round supersedes other classifications

**Yes/No Problems:**
- Only 94 total (8.3%)
- Don't have meaningful `choices` field
- `answer_value` is "Yes" or "No"

---

## Example Problem Structures

### Example 1: Simple MC Problem with Image

**From `inoi` collection:**
```javascript
{
  "_id": ObjectId("689678f87b0414c529b7b5c3"),
  "problem": "### Question 3.\n\nThe figure below shows $6$ cities...\n\n![](img-0.svg)",
  "context": "",  // Empty
  "choices": "1. $63$; 2. $69$; 3. $70$; 4. $76$; 5. $92$",
  "correct_option": 4,
  "answer_value": 76,
  "answer_type": "Multiple_Choice",
  "problem_number": 3,
  "exam_directory": "First Round\\10",
  "images_list": "![img-0.svg](img-0.svg)"
}
```

**From `synthetic_data` collection:**
```javascript
{
  "_id": ObjectId("68a257e91346e90ab3ccd2fa"),
  "problem_id": ObjectId("689678f87b0414c529b7b5c3"),  // Links to inoi
  "rewritten_solution": "**Option (4) is correct.**\n\nIt is optimal if...",
  "solution_technique_data": {
    "overall_classification": "optimization",
    "analysis_entries": [...]
  },
  "choice_dependency_data": {
    "label": "standalone",
    "confidence": null
  }
}
```

**Converted to HF:**
```python
{
  'id': 'combiz_0003',
  'problem': '### Question 3.\n\nThe figure below shows...\n\n![](fr10_p3_0.svg)',
  'images_list': ['fr10_p3_0.svg'],
  'solution': '**Option (4) is correct.**\n\nIt is optimal if...',
  'technique_label': 'optimization',
  'problem_type': 'mc-standalone-img',
  'choices': '1. $63$; 2. $69$; 3. $70$; 4. $76$; 5. $92$',
  'correct_option': 4,
  'answer_value': '76',
  'answer_type': 'Multiple_Choice',
  'exam_directory': 'First Round\\10',
  'problem_number': 3
}
```

---

### Example 2: Follow-up Question with Context

**From `inoi` collection:**
```javascript
{
  "_id": ObjectId("689678f87b0414c529b7b5e0"),
  "problem": "### Question 32.\n\nIn the previous question, how many corresponding strings...",
  "context": "### Question 31.\n\nA matrix $M$ with entries...\n\n![](img-6.svg)\n![](img-7.svg)\n\n...",
  "choices": "1. $16^{64}$; 2. $2^{16}$; 3. $2^{64}$; 4. $4^{256}$; 5. $4^{16}$",
  "correct_option": 1,
  "answer_value": "$16^{64}$",
  "answer_type": "Multiple_Choice",
  "problem_number": 32,
  "exam_directory": "First Round\\10",
  "images_list": "![img-6.svg](img-6.svg); ![img-7.svg](img-7.svg)"
}
```

**Converted to HF:**
```python
{
  'id': 'combiz_0032',
  'problem': '''### Question 31.

A matrix $M$ with entries...

![](fr10_p32_0.svg)
![](fr10_p32_1.svg)

...

---

### Question 32.

In the previous question, how many corresponding strings...''',
  'images_list': ['fr10_p32_0.svg', 'fr10_p32_1.svg'],
  'problem_type': 'mc-standalone-img',
  ...
}
```

**Note:** Images are in `context` but are extracted and renamed properly!

---

### Example 3: Multi-Problem Shared Context (Mamali DNA)

**Problems 18-20, First Round 29** share the SAME context with 6 images total.

**Context structure:**
```markdown
## Questions 18-20: Mamali DNA

[Explanation of Mamali DNA concept]
![](11.svg)

[XOR operation definition]

Example reproduction:
<img src="13.svg" style="width:45%;">
<img src="12.svg" style="width:45%;">

Result:
![](14.svg)

Answer the following 3 questions...
```

**Problem 18:** Uses context images (11, 13, 12, 14)  
**Problem 19:** Uses context images + own images (15, 16)  
**Problem 20:** Uses context images only

**After conversion:**
- All HTML converted to Markdown ✅
- Each problem gets its own renamed images ✅
- No duplicate images within a problem ✅

---

## Dataset Quality Metrics

### Data Completeness

| Metric | Status |
|--------|--------|
| **Cross-reference integrity** | 100% (all 1,135 matched) |
| **Image extraction coverage** | 100% (480/480 images) |
| **Solution availability** | 100% (all have rewritten_solution) |
| **Technique labels** | ~95% (1,078/1,135) |
| **Problem validation** | 100% (all validated) |

### Review Status

| Status | Count | Next Steps |
|--------|-------|------------|
| **Reviewed** | 600 (52.9%) | High quality ✅ |
| **Unreviewed** | 535 (47.1%) | May need QA |

**Priority for review:**
- First Round 15: 39/40 unreviewed
- First Round 13: 33/40 unreviewed
- Recent exams (31-34): Lower review rates

---

## Usage with Existing System

### After Conversion

The dataset is **fully compatible** with your existing `inoi.py` environment:

```python
# Load from HuggingFace
from environments.inoi import load_environment

env = load_environment(
    dataset_name='pmahdavi/inoi',  # or your-username/inoi
    num_train_examples=100,
    filter_multimodal=True,  # Only problems with images
)

# Works exactly as before!
# The format_prompt() function expects:
# - doc['problem'] ✅ (we provide this)
# - doc['images'] (added later by prepare_and_upload_hf.py)
# - doc['choices'] ✅
# - doc['answer_type'] ✅
```

### Workflow

```
MongoDB
  ↓
  ↓ (convert_mongodb_to_hf.py)
  ↓
HuggingFace Dataset (text + metadata)
  ↓
  ↓ (prepare_and_upload_hf.py - adds PIL Images)
  ↓
HuggingFace Dataset (with PIL Images)
  ↓
  ↓ (inoi.py - load_environment)
  ↓
Verifiers Environment
```

---

## Important MongoDB Queries

### Find Problems with Images
```javascript
db.inoi.find({
  images_list: { $ne: null, $ne: "" }
})
```

### Find Follow-up Questions
```javascript
db.inoi.find({
  context: { $ne: null, $ne: "" }
})
```

### Join inoi with synthetic_data
```javascript
db.inoi.aggregate([
  {
    $lookup: {
      from: "synthetic_data",
      localField: "_id",
      foreignField: "problem_id",
      as: "synthetic"
    }
  },
  { $unwind: "$synthetic" }
])
```

### Count by Problem Type
```javascript
db.inoi.aggregate([
  { $group: { _id: "$answer_type", count: { $sum: 1 } } }
])
```

### Extract Image References (MongoDB)
```javascript
db.inoi.aggregate([
  {
    $project: {
      combined: {
        $concat: [
          { $ifNull: ["$context", ""] },
          "\n\n",
          { $ifNull: ["$problem", ""] }
        ]
      }
    }
  },
  {
    $project: {
      images: {
        $regexFindAll: {
          input: "$combined",
          regex: "!\\[.*?\\]\\(([^)]+)\\)"
        }
      }
    }
  }
])
```

---

## Validation Checklist

Before running conversion:

- [ ] MongoDB connection working
- [ ] `inoi` collection has 1,135 documents
- [ ] `synthetic_data` collection has 1,135 documents
- [ ] All problems have matching synthetic_data (via `problem_id`)
- [ ] Required fields present in both collections

After conversion:

- [ ] Output has 1,135 rows
- [ ] All IDs are unique (combiz_0001 to combiz_1135)
- [ ] ~404 problems have non-empty `images_list`
- [ ] All image references updated to new names
- [ ] No HTML `<img>` tags in `problem` field
- [ ] Context properly concatenated where exists
- [ ] Problem types correctly classified
- [ ] Train/test split is 908/227 (80/20)

---

## Key Insights from Exploration

### 1. Database Purpose

This database appears to be an **AI solution evaluation system** for INOI problems:
- Original problems stored in `inoi`
- AI-generated solutions + analysis in `synthetic_data`
- 11 different AI models attempted each problem 8 times
- Results judged and stored with detailed metadata
- Backups track changes with references to synthetic data

### 2. Multimodal Content

- **35.6% of problems** have images
- Images are critical for understanding:
  - Graph problems
  - Geometric diagrams
  - Visual patterns
  - State transitions
- SVG preferred for diagrams (scalable, clean)
- PNG used for complex visuals

### 3. Problem Complexity

- Most problems are standalone Multiple Choice
- 13.8% require examining choices to solve (choice-dependent)
- Second Round problems significantly harder
- Many problems have mathematical expressions
- Some require multi-step reasoning

### 4. Data Quality

**Strengths:**
- Complete 1:1 mapping between collections
- Comprehensive AI evaluation data
- Bilingual (Persian + English)
- Well-structured metadata

**Areas for Improvement:**
- 47% still unreviewed
- 62% have validation issues flagged
- Some missing Persian solutions (years 31-34)
- 3 problems use HTML instead of Markdown

---

## Technical Specifications

### Image File Formats

**SVG (Vector):**
- Count: ~350 images (73%)
- Best for: Graphs, diagrams, geometric figures
- Scalable without quality loss
- May need conversion to PNG for some models

**PNG (Raster):**
- Count: ~130 images (27%)
- Best for: Screenshots, complex visuals, photos
- Fixed resolution
- Ready to use without conversion

### Text Encoding

- **Charset:** UTF-8 (supports Persian + English + mathematical symbols)
- **Markdown:** GitHub-flavored markdown
- **Math:** LaTeX format (`$...$` for inline, `$$...$$` for block)
- **Special characters:** Properly escaped in JSON

### MongoDB Schema

**BSON Types Used:**
- ObjectId: `_id`, `problem_id`, references
- String: Text fields
- Number: Integers and floats
- Boolean: Flags
- Array: Lists and collections
- Document: Nested objects
- Date: Timestamps
- Null: Optional fields

---

## Historical Context

### Exam Timeline

**First Round Coverage:**
- Oldest: Year 5
- Newest: Year 34
- **30 years** of exams
- **1,065 problems**

**Second Round Coverage:**
- Years: 24, 25, 26, 30, 32
- **5 years** of exams
- **115 problems**
- Generally more advanced problems

### Source Attribution

**Original Source:** https://opedia.ir/
- Persian olympiad problem repository
- Crawled and processed into structured format
- URLs tracked in `opedia_url` field

---

## Summary Statistics Table

| Metric | Value |
|--------|-------|
| **Total Problems** | 1,135 |
| **Total Collections** | 3 (using 2) |
| **Database Size** | 973 MB |
| **Unique Exams** | 35 |
| **Years Covered** | 30 (First Round 5-34, Second Round 24-32) |
| **Problems with Images** | 404 (35.6%) |
| **Total Image References** | 480 |
| **Unique Image Files** | ~380 |
| **Problems with Context** | 385 (33.9%) |
| **Reviewed Problems** | 600 (52.9%) |
| **AI Models Evaluated** | 11 |
| **Total AI Attempts** | ~99,880 (11 models × 8 attempts × 1,135 problems) |
| **Multiple Choice Problems** | 1,041 (91.7%) |
| **Yes/No Problems** | 94 (8.3%) |
| **Standalone MC** | 978 (86.2%) |
| **Choice-dependent MC** | 157 (13.8%) |
| **Problems with Validation Issues** | 710 (62.6%) |

---

## Quick Reference Commands

### Explore MongoDB
```bash
# List all databases
mongo --eval "db.adminCommand('listDatabases')"

# Count documents
mongo inoi --eval "db.inoi.count()"
mongo inoi --eval "db.synthetic_data.count()"

# Sample document
mongo inoi --eval "db.inoi.findOne()"
```

### Run Conversion
```bash
cd /scratch/pxm5426/repos/verifiers-fork/environments/inoi
pip install -r requirements_converter.txt
python convert_mongodb_to_hf.py
```

### Validate Output
```bash
# Check CSV
head -n 20 inoi_dataset_preview.csv

# Check image mapping
head -n 50 image_mapping.txt

# Load in Python
python -c "from datasets import load_from_disk; ds = load_from_disk('inoi_hf_dataset'); print(ds)"
```

---

## Contact & Maintenance

For questions about:
- **Database structure**: Refer to this document
- **Conversion script**: See `MONGODB_CONVERSION.md`
- **Integration**: See `README.md` and `inoi.py`

---

**Document Version:** 1.0  
**Last Updated:** October 21, 2025  
**Based on:** Complete MongoDB cluster exploration and analysis
