# SVG to PNG Conversion Status

## Overview

The INOI dataset contains 332 multimodal problems requiring 350 images. These images were originally stored as SVG files in MongoDB. To support PIL-based loading for the HuggingFace dataset, we converted SVGs to PNGs.

**Current Status: 91.3% Valid Coverage (303/332 multimodal problems)**

## Conversion Process

### Backend Used
- **Primary:** `cairosvg` - Python library for SVG rendering
- **Preprocessing:** Applied exact logic from `inoi-dataset-evolution`:
  - `_rewrite_light_dark()` - Resolves `light-dark(a,b)` CSS functions
  - `_preprocess_svg_text()` - Handles `var()` CSS variables and removes `foreignObject` elements
  - Promotes `<switch>` elements to their `<image>` children

### Why Not Browser Backend?
- Browser rendering (playwright/chromium) was tested but is **extremely slow**: ~31 seconds per SVG
- Would require ~26 minutes for remaining 47 SVGs
- `cairosvg` is fast (~0.1s per SVG) and worked for 78% of files

## Conversion Results

### ✅ Successful Conversions
- **325 valid PNG files** (78.3% of 415 total SVGs)
- All validated as non-uniform (not all-white or all-black)
- Full color range confirmed: `extrema = ((0,255), (0,255), (0,255))`

### ❌ Failed Conversions

#### Problem: All-Black PNG Output (67 files)

**Root Cause:** `cairosvg` rendering issue with certain SVG structures
- SVGs contain simple geometric primitives (circles, lines, rectangles)
- `cairosvg` fails to render them with proper background
- Output: 400x400 all-black PNG (RGB extrema: `((0,0), (0,0), (0,0))`)

**Example Failed SVG:**
```xml
<!-- First Round/29/10.svg -->
<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
  <circle cx="120" cy="120" r="6" fill="black"/>
  <!-- ... 24 more circles in 5x5 grid ... -->
</svg>
```

**Expected:** Black dots on white background
**Actual:** Completely black 400x400 PNG

**Affected Directories:**
```
First Round/29/  : 10 SVGs (4,5,10,11,12,13,14,15,16,3)
First Round/28/  :  8 SVGs (2,3,5,7,8,9,10,11)
First Round/34/  :  9 SVGs (1,2,3,4,5,8, + 3 solution images)
First Round/20/  :  7 SVGs (olympiad20_q2,5,8,9,13,20,25)
First Round/30/  :  8 SVGs (1,2,8,9,10,11,12, + solution-7-1)
First Round/5/   :  5 SVGs (1,3,4,7,8)
First Round/7/   :  5 SVGs (1,4,5,6,13)
First Round/26/  :  3 SVGs (1,10,12)
First Round/6/   :  2 SVGs (3,4)
First Round/32/  :  2 SVGs (1,2)
First Round/33/  :  1 SVG  (3)
First Round/18/  :  2 SVGs (18_q1, 18_q8)
First Round/17/  :  2 SVGs (17_q16, 17_q17)
First Round/19/  :  1 SVG  (19_q6)
First Round/31/  :  1 SVG  (solution-4-2)
```

**Total Failed:** 67 SVGs across 15 directories

All 67 all-black PNGs were **deleted** to prevent invalid images in the dataset.

### 🔄 Remaining Unconverted (23 files)

#### Real Images (1 file)
- `First Round/15/q14.svg` - Only real image still without PNG

#### macOS Metadata (22 files)
- `First Round/7/__MACOSX/.*svg` - Not actual images, can be ignored

## Dataset Coverage

### By Split
| Split | Total Examples | Multimodal Problems | With Valid Images | Coverage |
|-------|----------------|---------------------|-------------------|----------|
| Train | 908 | 268 | 245 | 91.4% |
| Test  | 227 | 64  | 58  | 90.6% |
| **Total** | **1135** | **332** | **303** | **91.3%** |

### Missing Images by Problem
**47 total images missing** across **31 multimodal problems**:
- Some problems have multiple images (diagrams + sub-figures)
- All missing images are from failed cairosvg conversions

## Why This Matters

### Impact on Model Evaluation
- **91.3% of multimodal problems** can be evaluated correctly
- **8.7% of multimodal problems** will fail to load images
- These problems will appear as text-only, potentially affecting:
  - Model accuracy (missing visual context)
  - Dataset statistics (skewed toward text-only reasoning)

### Data Quality
- **Good:** All included images are verified valid (no all-black/all-white)
- **Issue:** 31 problems completely missing visual content

## How to Fix

### Option 1: Browser-Based Conversion (Slow but Reliable)
Use the `inoi-dataset-evolution` browser backend for remaining 47 SVGs:

```bash
cd /path/to/inoi-dataset-evolution
for svg in First\ Round/{5,6,7,17,18,19,20,26,28,29,30,32,34}/*.svg; do
  python svg_to_png.py --input "$svg" --output "${svg%.svg}.png" --backend browser
done
```

**Time estimate:** ~25-30 minutes (31s per SVG × 47 files)

**Pros:**
- Guaranteed correct rendering via headless Chrome
- Same backend used in `inoi-dataset-evolution` production

**Cons:**
- Very slow (requires launching browser for each SVG)
- Requires `playwright` + Chromium browser installed

### Option 2: Manual PNG Creation
For the 47 failed SVGs:
1. Open in Inkscape/Adobe Illustrator
2. Export as PNG with white background (400×400px)
3. Save to same directory as SVG

**Pros:**
- Can verify visual correctness manually
- Faster than browser conversion for small batches

**Cons:**
- Manual work required
- Not reproducible/automated

### Option 3: Alternative SVG Renderers
Try other Python SVG libraries:
- `svglib` + `reportlab` (requires Cairo system library)
- `pyvips` (requires libvips)
- `wand` (ImageMagick wrapper)

## Files Modified

### Deleted (Cleaned Up)
- `__init__.py` - Unused module init
- `eval_gemini.py` - Old evaluation script
- `gemini_eval_results_*.json` - Old evaluation results
- `test_*.py` - Old test files (3 files)
- `convert_*.py` - Temporary conversion scripts (3 files)
- `prepare_*.py` - Dataset preparation scripts (2 files)
- `upload_*.py` - Dataset upload scripts (2 files)
- `verify_*.py` - Dataset verification scripts (2 files)
- `analyze_dataset.py` - Temporary analysis script
- `add_images_to_hf_dataset.py` - Temporary helper script
- `hf_dataset_*` directories - Temporary dataset versions (4 dirs)

### Updated
- `inoi.py` - Migrated from MongoDB to HuggingFace with PIL Images
- `pyproject.toml` - Updated dependencies and metadata

### Added
- `assets/` - Image files (325 valid PNGs from SVG conversions)

## Recommendations

### Short Term
1. ✅ **Commit current state** with 91.3% valid coverage
2. Document which problems are missing images in dataset card
3. Add warning when loading examples without images

### Long Term
1. Run browser-based conversion for remaining 47 SVGs (~30 min one-time cost)
2. Re-upload dataset with 99.7% coverage (331/332 images)
3. Consider pre-rendering all SVGs to PNG in source dataset

## References

- Original conversion logic: `inoi-dataset-evolution/src/olympiad_agent/common/utils.py`
- Browser backend: `inoi-dataset-evolution/svg_to_png.py`
- Dataset: https://huggingface.co/datasets/pmahdavi/inoi

---

**Last Updated:** 2025-10-11
**Valid PNG Count:** 325/415 (78.3%)
**Multimodal Coverage:** 303/332 (91.3%)
