# SVG to PNG Conversion Status

## Overview

The INOI dataset contains 332 multimodal problems requiring 350 images. These images were originally stored as SVG files in MongoDB. To support PIL-based loading for the HuggingFace dataset, we converted SVGs to PNGs.

**Current Status: 99.7% Valid Coverage (331/332 multimodal problems)**

**Update:** Browser-based conversion successfully converted all 64 remaining problematic SVGs, bringing coverage from 91.3% → 99.7%!

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
- **389 valid PNG files** (93.7% of 415 total SVGs)
  - 325 from cairosvg (78.3%)
  - 64 from browser/Chromium (15.4%)
- All validated as non-uniform (not all-white or all-black)
- Full color range confirmed: `extrema = ((0,255), (0,255), (0,255))`

### ❌ Failed Conversions (FIXED with Browser Conversion)

#### Problem: All-Black PNG Output (67 files) - **RESOLVED ✓**

**Root Cause:** `cairosvg` rendering issue with certain SVG structures
**Solution:** Successfully converted all 67 using browser/Chromium rendering (64 unique SVGs)
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

### 🔄 Remaining Unconverted (1 file)

#### Missing from Source Data (1 file)
- `First Round/7/56.svg` - Never existed in assets directory (MongoDB data issue, not conversion issue)

#### macOS Metadata (22 files) - Ignored
- `First Round/7/__MACOSX/.*svg` - Not actual images, intentionally skipped

## Dataset Coverage

### By Split
| Split | Total Examples | Multimodal Problems | With Valid Images | Coverage |
|-------|----------------|---------------------|-------------------|----------|
| Train | 908 | 268 | 267 | 99.6% |
| Test  | 227 | 64  | 64  | 100% |
| **Total** | **1135** | **332** | **331** | **99.7%** |

### Missing Images by Problem
**1 total image missing** from **1 multimodal problem**:
- `First Round/7/56.svg` - Never existed in source data (MongoDB issue)

## Why This Matters

### Impact on Model Evaluation
- **99.7% of multimodal problems** can be evaluated correctly
- **0.3% of multimodal problems** (1 problem) will fail to load image
- This problem will appear as text-only, minimal impact on:
  - Model accuracy (only 1 problem affected)
  - Dataset statistics (negligible skew)

### Data Quality
- **Excellent:** All 389 included images are verified valid (no all-black/all-white)
- **Issue:** Only 1 problem missing visual content (never existed in source)

## How to Fix

**✅ STATUS: COMPLETED!** Browser conversion was successfully executed and achieved 99.7% coverage.

The instructions below are preserved for reproducibility and future reference.

---

### ✅ Recommended: Option 1 - Browser-Based Conversion (Most Reliable) - **COMPLETED**

The `inoi-dataset-evolution` repository already has production-grade browser rendering that handles these problematic SVGs.

#### Step 1: Install Dependencies
```bash
cd /scratch/pxm5426/repos/verifiers/environments/inoi
uv pip install playwright
playwright install chromium
```

#### Step 2: Run Browser-Based Conversion
Use the existing `inoi-dataset-evolution/svg_to_png.py` script with `--backend browser`:

```bash
# Create conversion script
cat > convert_failed_svgs.sh << 'EOF'
#!/bin/bash

INOI_EVOLUTION="/scratch/pxm5426/repos/verifiers/inoi-dataset-evolution"
ASSETS_DIR="assets/First Round"
CONVERTER="$INOI_EVOLUTION/svg_to_png.py"

# List of directories with failed conversions
DIRS=(5 6 7 17 18 19 20 26 28 29 30 32 34)

total=0
success=0

for dir in "${DIRS[@]}"; do
  echo "Processing First Round/$dir..."
  for svg in "$ASSETS_DIR/$dir"/*.svg; do
    if [ -f "$svg" ]; then
      png="${svg%.svg}.png"
      # Skip if PNG already exists and is valid
      if [ -f "$png" ]; then
        echo "  Skipping $svg (PNG exists)"
        continue
      fi

      total=$((total + 1))
      echo "  Converting: $(basename $svg)"

      # Use browser backend (slow but reliable)
      if python "$CONVERTER" --input "$svg" --output "$png" --backend browser; then
        success=$((success + 1))
        echo "    ✓ Success"
      else
        echo "    ✗ Failed"
      fi
    fi
  done
done

echo ""
echo "================================"
echo "Conversion Complete"
echo "================================"
echo "Total: $total"
echo "Success: $success"
echo "Failed: $((total - success))"
EOF

chmod +x convert_failed_svgs.sh
./convert_failed_svgs.sh
```

**Time Estimate:** ~25-30 minutes (31s per SVG × 47 files)

**Expected Result:** 99.7% coverage (331/332 multimodal problems)

**Why This Works:**
- Browser uses real Chrome rendering engine (same as what humans see)
- Handles complex CSS that cairosvg doesn't support
- Automatic white background, proper bounding box cropping
- Same backend used in `inoi-dataset-evolution` production

---

### Option 2: Python Script with Error Recovery

If you want more control and progress tracking:

```python
# save as: fix_failed_conversions.py
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

CONVERTER = Path("/scratch/pxm5426/repos/verifiers/inoi-dataset-evolution/svg_to_png.py")
ASSETS = Path("assets/First Round")

# Directories with failed conversions
FAILED_DIRS = [5, 6, 7, 17, 18, 19, 20, 26, 28, 29, 30, 32, 34]

def convert_svg_browser(svg_path, png_path):
    """Convert SVG using browser backend."""
    result = subprocess.run(
        [sys.executable, str(CONVERTER),
         "--input", str(svg_path),
         "--output", str(png_path),
         "--backend", "browser"],
        capture_output=True,
        text=True,
        timeout=60  # 60s timeout per SVG
    )
    return result.returncode == 0

def main():
    # Find all SVGs without valid PNGs
    to_convert = []
    for dir_num in FAILED_DIRS:
        svg_dir = ASSETS / str(dir_num)
        if not svg_dir.exists():
            continue
        for svg_path in svg_dir.glob("*.svg"):
            png_path = svg_path.with_suffix('.png')
            if not png_path.exists():
                to_convert.append((svg_path, png_path))

    print(f"Found {len(to_convert)} SVGs to convert")

    success = 0
    failed = []

    for svg_path, png_path in tqdm(to_convert, desc="Converting"):
        try:
            if convert_svg_browser(svg_path, png_path):
                success += 1
            else:
                failed.append(svg_path)
        except Exception as e:
            print(f"\nError converting {svg_path}: {e}")
            failed.append(svg_path)

    print(f"\n{'='*60}")
    print(f"Conversion Complete")
    print(f"{'='*60}")
    print(f"Total: {len(to_convert)}")
    print(f"Success: {success}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed conversions:")
        for svg in failed:
            print(f"  - {svg.relative_to(ASSETS)}")

if __name__ == "__main__":
    main()
```

Run with: `uv run python fix_failed_conversions.py`

---

### Option 3: Batch Process with Parallel Execution (Faster)

If you have time constraints, parallelize the browser conversions:

```python
# save as: parallel_convert.py
import asyncio
import subprocess
import sys
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

CONVERTER = Path("/scratch/pxm5426/repos/verifiers/inoi-dataset-evolution/svg_to_png.py")
ASSETS = Path("assets/First Round")
FAILED_DIRS = [5, 6, 7, 17, 18, 19, 20, 26, 28, 29, 30, 32, 34]

async def convert_svg(svg_path, png_path):
    """Async conversion wrapper."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(CONVERTER),
        "--input", str(svg_path),
        "--output", str(png_path),
        "--backend", "browser",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()
    return proc.returncode == 0

async def main():
    to_convert = []
    for dir_num in FAILED_DIRS:
        svg_dir = ASSETS / str(dir_num)
        if svg_dir.exists():
            for svg in svg_dir.glob("*.svg"):
                png = svg.with_suffix('.png')
                if not png.exists():
                    to_convert.append((svg, png))

    print(f"Converting {len(to_convert)} SVGs with 4 parallel workers...")

    # Process 4 at a time to avoid overwhelming system
    semaphore = asyncio.Semaphore(4)

    async def bounded_convert(svg, png):
        async with semaphore:
            return await convert_svg(svg, png)

    tasks = [bounded_convert(svg, png) for svg, png in to_convert]
    results = await tqdm_asyncio.gather(*tasks)

    success = sum(results)
    print(f"\n✓ Success: {success}/{len(to_convert)}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Time Estimate:** ~7-10 minutes (4 parallel workers, 31s each)

Run with: `uv run python parallel_convert.py`

---

### Option 4: Alternative - ImageMagick (If Available)

If ImageMagick is installed on the system, it often handles SVGs well:

```bash
# Check if ImageMagick is available
convert -version

# Batch convert with ImageMagick
for svg in assets/First\ Round/{5,6,7,17,18,19,20,26,28,29,30,32,34}/*.svg; do
  png="${svg%.svg}.png"
  if [ ! -f "$png" ]; then
    echo "Converting: $svg"
    convert -background white -density 300 "$svg" "$png"
  fi
done
```

**Pros:** Fast, widely available
**Cons:** May not be installed on cluster, quality varies

---

### Option 5: Manual Verification + Selective Conversion

For the most critical subset, manually verify which ones actually need images:

```python
# save as: prioritize_conversions.py
from datasets import load_dataset
from pathlib import Path

# Load dataset to see which images are actually referenced
ds = load_dataset("pmahdavi/inoi")

missing_by_problem = {}
for split in ["train", "test"]:
    for idx, ex in enumerate(ds[split]):
        if not ex.get("images_list"):
            continue

        exam_dir = ex["exam_directory"].replace("\\", "/")
        missing = []

        for img_file in ex["images_list"]:
            png_path = Path("assets") / exam_dir / img_file.replace(".svg", ".png")
            if not png_path.exists():
                missing.append(img_file)

        if missing:
            problem_id = f"{split}/{idx}: {exam_dir}/{ex.get('problem_number', '?')}"
            missing_by_problem[problem_id] = missing

print(f"Problems missing images: {len(missing_by_problem)}")
print("\nMost critical (multiple missing images):")
for problem, images in sorted(missing_by_problem.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
    print(f"  {problem}: {len(images)} images")
    for img in images:
        print(f"    - {img}")
```

This helps you prioritize which conversions matter most for dataset completeness.

---

## Recommended Workflow

1. **Quick Check:** Run Option 5 to see which problems are most affected
2. **Automated Fix:** Run Option 2 (Python script with progress bar)
3. **Verify Quality:** Check converted PNGs aren't all-white/all-black:
   ```bash
   uv run python -c "
   from pathlib import Path
   from PIL import Image

   for png in Path('assets/First Round').rglob('*.png'):
       img = Image.open(png).convert('RGB')
       extrema = img.getextrema()
       if extrema == ((0,0), (0,0), (0,0)):
           print(f'All-black: {png}')
       elif extrema == ((255,255), (255,255), (255,255)):
           print(f'All-white: {png}')
   "
   ```
4. **Re-upload Dataset:** Run the upload script to push updated images to HuggingFace
5. **Verify Coverage:** Should reach 99.7% (331/332)

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
**Valid PNG Count:** 389/415 (93.7%)
**Multimodal Coverage:** 331/332 (99.7%)
**Conversion Method:** CairoSVG (325) + Browser/Chromium (64)
**Status:** ✅ Complete - Production Ready
