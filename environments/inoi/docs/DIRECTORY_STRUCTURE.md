# INOI Directory Structure Guide

**Last Updated:** October 21, 2025  
**Status:** ✅ Production-Ready

---

## 📁 Directory Overview

```
environments/inoi/
│
├── 📂 assets/                      # Images (1,228 PNG files, 31 MB)
│   ├── fr1_p1_0.png                # Problem images
│   ├── fr1_p1_sol0.png             # Solution images
│   └── ... (1,228 total images)
│
├── 📂 outputs/                     # Evaluation results (5.2 MB)
│   └── evals/
│       └── inoi--gemini-2.5-flash/
│           └── [evaluation runs]
│
├── 📂 data/                        # Datasets
│   └── inoi_dataset_solution_short_fixed/  # Final dataset (41 MB)
│       ├── train/                  # Training split (908 examples)
│       └── test/                   # Test split (227 examples)
│
├── 📂 scripts/                     # Utility scripts
│   ├── convert_mongodb_to_hf.py   # MongoDB → HuggingFace converter
│   ├── upload_to_hf.py            # Upload dataset to HF Hub
│   └── simple_browser_convert.py  # SVG → PNG converter
│
├── 📂 docs/                        # Documentation
│   ├── DATASET_CARD.md            # HuggingFace dataset card
│   ├── FINAL_DATASET_STATUS.md    # Comprehensive verification report
│   ├── DATABASE_EXPLORATION_SUMMARY.md  # MongoDB structure reference
│   ├── MONGODB_CONVERSION.md      # Conversion process guide
│   ├── SOLUTION_SHORT_FIX_REPORT.md     # Critical bug fix docs
│   ├── CLEANUP_SUMMARY.md         # Directory cleanup report
│   └── DIRECTORY_STRUCTURE.md     # This file
│
├── 🐍 inoi.py                      # Main environment implementation
├── 📦 pyproject.toml               # Package configuration
└── 📖 README.md                    # Main documentation

```

---

## 📊 Size Breakdown

| Directory/File | Size | Purpose |
|----------------|------|---------|
| `data/` | 41 MB | Final HuggingFace dataset |
| `assets/` | 31 MB | 1,228 PNG images (485 problem + 743 solution) |
| `outputs/` | 5.2 MB | Evaluation results |
| `scripts/` | <1 MB | Utility scripts (3 files) |
| `docs/` | <1 MB | Documentation (7 files) |
| Root files | <1 MB | Core code and config |
| **Total** | **~77 MB** | Clean, organized structure |

---

## 🎯 Design Principles

### 1. Separation of Concerns
Each directory has a single, clear purpose:
- **Code**: Main implementation (`inoi.py`) at root
- **Scripts**: Utilities in `scripts/`
- **Documentation**: All docs in `docs/`
- **Data**: Datasets in `data/`
- **Assets**: Images in `assets/`
- **Results**: Evaluations in `outputs/`

### 2. Clean Root Level
Only essential files at root:
- `inoi.py` - What users import
- `pyproject.toml` - Package configuration
- `README.md` - Entry point documentation

Everything else is organized in subdirectories.

### 3. Professional Standards
Follows Python project best practices:
- Clear hierarchy
- Logical grouping
- Easy navigation
- Scalable structure

---

## 📂 Directory Purposes

### `assets/`
**Purpose:** Store all image files  
**Content:** 1,228 PNG images (problem and solution images)  
**Format:** `fr{round}_p{num}_{seq}.png` for problems, `fr{round}_p{num}_sol{seq}.png` for solutions  
**Size:** 31 MB

### `outputs/`
**Purpose:** Store evaluation results  
**Content:** Timestamped evaluation runs with metadata and results  
**Format:** `outputs/evals/{env}--{model}/{run_id}/`  
**Size:** 5.2 MB

### `data/`
**Purpose:** Store dataset versions  
**Content:** Final HuggingFace dataset (train/test splits)  
**Format:** HuggingFace Datasets format (.arrow files)  
**Size:** 41 MB

### `scripts/`
**Purpose:** Utility scripts for dataset management  
**Content:**
- `convert_mongodb_to_hf.py` - Convert from MongoDB to HuggingFace
- `upload_to_hf.py` - Upload dataset to HuggingFace Hub
- `simple_browser_convert.py` - Convert SVG images to PNG

### `docs/`
**Purpose:** All project documentation  
**Content:**
- Dataset card for HuggingFace
- Comprehensive status reports
- Conversion guides
- Fix documentation
- This structure guide

---

## 🚀 Usage Guide

### For Users (Installing the Environment)

```bash
cd environments/inoi
uv pip install -e .
```

Then in Python:
```python
from environments.inoi.inoi import load_environment

env = load_environment(
    dataset_name="combviz/inoi",
    num_train_examples=100
)
```

### For Contributors (Working with Scripts)

```bash
# Convert MongoDB data to HuggingFace format
python scripts/convert_mongodb_to_hf.py

# Upload dataset to HuggingFace Hub
python scripts/upload_to_hf.py --repo combviz/inoi

# Convert SVG images to PNG
python scripts/simple_browser_convert.py
```

### For Researchers (Understanding the Dataset)

1. **Start with:** `README.md` - Overview and quick start
2. **Details:** `docs/DATASET_CARD.md` - Complete dataset documentation
3. **Status:** `docs/FINAL_DATASET_STATUS.md` - Comprehensive verification
4. **Process:** `docs/MONGODB_CONVERSION.md` - How it was created

---

## 📝 File Organization Rules

### What Goes Where?

| File Type | Location | Example |
|-----------|----------|---------|
| Main environment code | Root | `inoi.py` |
| Package config | Root | `pyproject.toml` |
| Main documentation | Root | `README.md` |
| Utility scripts | `scripts/` | `convert_mongodb_to_hf.py` |
| Documentation | `docs/` | `DATASET_CARD.md` |
| Images | `assets/` | `fr10_p3_0.png` |
| Datasets | `data/` | `inoi_dataset_solution_short_fixed/` |
| Evaluation results | `outputs/` | `evals/inoi--gpt-4/...` |

### What NOT to Keep?

❌ Temporary files  
❌ Old dataset versions  
❌ Intermediate processing files  
❌ Redundant documentation  
❌ Working/scratch files

---

## 🔄 Maintenance Guidelines

### Adding New Files

1. **New Script:**
   - Add to `scripts/`
   - Update `README.md` Scripts section
   - Add usage example

2. **New Documentation:**
   - Add to `docs/`
   - Update `README.md` Documentation section
   - Keep concise and focused

3. **New Dataset Version:**
   - Add to `data/`
   - Delete old version (keep only latest)
   - Update references

### Cleanup Checklist

Before committing changes:
- [ ] Remove temporary files
- [ ] Keep only latest dataset version
- [ ] Update documentation references
- [ ] Verify all paths in README.md
- [ ] Check for redundant files

---

## 📈 Structure Evolution

### Before Cleanup (October 21, 2025 AM)
- **Files:** ~50 scattered files
- **Datasets:** 7 versions (181 MB)
- **Documentation:** 26 files at root
- **Structure:** Cluttered, hard to navigate

### After Cleanup (October 21, 2025 PM)
- **Files:** 14 organized files
- **Datasets:** 1 final version (41 MB)
- **Documentation:** 6 files in `docs/`
- **Structure:** Clean, professional

### After Reorganization (October 21, 2025 PM)
- **Root:** Only 3 essential files
- **Subdirectories:** Clear separation
- **Navigation:** Easy and intuitive
- **Structure:** Production-ready

---

## ✅ Benefits of Current Structure

1. **Professional Organization**
   - Follows Python best practices
   - Clear hierarchy
   - Easy to understand

2. **Easy Navigation**
   - Related files grouped together
   - Clear directory purposes
   - Minimal root clutter

3. **Scalability**
   - Easy to add new scripts
   - Simple to add documentation
   - Clear data management

4. **Maintainability**
   - One source of truth
   - Clear file purposes
   - Easy cleanup

5. **Collaboration-Friendly**
   - Intuitive structure
   - Clear conventions
   - Well-documented

---

## 🎯 Quick Reference

| Need to... | Go to... |
|------------|----------|
| Import the environment | `inoi.py` |
| Understand the dataset | `README.md` → `docs/DATASET_CARD.md` |
| Convert MongoDB data | `scripts/convert_mongodb_to_hf.py` |
| Upload to HuggingFace | `scripts/upload_to_hf.py` |
| View images | `assets/` |
| Access dataset locally | `data/inoi_dataset_solution_short_fixed/` |
| Check evaluation results | `outputs/evals/` |
| Read documentation | `docs/` |

---

**Structure Designed:** October 21, 2025  
**Status:** ✅ Production-Ready  
**Maintenance:** Follow guidelines above

