# Changelog

All notable changes to the INOI environment will be documented in this file.

## [Unreleased] - 2025-10-21

### Major Restructuring & Cleanup

#### Directory Restructure
- **BREAKING**: Reorganized project structure for better maintainability
  - Moved all documentation to `docs/` directory
  - Moved all scripts to `scripts/` directory
  - Moved dataset to `data/` directory
  - Root now contains only 3 essential files: `inoi.py`, `pyproject.toml`, `README.md`

#### Cleanup (144 MB disk space freed, 36 files removed)
- Removed 6 old dataset versions (140 MB freed)
- Removed 21 redundant documentation files
- Removed 5 temporary/working files
- Removed `assets_svg/` directory (3.6 MB) - all SVGs converted to PNG
- Removed empty `inoi/` directory

#### Documentation Improvements
- Merged related documentation for clarity
  - `COMPLETE_COVERAGE_REPORT.md` + `MISSING_IMAGE_FIX.md` → `docs/FINAL_DATASET_STATUS.md`
  - `DATASET_EXAMPLES.md` + `example_usage.py` → Enhanced `README.md`
- Enhanced `README.md` with 7 comprehensive usage examples
- Created `docs/DIRECTORY_STRUCTURE.md` - Complete structure guide
- Created `docs/CLEANUP_SUMMARY.md` - Cleanup documentation
- Updated all documentation paths to reflect new structure

#### Dataset Enhancements
- ✅ **100% image coverage** achieved (1,228 images: 485 problem + 743 solution)
- Fixed 225 incorrect image references in `solution_short` field
- Removed all external image URLs (13 fixed)
- Added missing image for `combiz_0896` (fr7_p56_0.png)
- Uploaded complete dataset to HuggingFace Hub: `combviz/inoi`

#### .gitignore Updates
- Added `data/` directory to ignore list
- Added improved patterns for `outputs/` evaluation results
- Better organization of ignore patterns

### Impact Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Root Files | 47 | 3 | -94% |
| Total Files | ~50 | 14 | -72% |
| Disk Space | 220 MB | 77 MB | -65% (144 MB freed) |
| Dataset Versions | 7 | 1 | -86% |
| Documentation Files | 26 | 7 | -73% |

---

## [0.1.1] - 2025-10-21

### Added
- `solution_short` field with concise English solutions
- `solution_images` and `solution_images_list` fields
- Comprehensive image embedding (PIL objects)
- Enhanced dataset card on HuggingFace

### Fixed
- Image reference mismatches in `solution_short` (225 corrections)
- External image URL dependencies (13 fixes)
- Missing image for problem combiz_0896
- SVG to PNG conversion for all images
- Context and problem merging with `---` separator

### Changed
- Dataset repository moved to `combviz/inoi` organization
- All images standardized to PNG format
- Column ordering optimized for HuggingFace viewer
- Image naming convention standardized

---

## [0.1.0] - Initial Release

### Added
- Initial INOI environment implementation
- MongoDB to HuggingFace dataset conversion
- Support for multiple choice and yes/no questions
- Multimodal support with PIL Images
- Math verification using math-verify library
- Train/test splits (908/227 examples)
- 1,135 problems with solutions
- Integration with verifiers framework

### Features
- Single-turn environment for math olympiad problems
- Automatic image encoding to base64 for OpenAI API
- Flexible answer verification (multiple choice, yes/no)
- Rich metadata (problem types, techniques, answers)
- HuggingFace Datasets integration

---

## Format

This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

### Change Types
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

