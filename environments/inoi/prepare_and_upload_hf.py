"""
Prepare and upload INOI dataset to HuggingFace.

This script:
1. Loads an existing HF dataset structure (or creates from scratch)
2. Adds PIL Images from local assets directory
3. Uploads to HuggingFace with proper schema

Usage:
    python prepare_and_upload_hf.py

Requirements:
    - assets/ directory with PNG/SVG images
    - HuggingFace token configured (huggingface-cli login)
"""

from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image as HFImage
from pathlib import Path
from PIL import Image as PILImage
from tqdm import tqdm


def load_images_from_assets(image_filenames, exam_directory, assets_base="assets"):
    """
    Load PIL Images from assets directory.

    Args:
        image_filenames: List of image filenames (e.g., ['img-0.svg', 'img-1.png'])
        exam_directory: Exam directory path (e.g., 'First Round/10')
        assets_base: Base assets directory path

    Returns:
        List of PIL Images
    """
    # Normalize path separators (Windows \ to Unix /)
    exam_directory = exam_directory.replace('\\', '/')

    pil_images = []
    for img_filename in image_filenames:
        img_path = Path(assets_base) / exam_directory / img_filename

        # Prefer PNG over SVG (PNG is pre-converted and faster to load)
        if img_filename.endswith('.svg'):
            png_path = img_path.with_suffix('.png')
            if png_path.exists():
                img_path = png_path

        if img_path.exists():
            pil_img = PILImage.open(img_path)
            # Convert to RGB for consistency
            if pil_img.mode not in ('RGB', 'L'):
                pil_img = pil_img.convert('RGB')
            pil_images.append(pil_img)

    return pil_images


def main():
    print("="*60)
    print("INOI Dataset - HuggingFace Upload")
    print("="*60)

    # Step 1: Load existing dataset (or create from scratch)
    print("\n[1/5] Loading existing dataset structure...")
    from datasets import load_dataset

    try:
        existing = load_dataset("pmahdavi/inoi")
        print(f"   ✓ Loaded {len(existing['train'])} train + {len(existing['test'])} test examples")
    except Exception as e:
        print(f"   ✗ Could not load existing dataset: {e}")
        print("   Please ensure dataset exists or modify script to create from MongoDB")
        return

    # Step 2: Create schema with PIL Images
    print("\n[2/5] Creating schema with PIL Images...")
    existing_features = existing['train'].features

    new_features = Features({
        **{k: v for k, v in existing_features.items() if k != 'images'},
        "images": Sequence(HFImage())  # PIL Images
    })

    # Step 3: Load images from assets
    print("\n[3/5] Loading images from assets directory...")
    new_data = {"train": [], "test": []}

    for split in ["train", "test"]:
        print(f"\n   Processing {split} split...")
        for ex in tqdm(existing[split], desc=f"   {split}"):
            images_list = ex.get("images_list", [])
            exam_dir = ex.get("exam_directory", "")

            # Load PIL Images from assets
            pil_images = load_images_from_assets(images_list, exam_dir)

            # Create new example with PIL Images
            new_ex = {**ex, "images": pil_images}
            new_data[split].append(new_ex)

    # Step 4: Create HuggingFace datasets
    print("\n[4/5] Creating HuggingFace datasets...")
    train_ds = Dataset.from_list(new_data["train"], features=new_features)
    test_ds = Dataset.from_list(new_data["test"], features=new_features)
    dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})

    # Verify coverage
    print("\n[5/5] Verifying coverage...")
    train_multimodal = sum(1 for ex in new_data["train"]
                          if ex.get("images_list") and len(ex["images_list"]) > 0)
    train_has_images = sum(1 for ex in new_data["train"] if len(ex["images"]) > 0)

    test_multimodal = sum(1 for ex in new_data["test"]
                         if ex.get("images_list") and len(ex["images_list"]) > 0)
    test_has_images = sum(1 for ex in new_data["test"] if len(ex["images"]) > 0)

    total_multimodal = train_multimodal + test_multimodal
    total_has = train_has_images + test_has_images

    print(f"\n   Train: {train_has_images}/{train_multimodal} "
          f"({100*train_has_images/train_multimodal:.1f}%)")
    print(f"   Test: {test_has_images}/{test_multimodal} "
          f"({100*test_has_images/test_multimodal:.1f}%)")
    print(f"   Total: {total_has}/{total_multimodal} "
          f"({100*total_has/total_multimodal:.1f}%)")

    # Upload to HuggingFace
    print("\n" + "="*60)
    print("Ready to upload to HuggingFace")
    print("="*60)

    response = input("\nProceed with upload? (yes/no): ")
    if response.lower() != 'yes':
        print("Upload cancelled.")
        return

    print("\nUploading to HuggingFace (this may take a few minutes)...")
    dataset_dict.push_to_hub(
        "pmahdavi/inoi",
        commit_message=f"Updated with {total_has}/{total_multimodal} multimodal problems "
                      f"({100*total_has/total_multimodal:.1f}% coverage)"
    )

    print("\n" + "="*60)
    print("✓ UPLOAD COMPLETE!")
    print("="*60)
    print(f"\nFinal Coverage:")
    print(f"  Train: {train_has_images}/{train_multimodal} "
          f"({100*train_has_images/train_multimodal:.1f}%)")
    print(f"  Test: {test_has_images}/{test_multimodal} "
          f"({100*test_has_images/test_multimodal:.1f}%)")
    print(f"  Total: {total_has}/{total_multimodal} "
          f"({100*total_has/total_multimodal:.1f}%)")

    total_images = (sum(len(ex['images']) for ex in new_data['train']) +
                   sum(len(ex['images']) for ex in new_data['test']))
    print(f"\n  Total images: {total_images}")
    print(f"\nDataset: https://huggingface.co/datasets/pmahdavi/inoi")
    print("="*60)


if __name__ == "__main__":
    main()
