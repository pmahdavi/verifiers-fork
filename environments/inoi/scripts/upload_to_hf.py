#!/usr/bin/env python3
"""
Upload INOI Dataset to HuggingFace Hub

This script uploads the converted INOI dataset to HuggingFace Hub.
You can choose to upload:
1. Text-only version (with image references) - ~7 MB
2. With embedded PIL Images - ~35 MB

Usage:
    python upload_to_hf.py --repo pmahdavi/inoi-new
    python upload_to_hf.py --repo pmahdavi/inoi-new --with-images
    python upload_to_hf.py --repo ota-merge/inoi --org
"""

import argparse
from pathlib import Path
from datasets import load_from_disk, Features, Value, Sequence, Image as ImageFeature
from PIL import Image
import re


def upload_text_only(dataset_path: str, repo_id: str, private: bool = False):
    """Upload dataset with text references only (no embedded images)."""
    print("=" * 80)
    print("Uploading Text-Only Version (with image references)")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    print(f"   ✓ Loaded: {len(dataset['train'])} train + {len(dataset['test'])} test examples")
    
    # Display sample
    print(f"\n2. Sample record:")
    sample = dataset['train'][0]
    print(f"   ID: {sample['id']}")
    print(f"   Problem images: {sample['images_list']}")
    solution_imgs = re.findall(r'!\[.*?\]\(([^)]+)\)', sample['solution'])
    print(f"   Solution images: {[img for img in solution_imgs if not img.startswith('http')][:3]}")
    
    # Upload
    print(f"\n3. Uploading to {repo_id}...")
    print(f"   Private: {private}")
    
    dataset.push_to_hub(
        repo_id,
        private=private,
        commit_message="Upload INOI dataset with image references"
    )
    
    print(f"\n✓ Upload complete!")
    print(f"\n📦 Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"\nNote: Images are NOT embedded. You need to:")
    print(f"  1. Upload the assets/ directory separately, OR")
    print(f"  2. Run this script again with --with-images flag")


def upload_with_images(dataset_path: str, assets_path: str, repo_id: str, private: bool = False):
    """Upload dataset with embedded PIL Images."""
    print("=" * 80)
    print("Uploading Version with Embedded PIL Images")
    print("=" * 80)
    
    # Load dataset
    print(f"\n1. Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    print(f"   ✓ Loaded: {len(dataset['train'])} train + {len(dataset['test'])} test examples")
    
    # Check assets directory
    assets_dir = Path(assets_path)
    if not assets_dir.exists():
        print(f"\n❌ Error: Assets directory not found: {assets_dir}")
        return
    
    available_images = set(f.name for f in assets_dir.glob('*.png'))
    print(f"   ✓ Found {len(available_images)} images in {assets_dir}")
    
    # Add PIL Images
    print(f"\n2. Adding PIL Image objects...")
    
    def add_pil_images(example):
        """Add PIL Images to the example."""
        # Add problem images
        problem_pil_images = []
        for img_name in example['images_list']:
            img_path = assets_dir / img_name
            if img_path.exists():
                problem_pil_images.append(Image.open(img_path))
            else:
                # Create placeholder for missing images
                problem_pil_images.append(Image.new('RGB', (100, 100), color='white'))
        
        # Extract solution images
        solution_imgs = re.findall(r'!\[.*?\]\(([^)]+)\)', example['solution'])
        local_solution_imgs = [img for img in solution_imgs if not img.startswith('http')]
        
        solution_pil_images = []
        for img_name in local_solution_imgs:
            img_path = assets_dir / img_name
            if img_path.exists():
                solution_pil_images.append(Image.open(img_path))
            else:
                solution_pil_images.append(Image.new('RGB', (100, 100), color='white'))
        
        return {
            **example,
            'images': problem_pil_images,
            'solution_images': solution_pil_images
        }
    
    # Map to add images
    print("   Processing train split...")
    train_with_images = dataset['train'].map(add_pil_images)
    
    print("   Processing test split...")
    test_with_images = dataset['test'].map(add_pil_images)
    
    # Create new dataset with proper features
    from datasets import Dataset, DatasetDict
    
    # Define features with Image type
    features = Features({
        'id': Value('string'),
        'problem': Value('string'),
        'images_list': Sequence(Value('string')),
        'images': Sequence(ImageFeature()),  # PIL Images
        'solution': Value('string'),
        'solution_images': Sequence(ImageFeature()),  # PIL Images for solution
        'technique_label': Value('string'),
        'problem_type': Value('string'),
        'choices': Value('string'),
        'correct_option': Value('int32'),
        'answer_value': Value('string'),
        'answer_type': Value('string'),
        'exam_directory': Value('string'),
        'problem_number': Value('int32'),
        'original_problem_id': Value('string'),
    })
    
    # Create datasets with features
    train_final = Dataset.from_dict(train_with_images.to_dict(), features=features)
    test_final = Dataset.from_dict(test_with_images.to_dict(), features=features)
    
    final_dataset = DatasetDict({
        'train': train_final,
        'test': test_final
    })
    
    print(f"   ✓ Images added to all examples")
    
    # Display sample
    print(f"\n3. Sample record with images:")
    sample = final_dataset['train'][0]
    print(f"   ID: {sample['id']}")
    print(f"   Problem PIL images: {len(sample['images'])} images")
    print(f"   Solution PIL images: {len(sample['solution_images'])} images")
    
    # Upload
    print(f"\n4. Uploading to {repo_id}...")
    print(f"   Private: {private}")
    print(f"   This may take a while (~35 MB)...")
    
    final_dataset.push_to_hub(
        repo_id,
        private=private,
        commit_message="Upload INOI dataset with embedded PIL Images"
    )
    
    print(f"\n✓ Upload complete!")
    print(f"\n📦 Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print(f"\nDataset includes:")
    print(f"  - 'images': PIL Images for problem (from images_list)")
    print(f"  - 'solution_images': PIL Images for solution")
    print(f"  - All text fields with updated references")


def main():
    parser = argparse.ArgumentParser(description="Upload INOI dataset to HuggingFace Hub")
    parser.add_argument(
        '--repo',
        type=str,
        required=True,
        help='HuggingFace repo ID (e.g., pmahdavi/inoi-new or ota-merge/inoi)'
    )
    parser.add_argument(
        '--with-images',
        action='store_true',
        help='Upload with embedded PIL Images (~35 MB) instead of text-only (~7 MB)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the dataset private'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='inoi_hf_dataset',
        help='Path to the HF dataset directory'
    )
    parser.add_argument(
        '--assets-path',
        type=str,
        default='assets',
        help='Path to the assets directory with images'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(" " * 20 + "INOI Dataset Upload to HuggingFace Hub")
    print("=" * 80)
    print(f"\nTarget repository: {args.repo}")
    print(f"Mode: {'With embedded images' if args.with_images else 'Text-only (references)'}")
    print(f"Visibility: {'Private' if args.private else 'Public'}")
    
    # Confirm
    response = input("\nProceed with upload? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Upload cancelled.")
        return
    
    if args.with_images:
        upload_with_images(args.dataset_path, args.assets_path, args.repo, args.private)
    else:
        upload_text_only(args.dataset_path, args.repo, args.private)
    
    print("\n" + "=" * 80)
    print("Done! 🎉")
    print("=" * 80)


if __name__ == "__main__":
    main()

