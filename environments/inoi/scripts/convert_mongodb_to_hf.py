#!/usr/bin/env python3
"""
Convert INOI MongoDB collections to HuggingFace Dataset format.

This script:
1. Connects to MongoDB and loads inoi + synthetic_data collections
2. Processes and joins the data via problem_id cross-reference
3. Handles image references and unified naming
4. Creates a properly formatted HuggingFace dataset compatible with inoi.py

Requirements:
    - pymongo
    - datasets
    - pandas

Usage:
    python convert_mongodb_to_hf.py
"""

import re
from typing import Dict, List, Optional, Tuple
from pymongo import MongoClient
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import pandas as pd
from collections import Counter


# MongoDB connection configuration
MONGODB_URI = "mongodb+srv://mpouria80:pouriamahdavi@math-olympiad-db.tfheu.mongodb.net/"
DATABASE_NAME = "inoi"


class INOIMongoDBConverter:
    """Converter for INOI MongoDB data to HuggingFace format."""
    
    def __init__(self, connection_string: str):
        """Initialize MongoDB connection."""
        self.client = MongoClient(connection_string)
        self.db = self.client[DATABASE_NAME]
        self.inoi_collection = self.db['inoi']
        self.synthetic_collection = self.db['synthetic_data']
        
    def fix_html_images(self, text: str) -> str:
        """Convert HTML img tags to Markdown format."""
        if not text:
            return text
        # Pattern: <img src="filename.ext" ...>
        html_pattern = r'<img[^>]+src="([^"]+)"[^>]*>'
        return re.sub(html_pattern, r'![](\1)', text)
    
    def extract_images_from_text(self, text: str) -> List[str]:
        """Extract all image filenames from text using regex."""
        if not text:
            return []
        
        # First convert HTML to markdown
        text = self.fix_html_images(text)
        
        # Extract markdown images: ![...](filename)
        markdown_pattern = r'!\[.*?\]\(([^)]+)\)'
        images = re.findall(markdown_pattern, text)
        
        # Deduplicate while preserving order
        seen = set()
        unique_images = []
        for img in images:
            if img not in seen:
                unique_images.append(img)
                seen.add(img)
        
        return unique_images
    
    def create_new_image_name(self, exam_dir: str, problem_num: int, 
                             sequence: int, extension: str) -> str:
        """
        Create new standardized image name.
        Format: frXX_pYY_Z.ext or srXX_pYY_Z.ext
        """
        exam_dir = exam_dir.replace('\\', '/').lower()
        
        # Determine round type
        if 'second' in exam_dir:
            round_type = 'sr'
        else:
            round_type = 'fr'
        
        # Extract round number
        parts = exam_dir.split('/')
        round_num = ''.join(filter(str.isdigit, parts[-1])) if parts else '0'
        
        return f"{round_type}{round_num}_p{problem_num}_{sequence}.{extension}"
    
    def process_images_for_problem(self, problem_doc: dict) -> Tuple[List[str], Dict[str, str]]:
        """
        Process all images for a problem and create mapping.
        Returns: (list of new image names, mapping dict old->new)
        """
        # Get combined text (context + problem)
        context = (problem_doc.get('context') or '').strip()
        problem = (problem_doc.get('problem') or '').strip()
        combined = f"{context}\n\n{problem}" if context else problem
        
        # Extract all images from combined text
        images = self.extract_images_from_text(combined)
        
        # Create new standardized names
        exam_dir = problem_doc['exam_directory']
        problem_num = problem_doc['problem_number']
        new_names = []
        mapping = {}
        
        for idx, old_name in enumerate(images):
            # Extract extension
            ext = old_name.split('.')[-1] if '.' in old_name else 'png'
            new_name = self.create_new_image_name(exam_dir, problem_num, idx, ext)
            new_names.append(new_name)
            mapping[old_name] = new_name
        
        return new_names, mapping
    
    def replace_image_references(self, text: str, mapping: Dict[str, str]) -> str:
        """Replace old image names with new ones in text."""
        if not text:
            return text
        
        # First convert HTML to markdown
        text = self.fix_html_images(text)
        
        # Replace each image reference
        for old_name, new_name in mapping.items():
            # Replace in markdown format: ![...](old_name) -> ![...](new_name)
            pattern = r'(!\[.*?\]\()' + re.escape(old_name) + r'(\))'
            replacement = r'\1' + new_name + r'\2'
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def create_problem_field(self, problem_doc: dict, image_mapping: Dict[str, str]) -> str:
        """
        Create the 'problem' field by combining context and problem.
        Note: This is called 'problem' (not 'prompt') to match existing HF schema.
        """
        context = (problem_doc.get('context') or '').strip()
        problem = (problem_doc.get('problem') or '').strip()
        
        # Replace image references in both
        if context:
            context = self.replace_image_references(context, image_mapping)
        problem = self.replace_image_references(problem, image_mapping)
        
        # Combine with clear visual separation
        if context:
            # Use clear separator between context and problem
            result = f"{context}\n\n---\n\n{problem}"
        else:
            result = problem
        
        return result
    
    def get_technique_label(self, synthetic_doc: dict) -> Optional[str]:
        """Extract technique label from synthetic_data.solution_technique_data."""
        solution_technique = synthetic_doc.get('solution_technique_data', {})
        if not solution_technique or not isinstance(solution_technique, dict):
            return None
        
        # Get overall classification
        overall = solution_technique.get('overall_classification')
        return str(overall) if overall else None
    
    def determine_problem_type(self, problem_doc: dict, synthetic_doc: dict) -> str:
        """
        Determine problem type based on exam_directory, answer_type, and choice_dependency.
        Returns: yes-no, mc-standalone, mc-dependent, or second-round
        Appends '-img' if problem has images.
        """
        exam_dir = problem_doc.get('exam_directory', '')
        answer_type = problem_doc.get('answer_type', '')
        
        # Check if it's second round
        if 'second' in exam_dir.lower():
            base_type = 'second-round'
        elif answer_type == 'Yes/No':
            base_type = 'yes-no'
        elif answer_type == 'Multiple_Choice':
            # Check if standalone or choice-dependent
            choice_dep = synthetic_doc.get('choice_dependency_data', {})
            label = choice_dep.get('label', 'standalone')
            
            if label == 'choice-dependent':
                base_type = 'mc-dependent'
            else:
                base_type = 'mc-standalone'
        else:
            base_type = 'unknown'
        
        # Check if problem has images
        images_list = problem_doc.get('images_list')
        if images_list and images_list.strip():
            base_type += '-img'
        
        return base_type
    
    def process_single_problem(self, problem_doc: dict, 
                               synthetic_doc: dict, 
                               idx: int) -> dict:
        """Process a single problem and return HF dataset row."""
        # Create ID in format combiz_0001, combiz_0002, etc.
        hf_id = f"combiz_{idx:04d}"
        
        # Process images and create mapping
        new_image_names, image_mapping = self.process_images_for_problem(problem_doc)
        
        # Create 'problem' field (note: called 'problem' not 'prompt' in HF schema)
        problem_text = self.create_problem_field(problem_doc, image_mapping)
        
        # Get solution from synthetic_data (rewritten_solution)
        solution = synthetic_doc.get('rewritten_solution', '')
        
        # Get technique label
        technique_label = self.get_technique_label(synthetic_doc)
        
        # Determine problem type
        problem_type = self.determine_problem_type(problem_doc, synthetic_doc)
        
        # Get direct fields from inoi collection
        choices = problem_doc.get('choices', '')
        correct_option = problem_doc.get('correct_option')
        answer_value = problem_doc.get('answer_value')
        answer_type = problem_doc.get('answer_type', '')
        
        # Additional metadata for reference
        exam_directory = problem_doc.get('exam_directory', '')
        problem_number = problem_doc.get('problem_number')
        
        return {
            'id': hf_id,
            'problem': problem_text,  # Note: 'problem' not 'prompt'
            'images_list': new_image_names,
            'solution': solution,
            'technique_label': technique_label,
            'problem_type': problem_type,
            'choices': choices,
            'correct_option': correct_option,
            'answer_value': str(answer_value) if answer_value is not None else None,
            'answer_type': answer_type,  # Keep original for compatibility
            'exam_directory': exam_directory,
            'problem_number': problem_number,
            'original_problem_id': str(problem_doc['_id']),
        }
    
    def load_and_process_data(self) -> pd.DataFrame:
        """Load data from MongoDB and process into HF format."""
        print("=" * 80)
        print("Loading data from MongoDB...")
        print("=" * 80)
        
        # Load inoi collection sorted by exam and problem number
        inoi_docs = list(self.inoi_collection.find().sort([
            ('exam_directory', 1), 
            ('problem_number', 1)
        ]))
        print(f"✓ Loaded {len(inoi_docs)} documents from inoi collection")
        
        # Create mapping of problem_id -> synthetic_data
        synthetic_map = {}
        for doc in self.synthetic_collection.find():
            problem_id = doc.get('problem_id')
            if problem_id:
                synthetic_map[problem_id] = doc
        
        print(f"✓ Loaded {len(synthetic_map)} documents from synthetic_data collection")
        
        # Expected count check
        expected_count = 1135
        if len(inoi_docs) != expected_count:
            print(f"⚠ Warning: Expected {expected_count} problems, found {len(inoi_docs)}")
        
        # Process each problem
        processed_data = []
        missing_synthetic = []
        error_problems = []
        
        print("\nProcessing problems...")
        for idx, problem_doc in enumerate(inoi_docs, start=1):
            problem_id = problem_doc['_id']
            synthetic_doc = synthetic_map.get(problem_id)
            
            if not synthetic_doc:
                missing_synthetic.append({
                    'id': str(problem_id),
                    'exam': problem_doc.get('exam_directory'),
                    'problem_num': problem_doc.get('problem_number')
                })
                continue
            
            try:
                row = self.process_single_problem(problem_doc, synthetic_doc, idx)
                processed_data.append(row)
                
                if idx % 100 == 0:
                    print(f"  Processed {idx}/{len(inoi_docs)} problems...")
            except Exception as e:
                error_problems.append({
                    'id': str(problem_id),
                    'exam': problem_doc.get('exam_directory'),
                    'problem_num': problem_doc.get('problem_number'),
                    'error': str(e)
                })
        
        print(f"\n✓ Successfully processed {len(processed_data)} problems")
        
        if missing_synthetic:
            print(f"\n⚠ Warning: {len(missing_synthetic)} problems missing synthetic data")
        
        if error_problems:
            print(f"\n⚠ Warning: {len(error_problems)} problems had processing errors")
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df
    
    def create_hf_dataset(self, train_test_split: float = 0.8) -> Tuple[DatasetDict, pd.DataFrame]:
        """
        Create HuggingFace DatasetDict with train/test splits.
        
        Args:
            train_test_split: Fraction of data for training (default 0.8 = 80/20 split)
        
        Returns:
            Tuple of (DatasetDict with train/test, full DataFrame)
        """
        df = self.load_and_process_data()
        
        # Define features schema
        features = Features({
            'id': Value('string'),
            'problem': Value('string'),  # Note: 'problem' not 'prompt'
            'images_list': Sequence(Value('string')),
            'solution': Value('string'),
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
        
        # Split into train/test
        split_idx = int(len(df) * train_test_split)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"\nCreating train/test splits:")
        print(f"  Train: {len(train_df)} examples ({train_test_split*100:.0f}%)")
        print(f"  Test: {len(test_df)} examples ({(1-train_test_split)*100:.0f}%)")
        
        # Convert to HF Datasets
        train_dataset = Dataset.from_pandas(train_df, features=features)
        test_dataset = Dataset.from_pandas(test_df, features=features)
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
        })
        
        return dataset_dict, df
    
    def generate_statistics(self, df: pd.DataFrame) -> dict:
        """Generate statistics about the processed dataset."""
        stats = {
            'total_problems': len(df),
            'problems_with_images': len(df[df['images_list'].apply(lambda x: len(x) > 0)]),
            'problem_type_distribution': df['problem_type'].value_counts().to_dict(),
            'answer_type_distribution': df['answer_type'].value_counts().to_dict(),
            'unique_exams': df['exam_directory'].nunique(),
            'avg_problem_length_chars': int(df['problem'].str.len().mean()),
            'avg_solution_length_chars': int(df['solution'].str.len().mean()),
            'problems_with_technique_label': int(df['technique_label'].notna().sum()),
            'problems_with_context': int(df['problem'].str.contains('---').sum()),
        }
        return stats


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print(" " * 15 + "INOI MongoDB → HuggingFace Converter")
    print("=" * 80 + "\n")
    
    # Initialize converter
    print(f"Connecting to MongoDB at {MONGODB_URI}...")
    converter = INOIMongoDBConverter(MONGODB_URI)
    
    # Create dataset
    print("\nCreating HuggingFace dataset...")
    dataset_dict, df = converter.create_hf_dataset(train_test_split=0.8)
    
    print("\n" + "=" * 80)
    print("DATASET CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nTrain examples: {len(dataset_dict['train'])}")
    print(f"Test examples: {len(dataset_dict['test'])}")
    print(f"Total: {len(df)}")
    print(f"\nColumns: {', '.join(dataset_dict['train'].column_names)}")
    
    # Generate and display statistics
    stats = converter.generate_statistics(df)
    
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Detailed problem type breakdown
    print("\nProblem Type Distribution:")
    for ptype, count in stats['problem_type_distribution'].items():
        percentage = (count / stats['total_problems']) * 100
        print(f"  {ptype:25s}: {count:4d} ({percentage:5.1f}%)")
    
    # Save dataset
    print("\n" + "=" * 80)
    print("SAVING FILES...")
    print("=" * 80)
    
    output_dir = "inoi_hf_dataset"
    dataset_dict.save_to_disk(output_dir)
    print(f"✓ HuggingFace dataset saved to: {output_dir}/")
    
    df.to_csv("inoi_dataset_preview.csv", index=False)
    print("✓ Preview CSV saved to: inoi_dataset_preview.csv")
    
    # Save image mapping for reference
    with open("image_mapping.txt", "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("IMAGE MAPPING: Old Names → New Names\n")
        f.write("=" * 80 + "\n\n")
        
        for row in df.itertuples():
            if row.images_list:
                f.write(f"Problem {row.id} ({row.exam_directory} P{row.problem_number}):\n")
                for idx, img in enumerate(row.images_list):
                    f.write(f"  [{idx}] {img}\n")
                f.write("\n")
    
    print("✓ Image mapping saved to: image_mapping.txt")
    
    # Sample output
    print("\n" + "=" * 80)
    print("SAMPLE RECORD (First Problem)")
    print("=" * 80)
    print(f"\nID: {df.iloc[0]['id']}")
    print(f"Exam: {df.iloc[0]['exam_directory']}, Problem: {df.iloc[0]['problem_number']}")
    print(f"Type: {df.iloc[0]['problem_type']}")
    print(f"Answer Type: {df.iloc[0]['answer_type']}")
    print(f"Images: {df.iloc[0]['images_list']}")
    print(f"Problem (first 300 chars):\n{df.iloc[0]['problem'][:300]}...")
    print(f"\nSolution (first 200 chars):\n{df.iloc[0]['solution'][:200]}...")
    
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the CSV file: inoi_dataset_preview.csv")
    print("2. Check image mapping: image_mapping.txt")
    print("3. Load dataset:")
    print("   from datasets import load_from_disk")
    print("   dataset = load_from_disk('inoi_hf_dataset')")
    print("4. Upload to HuggingFace Hub:")
    print("   dataset.push_to_hub('your-username/inoi')")
    print("5. Use with inoi.py environment:")
    print("   from environments.inoi import load_environment")
    print("   env = load_environment(dataset_name='your-username/inoi')")
    print("=" * 80 + "\n")
    
    return dataset_dict, df


if __name__ == "__main__":
    try:
        dataset_dict, df = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
