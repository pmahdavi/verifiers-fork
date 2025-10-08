import base64
import os
import re
from io import BytesIO
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
import requests
from PIL import Image
import pymongo
from datasets import Dataset
import verifiers as vf
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
)
from math_verify import parse, verify
from math_verify import LatexExtractionConfig, ExprExtractionConfig


def download_image(url: str) -> Optional[str]:
    """Download image from URL and convert to base64."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        buffer = BytesIO()
        # Convert to PNG for consistency
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None


def extract_and_encode_images(text: str, image_mapping: Dict[str, str]) -> tuple[str, List[Dict[str, Any]]]:
    """Extract image references from text and prepare them for multimodal prompt."""
    # Pattern to match image references like ![](img-0.svg) or ![...](http://...)
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    images = []
    
    def replace_image(match):
        alt_text = match.group(1)
        img_ref = match.group(2)
        
        # Check if it's a URL
        if img_ref.startswith(('http://', 'https://')):
            b64_img = download_image(img_ref)
            if b64_img:
                images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                })
                return f"[Image: {alt_text or 'Figure'}]"
        else:
            # It's a local reference, check if we have it in mapping
            if img_ref in image_mapping:
                b64_img = image_mapping[img_ref]
                images.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                })
                return f"[Image: {alt_text or 'Figure'}]"
        
        # If we can't load the image, keep the reference
        return match.group(0)
    
    # Replace image references
    processed_text = re.sub(img_pattern, replace_image, text)
    
    return processed_text, images


def format_prompt(doc: Dict[str, Any], image_mapping: Dict[str, str] = {}) -> List[Dict[str, Any]]:
    """Format a document into a multimodal prompt."""
    problem_text = doc.get("problem", "")
    choices = doc.get("choices", "")
    answer_type = doc.get("answer_type", "Multiple_Choice")
    
    # Extract and process images from problem text
    processed_problem, problem_images = extract_and_encode_images(problem_text, image_mapping)
    
    # Build the text content
    if answer_type == "Multiple_Choice":
        text_content = f"{processed_problem}\n\n{choices}\n\nProvide the number of the correct option (1-5) inside \\boxed{{}}."
    else:  # Yes/No
        text_content = f"{processed_problem}\n\nAnswer with Yes or No inside \\boxed{{}}."
    
    # Build content - use simple string if no images, multimodal list if images present
    if problem_images:
        # Multimodal format (images present)
        content = [{"type": "text", "text": text_content}]
        content.extend(problem_images)
    else:
        # Simple string format (no images) - compatible with vf-tui
        content = text_content

    prompt = [
        {
            "role": "user",
            "content": content
        }
    ]

    return prompt


# ========================= Math Verification Helpers =========================
_COMBINED_EXTRACT_CONFIGS = [LatexExtractionConfig(), ExprExtractionConfig()]


def _strip_math_wrappers(text: str) -> str:
    """Remove common LaTeX math wrappers."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = re.sub(r"^\s*\$\$(.*)\$\$\s*$", r"\1", s, flags=re.S)
    s = re.sub(r"^\s*\$(.*)\$\s*$", r"\1", s, flags=re.S)
    s = re.sub(r"^\s*\\\((.*)\\\)\s*$", r"\1", s, flags=re.S)
    s = re.sub(r"^\s*\\\[(.*)\\\]\s*$", r"\1", s, flags=re.S)
    s = s.replace("\\left", "").replace("\\right", "")
    return s.strip()


def _parse_expr_with_configs(text: str):
    """Parse expression with multiple config attempts."""
    last_err = None
    try:
        return parse(text, extraction_config=_COMBINED_EXTRACT_CONFIGS, parsing_timeout=None, raise_on_error=True)
    except Exception as e:
        last_err = e
    try:
        return parse(text, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()], parsing_timeout=None, raise_on_error=True)
    except Exception as e:
        last_err = e
    stripped = _strip_math_wrappers(text)
    if stripped and stripped != text:
        try:
            return parse(stripped, extraction_config=[ExprExtractionConfig()], parsing_timeout=None, raise_on_error=True)
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("parse failed without exception")


def expressions_equivalent(gold_text: str, pred_text: str) -> bool:
    """Check if two mathematical expressions are equivalent."""
    try:
        gold_node = _parse_expr_with_configs(str(gold_text))
        pred_node = _parse_expr_with_configs(str(pred_text))
        return bool(verify(gold_node, pred_node, timeout_seconds=None, raise_on_error=True))
    except Exception:
        return False


def _to_float_simple(expr: str) -> Optional[float]:
    """Convert simple numeric expressions to float."""
    if not isinstance(expr, str):
        return None
    s = expr.strip().replace(",", "")
    if "=" in s and s.count("=") == 1:
        left, right = s.split("=", 1)
        if left.strip() and right.strip():
            s = right.strip()
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)", s):
        try:
            return float(s)
        except ValueError:
            return None
    if re.fullmatch(r"[+-]?\d+/[+-]?\d+", s):
        try:
            from fractions import Fraction
            return float(Fraction(s))
        except Exception:
            return None
    return None


def _looks_like_latex(text: str) -> bool:
    """Check if text looks like LaTeX."""
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if "$" in s or "\\(" in s or "\\[" in s:
        return True
    latex_tokens = ["\\frac", "\\sqrt", "\\binom", "\\pi", "\\times", "\\cdot", "^{", "_{", "\\left", "\\right"]
    return any(tok in s for tok in latex_tokens)


def _remove_left_right(text: str) -> str:
    """Remove \\left and \\right commands."""
    if not isinstance(text, str):
        return ""
    return text.replace("\\left", "").replace("\\right", "")


def _preprocess_variants(text: str) -> List[str]:
    """Generate preprocessing variants for matching."""
    variants: List[str] = []
    if not isinstance(text, str):
        return variants
    base = text.strip()
    if not base:
        return variants

    def _add(v: Optional[str]):
        if v is None:
            return
        v2 = v.strip()
        if v2 and v2 not in variants:
            variants.append(v2)

    _add(base)
    stripped = _strip_math_wrappers(base)
    _add(stripped)

    _add(_remove_left_right(base))
    _add(_remove_left_right(stripped))

    if _looks_like_latex(base) and not (base.startswith("$") and base.endswith("$")):
        _add(f"${base}$")
    if _looks_like_latex(stripped) and not (stripped.startswith("$") and stripped.endswith("$")):
        _add(f"${stripped}$")

    return variants


def verify_expression_with_math_verify(extracted_answer: Any, ground_truth: Any) -> bool:
    """Verify if extracted answer matches ground truth using math_verify."""
    if ground_truth in (None, ""):
        return False

    gold_text = str(ground_truth)
    pred_text = str(extracted_answer)

    gold_variants = _preprocess_variants(gold_text)
    pred_variants = _preprocess_variants(pred_text)

    for gv in gold_variants or [gold_text]:
        for pv in pred_variants or [pred_text]:
            if expressions_equivalent(gv, pv):
                return True

    a = _to_float_simple(pred_text)
    b = _to_float_simple(gold_text)
    if a is not None and b is not None:
        return abs(a - b) < 1e-6

    return False


# ========================= Choice Parsing =========================
RE_TEXT_WRAP = re.compile(r'^\s*\\text\s*\{\s*(.*)\s*\}\s*$', re.DOTALL)


def unwrap_text_wrapper(s: str) -> str:
    """Unwrap \\text{} wrapper."""
    m = RE_TEXT_WRAP.match(s)
    return m.group(1) if m else s


CHOICE_PREFIX = re.compile(r'^\s*(?:([1-5])|([A-Ea-e]))[.)]\s*(.+?)(?:\s*\.\s*)?$', re.UNICODE)
CHOICE_ANY_EXPR = re.compile(r'^\s*(.+?)(?:\s*\.\s*)?$', re.UNICODE)


def parse_choice(line: str):
    """Parse a choice line to extract option number and value."""
    inner = unwrap_text_wrapper(line)

    m = CHOICE_PREFIX.match(inner)
    if m:
        num, letter, expr = m.groups()
        n = int(num) if num else (ord(letter.upper()) - 64)  # A→1 … E→5
        return [n, expr]

    m = CHOICE_ANY_EXPR.match(inner)
    if not m:
        return None
    expr = m.group(1)
    return [None, expr]


def parse_choices_block(s: str) -> dict:
    """Parse a block of choices into a dictionary."""
    parts = [p.strip() for p in s.split(";") if p.strip()]
    result: dict = {}
    unlabeled = []
    for part in parts:
        res = parse_choice(part)
        if not res:
            continue
        num, expr = res
        if num is None:
            unlabeled.append(expr)
        else:
            result[expr.strip()] = num  # standardized to 1..5
    if unlabeled:
        result["_unlabeled"] = unlabeled
    return result


# ========================= MongoDB Data Loading =========================


def load_mongodb_data(
    connection_string: str,
    database: str = "inoi",
    collection: str = "inoi",
    exam_directory: Optional[str] = None,
    answer_type: Optional[str] = None,
    limit: Optional[int] = None,
    reviewed_only: bool = False
) -> List[Dict[str, Any]]:
    """Load data from MongoDB with optional filters."""
    client = pymongo.MongoClient(connection_string)
    db = client[database]
    coll = db[collection]
    
    # Build filter
    filter_query = {}
    if exam_directory:
        filter_query["exam_directory"] = exam_directory
    if answer_type:
        filter_query["answer_type"] = answer_type
    if reviewed_only:
        filter_query["is_reviewed"] = True
    
    # Query data
    cursor = coll.find(filter_query)
    if limit:
        cursor = cursor.limit(limit)
    
    data = list(cursor)
    client.close()
    
    return data


def load_environment(
    connection_string: Optional[str] = None,
    exam_directory: Optional[str] = None,
    answer_type: Optional[str] = None,  # "Multiple_Choice" or "Yes/No"
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    split_ratio: float = 0.8,
    reviewed_only: bool = False,
    use_think: bool = True,
    system_prompt: str = BOXED_SYSTEM_PROMPT,
) -> vf.Environment:
    """
    Load INOI environment from MongoDB.
    
    Args:
        connection_string: MongoDB connection string (defaults to localhost)
        exam_directory: Filter by exam directory (e.g., "First Round\\10")
        answer_type: Filter by answer type ("Multiple_Choice" or "Yes/No")
        num_train_examples: Number of training examples (-1 for all)
        num_eval_examples: Number of evaluation examples (-1 for all)
        split_ratio: Train/eval split ratio if not using separate splits
        reviewed_only: Only use reviewed problems
        use_think: Whether to use ThinkParser
        system_prompt: System prompt to use
    """
    if connection_string is None:
        connection_string = "mongodb://localhost:27017/"
    
    # Load all data
    all_data = load_mongodb_data(
        connection_string=connection_string,
        exam_directory=exam_directory,
        answer_type=answer_type,
        reviewed_only=reviewed_only
    )
    
    if not all_data:
        raise ValueError("No data found with the given filters")
    
    # TODO: In a real implementation, we'd need to handle image loading
    # For now, we'll create an empty image mapping
    image_mapping = {}
    
    # Convert to dataset format
    dataset_items = []
    for doc in all_data:
        # Get the correct answer
        if doc["answer_type"] == "Multiple_Choice":
            answer = str(doc["correct_option"])
        else:  # Yes/No
            # Assuming answer_value contains Yes/No for these questions
            answer = str(doc["answer_value"])

        # Put grading-related fields in info dict for rubric access
        info_dict = {
            "answer_type": doc.get("answer_type"),
            "choices": doc.get("choices", ""),
            "answer_value": str(doc.get("answer_value", "")),
            "correct_option": str(doc.get("correct_option", "")),
            "problem_number": doc.get("problem_number"),
            "exam_directory": doc.get("exam_directory"),
        }

        item = {
            "prompt": format_prompt(doc, image_mapping),
            "answer": answer,
            "info": info_dict,
        }
        dataset_items.append(item)
    
    # Create dataset
    full_dataset = Dataset.from_list(dataset_items)
    
    # Split into train/eval
    if len(full_dataset) > 1:
        split_point = int(len(full_dataset) * split_ratio)
        train_dataset = full_dataset.select(range(split_point))
        eval_dataset = full_dataset.select(range(split_point, len(full_dataset)))
    else:
        train_dataset = full_dataset
        eval_dataset = full_dataset
    
    # Apply limits
    if num_train_examples != -1:
        train_dataset = train_dataset.select(range(min(num_train_examples, len(train_dataset))))
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(min(num_eval_examples, len(eval_dataset))))
    
    # Setup parser
    if use_think:
        parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    else:
        parser = vf.Parser(extract_fn=extract_boxed_answer)
    
    # Define sophisticated reward function matching dataset-evolution grading
    def correct_answer_reward_func(parser, completion, answer, info=None, **kwargs):
        """
        Sophisticated grading that matches the dataset-evolution logic.
        Handles choice parsing, expression equivalence, and multiple answer formats.
        """
        # Extract grading fields from info dict
        if info is None:
            info = {}
        answer_type = info.get("answer_type")
        choices = info.get("choices", "")
        answer_value = info.get("answer_value", "")
        correct_option = info.get("correct_option", "")

        # Extract the boxed answer from completion
        boxed_match = parser.parse_answer(completion)
        if not boxed_match:
            return 0.0

        # Parse the extracted answer
        parsed = parse_choice(boxed_match)
        if not parsed:
            # Try direct string matching
            if answer_type == "Yes/No":
                response = boxed_match.strip().lower()
                target = answer_value.strip().lower()
                if response in ['yes', 'y', '1', 'true'] and target in ['yes', 'y', '1', 'true']:
                    return 1.0
                if response in ['no', 'n', '0', 'false'] and target in ['no', 'n', '0', 'false']:
                    return 1.0
            return 0.0

        standardized_choice_key, raw_value = parsed

        # If we got a direct choice number (1-5 or A-E), check against correct_option
        if standardized_choice_key is not None:
            if str(standardized_choice_key) == correct_option:
                return 1.0

        # If raw_value is a plain number 1-5, treat it as a choice number
        if raw_value and raw_value.strip() in ['1', '2', '3', '4', '5']:
            if raw_value.strip() == correct_option:
                return 1.0

        # Try to match against answer_value using expression equivalence
        if raw_value and answer_value:
            if verify_expression_with_math_verify(raw_value, answer_value):
                return 1.0

        # If no direct match, try matching the value against choices
        if choices and not standardized_choice_key:
            parsed_choices = parse_choices_block(choices)
            for choice_expr, choice_num in parsed_choices.items():
                if choice_expr == "_unlabeled":
                    continue
                if verify_expression_with_math_verify(choice_expr, raw_value):
                    if str(choice_num) == correct_option:
                        return 1.0

        return 0.0
    
    # Create rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[correct_answer_reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],
    )
    
    # Create environment
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    
    return vf_env
