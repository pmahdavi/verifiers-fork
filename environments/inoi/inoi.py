"""
INOI Environment for mathematical olympiad problems with multimodal support.

This environment implements the evaluation logic from the INOI dataset evolution project,
which was used to evaluate multiple LLMs (Gemini, GPT, Gemma) on 1,135 mathematical
olympiad problems from the Iranian National Olympiad in Informatics (2006-2024).

## Key Features

### Choice Dependency Logic
The environment implements a critical distinction between two types of multiple choice problems:

1. **mc-standalone**: Problems where choices should NOT be shown to the model
   - Model must solve the problem independently and provide the numerical answer
   - Grading checks if the answer matches the `answer_value` field
   - Example: "Find the sum of all positive integers n such that..."
   
2. **mc-dependent**: Problems where choices MUST be shown to the model
   - Choices are integral to the problem (e.g., "Which of the following...")
   - Model must select the correct choice number (1-5)
   - Grading checks if the choice number matches the `correct_option` field

This distinction prevents the model from accidentally seeing answer options when it
should solve the problem independently, matching the evaluation methodology.

### Problem Types
- `mc-standalone`: Standalone multiple choice (no choices shown)
- `mc-dependent`: Choice-dependent multiple choice (choices shown)
- `mc-standalone-img`: Standalone with images
- `mc-dependent-img`: Choice-dependent with images
- `yes-no`: Yes/No questions (no images)
- `yes-no-img`: Yes/No questions with images

### Grading System
The environment uses a multi-strategy grading approach:

1. **Direct match**: For choice numbers (1-5) and Yes/No answers
2. **Math-verify symbolic**: Algebraic equivalence check
   - Handles: x^2 + 2x + 1 ≡ (x+1)^2
   - Supports: LaTeX, plain math, various notations
3. **Numeric fallback**: Float comparison with 1e-6 tolerance
   - Handles: 1/2 ≡ 0.5, fractions, decimals

### Supported Features
- Multiple choice questions with mathematical expressions
- Yes/No questions
- Multimodal problems with images (PNG format)
- Math expression verification using math_verify
- Nested brace support in \\boxed{} answers
- LaTeX and plain math notation

## Dataset Source
HuggingFace: combviz/inoi
- 1,135 problems (908 train / 227 test)
- 100% image coverage (1,228 embedded PNG images)
- Full solutions with step-by-step explanations
"""

import base64
import logging
import re
from fractions import Fraction
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
from PIL import Image

import verifiers as vf
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, extract_boxed_answer

logger = logging.getLogger("verifiers.envs.inoi")

# ========================= Image Handling =========================


def pil_image_to_base64(pil_img: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        pil_img: PIL Image object

    Returns:
        Base64-encoded PNG image string

    Raises:
        ValueError: If image conversion fails
    """
    try:
        buffer = BytesIO()
        # Convert to RGB if needed (for consistency)
        if pil_img.mode not in ('RGB', 'L'):
            pil_img = pil_img.convert('RGB')
        # Save as PNG
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to convert PIL Image to base64: {e}") from e


def validate_dataset_example(doc: Dict[str, Any]) -> None:
    """
    Validate that a dataset example has required fields.

    Args:
        doc: Dataset example to validate

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ["problem", "answer_type", "problem_type"]
    missing = [f for f in required_fields if f not in doc]
    if missing:
        raise ValueError(f"Dataset example missing required fields: {missing}")

    # Validate answer_type value
    valid_answer_types = ["Multiple_Choice", "Yes/No"]
    answer_type = doc.get("answer_type", "")
    if answer_type not in valid_answer_types:
        logger.warning(f"Unknown answer_type '{answer_type}'. Valid types: {valid_answer_types}")
    
    # Validate problem_type value
    valid_problem_types = [
        "mc-standalone", "mc-dependent", "mc-standalone-img", 
        "mc-dependent-img", "yes-no", "yes-no-img"
    ]
    problem_type = doc.get("problem_type", "")
    if problem_type not in valid_problem_types:
        logger.warning(f"Unknown problem_type '{problem_type}'. Valid types: {valid_problem_types}")


def format_prompt(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format a document into a multimodal prompt.

    Handles PIL Images from HuggingFace Image feature.
    Implements choice dependency logic: shows choices only for mc-dependent problems.

    Args:
        doc: Dataset example with 'problem', 'choices', 'answer_type', 'problem_type', and 'images' (PIL Images)

    Returns:
        List of chat messages (OpenAI format)

    Raises:
        ValueError: If dataset example is invalid or image conversion fails
    """
    # Validate input
    validate_dataset_example(doc)

    problem_text = doc["problem"]
    choices = doc.get("choices", "")
    answer_type = doc.get("answer_type", "")
    problem_type = doc.get("problem_type", "")
    pil_images = doc.get("images", [])

    # Remove markdown image references from problem text (images are separate now)
    # Pattern: ![...](filename.ext)
    processed_problem = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'[Image: \1]', problem_text)

    # Build the text content based on answer type and problem type
    if answer_type == "Multiple_Choice":
        # CRITICAL: Choice dependency logic matching the evaluation system
        # mc-standalone: Do NOT show choices (evaluate based on answer_value)
        # mc-dependent: SHOW choices (evaluate based on correct_option)
        is_standalone = "standalone" in problem_type
        
        if is_standalone:
            # Standalone MC: Don't show choices, expect numeric answer
            text_content = f"{processed_problem}\n\nProvide your answer inside \\boxed{{}}."
        else:
            # Choice-dependent MC: Show choices, expect choice number
            text_content = f"{processed_problem}\n\n{choices}\n\nProvide the number of the correct option (1-5) inside \\boxed{{}}."
    
    elif answer_type == "Yes/No":
        text_content = f"{processed_problem}\n\nAnswer with Yes or No inside \\boxed{{}}."
    
    else:
        # Fallback for unknown format
        text_content = f"{processed_problem}\n\nProvide your answer inside \\boxed{{}}."

    # Always use list format for consistency (required for HuggingFace datasets)
    # This prevents PyArrow errors when mixing string and list types
    content: List[Dict[str, Any]] = [{"type": "text", "text": text_content}]
    
    # Add images if present
    has_images = pil_images and len(pil_images) > 0
    if has_images:
        # Add PIL Images as base64 (HuggingFace Image feature automatically decodes to PIL)
        for idx, pil_img in enumerate(pil_images):
            try:
                b64_img = pil_image_to_base64(pil_img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                })
            except ValueError as e:
                logger.error(f"Failed to encode image {idx}: {e}")
                raise

    prompt = [{"role": "user", "content": content}]
    return prompt


# ========================= Math Verification Helpers =========================

_COMBINED_EXTRACT_CONFIGS = [LatexExtractionConfig(), ExprExtractionConfig()]


def _strip_math_wrappers(text: str) -> str:
    """Remove common LaTeX math wrappers like $...$ and \\[...\\]."""
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

    # Try combined configs
    try:
        return parse(text, extraction_config=_COMBINED_EXTRACT_CONFIGS, parsing_timeout=None, raise_on_error=True)
    except Exception as e:
        last_err = e

    # Try reversed order
    try:
        return parse(text, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()], parsing_timeout=None, raise_on_error=True)
    except Exception as e:
        last_err = e

    # Try stripped version
    stripped = _strip_math_wrappers(text)
    if stripped and stripped != text:
        try:
            return parse(stripped, extraction_config=[ExprExtractionConfig()], parsing_timeout=None, raise_on_error=True)
        except Exception as e:
            last_err = e

    raise last_err if last_err else RuntimeError("parse failed without exception")


def expressions_equivalent(gold_text: str, pred_text: str) -> bool:
    """
    Check if two mathematical expressions are equivalent.

    Args:
        gold_text: Ground truth expression
        pred_text: Predicted expression

    Returns:
        True if expressions are mathematically equivalent
    """
    try:
        gold_node = _parse_expr_with_configs(str(gold_text))
        pred_node = _parse_expr_with_configs(str(pred_text))
        return bool(verify(gold_node, pred_node, timeout_seconds=None, raise_on_error=True))
    except Exception:
        return False


def _to_float_simple(expr: str) -> Optional[float]:
    """
    Convert simple numeric expressions to float.

    Handles:
    - Decimal numbers: "3.14", ".5", "42"
    - Fractions: "1/2", "3/4"
    - Equations: "x = 5" -> 5
    """
    if not isinstance(expr, str):
        return None

    s = expr.strip().replace(",", "")

    # Handle equations like "x = 5"
    if "=" in s and s.count("=") == 1:
        left, right = s.split("=", 1)
        if left.strip() and right.strip():
            s = right.strip()

    # Handle decimal numbers
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)", s):
        try:
            return float(s)
        except ValueError:
            return None

    # Handle fractions
    if re.fullmatch(r"[+-]?\d+/[+-]?\d+", s):
        try:
            return float(Fraction(s))
        except Exception:
            return None

    return None


def _looks_like_latex(text: str) -> bool:
    """Check if text contains LaTeX formatting."""
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
    """
    Generate preprocessing variants for robust matching.

    Returns multiple normalized versions of the input text to increase
    the chance of successful mathematical equivalence checking.
    """
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

    # Add base and stripped versions
    _add(base)
    stripped = _strip_math_wrappers(base)
    _add(stripped)

    # Add versions without \left and \right
    _add(_remove_left_right(base))
    _add(_remove_left_right(stripped))

    # Add dollar-wrapped versions if it looks like LaTeX
    if _looks_like_latex(base) and not (base.startswith("$") and base.endswith("$")):
        _add(f"${base}$")
    if _looks_like_latex(stripped) and not (stripped.startswith("$") and stripped.endswith("$")):
        _add(f"${stripped}$")

    return variants


def verify_expression_with_math_verify(extracted_answer: Any, ground_truth: Any) -> bool:
    """
    Verify if extracted answer matches ground truth using math_verify.

    Tries multiple strategies:
    1. Expression equivalence via math_verify
    2. Numeric comparison for simple numbers

    Args:
        extracted_answer: Student's answer
        ground_truth: Correct answer

    Returns:
        True if answers match
    """
    if ground_truth in (None, ""):
        return False

    gold_text = str(ground_truth)
    pred_text = str(extracted_answer)

    # Try expression equivalence with multiple variants
    gold_variants = _preprocess_variants(gold_text)
    pred_variants = _preprocess_variants(pred_text)

    for gv in gold_variants or [gold_text]:
        for pv in pred_variants or [pred_text]:
            if expressions_equivalent(gv, pv):
                return True

    # Fallback: numeric comparison
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


def parse_choice(line: str) -> Optional[Tuple[Optional[int], str]]:
    """
    Parse a choice line to extract option number and value.

    Args:
        line: Choice text like "1) x=5" or "A. 42" or just "5"

    Returns:
        Tuple of (option_number, expression) or None if parsing fails
        Option number is 1-5, or None if not found
    """
    inner = unwrap_text_wrapper(line)

    # Try to match numbered/lettered choice
    m = CHOICE_PREFIX.match(inner)
    if m:
        num, letter, expr = m.groups()
        n = int(num) if num else (ord(letter.upper()) - 64)  # A→1 … E→5
        return (n, expr)

    # Try to match any expression
    m = CHOICE_ANY_EXPR.match(inner)
    if not m:
        return None
    expr = m.group(1)
    return (None, expr)


# ========================= Grading Functions =========================


def grade_multiple_choice(
    extracted: str,
    correct_option: str,
    answer_value: str
) -> float:
    """
    Grade a multiple choice response.

    Checks in order:
    1. Direct choice number match (e.g., "3" matches option 3)
    2. Math verification of expression against answer_value
    3. Direct string match

    Args:
        extracted: Student's extracted answer from \\boxed{}
        correct_option: Correct option number ("1"-"5")
        answer_value: Actual mathematical value of correct answer

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    # Parse extracted answer for choice number
    parsed = parse_choice(extracted)
    if parsed:
        choice_num, raw_value = parsed

        # Direct choice number match (1-5 or A-E)
        if choice_num and str(choice_num) == correct_option:
            return 1.0

        # Math verification against ground truth answer_value
        if raw_value and verify_expression_with_math_verify(raw_value, answer_value):
            return 1.0

    # Fallback: direct string match for simple numeric answers
    if extracted.strip() == correct_option:
        return 1.0

    return 0.0


def grade_yes_no(extracted: str, answer_value: str) -> float:
    """
    Grade a yes/no response.

    Args:
        extracted: Student's extracted answer from \\boxed{}
        answer_value: Correct answer ("Yes", "No", or variants)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    target = answer_value.strip().lower()
    response = extracted.strip().lower()

    # Flexible yes/no matching
    if response in ['yes', 'y'] and target in ['yes', 'y']:
        return 1.0
    if response in ['no', 'n'] and target in ['no', 'n']:
        return 1.0

    return 0.0


def create_inoi_reward_func(parser: vf.Parser):
    """
    Create the INOI grading function with parser closure.

    Args:
        parser: Parser instance to use for answer extraction

    Returns:
        Reward function compatible with verifiers.Rubric
    """
    def correct_answer_reward_func(parser, completion, answer, info=None, **kwargs) -> float:
        """
        Grading function for INOI environment.

        - Assumes model follows \\boxed{} format (enforced by system prompt)
        - Handles multiple choice and yes/no questions
        - Uses math_verify for expression equivalence

        Args:
            parser: Parser to extract answer
            completion: Model's completion text
            answer: Ground truth answer (for compatibility, not used directly)
            info: Metadata dict with answer_type, correct_option, answer_value

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        if info is None:
            info = {}

        # Extract boxed answer (returns None if not found)
        extracted = parser.parse_answer(completion)
        if not extracted:
            return 0.0

        answer_type = info.get("answer_type", "")

        # Multiple Choice Grading
        if answer_type in ["Multiple_Choice"] or answer_type.startswith("mc-"):
            return grade_multiple_choice(
                extracted,
                info.get("correct_option", ""),
                info.get("answer_value", "")
            )

        # Yes/No Question Grading
        elif answer_type in ["Yes/No"] or answer_type.startswith("yes-no"):
            return grade_yes_no(extracted, info.get("answer_value", ""))

        # Unknown answer type
        logger.warning(f"Unknown answer_type '{answer_type}' in grading")
        return 0.0

    return correct_answer_reward_func


# ========================= Dataset Preparation =========================


def prepare_dataset_from_hf(
    dataset: Dataset,
    question_key: str = "problem",
    answer_key: str = "answer",
) -> Dataset:
    """
    Prepare dataset from HuggingFace format.

    Maps raw dataset to standardized format with prompt, answer, and info.

    Args:
        dataset: Raw HuggingFace dataset
        question_key: Column name for question text
        answer_key: Column name for answer

    Returns:
        Formatted dataset ready for environment
    """
    def format_example(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single example."""
        # Determine answer based on answer_type and problem_type
        answer_type = doc.get("answer_type", "")
        problem_type = doc.get("problem_type", "")
        
        if answer_type == "Multiple_Choice":
            # For mc-dependent: answer is the choice number (correct_option)
            # For mc-standalone: answer is the actual value (answer_value)
            if "dependent" in problem_type:
                answer = str(doc.get("correct_option", ""))
            else:
                answer = str(doc.get("answer_value", ""))
        elif answer_type == "Yes/No":
            answer = str(doc.get("answer_value", ""))
        else:
            # Fallback to answer_key
            answer = str(doc.get(answer_key, ""))

        # Build info dict with grading metadata
        info_dict = {
            "answer_type": doc.get("answer_type", ""),
            "problem_type": doc.get("problem_type", ""),
            "choices": doc.get("choices", ""),
            "answer_value": str(doc.get("answer_value", "")),
            "correct_option": str(doc.get("correct_option", "")),
            "problem_number": doc.get("problem_number"),
            "exam_directory": doc.get("exam_directory"),
        }

        return {
            "prompt": format_prompt(doc),
            "answer": answer,
            "info": info_dict,
        }

    return dataset.map(format_example)


# ========================= Environment Factory =========================


def load_environment(
    dataset_name: str = "combviz/inoi",
    split_train: str = "train",
    split_eval: Optional[str] = "test",
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    use_think: bool = True,
    system_prompt: str = BOXED_SYSTEM_PROMPT,
    filter_multimodal: Optional[bool] = None,
    filter_problem_type: Optional[str] = None,
    **kwargs
) -> vf.Environment:
    """
    Load INOI environment from HuggingFace datasets.

    The environment implements the evaluation logic from the INOI dataset evolution project:
    - mc-standalone: Choices are NOT shown to the model
    - mc-dependent: Choices ARE shown to the model
    - Grading uses math-verify for symbolic/numeric equivalence

    Args:
        dataset_name: HuggingFace dataset identifier (default: "combviz/inoi")
        split_train: Training split name
        split_eval: Evaluation split name (None to use train split)
        num_train_examples: Number of training examples (-1 for all)
        num_eval_examples: Number of evaluation examples (-1 for all)
        use_think: Whether to use ThinkParser for chain-of-thought
        system_prompt: System prompt to use
        filter_multimodal: If True, keep only multimodal; if False, keep only text; if None, keep all
        filter_problem_type: If specified, filter to specific problem type (e.g., "mc-standalone")
        **kwargs: Additional arguments passed to SingleTurnEnv

    Returns:
        Configured INOI environment

    Example:
        >>> env = load_environment(
        ...     dataset_name="combviz/inoi",
        ...     num_train_examples=100,
        ...     filter_problem_type="mc-standalone",
        ...     use_think=True
        ... )
    """
    # Load datasets
    logger.info(f"Loading dataset '{dataset_name}' from HuggingFace")
    train_dataset = load_dataset(dataset_name, split=split_train)

    if split_eval:
        eval_dataset = load_dataset(dataset_name, split=split_eval)
    else:
        eval_dataset = None

    # Apply problem type filtering if specified
    if filter_problem_type is not None:
        logger.info(f"Filtering to problem_type: {filter_problem_type}")
        train_dataset = train_dataset.filter(lambda x: x.get("problem_type") == filter_problem_type)
        if eval_dataset:
            eval_dataset = eval_dataset.filter(lambda x: x.get("problem_type") == filter_problem_type)

    # Apply multimodal filtering if specified
    if filter_multimodal is not None:
        def has_images(example):
            images = example.get("images", [])
            return len(images) > 0 if images is not None else False

        if filter_multimodal:
            logger.info("Filtering to keep only multimodal examples")
            train_dataset = train_dataset.filter(has_images)
            if eval_dataset:
                eval_dataset = eval_dataset.filter(has_images)
        else:
            logger.info("Filtering to keep only text-only examples")
            train_dataset = train_dataset.filter(lambda x: not has_images(x))
            if eval_dataset:
                eval_dataset = eval_dataset.filter(lambda x: not has_images(x))

    # Prepare datasets
    train_dataset = prepare_dataset_from_hf(train_dataset)
    if eval_dataset:
        eval_dataset = prepare_dataset_from_hf(eval_dataset)

    # Apply limits
    if num_train_examples > 0:
        train_dataset = train_dataset.select(range(min(num_train_examples, len(train_dataset))))
    if num_eval_examples > 0 and eval_dataset:
        eval_dataset = eval_dataset.select(range(min(num_eval_examples, len(eval_dataset))))

    logger.info(f"Loaded {len(train_dataset)} training examples")
    if eval_dataset:
        logger.info(f"Loaded {len(eval_dataset)} evaluation examples")

    # Setup parser
    if use_think:
        parser = vf.ThinkParser(extract_fn=extract_boxed_answer)
    else:
        parser = vf.Parser(extract_fn=extract_boxed_answer)

    # Create reward function
    reward_func = create_inoi_reward_func(parser)

    # Create rubric
    rubric = vf.Rubric(
        parser=parser,
        funcs=[reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.0],  # Only reward correct answers, not format
    )

    # Create environment
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )

    logger.info("INOI environment loaded successfully")
    return vf_env
