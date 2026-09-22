"""
Evaluation metrics: Exact Match (EM), Accuracy, F1.
"""
import re
import string
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalized prediction exactly matches normalized ground truth."""
    return normalize_text(prediction) == normalize_text(ground_truth)


def contains_match(prediction: str, ground_truth: str) -> bool:
    """Check if ground truth appears in prediction (or vice versa)."""
    norm_pred = normalize_text(prediction)
    norm_gt = normalize_text(ground_truth)
    return norm_gt in norm_pred or norm_pred in norm_gt


def f1_score_tokens(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(predictions: list[str], ground_truths: list[str]) -> dict:
    """Compute all metrics for a set of predictions."""
    n = len(predictions)
    em_count = sum(1 for p, g in zip(predictions, ground_truths) if exact_match(p, g))
    contains_count = sum(1 for p, g in zip(predictions, ground_truths) if contains_match(p, g))
    f1_scores = [f1_score_tokens(p, g) for p, g in zip(predictions, ground_truths)]

    return {
        "total": n,
        "exact_match": round(em_count / n * 100, 2) if n else 0,
        "accuracy": round(contains_count / n * 100, 2) if n else 0,
        "f1": round(sum(f1_scores) / n * 100, 2) if n else 0,
    }
