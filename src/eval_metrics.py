# -*- coding: utf-8 -*-
"""
Evaluation Metrics for Extractive MRC: Exact Match (EM) & Token-level F1
Author: Nhom 7 - CS116 UIT
"""
import re
import string
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def compute_exact_match(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_predictions(predictions, ground_truths):
    total_em = 0.0
    total_f1 = 0.0
    count = len(predictions)

    for qid, pred in predictions.items():
        gts = ground_truths.get(qid, [])
        if not gts:
            continue
        em = max(compute_exact_match(pred, gt) for gt in gts)
        f1 = max(compute_f1(pred, gt) for gt in gts)
        total_em += em
        total_f1 += f1

    em_score = 100.0 * total_em / count if count > 0 else 0.0
    f1_score = 100.0 * total_f1 / count if count > 0 else 0.0

    return {"ExactMatch": em_score, "F1": f1_score}


def compute_answer_length_stats(predictions, ground_truths):
    """Compute statistics on answer lengths and prediction accuracy correlation."""
    pred_lengths = []
    gt_lengths = []
    
    for qid, pred in predictions.items():
        gts = ground_truths.get(qid, [])
        if not gts:
            continue
        pred_lengths.append(len(normalize_text(pred).split()))
        gt_lengths.append(len(normalize_text(gts[0]).split()))
    
    return {
        "pred_avg_length": sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0,
        "gt_avg_length": sum(gt_lengths) / len(gt_lengths) if gt_lengths else 0,
        "pred_lengths": pred_lengths,
        "gt_lengths": gt_lengths
    }


def detailed_error_analysis(predictions, contexts, questions, ground_truths, 
                            output_file="error_analysis.md"):
    """
    Deep Error Analysis for CS221 requirement.
    Classifies errors into categories:
    - Word Boundary Error
    - Context Length Overhead
    - Multi-hop & Paraphrase Failure
    - Ambiguous Ground Truth
    - Model Failure (other)
    """
    errors = []
    
    for i, (qid, pred) in enumerate(predictions.items()):
        gt_list = ground_truths.get(qid, [])
        if not gt_list:
            continue
            
        gt = gt_list[0]
        ctx = contexts.get(qid, "")
        q = questions.get(qid, "")
        
        em = compute_exact_match(pred, gt)
        f1 = compute_f1(pred, gt)
        
        if em < 1:  # Model made an error
            error_type = classify_error(pred, gt, ctx, q)
            errors.append({
                "index": i,
                "question": q,
                "context_preview": ctx[:200] + "..." if len(ctx) > 200 else ctx,
                "prediction": pred,
                "ground_truth": gt,
                "exact_match": em,
                "f1_score": round(f1, 4),
                "error_type": error_type
            })
    
    # Generate report
    _generate_error_report(errors, output_file)
    
    return errors


def classify_error(prediction, ground_truth, context, question):
    """Classify error type based on characteristics."""
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    
    # Check for word boundary issues
    pred_tokens = set(pred_norm.split())
    gt_tokens = set(gt_norm.split())
    
    if pred_tokens and gt_tokens:
        overlap = pred_tokens & gt_tokens
        if len(overlap) > 0 and len(overlap) < len(gt_tokens):
            return "Word Boundary Error"
    
    # Check for context length issues
    if len(context.split()) > 200:
        return "Context Length Overhead"
    
    # Check for multi-hop reasoning
    question_lower = question.lower()
    multi_hop_keywords = ["tại sao", "vì sao", "do đó", "cho nên", "liệu có", "khi nào", "tại sao"]
    if any(w in question_lower for w in multi_hop_keywords):
        return "Multi-hop & Paraphrase Failure"
    
    # Check for ambiguous ground truth
    if len(gt_tokens) == 0 or len(pred_tokens) == 0:
        return "Ambiguous Ground Truth"
    
    return "Model Failure"


def _generate_error_report(errors, output_file):
    """Generate a markdown report of error analysis."""
    error_type_counts = Counter(e["error_type"] for e in errors)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Phân tích lỗi mô hình MRC tiếng Việt\n\n")
        f.write(f"**Tổng số lỗi:** {len(errors)}\n\n")
        f.write("## Phân phối loại lỗi:\n\n")
        
        for error_type, count in error_type_counts.items():
            percentage = (count / len(errors)) * 100 if len(errors) > 0 else 0
            f.write(f"- **{error_type}**: {count} ({percentage:.1f}%)\n")
        
        f.write("\n## Chi tiết lỗi (50 mẫu đầu tiên):\n\n")
        
        for error in errors[:50]:
            f.write(f"### Lỗi {error['index'] + 1}: {error['error_type']}\n")
            f.write(f"- **Câu hỏi:** {error['question']}\n")
            f.write(f"- **Dự đoán:** {error['prediction']}\n")
            f.write(f"- **Thực tế:** {error['ground_truth']}\n")
            f.write(f"- **F1-Score:** {error['f1_score']}\n")
            f.write(f"- **Ngữ cảnh:** {error['context_preview']}\n\n")


def plot_metrics_curves(train_losses, val_em_scores, val_f1_scores, save_path="training_metrics.png"):
    """Plot training loss and validation metrics curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot validation metrics
    axes[1].plot(epochs, val_em_scores, 'r-', marker='o', label='Exact Match')
    axes[1].plot(epochs, val_f1_scores, 'g-', marker='s', label='F1-Score')
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Training metrics plot saved to {save_path}")