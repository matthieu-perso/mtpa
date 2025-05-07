from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

def compute_classification_metrics(
    y_true: List[Any],
    y_pred: List[Any],
    average: str = "macro"
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dictionary of metric names and values
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average)
    }

def compute_detailed_metrics(
    y_true: List[Any],
    y_pred: List[Any],
    labels: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Compute detailed metrics including per-class statistics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label names
        
    Returns:
        Dictionary containing detailed metrics
    """
    # Get classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True
    )
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"]
            }
            for label in report.keys()
            if label not in ["accuracy", "macro avg", "weighted avg"]
        }
    }

def compute_generation_metrics(
    references: List[str],
    predictions: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute metrics for text generation tasks.
    
    Args:
        references: Reference texts
        predictions: Generated texts
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ["exact_match", "f1"]
        
    results = {}
    
    if "exact_match" in metrics:
        results["exact_match"] = np.mean([
            ref.strip() == pred.strip()
            for ref, pred in zip(references, predictions)
        ])
        
    if "f1" in metrics:
        # Simple token-level F1
        f1_scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = set(ref.lower().split())
            pred_tokens = set(pred.lower().split())
            
            if not ref_tokens or not pred_tokens:
                f1_scores.append(0.0)
                continue
                
            common = ref_tokens.intersection(pred_tokens)
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
                
        results["f1"] = np.mean(f1_scores)
        
    return results 