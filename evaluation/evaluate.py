from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging
from tqdm import tqdm

from .models.model_registry import registry
from .metrics import (
    compute_classification_metrics,
    compute_detailed_metrics,
    compute_generation_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    """Main evaluation orchestrator."""
    
    def __init__(self, results_dir: str = "evaluation/results"):
        """Initialize the evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_model(
        self,
        model_name: str,
        dataset: List[Dict[str, Any]],
        task_type: str = "text-classification",
        compute_detailed: bool = False,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Evaluate a model on a dataset.
        
        Args:
            model_name: Name of the registered model
            dataset: List of examples with 'input' and 'label' keys
            task_type: Type of task (text-classification, text-generation)
            compute_detailed: Whether to compute detailed metrics
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model {model_name} on dataset")
        
        # Get model
        model = registry.get_model(model_name)
        
        # Extract inputs and labels
        inputs = [ex["input"] for ex in dataset]
        labels = [ex["label"] for ex in dataset]
        
        # Make predictions
        predictions = model.batch_predict(inputs, batch_size=batch_size)
        
        # Compute metrics
        if task_type == "text-classification":
            metrics = compute_classification_metrics(labels, predictions)
            
            if compute_detailed:
                detailed = compute_detailed_metrics(labels, predictions)
                metrics.update(detailed)
                
        elif task_type == "text-generation":
            metrics = compute_generation_metrics(labels, predictions)
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        # Save results
        results = {
            "model": model_name,
            "task_type": task_type,
            "metrics": metrics,
            "config": model.config
        }
        
        out_path = self.results_dir / f"{model_name}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {out_path}")
        return results
    
    def evaluate_all(
        self,
        dataset_path: str,
        model_names: Optional[List[str]] = None,
        task_type: str = "text-classification",
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple models on a dataset.
        
        Args:
            dataset_path: Path to dataset file
            model_names: List of model names to evaluate
            task_type: Type of task
            **kwargs: Additional arguments passed to evaluate_model
            
        Returns:
            Dictionary mapping model names to results
        """
        # Load dataset
        with open(dataset_path) as f:
            dataset = json.load(f)
            
        # Get model names
        if model_names is None:
            model_names = list(registry.list_models().keys())
            
        # Evaluate each model
        results = {}
        for model_name in tqdm(model_names, desc="Evaluating models"):
            try:
                result = self.evaluate_model(
                    model_name,
                    dataset,
                    task_type=task_type,
                    **kwargs
                )
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
                
        return results
    
    def generate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """Generate a human-readable evaluation report.
        
        Args:
            results: Dictionary of evaluation results
            output_path: Optional path to save report
            
        Returns:
            Report as a string
        """
        report = ["# Evaluation Report\n"]
        
        for model_name, result in results.items():
            report.append(f"## {model_name}\n")
            report.append(f"Task Type: {result['task_type']}\n")
            
            metrics = result["metrics"]
            if "accuracy" in metrics:
                report.append("\n### Classification Metrics\n")
                report.append(f"- Accuracy: {metrics['accuracy']:.4f}")
                report.append(f"- F1 Score: {metrics['f1']:.4f}")
                report.append(f"- Precision: {metrics['precision']:.4f}")
                report.append(f"- Recall: {metrics['recall']:.4f}\n")
                
            if "per_class_metrics" in metrics:
                report.append("\n### Per-Class Metrics\n")
                for label, class_metrics in metrics["per_class_metrics"].items():
                    report.append(f"\n#### {label}")
                    report.append(f"- Precision: {class_metrics['precision']:.4f}")
                    report.append(f"- Recall: {class_metrics['recall']:.4f}")
                    report.append(f"- F1: {class_metrics['f1']:.4f}")
                    report.append(f"- Support: {class_metrics['support']}")
                    
        report = "\n".join(report)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
                
        return report 