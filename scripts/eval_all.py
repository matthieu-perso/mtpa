#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import logging

from evaluation.evaluate import Evaluator
from evaluation.models.model_registry import registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file or directory"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="List of model names to evaluate (default: all registered models)"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="text-classification",
        choices=["text-classification", "text-generation"],
        help="Type of task"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for prediction"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Compute detailed metrics"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation report"
    )
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Get dataset path(s)
    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        dataset_paths = list(dataset_path.glob("*.json"))
    else:
        dataset_paths = [dataset_path]
        
    # Run evaluation
    all_results = {}
    for dataset_path in dataset_paths:
        logger.info(f"Evaluating on dataset: {dataset_path}")
        
        results = evaluator.evaluate_all(
            str(dataset_path),
            model_names=args.models,
            task_type=args.task_type,
            compute_detailed=args.detailed,
            batch_size=args.batch_size
        )
        
        all_results[dataset_path.stem] = results
        
        # Generate report
        report = evaluator.generate_report(
            results,
            output_path=args.output and f"{args.output}_{dataset_path.stem}.md"
        )
        
        print(f"\nResults for {dataset_path.stem}:")
        print(report)
        
    # Save combined results
    if args.output:
        with open(f"{args.output}_all.json", "w") as f:
            json.dump(all_results, f, indent=2)
            
if __name__ == "__main__":
    main() 