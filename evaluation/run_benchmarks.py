import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .datasets import load_all_datasets
from .models import ModelRegistry
from .personas import load_data, create_prompt


def create_persona_prompt(persona_type: str, respondent: Dict, target_questions: list) -> str:
    return create_prompt(persona_type, respondent, target_questions)


def evaluate_truthfulqa_with_runner(runner, limit: int = None) -> Dict:
    ds = load_all_datasets().get("truthful_qa")
    if ds is None:
        return {"dataset": "truthful_qa", "error": "load_failed"}
    dataset = ds["data"]["validation"] if "validation" in ds["data"] else ds["data"].get("test", ds["data"]["train"])  # type: ignore

    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        question = ex.get("question", "")
        choices = ex.get("choices", [])
        correct_idx = ex.get("labels")
        if not choices or correct_idx is None:
            # Some mirrors store MC fields differently
            choices = ex.get("mc1_targets", {}).get("choices", [])
            correct_idx = ex.get("mc1_targets", {}).get("labels")
        if not choices or correct_idx is None:
            continue
        pred_idx, scores = runner.score_mc(question, choices)
        is_correct = int(pred_idx == int(correct_idx))
        correct += is_correct
        total += 1
        results.append({
            "question": question,
            "choices": choices,
            "pred_idx": pred_idx,
            "gold_idx": int(correct_idx),
            "correct": is_correct,
            "scores": scores,
        })

    accuracy = (correct / total) if total else 0.0
    return {"dataset": "truthful_qa", "model": runner.name, "accuracy": accuracy, "num": total, "examples": results}


def evaluate_bbq_with_runner(runner, limit: int = None) -> Dict:
    ds = load_all_datasets().get("bbq")
    if ds is None:
        return {"dataset": "bbq", "error": "load_failed"}
    hfds = ds["data"]
    # Best-effort: collect any available split
    split = "validation" if "validation" in hfds else ("test" if "test" in hfds else "train")

    # Some mirrors expose multiple domain subsets as dict of datasets
    if isinstance(hfds, dict) and hasattr(next(iter(hfds.values())), split):
        aggregated = []
        for sub in hfds.values():
            try:
                aggregated.extend(list(sub[split]))
            except Exception:
                continue
        dataset = aggregated
    else:
        dataset = hfds[split]

    def build_question(ex: Dict) -> Tuple[str, List[str], int]:
        context = ex.get("context", "")
        question = ex.get("question", "")
        answers = [ex.get(f"answer{i}", "") for i in range(3) if f"answer{i}" in ex]
        label = ex.get("label")
        if not answers and "choices" in ex:
            answers = ex["choices"]
        return (f"{context}\n{question}".strip(), answers, label)

    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        q, options, gold = build_question(ex)
        if not options or gold is None:
            continue
        pred_idx, scores = runner.score_mc(q, options)
        is_correct = int(pred_idx == int(gold))
        correct += is_correct
        total += 1
        results.append({
            "question": q,
            "choices": options,
            "pred_idx": pred_idx,
            "gold_idx": int(gold),
            "correct": is_correct,
            "scores": scores,
        })

    accuracy = (correct / total) if total else 0.0
    return {"dataset": "bbq", "model": runner.name, "accuracy": accuracy, "num": total, "examples": results}


def main():
    parser = argparse.ArgumentParser(description="Run MC benchmarks (TruthfulQA, BBQ) using ModelRegistry")
    parser.add_argument("--models", nargs="+", required=True, help="Model identifiers or registry names")
    parser.add_argument("--datasets", nargs="+", default=["truthful_qa", "bbq"], help="Datasets to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    registry = ModelRegistry()

    results = {}
    for model_name in args.models:
        runner = registry.create(model_name)
        results[model_name] = {}
        if "truthful_qa" in args.datasets:
            print(f"Running TruthfulQA on {runner.name}...")
            results[model_name]["truthful_qa"] = evaluate_truthfulqa_with_runner(runner, args.limit)
        if "bbq" in args.datasets:
            print(f"Running BBQ on {runner.name}...")
            results[model_name]["bbq"] = evaluate_bbq_with_runner(runner, args.limit)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
