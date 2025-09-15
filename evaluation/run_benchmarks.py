import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .datasets import load_all_datasets


def compute_option_logprob(model, tokenizer, prompt: str, option_text: str) -> float:
    """Compute average token logprob of generating option_text after prompt."""
    device = next(model.parameters()).device
    with torch.no_grad():
        # Concatenate prompt + option and score only the option tokens
        full = prompt + option_text
        inputs = tokenizer(full, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attn = inputs.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits  # [B, T, V]
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        # Identify the span corresponding to option_text
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]
        option_len = shift_labels.shape[1] - (prompt_len - 1)
        # Slice option positions
        option_logits = shift_logits[:, prompt_len - 1 : prompt_len - 1 + option_len, :]
        option_labels = shift_labels[:, prompt_len - 1 : prompt_len - 1 + option_len]
        # Gather logprobs
        logprobs = torch.log_softmax(option_logits, dim=-1)
        token_logprobs = logprobs.gather(-1, option_labels.unsqueeze(-1)).squeeze(-1)
        avg_logprob = token_logprobs.mean().item()
        return avg_logprob


def select_mc_answer(model, tokenizer, question: str, options: List[str], preface: str = "") -> Tuple[int, List[float]]:
    """Select multiple-choice answer by scoring options with logprobs. Returns (best_index, scores)."""
    prompt = f"{preface}{question}\nAnswer: "
    scores = [compute_option_logprob(model, tokenizer, prompt, opt) for opt in options]
    best_idx = int(torch.tensor(scores).argmax().item())
    return best_idx, scores


def evaluate_truthfulqa(model_name: str, limit: int = None) -> Dict:
    ds = load_all_datasets().get("truthful_qa")
    if ds is None:
        return {"dataset": "truthful_qa", "error": "load_failed"}
    dataset = ds["data"]["validation"] if "validation" in ds["data"] else ds["data"]["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

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
            # Try alternative field names used by some mirrors
            choices = ex.get("mc1_targets", {}).get("choices", [])
            correct_idx = ex.get("mc1_targets", {}).get("labels")
        if not choices or correct_idx is None:
            continue
        best_idx, scores = select_mc_answer(model, tokenizer, question, choices)
        is_correct = int(best_idx == correct_idx)
        correct += is_correct
        total += 1
        results.append({
            "question": question,
            "choices": choices,
            "pred_idx": best_idx,
            "gold_idx": int(correct_idx),
            "correct": is_correct,
            "scores": scores,
        })

    accuracy = (correct / total) if total else 0.0
    return {"dataset": "truthful_qa", "model": model_name, "accuracy": accuracy, "num": total, "examples": results}


def evaluate_bbq(model_name: str, limit: int = None, subset: str = None) -> Dict:
    ds = load_all_datasets().get("bbq")
    if ds is None:
        return {"dataset": "bbq", "error": "load_failed"}
    # BBQ is typically split into many domain subsets; consolidate to a single iterable
    hfds = ds["data"]
    split = "validation" if "validation" in hfds else ("test" if "test" in hfds else "train")
    if subset and subset in hfds:
        dataset = hfds[subset][split]
    else:
        # Merge all configs if available
        if isinstance(hfds, dict) and len(hfds.keys()) > 0 and hasattr(next(iter(hfds.values())), split):
            # Concatenate across subsets
            all_parts = []
            for name, sub in hfds.items():
                try:
                    all_parts.extend(list(sub[split]))
                except Exception:
                    continue
            dataset = all_parts
        else:
            dataset = hfds[split]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")

    def build_question(ex: Dict) -> Tuple[str, List[str], int]:
        context = ex.get("context", "")
        question = ex.get("question", "")
        answers = [ex.get(f"answer{i}", "") for i in range(3) if f"answer{i}" in ex]
        label = ex.get("label")
        # Some variants store options under 'choices'
        if not answers and "choices" in ex:
            answers = ex["choices"]
        return (f"{context}\n{question}", answers, label)

    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        q, options, gold = build_question(ex)
        if not options or gold is None:
            continue
        best_idx, scores = select_mc_answer(model, tokenizer, q, options)
        is_correct = int(best_idx == gold)
        correct += is_correct
        total += 1
        results.append({
            "question": q,
            "choices": options,
            "pred_idx": best_idx,
            "gold_idx": int(gold),
            "correct": is_correct,
            "scores": scores,
        })

    accuracy = (correct / total) if total else 0.0
    return {"dataset": "bbq", "model": model_name, "accuracy": accuracy, "num": total, "examples": results}


def main():
    parser = argparse.ArgumentParser(description="Run MC benchmarks (TruthfulQA, BBQ)")
    parser.add_argument("--models", nargs="+", required=True, help="HF model names")
    parser.add_argument("--datasets", nargs="+", default=["truthful_qa", "bbq"], help="Datasets to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    results = {}
    for model_name in args.models:
        results[model_name] = {}
        if "truthful_qa" in args.datasets:
            print(f"Running TruthfulQA on {model_name}...")
            results[model_name]["truthful_qa"] = evaluate_truthfulqa(model_name, args.limit)
        if "bbq" in args.datasets:
            print(f"Running BBQ on {model_name}...")
            results[model_name]["bbq"] = evaluate_bbq(model_name, args.limit)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
