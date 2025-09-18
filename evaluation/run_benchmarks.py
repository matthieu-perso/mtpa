import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .datasets import load_all_datasets
from .models import ModelRegistry
from .personas import load_data, create_prompt
from typing import Optional  
import random


class PersonaProvider:
    """
    Persona selector:
      mode = 'first' | 'cycle' | 'random'
      persona_type = 'bullets' | 'json' | 'oneliner' (passed to create_prompt)
    """
    def __init__(
        self,
        people: Optional[List[Dict]] = None,
        mode: str = "cycle",
        persona_type: str = "bullets",
        seed: int = 0
    ):
        self.enabled = bool(people)
        self.people = people or []
        self.mode = mode
        self.persona_type = persona_type
        self._i = 0
        random.seed(seed)

    def next(self) -> Tuple[str, str, Dict[str, str]]:
        """
        Returns (pid, preface, traits). If disabled/empty, returns ("", "", {}).
        """
        if not self.enabled or not self.people:
            return "", "", {}
        if self.mode == "first":
            person = self.people[0]
        elif self.mode == "random":
            person = random.choice(self.people)
        else:  
            person = self.people[self._i % len(self.people)]
            self._i += 1

        pid = str(person.get("pid", ""))
        preface = create_prompt(self.persona_type, person, target_questions=None) + "\n\n"
        traits = person.get("traits", {}) or {}
        return pid, preface, traits




def _tqa_mc_from_row(row) -> tuple[str, list[str], int]:
    """
    domenicrosati/TruthfulQA schema:
      ['Type','Category','Question','Best Answer','Correct Answers','Incorrect Answers','Source']

    Build a 4-option MC:
      - correct option: 'Best Answer' (fallback to first of 'Correct Answers')
      - distractors: up to 3 from 'Incorrect Answers'
    """
    import re

    def _to_list(x):
        # Already a list?
        if isinstance(x, list):
            return [s for s in x if isinstance(s, str)]
        if isinstance(x, str):
            # Split on common delimiters: ; | , and newlines
            parts = re.split(r"[;\|\n,]+", x)
            return [p.strip() for p in parts if p and p.strip()]
        return []

    q = (row.get("Question") or "").strip()

    best = (row.get("Best Answer") or "").strip()
    correct_list = _to_list(row.get("Correct Answers"))
    incorrect_list = _to_list(row.get("Incorrect Answers"))

    # Choose the gold option
    correct = best or (correct_list[0] if correct_list else "")

    # Clean & dedupe
    distractors = [s for s in incorrect_list if s and s != correct]
    # Shuffle & take up to 3
    random.shuffle(distractors)
    distractors = distractors[:3]

    # If we still don’t have enough distractors, backfill from other correct answers
    if len(distractors) < 3 and len(correct_list) > 1:
        backfill = [s for s in correct_list[1:] if s and s != correct and s not in distractors]
        random.shuffle(backfill)
        distractors += backfill[: (3 - len(distractors))]

    # Final options
    options = [correct] + distractors
    # As a final safety, ensure we have at least 2 options
    if len(options) < 2:
        # fabricate a neutral distractor to avoid degenerate items
        options.append("None of the above")

    gold_idx = 0
    return q, options, gold_idx


def _bbq_mc_from_row(row) -> tuple[str, list[str], int]:
    """
    walledai/BBQ subset row schema:
      ['context','question','choices','answer','category']
    'answer' may be an index OR a string contained in 'choices'.
    """
    ctx = row.get("context", "") or ""
    q = row.get("question", "") or ""
    choices = row.get("choices", []) or []
    ans = row.get("answer")

    if isinstance(ans, int):
        gold = int(ans)
    else:
        try:
            gold = choices.index(ans)
        except Exception:
            gold = 0  # fallback
    full_q = f"{ctx}\n{q}".strip()
    return full_q, choices, gold


def _normad_mc_from_row(row, label_vocab: list[str]) -> tuple[str, list[str], int]:
    """
    akhilayerukola/NormAd row schema:
      ['ID','Country','Background','Axis','Subaxis','Value','Rule-of-Thumb','Story','Explanation','Gold Label']
    We’ll build a question using Background + Rule-of-Thumb + Story.
    Choices are the global label vocabulary (unique Gold Label values).
    Gold index is index of this row's 'Gold Label' in that vocab.
    """
    bg = row.get("Background", "") or ""
    rot = row.get("Rule-of-Thumb", "") or ""
    story = row.get("Story", "") or ""
    gold_lab = (row.get("Gold Label", "") or "").strip()

    q = (f"{bg}\nRule-of-Thumb: {rot}\nSituation: {story}\n"
         "Based on the rule-of-thumb, select the correct label.").strip()

    try:
        gold_idx = label_vocab.index(gold_lab)
    except ValueError:
        gold_idx = 0
    return q, label_vocab, gold_idx


def evaluate_truthfulqa_with_runner(runner, limit: int = None, personas: Optional[PersonaProvider] = None) -> Dict:
    ds = load_all_datasets().get("truthful_qa")
    if ds is None:
        return {"dataset": "truthful_qa", "error": "load_failed"}
    dataset = list(ds["data"]["train"])

    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            break

        row = ex
        q, options, gold = _tqa_mc_from_row(row)
        if not options:
            continue

        pid, preface, traits = personas.next() if personas else ("", "", {})
        pred_idx, scores = runner.score_mc(q, options, preface=preface)
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
            **({"pid": pid, "persona_traits": traits} if personas and personas.enabled else {}),
        })


    accuracy = (correct / total) if total else 0.0
    meta = {"dataset": "truthful_qa", "model": runner.name, "accuracy": accuracy, "num": total, "examples": results}
    if personas and personas.enabled:
        meta["persona_mode"] = personas.mode
        meta["persona_type"] = personas.persona_type
    return meta


def evaluate_bbq_with_runner(runner, limit: int = None, personas: Optional[PersonaProvider] = None) -> Dict:
    ds = load_all_datasets().get("bbq")
    if ds is None:
        return {"dataset": "bbq", "error": "load_failed"}
    hfds = ds["data"]
    if isinstance(hfds, dict):
        dataset = []
        for subset_ds in hfds.values():
            dataset.extend(list(subset_ds))
    else:
        dataset = list(hfds["train"] if "train" in hfds else hfds[next(iter(hfds.keys()))])

    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        q, options, gold = _bbq_mc_from_row(ex)
        if not options:
            continue

        pid, preface, traits = personas.next() if personas else ("", "", {})

        pred_idx, scores = runner.score_mc(q, options, preface=preface)
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
            **({"pid": pid, "persona_traits": traits} if personas and personas.enabled else {}),
        })

    accuracy = (correct / total) if total else 0.0
    meta = {"dataset": "bbq", "model": runner.name, "accuracy": accuracy, "num": total, "examples": results}
    if personas and personas.enabled:
        meta["persona_mode"] = personas.mode
        meta["persona_type"] = personas.persona_type
    return meta


def evaluate_normad_with_runner(runner, limit: int = None, personas: Optional[PersonaProvider] = None) -> Dict:
    ds = load_all_datasets().get("normad")
    if ds is None:
        return {"dataset": "normad", "error": "load_failed"}

    hfds = ds["data"]
    dataset = list(hfds["train"] if "train" in hfds else hfds[next(iter(hfds.keys()))])

    # Build a stable label vocabulary
    labels = sorted({ (row.get("Gold Label") or "").strip() for row in dataset if (row.get("Gold Label") or "").strip() })
    if not labels:
        labels = ["LabelA", "LabelB"]

    results = []
    correct = 0
    total = 0

    for i, ex in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        q, options, gold = _normad_mc_from_row(ex, labels)
        if not options:
            continue

        pid, preface, traits = personas.next() if personas else ("", "", {})

        pred_idx, scores = runner.score_mc(q, options, preface=preface)
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
            **({"pid": pid, "persona_traits": traits} if personas and personas.enabled else {}),
        })

    accuracy = (correct / total) if total else 0.0
    meta = {"dataset": "normad", "model": runner.name, "accuracy": accuracy, "num": total, "examples": results}
    if personas and personas.enabled:
        meta["persona_mode"] = personas.mode
        meta["persona_type"] = personas.persona_type
    return meta



def main():
    parser = argparse.ArgumentParser(description="Run MC benchmarks (TruthfulQA, BBQ) using ModelRegistry")
    parser.add_argument("--models", nargs="+", required=True, help="Model identifiers or registry names")
    parser.add_argument("--datasets", nargs="+", default=["truthful_qa", "bbq", "normad"], help="Datasets to run (choose from: truthful_qa, bbq, normad)")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    parser.add_argument("--with-persona", action="store_true", help="Enable persona-prefaced prompts")
    parser.add_argument("--mtpa", type=str, default=None, help="Path to MTPA json/jsonl")
    parser.add_argument("--persona-type", type=str, default="bullets", choices=["bullets", "json", "oneliner"], help="Persona rendering style")
    parser.add_argument("--persona-mode", type=str, default="cycle", choices=["first", "cycle", "random"], help="How to assign personas across examples")
    parser.add_argument("--persona-limit", type=int, default=None, help="Use only first N personas (after load)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for persona assignment (random mode)")

    args = parser.parse_args()

    registry = ModelRegistry()
    registry.register_gemini("gemini-2.0-flash", "gemini-2.0-flash")

    personas_provider = None
    if args.with_persona and args.mtpa:
        people = load_data(args.mtpa, max_n=args.persona_limit)
        personas_provider = PersonaProvider(
            people=people,
            mode=args.persona_mode,
            persona_type=args.persona_type,
            seed=args.seed,
        )

    results = {}
    for model_name in args.models:
        runner = registry.create(model_name)
        results[model_name] = {}
        if "truthful_qa" in args.datasets:
            print(f"Running TruthfulQA on {runner.name}..." f"{' with personas' if personas_provider else ''}")
            results[model_name]["truthful_qa"] = evaluate_truthfulqa_with_runner(
                runner, args.limit, personas=personas_provider
            )
        if "bbq" in args.datasets:
            print(f"Running BBQ on {runner.name}..." f"{' with personas' if personas_provider else ''}")
            results[model_name]["bbq"] = evaluate_bbq_with_runner(
                runner, args.limit, personas=personas_provider
            )
        if "normad" in args.datasets:
            print(f"Running Normad on {runner.name}..." f"{' with personas' if personas_provider else ''}")
            results[model_name]["normad"] = evaluate_normad_with_runner(
                runner, args.limit, personas=personas_provider
            )


    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
