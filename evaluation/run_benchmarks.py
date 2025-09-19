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
from tqdm import tqdm
import os
import glob
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


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

    def pick_k(self, k: int, mode: str, seed: int, start_index: int = 0) -> List[Tuple[str, str, Dict[str, str]]]:
        """Pick K personas deterministically.
        - first: take first K
        - cycle: rotate by start_index, then take K
        - random: shuffle with given seed, take K
        Returns list of (pid, preface, traits).
        """
        if not self.enabled or not self.people or k <= 0:
            return []
        total = len(self.people)
        k = min(k, total)
        selection: List[Dict] = []
        if mode == "first":
            selection = self.people[:k]
        elif mode == "cycle":
            offset = start_index % total
            selection = [self.people[(offset + i) % total] for i in range(k)]
        else:  # random
            rng = random.Random(seed)
            idxs = list(range(total))
            rng.shuffle(idxs)
            selection = [self.people[idxs[i]] for i in range(k)]
        result: List[Tuple[str, str, Dict[str, str]]] = []
        for person in selection:
            pid = str(person.get("pid", ""))
            preface = create_prompt(self.persona_type, person, target_questions=None) + "\n\n"
            traits = person.get("traits", {}) or {}
            result.append((pid, preface, traits))
        return result


def _resolve_mtpa_path(mtpa_arg: Optional[str]) -> str:
    """Return a local path to the MTPA JSON file.
    If mtpa_arg is an HF repo id or missing file, download and locate mpta.json (or base_mpta.json).
    Accepted forms:
      - path/to/mpta.json (existing file)
      - directory containing mpta.json (or base_mpta.json)
      - 'hf:matthieunlp/MTPA' or 'matthieunlp/MTPA' (dataset repo id with mpta.json)
      - None → defaults to 'matthieunlp/MTPA'
    """
    # If a concrete file exists, use it
    if mtpa_arg:
        p = Path(mtpa_arg)
        if p.exists() and p.is_file():
            return str(p)
        if p.exists() and p.is_dir():
            # search inside
            cands = []
            cands += glob.glob(os.path.join(str(p), "**", "mpta.json"), recursive=True)
            cands += glob.glob(os.path.join(str(p), "**", "base_mpta.json"), recursive=True)
            if cands:
                return cands[0]
    # Otherwise treat as HF repo id
    repo_id = (mtpa_arg or "matthieunlp/MTPA").replace("hf:", "")
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required to download MTPA; please install it.")
    local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    # Prefer mpta.json at root; else any mpta.json; else base_mpta.json variants
    cands = glob.glob(os.path.join(local_dir, "mpta.json"))
    if not cands:
        cands = glob.glob(os.path.join(local_dir, "**", "mpta.json"), recursive=True)
    if not cands:
        cands = glob.glob(os.path.join(local_dir, "**", "full", "base_mpta.json"), recursive=True)
    if not cands:
        cands = glob.glob(os.path.join(local_dir, "**", "base_mpta.json"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"MTPA json not found under downloaded dataset {local_dir}")
    return cands[0]


def _age_bin_from_traits(traits: Dict[str, str]) -> str:
    a = traits.get("age")
    try:
        ai = int(float(a)) if a is not None and str(a).strip() != "" else 0
    except Exception:
        ai = 0
    return f"{(ai//10)*10:02d}s" if ai > 0 else "UNK"


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _choose_stratified_personas(
    people: List[Dict[str, Dict[str, str]]],
    countries: List[str],
    total_n: int,
    seed: int,
) -> List[Dict[str, Dict[str, str]]]:
    """Select personas equally across given countries, 50/50 gender, spread across age bins.
    Reproducible via seed. Falls back within country if some strata lack members.
    """
    if total_n <= 0 or not countries:
        return []
    rng = random.Random(seed)
    # Normalize country targets
    target_countries_norm = [_norm(c) for c in countries]
    # Build index: by country -> gender -> bin -> list of indices
    by_country: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    for idx, p in enumerate(people):
        t = p.get("traits", {}) or {}
        c = _norm(t.get("country"))
        if not c:
            continue
        # Only consider if country is in target list (lenient: exact or substring match either way)
        if c not in target_countries_norm and not any(c in tc or tc in c for tc in target_countries_norm):
            continue
        g_raw = _norm(t.get("gender"))
        gender = "female" if g_raw == "female" else ("male" if g_raw == "male" else "other")
        if gender not in ("female", "male"):
            continue
        bin_label = _age_bin_from_traits(t)
        by_country.setdefault(c, {}).setdefault(gender, {}).setdefault(bin_label, []).append(idx)
    # Prepare quotas per country
    K = len(target_countries_norm)
    base = total_n // K
    rem = total_n - base * K
    selected: List[int] = []
    # Fixed bin order for spread
    bin_order = [f"{k:02d}s" for k in range(10, 90, 10)]
    # Iterate countries in given order
    for i, c_in in enumerate(target_countries_norm):
        # Find best matching key present
        candidates_keys = [ck for ck in by_country.keys() if ck == c_in or c_in in ck or ck in c_in]
        if not candidates_keys:
            continue
        ckey = sorted(candidates_keys, key=len)[0]
        quota_country = base + (1 if i < rem else 0)
        # 50/50 gender split
        gq_f = quota_country // 2
        gq_m = quota_country - gq_f
        for gender, gq in (("female", gq_f), ("male", gq_m)):
            taken = 0
            # Shuffle lists per bin for randomness
            bins_map = by_country.get(ckey, {}).get(gender, {})
            # Round-robin over bins
            while taken < gq and bins_map:
                progressed = False
                for b in bin_order:
                    lst = bins_map.get(b, [])
                    if not lst:
                        continue
                    rng.shuffle(lst)
                    idx = lst.pop()  # take one
                    if idx not in selected:
                        selected.append(idx)
                        taken += 1
                        progressed = True
                        if taken >= gq:
                            break
                if not progressed:
                    break
            # Backfill within gender in this country regardless of bin
            if taken < gq:
                pool = []
                for arr in bins_map.values():
                    pool.extend(arr)
                rng.shuffle(pool)
                for idx in pool:
                    if taken >= gq:
                        break
                    if idx not in selected:
                        selected.append(idx)
                        taken += 1
        # If still short for this country, try other gender pools in the same country
        current_country_count = sum(1 for idx in selected if _norm(people[idx]["traits"].get("country")) in (ckey,))
        if current_country_count < quota_country:
            need = quota_country - current_country_count
            pool = []
            for g in ("female", "male"):
                for arr in by_country.get(ckey, {}).get(g, {}).values():
                    pool.extend(arr)
            rng.shuffle(pool)
            for idx in pool:
                if need <= 0:
                    break
                if idx not in selected:
                    selected.append(idx)
                    need -= 1
    # If overall shortfall, backfill from any remaining candidates among targets
    if len(selected) < total_n:
        all_pool = []
        for ckey in list(by_country.keys()):
            if not any(ckey == tc or tc in ckey or ckey in tc for tc in target_countries_norm):
                continue
            for g in ("female", "male"):
                for arr in by_country.get(ckey, {}).get(g, {}).values():
                    all_pool.extend(arr)
        rng.shuffle(all_pool)
        for idx in all_pool:
            if len(selected) >= total_n:
                break
            if idx not in selected:
                selected.append(idx)
    # Return selected personas maintaining deterministic order by (country order, then selection order)
    return [people[i] for i in selected[:total_n]]


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


def _tqa_mc2_from_row(row) -> tuple[str, list[str], int]:
    """Prefer official TruthfulQA MC2 targets; fall back to MC1 or CSV-style if needed."""
    # Official fields
    if isinstance(row.get("mc2_targets"), dict):
        q = (row.get("question") or "").strip()
        choices = row["mc2_targets"].get("choices", []) or []
        gold = row["mc2_targets"].get("labels")
        if choices and gold is not None:
            try:
                return q, list(choices), int(gold)
            except Exception:
                pass
    # Fallback to MC1 if MC2 missing
    if isinstance(row.get("mc1_targets"), dict):
        q = (row.get("question") or "").strip()
        choices = row["mc1_targets"].get("choices", []) or []
        gold = row["mc1_targets"].get("labels")
        if choices and gold is not None:
            try:
                return q, list(choices), int(gold)
            except Exception:
                pass
    # Last resort: build from CSV-style
    return _tqa_mc_from_row(row)


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


def _sample_indices(n_total: int, limit: Optional[int], seed: int) -> List[int]:
    """Return a deterministic sample of indices of size limit (or all) using seed."""
    if limit is None or limit >= n_total:
        return list(range(n_total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), k=limit))


def evaluate_truthfulqa_with_runner(runner, limit: int = None, personas: Optional[PersonaProvider] = None,
                                    sweep: bool = False, sweep_k: int = 0, sweep_mode: str = "random", seed: int = 0) -> Dict:
    ds = load_all_datasets().get("truthful_qa")
    if ds is None:
        return {"dataset": "truthful_qa", "error": "load_failed"}
    # Prefer validation/test from official multiple_choice split
    data = ds["data"]
    if "validation" in data:
        dataset = list(data["validation"])  # type: ignore
    elif "test" in data:
        dataset = list(data["test"])  # type: ignore
    else:
        dataset = list(data.get("train", []))  # type: ignore

    # Deterministic sampling of questions
    idxs = _sample_indices(len(dataset), limit, seed=seed + 11)

    results = []
    correct = 0
    total = 0

    for j, i in enumerate(tqdm(idxs, total=len(idxs), desc=f"TruthfulQA [{runner.name}]") ):
        ex = dataset[i]
        q, options, gold = _tqa_mc2_from_row(ex)
        if not options:
            continue
        example_id = i

        persona_entries: List[Tuple[str, str, Dict[str, str]]] = [("", "", {})]
        if personas and personas.enabled:
            if sweep and sweep_k > 0:
                persona_entries = personas.pick_k(sweep_k, sweep_mode, seed=seed + 1000 + i, start_index=j)
            else:
                persona_entries = [personas.next()]

        for p_idx, (pid, preface, traits) in enumerate(persona_entries):
            pred_idx, scores = runner.score_mc(q, options, preface=preface)
            is_correct = int(pred_idx == int(gold))
            correct += is_correct
            total += 1
            results.append({
                "example_id": example_id,
                "persona_idx": p_idx,
                "pid": pid,
                "persona_traits": traits if (personas and personas.enabled) else {},
                "question": q,
                "choices": options,
                "pred_idx": pred_idx,
                "gold_idx": int(gold),
                "correct": is_correct,
                "scores": scores,
            })

    accuracy = (correct / total) if total else 0.0
    meta = {"dataset": "truthful_qa", "model": runner.name, "accuracy": accuracy, "num": total, "examples": results}
    if personas and personas.enabled:
        meta["persona_mode"] = personas.mode
        meta["persona_type"] = personas.persona_type
        meta["persona_sweep"] = bool(sweep and sweep_k > 0)
        meta["persona_sweep_k"] = sweep_k
        meta["persona_sweep_mode"] = sweep_mode
    return meta


def evaluate_bbq_with_runner(runner, limit: int = None, personas: Optional[PersonaProvider] = None,
                             sweep: bool = False, sweep_k: int = 0, sweep_mode: str = "random", seed: int = 0) -> Dict:
    ds = load_all_datasets().get("bbq")
    if ds is None:
        return {"dataset": "bbq", "error": "load_failed"}
    hfds = ds["data"]
    # Deterministic aggregation across subsets by sorted key name if dict-like
    if isinstance(hfds, dict):
        dataset = []
        for key in sorted(hfds.keys()):
            subset_ds = hfds[key]
            try:
                dataset.extend(list(subset_ds))
            except Exception:
                # Handle HF DatasetDict with splits
                for split in ["validation", "test", "train"]:
                    if split in subset_ds:
                        dataset.extend(list(subset_ds[split]))
                        break
    else:
        dataset = list(hfds["train"] if "train" in hfds else hfds[next(iter(hfds.keys()))])

    idxs = _sample_indices(len(dataset), limit, seed=seed + 22)

    results = []
    correct = 0
    total = 0

    for j, i in enumerate(tqdm(idxs, total=len(idxs), desc=f"BBQ [{runner.name}]") ):
        ex = dataset[i]
        q, options, gold = _bbq_mc_from_row(ex)
        if not options:
            continue
        example_id = i

        persona_entries: List[Tuple[str, str, Dict[str, str]]] = [("", "", {})]
        if personas and personas.enabled:
            if sweep and sweep_k > 0:
                persona_entries = personas.pick_k(sweep_k, sweep_mode, seed=seed + 2000 + i, start_index=j)
            else:
                persona_entries = [personas.next()]

        for p_idx, (pid, preface, traits) in enumerate(persona_entries):
            pred_idx, scores = runner.score_mc(q, options, preface=preface)
            is_correct = int(pred_idx == int(gold))
            correct += is_correct
            total += 1
            results.append({
                "example_id": example_id,
                "persona_idx": p_idx,
                "pid": pid,
                "persona_traits": traits if (personas and personas.enabled) else {},
                "question": q,
                "choices": options,
                "pred_idx": pred_idx,
                "gold_idx": int(gold),
                "correct": is_correct,
                "scores": scores,
            })

    accuracy = (correct / total) if total else 0.0
    meta = {"dataset": "bbq", "model": runner.name, "accuracy": accuracy, "num": total, "examples": results}
    if personas and personas.enabled:
        meta["persona_mode"] = personas.mode
        meta["persona_type"] = personas.persona_type
        meta["persona_sweep"] = bool(sweep and sweep_k > 0)
        meta["persona_sweep_k"] = sweep_k
        meta["persona_sweep_mode"] = sweep_mode
    return meta


def evaluate_normad_with_runner(runner, limit: int = None, personas: Optional[PersonaProvider] = None,
                                sweep: bool = False, sweep_k: int = 0, sweep_mode: str = "random", seed: int = 0) -> Dict:
    ds = load_all_datasets().get("normad")
    if ds is None:
        return {"dataset": "normad", "error": "load_failed"}

    hfds = ds["data"]
    dataset = list(hfds["train"] if "train" in hfds else hfds[next(iter(hfds.keys()))])

    # Build a stable label vocabulary
    labels = sorted({ (row.get("Gold Label") or "").strip() for row in dataset if (row.get("Gold Label") or "").strip() })
    if not labels:
        labels = ["LabelA", "LabelB"]

    idxs = _sample_indices(len(dataset), limit, seed=seed + 33)

    results = []
    correct = 0
    total = 0

    for j, i in enumerate(tqdm(idxs, total=len(idxs), desc=f"NormAd [{runner.name}]") ):
        ex = dataset[i]
        q, options, gold = _normad_mc_from_row(ex, labels)
        if not options:
            continue
        example_id = i

        persona_entries: List[Tuple[str, str, Dict[str, str]]] = [("", "", {})]
        if personas and personas.enabled:
            if sweep and sweep_k > 0:
                persona_entries = personas.pick_k(sweep_k, sweep_mode, seed=seed + 3000 + i, start_index=j)
            else:
                persona_entries = [personas.next()]

        for p_idx, (pid, preface, traits) in enumerate(persona_entries):
            pred_idx, scores = runner.score_mc(q, options, preface=preface)
            is_correct = int(pred_idx == int(gold))
            correct += is_correct
            total += 1
            results.append({
                "example_id": example_id,
                "persona_idx": p_idx,
                "pid": pid,
                "persona_traits": traits if (personas and personas.enabled) else {},
                "question": q,
                "choices": options,
                "pred_idx": pred_idx,
                "gold_idx": int(gold),
                "correct": is_correct,
                "scores": scores,
            })

    accuracy = (correct / total) if total else 0.0
    meta = {"dataset": "normad", "model": runner.name, "accuracy": accuracy, "num": total, "examples": results}
    if personas and personas.enabled:
        meta["persona_mode"] = personas.mode
        meta["persona_type"] = personas.persona_type
        meta["persona_sweep"] = bool(sweep and sweep_k > 0)
        meta["persona_sweep_k"] = sweep_k
        meta["persona_sweep_mode"] = sweep_mode
    return meta



def main():
    parser = argparse.ArgumentParser(description="Run MC benchmarks (TruthfulQA, BBQ) using ModelRegistry")
    parser.add_argument("--models", nargs="+", required=True, help="Model identifiers or registry names")
    parser.add_argument("--datasets", nargs="+", default=["truthful_qa", "bbq", "normad"], help="Datasets to run (choose from: truthful_qa, bbq, normad)")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset (sampled deterministically)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")
    parser.add_argument("--with-persona", action="store_true", help="Enable persona-prefaced prompts")
    parser.add_argument("--mtpa", type=str, default=None, help="Path to MTPA json/jsonl, dir, or HF repo id (e.g., hf:matthieunlp/MTPA)")
    parser.add_argument("--persona-type", type=str, default="bullets", choices=["bullets", "json", "oneliner"], help="Persona rendering style")
    parser.add_argument("--persona-mode", type=str, default="cycle", choices=["first", "cycle", "random"], help="How to assign personas across examples (non-sweep mode)")
    parser.add_argument("--persona-limit", type=int, default=None, help="Use only first N personas (after load)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for persona and question sampling")
    # Persona sweep flags
    parser.add_argument("--persona-sweep", action="store_true", help="Evaluate each question with multiple personas")
    parser.add_argument("--persona-sweep-k", type=int, default=0, help="Number of personas per question when sweep is enabled")
    parser.add_argument("--persona-sweep-mode", type=str, default="random", choices=["first", "cycle", "random"], help="Selection mode for personas per question in sweep")
    # API registration convenience
    parser.add_argument("--openai-models", nargs="*", default=[], help="OpenAI model ids to register (e.g., gpt-4o)")
    parser.add_argument("--gemini-models", nargs="*", default=[], help="Gemini model ids to register (e.g., gemini-2.0-flash)")
    # Stratified persona sampling
    parser.add_argument("--persona-stratified", action="store_true", help="Select personas stratified by countries, gender 50/50, and age bins")
    parser.add_argument("--persona-n", type=int, default=None, help="Total personas to select when using --persona-stratified")
    parser.add_argument("--persona-countries", type=str, default="", help="Comma-separated list of countries for stratified sampling")

    args = parser.parse_args()

    registry = ModelRegistry()
    # Register API models if requested
    for mid in args.openai_models:
        registry.register_openai(mid, mid)
    for mid in args.gemini_models:
        registry.register_gemini(mid, mid)

    personas_provider = None
    if args.with_persona:
        mtpa_path = _resolve_mtpa_path(args.mtpa)
        people_all = load_data(mtpa_path, max_n=None)
        # Optional filtering/stratification
        if args.persona_stratified:
            countries = [c.strip() for c in (args.persona_countries or "").split(",") if c.strip()]
            total_n = args.persona_n or args.persona_limit or 100
            selected_people = _choose_stratified_personas(people_all, countries, total_n, seed=args.seed)
            people = selected_people
        else:
            # Fall back to simple head limit
            limit = args.persona_limit
            people = people_all[:limit] if limit else people_all
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
                runner, args.limit, personas=personas_provider, sweep=args.persona_sweep,
                sweep_k=args.persona_sweep_k, sweep_mode=args.persona_sweep_mode, seed=args.seed
            )
        if "bbq" in args.datasets:
            print(f"Running BBQ on {runner.name}..." f"{' with personas' if personas_provider else ''}")
            results[model_name]["bbq"] = evaluate_bbq_with_runner(
                runner, args.limit, personas=personas_provider, sweep=args.persona_sweep,
                sweep_k=args.persona_sweep_k, sweep_mode=args.persona_sweep_mode, seed=args.seed
            )
        if "normad" in args.datasets:
            print(f"Running Normad on {runner.name}..." f"{' with personas' if personas_provider else ''}")
            results[model_name]["normad"] = evaluate_normad_with_runner(
                runner, args.limit, personas=personas_provider, sweep=args.persona_sweep,
                sweep_k=args.persona_sweep_k, sweep_mode=args.persona_sweep_mode, seed=args.seed
            )


    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
