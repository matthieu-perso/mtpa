import json, argparse, itertools, statistics as stats
from pathlib import Path
from collections import defaultdict, Counter

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def _examples(results, model, dataset):
    return results[model][dataset]["examples"]

def group_key(traits, keys=("country","gender")):
    # Extend keys as needed, e.g. ("country","gender","age_bin")
    vals = []
    for k in keys:
        v = traits.get(k)
        if k == "age" and v and v.isdigit():
            v = f"{(int(v)//10)*10}s"  # 20s, 30s, ...
        vals.append(v or "NA")
    return tuple(vals)

def subgroup_accuracy(examples, keys=("country","gender")):
    buckets = defaultdict(lambda: {"correct":0, "total":0})
    for ex in examples:
        traits = ex.get("persona_traits", {}) or {}
        g = group_key(traits, keys)
        buckets[g]["correct"] += int(ex["correct"])
        buckets[g]["total"] += 1
    out = []
    for g, agg in buckets.items():
        if agg["total"] == 0: continue
        out.append({
            "group": " | ".join(g),
            "correct": agg["correct"],
            "total": agg["total"],
            "accuracy": agg["correct"] / agg["total"],
        })
    return sorted(out, key=lambda r: r["accuracy"], reverse=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-persona", required=True)
    ap.add_argument("--with-persona", required=True)
    ap.add_argument("--group-by", nargs="+", default=["country","gender"])
    args = ap.parse_args()

    base = load_results(args.no_persona)
    pers = load_results(args.with_persona)

    for model in pers.keys():
        for dataset in pers[model].keys():
            ex_p = _examples(pers, model, dataset)
            ex_b = _examples(base, model, dataset)

            acc_p = pers[model][dataset]["accuracy"]
            acc_b = base[model][dataset]["accuracy"]
            print(f"\n== {model} / {dataset} ==")
            print(f"overall: persona {acc_p:.3f} vs baseline {acc_b:.3f} (Δ {acc_p-acc_b:+.3f})")

            # subgroup on persona run (has traits); compare to baseline where possible by index
            subs = subgroup_accuracy(ex_p, tuple(args.group_by))
            for row in subs[:20]:  # top n by accuracy
                print(f"{row['group']:<40} n={row['total']:<4} acc={row['accuracy']:.3f}")

if __name__ == "__main__":
    main()