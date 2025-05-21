"""visualize_results.py

Generate publication‑ready charts from survey‑benchmark output files.
Assumes the directory contains:
    benchmark_results_*.json   # one file per target question (per the runner)

The script produces PNGs:
    overall_accuracy.png
    accuracy_scatter.png
    accuracy_heatmap.png
    gender_accuracy_bar.png
    gender_gap.png

Only matplotlib is used (no seaborn).  Axis labels are visible; titles are
omitted for a clean, publication‑ready look.  All figures use tight_layout().
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1 · Load & tidy every benchmark_results_*.json into a single DataFrame
# ---------------------------------------------------------------------------

rows: list[dict] = []
for fp in glob.glob("benchmark_results_*.json"):
    with open(fp) as f:
        block = json.load(f)
    for key, val in block.items():
        model, qid = key.rsplit("_", 1)
        rows.append({
            "model":       model,
            "question_id": qid,
            "accuracy":    val["accuracy"],
            "category":    val["predictions"][0]["category"],
            # try to capture demographic group if present in first record
            "sex":         val["predictions"][0].get("sex", None),
        })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 2 · Overall accuracy per model (bar chart)
# ---------------------------------------------------------------------------

overall = df.groupby("model")["accuracy"].mean().sort_values()
plt.figure(figsize=(6, 4))
plt.barh(overall.index, overall.values)
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "overall_accuracy.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# 3 · Scatter plot – item difficulty vs. model accuracy
#      Difficulty = mean accuracy across models for that question.
# ---------------------------------------------------------------------------

difficulty = df.groupby("question_id")["accuracy"].mean()
df = df.join(difficulty.rename("difficulty"), on="question_id")
plt.figure(figsize=(6, 4))
plt.scatter(df["difficulty"], df["accuracy"], s=10, alpha=0.6)
plt.xlabel("Question mean accuracy (difficulty → low)")
plt.ylabel("Model accuracy on question")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_scatter.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# 4 · Heat‑map – accuracy matrix (questions × models)
# ---------------------------------------------------------------------------

pivot = df.pivot_table(index="question_id", columns="model", values="accuracy")
plt.figure(figsize=(8, max(4, 0.25 * len(pivot))))
plt.imshow(pivot, aspect="auto", interpolation="nearest")
plt.xlabel("Model")
plt.ylabel("Question")
plt.colorbar(label="Accuracy")
plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_heatmap.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# 5 · Gender accuracy bar chart for importance_of_family
# ---------------------------------------------------------------------------

topic = "importance_of_family"
gender_subset = df[df["question_id"] == topic]
gender_acc = gender_subset.groupby(["model", "sex"])["accuracy"].mean().unstack()
plt.figure(figsize=(8, 4))
for idx, sex in enumerate(sorted(gender_acc.columns)):
    plt.bar(np.arange(len(gender_acc)) + idx * 0.3, gender_acc[sex], width=0.3, label=str(sex))
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(np.arange(len(gender_acc)) + 0.3 / 2, gender_acc.index, rotation=45, ha="right")
plt.legend(title="Sex", frameon=False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gender_accuracy_bar.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# 6 · Parity gap bar chart (max – min gender accuracy per model)
# ---------------------------------------------------------------------------

gap = (gender_acc.max(axis=1) - gender_acc.min(axis=1)).sort_values()
plt.figure(figsize=(6, 4))
plt.barh(gap.index, gap.values)
plt.xlabel("Gender accuracy gap")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gender_gap.png", dpi=300)
plt.close()

print(f"Saved figures to {OUTPUT_DIR.resolve()}")
