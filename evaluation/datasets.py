#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline:
1.  Load MTPA profiles and PRISM records.
2.  For each PRISM record, mask a subset of persona traits (t_hidden).
3.  Use an LLM to infer the hidden traits from the visible ones (MTPA task).
4.  Re-assemble a full persona (observed + inferred) -> z_hat.
5.  Feed z_hat + PRISM prompt to an LLM to generate personalised text.
6.  Evaluate generation against gold PRISM output.
"""
import argparse, json, random, pathlib, os, time
from dataclasses import dataclass, field
from typing import List, Dict, Any

# --- external libs -----------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from bert_score import score as bert_score
import sacrebleu

# ============== data classes =================================================
@dataclass
class Trait:
    question_id: str
    answer_value: str
    # extra metadata omitted

@dataclass
class Profile:
    pid: str
    traits: Dict[str, Trait]  # keyed by question_id

@dataclass
class PrismSample:
    pid: str
    prompt: str
    gold_text: str
    persona: Dict[str, str]   # full key→value map

# ============== I/O helpers ==================================================
def load_jsonl(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f]

def load_mpta(path: str) -> Dict[str, Profile]:
    profiles = {}
    for obj in load_jsonl(path):
        traits = {t['question_id']: Trait(**t) for t in obj['data']}
        profiles[obj['id']] = Profile(pid=obj['id'], traits=traits)
    return profiles

def load_prism(path: str) -> List[PrismSample]:
    samples = []
    for obj in load_jsonl(path):
        samples.append(
            PrismSample(
                pid=obj['id'],
                prompt=obj['prompt'],
                gold_text=obj['target_text'],
                persona=obj['persona']    # dict[str,str]
            )
        )
    return samples

# ============== LLM wrappers =================================================
class LLM:
    """HuggingFace auto model wrapper (can be replaced with OpenAI)."""
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        print("Loading model …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.pipe = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer,
            max_new_tokens=256, temperature=0.7, do_sample=True
        )

    def generate(self, prompt: str) -> str:
        out = self.pipe(prompt, num_return_sequences=1)[0]["generated_text"]
        return out[len(prompt):].strip()

# ============== trait masking & inference ====================================
def mask_persona(persona: Dict[str, str], pct_hide: float = 0.3):
    keys = list(persona.keys())
    k_hide = max(1, int(len(keys) * pct_hide))
    hidden = set(random.sample(keys, k_hide))
    obs    = {k:v for k,v in persona.items() if k not in hidden}
    tgt    = {k:v for k,v in persona.items() if k in hidden}
    return obs, tgt

def trait_inference_prompt(obs_traits: Dict[str,str], missing_keys: List[str]) -> str:
    """Craft a natural-language prompt for trait prediction."""
    obs_list = [f"{k.replace('_',' ')}: {v}" for k,v in obs_traits.items()]
    missing  = ', '.join(missing_keys)
    return (
        "Below is partial information about a user.\n\n"
        "Known traits:\n" +
        '\n'.join(f"  - {t}" for t in obs_list) +
        f"\n\nPredict the following missing traits: {missing}.\n"
        "Respond with a JSON object mapping trait → predicted value."
    )

def infer_traits(llm: LLM, obs: Dict[str,str], tgt: Dict[str,str]) -> Dict[str,str]:
    prompt = trait_inference_prompt(obs, list(tgt.keys()))
    raw    = llm.generate(prompt)
    try:
        pred = json.loads(raw.split('\n')[0])
    except Exception:
        pred = {}
    # ensure we output *something* for every key (fallback = most common guess)
    for k in tgt:
        pred.setdefault(k, "unknown")
    return pred

# ============== personalised generation ======================================
def generation_prompt(persona: Dict[str,str], task_prompt: str) -> str:
    traits_txt = '\n'.join(f"{k.replace('_',' ')}: {v}" for k,v in persona.items())
    return (
        "You are a personalised assistant.\n"
        "User profile:\n" + traits_txt +
        "\n\nTask: " + task_prompt + "\n\nResponse:"
    )

# ============== evaluation ====================================================
def bertscore_single(ref: str, hyp: str) -> float:
    P, R, F = bert_score([hyp], [ref], lang="en", verbose=False)
    return F.item()

def run_bleu(refs: List[str], hyps: List[str]) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs]).score

# ============== full pipeline loop ============================================
def main(args):
    random.seed(0)

    profiles = load_mpta(args.mpta)
    prism    = load_prism(args.prism)
    llm      = LLM(args.model)

    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    refs, hyps = [], []
    for i, sample in enumerate(prism):
        if sample.pid not in profiles:
            # fall back to PRISM persona only
            mpta_traits = {}
        else:
            mpta_traits = {k:t.answer_value for k,t in profiles[sample.pid].traits.items()}

        # 1. mask part of persona (simulate partial knowledge)
        observed, hidden = mask_persona(sample.persona, pct_hide=0.3)

        # 2. infer hidden traits using MPTA-style prompt
        inferred = infer_traits(llm, {**observed, **mpta_traits}, hidden)

        # 3. assemble inferred persona
        persona_hat = {**observed, **inferred}

        # 4. generate personalised text
        prompt = generation_prompt(persona_hat, sample.prompt)
        gen    = llm.generate(prompt)

        refs.append(sample.gold_text.strip())
        hyps.append(gen.strip())

        # optional: write each instance
        with open(out_dir/f"{i:04d}.json", "w", encoding="utf-8") as f:
            json.dump(
                dict(pid=sample.pid, prompt=sample.prompt, ref=sample.gold_text,
                     hyp=gen, observed=observed, hidden=hidden, inferred=inferred),
                f, ensure_ascii=False, indent=2
            )
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(prism)}…")

    # 5. compute automatic metrics
    bleu  = run_bleu(refs, hyps)
    bsc   = sum(bertscore_single(r, h) for r,h in zip(refs, hyps)) / len(refs)

    with open(out_dir/"summary.txt", "w") as f:
        f.write(f"BLEU  = {bleu:.2f}\n")
        f.write(f"BERTScore F1 = {bsc:.4f}\n")

    print("\nFinished")
    print(f"BLEU         : {bleu:.2f}")
    print(f"BERTScore F1 : {bsc:.4f}")
    print(f"Per-sample outputs in {out_dir}")

# ============== CLI ===========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpta",  required=True, help="MTPA jsonl file")
    ap.add_argument("--prism", required=True, help="PRISM jsonl file")
    ap.add_argument("--out",   required=True, help="output directory")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()
    main(args)
