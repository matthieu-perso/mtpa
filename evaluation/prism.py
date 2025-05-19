
import json, torch, random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score as bert_score
from tqdm import tqdm
import pandas as pd

# ---------- Config -----------------
MODEL_ID      = "microsoft/Phi-3-mini-4k-instruct"  # small enough for 16 GB GPU
MAX_EXAMPLES  = 100         # increase once it runs
MAX_NEW_TOK   = 120         # generation length
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MTPA_ACCURACY = 0.557       # plug in your model’s MTPA score
# -----------------------------------

# ---------- Load PRISM -------------
survey_ds = load_dataset("HannahRoseKirk/prism-alignment", name="survey",        split="train")
conv_ds   = load_dataset("HannahRoseKirk/prism-alignment", name="conversations", split="train")

ADMIN = {"user_id","survey_only","num_completed_conversations","consent","consent_age"}
user_profiles = {
    row["user_id"]: {k: str(v) for k,v in row.items() if k not in ADMIN}
    for row in survey_ds
}

print(f"✅ Loaded {len(user_profiles)} user profiles and {len(conv_ds)} conversations")

# ---------- Load model -------------
tok   = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else "auto"
)

def generate_reply(profile: dict, opening_prompt: str) -> str:
    traits_txt = "; ".join(f"{k.replace('_',' ')}: {v}" for k,v in profile.items())
    prompt = (
        "You are a personalised assistant.\n"
        f"User profile: {traits_txt}\n\n"
        f"Task: {opening_prompt}\n\n"
        "Response:"
    )
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOK)
    return tok.decode(out[0], skip_special_tokens=True).split("Response:")[-1].strip()

# ---------- Evaluate ---------------
rows = conv_ds.shuffle(seed=0)[:MAX_EXAMPLES]
gens, refs = [], []
meta = []

for row in tqdm(rows, desc="Conversations"):
    uid   = row["user_id"]
    prof  = user_profiles.get(uid, None)
    if not prof:
        continue

    prompt = row["opening_prompt"]
    # target = first USER turn or first ASSISTANT? We'll compare with user utterance
    tgt = next((t["text"] for t in row["conversation_turns"] if t["speaker"]=="user"), "")
    if not tgt.strip():
        continue

    gen = generate_reply(prof, prompt)
    gens.append(gen)
    refs.append(tgt)
    meta.append(dict(uid=uid, prompt=prompt, generated=gen, reference=tgt))

# BERTScore
_, _, F1 = bert_score(gens, refs, lang="en", verbose=True, batch_size=8)
bert_scores = F1.tolist()

# ---------- Results table ----------
df = pd.DataFrame(meta)
df["bert_score"] = bert_scores
df["mpta_acc"]   = MTPA_ACCURACY
display(df.head())

print(f"\nAverage BERTScore-F1 on {len(df)} samples: {df['bert_score'].mean():.4f}")
