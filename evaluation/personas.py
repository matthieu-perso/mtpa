from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import pycountry  


DEMOGRAPHIC_QIDS = ["sex", "age", "country", "birth_country"]

IMPORTANCE_QIDS = [
    "importance_family", "importance_friends", "importance_leisure",
    "importance_politics", "importance_religion", "importance_work",
]

AGREE_QIDS_1TO4 = [
    "men_better_political_leaders",
    "men_better_business_executives",
    "university_more_important_boys",
]
AGREE_QIDS_1TO5 = [
    "men_priority_jobs",
    "citizens_priority_jobs",
]

CHILDREN_QIDS = [
    "children_learn_manners", "children_learn_independence", "children_learn_hard_work",
    "children_learn_responsibility", "children_learn_imagination", "children_learn_tolerance",
    "children_learn_thrift", "children_learn_perseverance", "children_learn_faith",
    "children_learn_unselfishness", "children_learn_obedience",
]

NEIGHBOR_QIDS = [
    "uncomfortable_drug_addicts", "uncomfortable_different_race",
    "uncomfortable_immigrants", "uncomfortable_homosexuals",
    "uncomfortable_drinkers",
]

TECHNICAL_QIDS = {
    "year",
    "interview_date",
    "interview_language",
    "birth_year",
    "father_immigrant",
    "mother_birth_country",
    "father_birth_country",
}


OPINION_QIDS = ["polviews", "cappun", "postlife"]

GOAL_QIDS = [
    "goal_make_parents_proud",
    "mother_work_children_suffer",
]

EXTRA_OPINION_QIDS = ["homosex", "abany"]

ALL_QIDS = (
    DEMOGRAPHIC_QIDS
    + IMPORTANCE_QIDS
    + AGREE_QIDS_1TO4
    + AGREE_QIDS_1TO5
    + OPINION_QIDS
    + EXTRA_OPINION_QIDS
    + CHILDREN_QIDS
    + NEIGHBOR_QIDS
    + GOAL_QIDS
)


def _load_json_any(path: str | Path) -> List[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        return json.load(f) if first == "[" else [json.loads(l) for l in f]

NOISY = {
    "-1", "No answer", "Missing", "Missing; Not available",
    "Not asked", "Don't know", "DK", "Refused", ""
}

def _clean_cat(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return None if s in NOISY else s

def _to_int(x: Any) -> int | None:
    if x is None: return None
    try: return int(float(x))
    except: return None

def _decode_country(code_or_name: Any) -> str | None:
    if code_or_name is None:
        return None
    s = str(code_or_name).strip()
    if s in NOISY or not s:
        return None
    if not s.isdigit():  
        return s
    if pycountry is not None:
        try:
            c = pycountry.countries.get(numeric=s.zfill(3))
            if c:
                return c.name
        except Exception:
            pass
    return s  

def _verbalize_importance(val: Any) -> str | None:
    m = {1: "very important", 2: "quite important", 3: "not very important", 4: "not at all important"}
    v = _to_int(val)
    return m.get(v)

def _verbalize_yesno(val: Any) -> str | None:
    m = {1: "yes", 2: "no"}
    v = _to_int(val)
    return m.get(v)

def _verbalize_agree_1to4(val: Any) -> str | None:
    m = {1: "strongly agree", 2: "agree", 3: "disagree", 4: "strongly disagree"}
    v = _to_int(val)
    return m.get(v)

def _verbalize_agree_1to5(val: Any) -> str | None:
    m = {1: "strongly agree", 2: "agree", 3: "neither", 4: "disagree", 5: "strongly disagree"}
    v = _to_int(val)
    return m.get(v)

def _index_data(raw_obj: dict) -> Dict[str, dict]:
    return {d.get("question_id"): d for d in (raw_obj.get("data") or [])}

def _get_ans(idx: Dict[str, dict], qid: str):
    rec = idx.get(qid) or {}
    return rec.get("answer_value", None)

def _yesno(val: Any) -> bool | None:
    """Map 'yes'/'no' style answers into bool."""
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in {"1", "yes", "y", "true"}:
        return True
    if v in {"2", "no", "n", "false"}:
        return False
    return None

def _extract_traits(raw_obj: dict) -> Dict[str, str]:
    idx = _index_data(raw_obj)
    traits: Dict[str, str] = {}

    sex = _clean_cat(_get_ans(idx, "sex"))
    age_raw = _clean_cat(_get_ans(idx, "age"))
    if sex: traits["gender"] = sex
    if age_raw:
        iv = _to_int(age_raw)
        if iv: traits["age"] = str(iv)
    country = _decode_country(_get_ans(idx, "country"))
    birth_country = _decode_country(_get_ans(idx, "birth_country"))
    if country:
        traits["country"] = country
    if birth_country:
        traits["birth_country"] = birth_country


    for qid in IMPORTANCE_QIDS:
        v = _get_ans(idx, qid)
        lab = _verbalize_importance(v)
        if lab: traits[qid] = lab

    for qid in AGREE_QIDS_1TO4:
        v = _get_ans(idx, qid)
        lab = _verbalize_agree_1to4(v)
        if lab: traits[qid] = lab
    for qid in AGREE_QIDS_1TO5:
        v = _get_ans(idx, qid)
        lab = _verbalize_agree_1to5(v)
        if lab: traits[qid] = lab

    for qid in OPINION_QIDS + EXTRA_OPINION_QIDS:
        v = _clean_cat(_get_ans(idx, qid))
        if v:
            traits[qid] = v

    for qid in GOAL_QIDS:
        v = _get_ans(idx, qid)
        lab = _verbalize_agree_1to4(v)
        if lab: traits[qid] = lab

    for qid in CHILDREN_QIDS:
        v = _get_ans(idx, qid)
        lab = _verbalize_yesno(v)
        if lab: traits[qid] = lab

    for qid in NEIGHBOR_QIDS:
        v = _get_ans(idx, qid)
        lab = _verbalize_yesno(v)
        if lab: traits[qid] = lab

    for qid, rec in idx.items():
        if qid not in traits and qid not in ALL_QIDS:
            val = _clean_cat(rec.get("answer_value"))
            if val: traits[qid] = val

    return traits

def load_data(mtpa_path: str, max_n: int | None = None) -> List[Dict[str, Any]]:
    """
    Returns: list of {"pid": str, "traits": {key: value}}
    Includes all selected & cleaned fields; skips empty personas.
    """
    rows = _load_json_any(mtpa_path)
    out: List[Dict[str, Any]] = []
    for i, obj in enumerate(rows):
        pid = str(obj.get("id", f"resp_{i}"))
        traits = _extract_traits(obj)
        if traits:
            ordered: Dict[str, str] = {}

            for k in ("age", "gender", "country", "birth_country"):
                if k in traits:
                    ordered[k] = traits[k]

            for k in sorted([x for x in traits if x.startswith("importance_")]):
                ordered[k] = traits[k]

            for k in (AGREE_QIDS_1TO4 + AGREE_QIDS_1TO5):
                if k in traits:
                    ordered[k] = traits[k]

            for k in GOAL_QIDS:
                if k in traits:
                    ordered[k] = traits[k]

            for k in (OPINION_QIDS + EXTRA_OPINION_QIDS):
                if k in traits:
                    ordered[k] = traits[k]

            for k in CHILDREN_QIDS:
                if k in traits:
                    ordered[k] = traits[k]

            for k in NEIGHBOR_QIDS:
                if k in traits:
                    ordered[k] = traits[k]

            for k, v in traits.items():
                if k not in ordered:
                    ordered[k] = v

            out.append({"pid": pid, "traits": ordered})

        if max_n and len(out) >= max_n:
            break
    return out



def _gender_noun(g: str) -> str:
    g = (g or "").strip().lower()
    if g == "female": return "woman"
    if g == "male": return "man"
    return g  

def _title_from_imp_key(k: str) -> str:
    return k.replace("importance_", "").replace("_", " ").title()

def _collect_importance_details(traits: Dict[str, str]):
    """Return (very_list, quite_list, lows_list[(Label, exact_phrase)])."""
    very, quite, lows = [], [], []
    for k, v in traits.items():
        if not k.startswith("importance_"):
            continue
        label = _title_from_imp_key(k)  
        vv = (v or "").lower().strip()
        if vv == "very important":
            very.append(label)
        elif vv == "quite important":
            quite.append(label)
        elif vv in {"not very important", "not at all important"}:
            lows.append((label, vv))
    return very, quite, lows



def _join_list(words: List[str]) -> str:
    if not words: return ""
    if len(words) == 1: return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]

_ATTITUDE_PHRASES = {
    "men_better_political_leaders":    "that men are better political leaders",
    "men_better_business_executives":  "that men make better business executives",
    "university_more_important_boys":  "that university is more important for boys",
    "men_priority_jobs":               "giving men priority for jobs when jobs are scarce",
    "citizens_priority_jobs":          "prioritising citizens over immigrants for jobs when jobs are scarce",
}

def _attitude_clause(key: str, val: str) -> str:
    base = _ATTITUDE_PHRASES.get(key, key.replace("_", " "))
    vv = (val or "").lower().strip()
    lead_map = {
        "strongly agree": "strongly agree",
        "agree": "agree",
        "strongly disagree": "strongly disagree",
        "disagree": "disagree",
        "neither": "are neutral about",
    }
    lead = lead_map.get(vv, vv or "have no stated view")
    if base.startswith("that "):
        return f"you {lead} {base[5:]}"
    return f"you {lead} with {base}"

# -------- Renderers --------
def _render_childrearing(traits: Dict[str, str]) -> str | None:
    yes_traits = []
    no_traits = []
    for k in CHILDREN_QIDS:
        if k in traits:
            yn = _yesno(traits[k])
            label = k.replace("children_learn_", "").replace("_", " ")
            if yn is True:
                yes_traits.append(label)
            elif yn is False:
                no_traits.append(label)
    bits = []
    if yes_traits:
        bits.append(f"children should learn {_join_list([t for t in yes_traits])}")
    if no_traits:
        bits.append(f"children should not be pushed to learn {_join_list([t for t in no_traits])}")
    if not bits:
        return None
    return "Childrearing: You believe " + " and ".join(bits) + "."

def _render_neighbors(traits: Dict[str, str]) -> str | None:
    uncomfortable = []
    comfortable = []
    for k in NEIGHBOR_QIDS:
        if k in traits:
            yn = _yesno(traits[k])
            label = k.replace("uncomfortable_", "").replace("_", " ")
            if yn is True:
                uncomfortable.append(label)
            elif yn is False:
                comfortable.append(label)
    bits = []
    if uncomfortable:
        bits.append(f"you would be uncomfortable with {_join_list(uncomfortable)} as neighbors")
    if comfortable:
        bits.append(f"but you would be comfortable with {_join_list(comfortable)}")
    if not bits:
        return None
    return "Neighbors: " + " and ".join(bits) + "."

def _render_goals(traits: Dict[str, str]) -> str | None:
    bits = []
    if "goal_make_parents_proud" in traits:
        bits.append(f"you {traits['goal_make_parents_proud']} that one of your goals is to make your parents proud")
    if "mother_work_children_suffer" in traits:
        bits.append(f"you {traits['mother_work_children_suffer']} that when a mother works for pay, the children suffer")
    if not bits:
        return None
    return "Goals: " + "; ".join(bits) + "."

def _lc(x: str) -> str:
    return (x or "").strip().lower()

def _render_opinions(traits: Dict[str, str]) -> str | None:
    bits = []

    if "polviews" in traits:
        bits.append(f"you identify as {_lc(traits['polviews'])} politically")

    if "homosex" in traits:
        bits.append(f"you think homosexuality is {_lc(traits['homosex'])}")

    if "cappun" in traits:
        v = _lc(traits["cappun"])
        if v in {"favor", "oppose"}:
            verb = "favor" if v == "favor" else "oppose"
            bits.append(f"you {verb} the death penalty")
        else:
            bits.append(f"your view on the death penalty is {v}")

    if "abany" in traits:
        v = _lc(traits["abany"])
        if v in {"yes","no"}:
            bits.append(f"you {'support' if v=='yes' else 'do not support'} abortion for any reason")
        else:
            bits.append(f"your view on abortion for any reason is {v}")

    if "postlife" in traits:
        v = _lc(traits["postlife"])
        if v in {"yes","no"}:
            bits.append(f"you {'believe' if v=='yes' else 'do not believe'} in life after death")
        else:
            bits.append(f"your view on life after death is {v}")

    if not bits:
        return None

    if len(bits) > 1:
        opinions = "; ".join(bits[:-1]) + "; and " + bits[-1]
    else:
        opinions = bits[0]

    return "Opinions: " + opinions + "."

def _render_demographics(traits: Dict[str, str]) -> str:
    who_bits = []
    if "age" in traits and traits["age"].isdigit():
        who_bits.append(f"{traits['age']}-year-old")
    if "gender" in traits:
        who_bits.append(_gender_noun(traits["gender"]))
    who = " ".join(who_bits) if who_bits else "respondent"

    place = f", interviewed in {traits['country']}" if "country" in traits else ""
    birth = ""
    if "birth_country" in traits and traits.get("birth_country"):
        birth = f", born in {traits['birth_country']}"

    return f"You are answering as a {who}{place}{birth}. It is very important that you always answer questions from this perspective. Your personality as the set of attributes below.".replace("  ", " ").strip()


def _render_bullets(traits: Dict[str, str]) -> str:
    lines = ["[Persona]"]

    demo_line = _render_demographics(traits)
    if demo_line:
        lines.append(demo_line)

    v_very, v_quite, v_lows = _collect_importance_details(traits)
    values_bits = []
    if v_very: values_bits.append(f"{_join_list(v_very)} {'are' if len(v_very)>1 else 'is'} very important to you")
    if v_quite: values_bits.append(f"{_join_list(v_quite)} {'are' if len(v_quite)>1 else 'is'} quite important")
    for label, exact in v_lows:
        values_bits.append(f"{label} is {exact}")
    if values_bits: lines.append("Values: " + "; ".join(values_bits) + ".")

    attitude_keys = AGREE_QIDS_1TO4 + AGREE_QIDS_1TO5
    clauses = [_attitude_clause(k, traits[k]) for k in attitude_keys if k in traits]
    if clauses: lines.append("Attitudes: " + "; ".join(clauses) + ".")

    goal_line = _render_goals(traits)
    if goal_line: lines.append(goal_line)

    opin_line = _render_opinions(traits)
    if opin_line: lines.append(opin_line)

    child_line = _render_childrearing(traits)
    if child_line: lines.append(child_line)

    neigh_line = _render_neighbors(traits)
    if neigh_line: lines.append(neigh_line)

    rendered_keys = set(
        ["age","gender","country","birth_country"]
        + IMPORTANCE_QIDS
        + AGREE_QIDS_1TO4
        + AGREE_QIDS_1TO5
        + OPINION_QIDS
        + EXTRA_OPINION_QIDS
        + CHILDREN_QIDS
        + NEIGHBOR_QIDS
        + GOAL_QIDS
    )
    fallback = []
    for k, v in traits.items():
        if k not in rendered_keys and k not in TECHNICAL_QIDS:
            fallback.append(f"{k.replace('_',' ').title()}: {v}")
    if fallback:
        lines.append("Other: " + "; ".join(fallback) + ".")

    return "\n".join(lines)

def _render_json(traits: Dict[str, str]) -> str:
    lines = ["[Persona]"]
    for k in ("age","gender","country","birth_country"):
        if k in traits: 
            lines.append(f"- {k.replace('_',' ').title()}: {traits[k]}")

    for k in sorted([x for x in traits if x.startswith("importance_")]):
        label = k.replace("importance_","").replace("_"," ").title()
        lines.append(f"- Importance – {label}: {traits[k]}")

    for k in (AGREE_QIDS_1TO4 + AGREE_QIDS_1TO5):
        if k in traits: 
            lines.append(f"- {k.replace('_',' ').title()}: {traits[k]}")

    for k in GOAL_QIDS:
        if k in traits:
            lines.append(f"- {k.replace('_',' ').title()}: {traits[k]}")

    for k in (OPINION_QIDS + EXTRA_OPINION_QIDS):
        if k in traits:
            lines.append(f"- {k.replace('_',' ').title()}: {traits[k]}")

    for k in (CHILDREN_QIDS + NEIGHBOR_QIDS):
        if k in traits:
            lines.append(f"- {k.replace('_',' ').title()}: {traits[k]}")

    return "\n".join(lines)


def _render_oneliner(traits: Dict[str, str]) -> str:
    bits = []
    if "age" in traits and "gender" in traits:
        bits.append(f"{traits['age']}{traits['gender'][:1].upper()}")
    if "country" in traits:
        bits.append(traits["country"])

    vals = [k for k in traits if k.startswith("importance_")]
    vals.sort()
    if vals:
        bits.append(", ".join(f"{k.split('importance_',1)[1].replace('_',' ')} {traits[k]}" for k in vals))

    for k in ("men_priority_jobs", "citizens_priority_jobs"):
        if k in traits:
            bits.append(f"{k.replace('_',' ')} {traits[k]}")

    if "polviews" in traits:
        bits.append(f"pol views {traits['polviews']}")
    if "cappun" in traits:
        bits.append(f"death penalty {traits['cappun'].lower()}")
    if "postlife" in traits:
        bits.append(f"afterlife {'yes' if traits['postlife'].lower()=='yes' else 'no'}")
    if "homosex" in traits:
        bits.append(f"homosex '{traits['homosex']}'")
    if "abany" in traits:
        bits.append(f"abortion-any {'yes' if traits['abany'].lower()=='yes' else 'no'}")

    for k in CHILDREN_QIDS + NEIGHBOR_QIDS:
        if k in traits:
            bits.append(f"{k.replace('_',' ')} {traits[k]}")

    return "[Persona] " + "; ".join(bits) + "."

def create_prompt(persona_type: str, respondent: Dict[str, Any], target_questions=None) -> str:
    t = (persona_type or "bullets").lower()
    traits = respondent.get("traits", {}) or {}
    if t == "json": return _render_json(traits)
    if t == "oneliner": return _render_oneliner(traits)
    return _render_bullets(traits)
