"""Normalization to format the data into a unified schema"""


from src.preprocessing.logic import resolve

def normalize(example, config):
<<<<<<< HEAD
    m = config["mappings"]
    is_merged = config.get("type") == "merge"

    return {
        "task_type": resolve(m["task_type"], example, is_merged),
        "prompt": resolve(m["prompt"], example, is_merged),
        "split": config.get("split", "train"),
        "annotation": {
            "type": resolve(m["annotation"]["type"], example, is_merged),
            "value": resolve(m["annotation"]["value"], example, is_merged),
            "choices": resolve(m["annotation"].get("choices", []), example, is_merged)
=======
    m = config.get("mappings", {})

    return {
        "user_id": resolve(m.get("user_id", ""), example),
        "task_type": resolve(m.get("task_type", ""), example),
        "prompt": resolve(m.get("prompt", ""), example),
        "split": config.get("split", "train"),
        "target": {
            "input": resolve(m.get("target", {}).get("input", ""), example),
            "value": resolve(m.get("target", {}).get("value", ""), example),
            "choices": resolve(m.get("target", {}).get("choices", []), example)
>>>>>>> 36e17b3 (feat: further additions)
        },
        "user_profile": {
            "characteristics": resolve(m.get("user_profile", {}).get("characteristics", {}), example),
            "stated_preferences": {
<<<<<<< HEAD
                "pairwise": resolve(m["user_profile"]["stated_preferences"].get("pairwise", []), example, is_merged),
                "scored": resolve(m["user_profile"]["stated_preferences"].get("scored", []), example, is_merged),
                "categorical": resolve(m["user_profile"]["stated_preferences"].get("categorical", []), example, is_merged),
                "ranked_lists": resolve(m["user_profile"]["stated_preferences"].get("ranked_lists", []), example, is_merged),
                "freeform_text": resolve(m["user_profile"]["stated_preferences"].get("freeform_text", []), example, is_merged)
            },
            "observed_preferences": {
                "click_data": resolve(m["user_profile"]["observed_preferences"].get("click_data", []), example, is_merged),
                "chosen_options": resolve(m["user_profile"]["observed_preferences"].get("chosen_options", []), example, is_merged)
            },
            "behavior": resolve(m["user_profile"].get("behavior", []), example, is_merged),
            "characteristics": resolve(m["user_profile"].get("characteristics", {}), example, is_merged)
=======
                "pairwise": resolve(m.get("user_profile", {}).get("stated_preferences", {}).get("pairwise", []), example),
                "scored": resolve(m.get("user_profile", {}).get("stated_preferences", {}).get("scored", []), example),
                "categorical": resolve(m.get("user_profile", {}).get("stated_preferences", {}).get("categorical", []), example),
                "ranked_lists": resolve(m.get("user_profile", {}).get("stated_preferences", {}).get("ranked_lists", []), example),
                "freeform_text": resolve(m.get("user_profile", {}).get("stated_preferences", {}).get("freeform_text", []), example)
            },
            "observed_preferences": {
                "pairwise": resolve(m.get("user_profile", {}).get("observed_preferences", {}).get("pairwise", []), example),
                "scored": resolve(m.get("user_profile", {}).get("observed_preferences", {}).get("scored", []), example),
                "categorical": resolve(m.get("user_profile", {}).get("observed_preferences", {}).get("categorical", []), example),
                "ranked_lists": resolve(m.get("user_profile", {}).get("observed_preferences", {}).get("ranked_lists", []), example),
                "freeform_text": resolve(m.get("user_profile", {}).get("observed_preferences", {}).get("freeform_text", []), example)
            },
            "behavior": resolve(m.get("user_profile", {}).get("behavior", []), example)
>>>>>>> 36e17b3 (feat: further additions)
        }
    }

def to_list(x):
    return x if isinstance(x, list) else [x]
