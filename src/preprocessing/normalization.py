"""Normalization to format the data into a unified schema"""


from src.preprocessing.logic import resolve

def normalize(example, config):
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
        },
        "user_profile": {
            "stated_preferences": {
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
        }
    }

def to_list(x):
    return x if isinstance(x, list) else [x]
