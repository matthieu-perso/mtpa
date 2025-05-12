"""Normalization to format the data into a unified schema"""


from src.preprocessing.logic import resolve

def normalize(example, config):
    m = config["mappings"]

    return {
        "task_type": resolve(m["task_type"], example),
        "prompt": resolve(m["prompt"], example),
        "split": config.get("split", "train"),
        "annotation": {
            "type": resolve(m["annotation"]["type"], example),
            "value": resolve(m["annotation"]["value"], example),
            "choices": resolve(m["annotation"].get("choices", []), example)
        },
        "user_profile": {
            "stated_preferences": {
                "pairwise": resolve(m["user_profile"]["stated_preferences"].get("pairwise", []), example),
                "scored": resolve(m["user_profile"]["stated_preferences"].get("scored", []), example),
                "categorical": resolve(m["user_profile"]["stated_preferences"].get("categorical", []), example),
                "ranked_lists": resolve(m["user_profile"]["stated_preferences"].get("ranked_lists", []), example),
                "freeform_text": resolve(m["user_profile"]["stated_preferences"].get("freeform_text", []), example)
            },
            "observed_preferences": {
                "click_data": resolve(m["user_profile"]["observed_preferences"].get("click_data", []), example),
                "chosen_options": resolve(m["user_profile"]["observed_preferences"].get("chosen_options", []), example)
            },
            "behavior": resolve(m["user_profile"].get("behavior", []), example),
            "characteristics": resolve(m["user_profile"].get("characteristics", {}), example)
        }
    }

def to_list(x):
    return x if isinstance(x, list) else [x]
