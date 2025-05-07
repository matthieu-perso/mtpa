"""Normalization to format the data into a unified schema"""


def normalize(example, config):
    print(f"Normalizing example: {example}")
    text = example[config["input_column"]]
    raw_label = example[config["label_column"]]
    label = config.get("label_mapping", {}).get(str(raw_label), raw_label)
    return {"text": text, "label": label}
