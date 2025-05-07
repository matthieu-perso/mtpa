from datasets import load_dataset

def load_dataset_hf(config):
    dataset = config["source"]
    subset = config.get("subset", None)
    split = config.get("split", "train")

    if subset:
        return load_dataset(dataset, subset, split=split)
    else:
        return load_dataset(dataset, split=split)
