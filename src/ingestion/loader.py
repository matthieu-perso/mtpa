from datasets import load_dataset

def load_dataset_hf(config):
    if "subset" in config:
        return load_dataset(config["source"], config["subset"], split=config.get("split", "train"))
    else:
        return load_dataset(config["source"], split=config.get("split", "train"))
