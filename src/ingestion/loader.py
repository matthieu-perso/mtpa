from datasets import load_dataset

def load_dataset_hf(config):
    return load_dataset(config["source"], config["subset"], split=config.get("split", "train"))
