<<<<<<< HEAD
import os
import json
from pathlib import Path
from typing import Dict, Any
from urllib.request import urlretrieve
from datasets import load_dataset

def download_lamp_files(config: Dict[str, Any]) -> Dict[str, Path]:
    """Download LaMP dataset files from CIIR server."""
    base_url = config["base_url"]
    data_files = config["data_files"]
    base_dir = Path(f"data/lamp_{config['name'].lower()}")
    
    downloaded_files = {}
    for split, files in data_files.items():
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for file_type, file_path in files.items():
            dest = split_dir / Path(file_path).name
            if not dest.exists():
                url = f"{base_url}/{file_path}"
                urlretrieve(url, dest)
                print(f"Downloaded: {dest}")
            else:
                print(f"Already exists: {dest}")
            downloaded_files[f"{split}_{file_type}"] = dest
    
    return downloaded_files

def load_lamp_dataset(config: Dict[str, Any]):
    """Load and process LaMP dataset from downloaded files."""
    files = download_lamp_files(config)
    data = []
    
    for split in config["data_files"].keys():
        with open(files[f"{split}_questions"], 'r', encoding='utf-8') as f:
            questions = json.load(f)
        with open(files[f"{split}_outputs"], 'r', encoding='utf-8') as f:
            outputs = json.load(f)
            
        output_lookup = {item["id"]: item["output"] for item in outputs["golds"]}
        
        for entry in questions:
            entry_id = entry["id"]
            transformed_entry = {
                "input": entry["input"].strip(),
                "output": output_lookup.get(entry_id, ""),
                "split": split,
                "profile": entry.get("profile", [])
            }
            data.append(transformed_entry)
    
    return data

def load_dataset_hf(config: Dict[str, Any]):
    """Load a dataset based on configuration."""
    # Check if this is a LaMP dataset
    if config["source"].startswith("LaMP_"):
        return load_lamp_dataset(config)
        
    # Otherwise load from HuggingFace
    dataset_args = {
        "path": config["source"]
    }
    
    if "subset" in config:
        dataset_args["name"] = config["subset"]
        
    return load_dataset(**dataset_args)["train"]  # Default to train split
=======
import pandas as pd
from datasets import load_dataset


def load_csv_file(config):
    return pd.read_csv(config["source"])

def load_dataset_hf(config):
    return load_dataset(config["source"], config["subset"], split=config.get("split", "train"))
>>>>>>> 36e17b3 (feat: further additions)
