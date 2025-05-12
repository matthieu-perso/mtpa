import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any

from src.ingestion.loader import load_dataset_hf
from src.preprocessing.normalization import normalize
#from src.preprocessing.processing import process_combined_dataset
#from src.transformation.transform import transform


def process_merged_dataset(config):
    # 1. Load and filter sources
    datasets = {}
    for src_cfg in config["merge_sources"]:
        raw = load_dataset_hf(src_cfg)

        filtered = raw.map(lambda ex: {
            col: ex[col] for col in src_cfg["keep_columns"]
        })
        datasets[src_cfg["name"]] = {ex[config["merge_key"]]: ex for ex in filtered}

    # 2. Find common keys
    common_keys = set.intersection(*(set(d.keys()) for d in datasets.values()))

    # 3. Merge and normalize
    results = []
    for key in common_keys:
        merged_example = {src_name: datasets[src_name][key] for src_name in datasets}
        merged_example[config["merge_key"]] = key  # include key at top-level
        try:
            normalized = normalize(merged_example, config)
            if normalized:
                results.append(normalized)
        except Exception as e:
            print(f"Error normalizing user_id={key}: {e}")
    return results


def process_dataset(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config.get("type") == "merge":
        return process_merged_dataset(config)
    else:
        return process_dataset(config)


class DatasetPipeline:
    def __init__(self, config_path: str):
        """Initialize the pipeline with a configuration file."""
        self.config = self._load_config(config_path)
        self.dataset_name = Path(config_path).stem

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def run(self, save_results: bool = True):
        print(f"Processing dataset: {self.dataset_name}")
        
        if self.config.get("type") == "merge":
            results = process_merged_dataset(self.config)
        else:
            raw = load_dataset_hf(self.config)
            #results = raw.map(lambda ex: normalize(ex, self.config))
        
        if save_results:
            # Save results as JSON
            output_dir = Path("results")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{self.dataset_name}_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"Results saved to {output_path}")
        
        return results


def main():
    """Main entry point for the pipeline."""
    config_dir = Path("configs")
    
    # Process all datasets in the configs directory
    for config_file in config_dir.glob("*.yaml"):
        pipeline = DatasetPipeline(str(config_file))
        pipeline.run(save_results=True)

if __name__ == "__main__":
    main() 