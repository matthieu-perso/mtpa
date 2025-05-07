import os
import yaml
from pathlib import Path
from typing import Dict, Any

from src.ingestion.loader import load_dataset_hf
from src.preprocessing.normalization import normalize
#from src.transformation.transform import transform

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

    def run(self):
        """Execute the full pipeline for a dataset."""
        print(f"Processing dataset: {self.dataset_name}")
        
        raw = load_dataset_hf(self.config)
        normed = raw.map(lambda ex: normalize(ex, self.config))
        #transformed = normed.map(transform)
        return normed
        
        print(f"Completed processing dataset: {self.dataset_name}")

def main():
    """Main entry point for the pipeline."""
    config_dir = Path("configs")
    
    # Process all datasets in the configs directory
    for config_file in config_dir.glob("*.yaml"):
        pipeline = DatasetPipeline(str(config_file))
        pipeline.run()

if __name__ == "__main__":
    main() 