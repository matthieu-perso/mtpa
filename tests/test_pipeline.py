import pytest
from pathlib import Path
from src.pipeline import DatasetPipeline

def test_pipeline_initialization():
    """Test pipeline initialization with a config file."""
    config_path = "configs/dataset_a.yaml"
    pipeline = DatasetPipeline(config_path)
    
    assert pipeline.dataset_name == "dataset_a"
    assert "dataset" in pipeline.config
    assert "ingestion" in pipeline.config

def test_config_loading():
    """Test configuration loading from YAML."""
    config_path = "configs/dataset_a.yaml"
    pipeline = DatasetPipeline(config_path)
    
    # Test specific config values
    assert pipeline.config["dataset"]["name"] == "dataset_a"
    assert pipeline.config["preprocessing"]["drop_na"] is True
    assert pipeline.config["transformation"]["vectorization"]["method"] == "tfidf"

@pytest.mark.parametrize("config_path", [
    "nonexistent_config.yaml",
    "invalid_config.yaml"
])
def test_invalid_config(config_path):
    """Test pipeline behavior with invalid config files."""
    with pytest.raises(Exception):
        DatasetPipeline(config_path) 