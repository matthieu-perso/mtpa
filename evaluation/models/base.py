from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

class BaseModel(ABC):
    """Abstract base class defining the interface for all model implementations."""
    
    def __init__(self, model_name: str, model_config: Optional[Dict[str, Any]] = None):
        """Initialize the model with a name and optional configuration.
        
        Args:
            model_name: Unique identifier for the model
            model_config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = model_config or {}
        
    @abstractmethod
    def predict(self, texts: List[str]) -> List[Any]:
        """Make predictions for a list of input texts.
        
        Args:
            texts: List of input texts to process
            
        Returns:
            List of predictions (format depends on task type)
        """
        pass
    
    def batch_predict(self, texts: List[str], batch_size: int = 32) -> List[Any]:
        """Process texts in batches to avoid memory issues.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of predictions
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend(self.predict(batch))
        return results
    
    def save_results(self, results: Dict[str, Any], dataset_name: str) -> Path:
        """Save evaluation results to disk.
        
        Args:
            results: Dictionary of metrics and metadata
            dataset_name: Name of the dataset evaluated
            
        Returns:
            Path to the saved results file
        """
        results_dir = Path("evaluation/results")
        results_dir.mkdir(exist_ok=True)
        
        out_path = results_dir / f"{dataset_name}_{self.model_name}.json"
        
        with open(out_path, "w") as f:
            json.dump({
                "model": self.model_name,
                "dataset": dataset_name,
                "config": self.config,
                "metrics": results
            }, f, indent=2)
            
        return out_path 