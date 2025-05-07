from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from .base import BaseModel

class HuggingFaceModel(BaseModel):
    """Wrapper for HuggingFace models supporting various tasks."""
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "text-classification",
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize a HuggingFace model.
        
        Args:
            model_name: Name or path of the HuggingFace model
            task_type: Type of task (text-classification, text-generation, etc.)
            device: Device to run the model on (cuda, cpu, etc.)
            **kwargs: Additional arguments passed to the pipeline
        """
        super().__init__(model_name, {"task_type": task_type, **kwargs})
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create pipeline
        self.pipeline = pipeline(
            task_type,
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            **kwargs
        )
    
    def predict(self, texts: List[str]) -> List[Any]:
        """Make predictions using the HuggingFace pipeline.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of predictions (format depends on task type)
        """
        results = self.pipeline(texts)
        
        # Handle different task types
        if self.config["task_type"] == "text-classification":
            return [int(pred["label"].split("_")[-1]) for pred in results]
        elif self.config["task_type"] == "text-generation":
            return [pred["generated_text"] for pred in results]
        else:
            return results 