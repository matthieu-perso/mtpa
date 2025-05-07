from typing import Dict, Any, Optional
from .base import BaseModel
from .huggingface_model import HuggingFaceModel
from .openai_model import OpenAIModel

class ModelRegistry:
    """Registry for managing model instantiation and configuration."""
    
    def __init__(self):
        """Initialize the model registry."""
        self._models: Dict[str, BaseModel] = {}
        self._configs: Dict[str, Dict[str, Any]] = {}
        
    def register_model(
        self,
        model_name: str,
        model_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a model configuration.
        
        Args:
            model_name: Unique identifier for the model
            model_type: Type of model (huggingface, openai, etc.)
            config: Model configuration dictionary
        """
        self._configs[model_name] = {
            "type": model_type,
            "config": config or {}
        }
    
    def get_model(self, model_name: str) -> BaseModel:
        """Get or create a model instance.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model is not registered
        """
        if model_name not in self._configs:
            raise ValueError(f"Model {model_name} not registered")
            
        if model_name not in self._models:
            config = self._configs[model_name]
            model_type = config["type"]
            
            if model_type == "huggingface":
                self._models[model_name] = HuggingFaceModel(
                    model_name=model_name,
                    **config["config"]
                )
            elif model_type == "openai":
                self._models[model_name] = OpenAIModel(
                    model_name=model_name,
                    **config["config"]
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        return self._models[model_name]
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models and their configurations.
        
        Returns:
            Dictionary of model configurations
        """
        return self._configs.copy()

# Global registry instance
registry = ModelRegistry()

# Register some default models
registry.register_model(
    "distilbert-base-uncased",
    "huggingface",
    {
        "task_type": "text-classification",
        "device": "cuda"
    }
)

registry.register_model(
    "gpt-3.5-turbo",
    "openai",
    {
        "task_type": "text-classification",
        "temperature": 0
    }
) 