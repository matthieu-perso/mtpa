from typing import List, Dict, Any, Optional
import openai
from .base import BaseModel

class OpenAIModel(BaseModel):
    """Wrapper for OpenAI API models."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        task_type: str = "text-classification",
        prompt_template: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize an OpenAI model.
        
        Args:
            model_name: Name of the OpenAI model
            task_type: Type of task (text-classification, text-generation, etc.)
            prompt_template: Template for constructing prompts
            api_key: OpenAI API key (if not set in environment)
            **kwargs: Additional arguments passed to the API
        """
        super().__init__(model_name, {
            "task_type": task_type,
            "prompt_template": prompt_template,
            **kwargs
        })
        
        # Set API key
        if api_key:
            openai.api_key = api_key
            
        # Set default prompt template based on task
        if not prompt_template:
            if task_type == "text-classification":
                self.prompt_template = "Classify the following text into one of the categories: {categories}\n\nText: {text}\n\nCategory:"
            elif task_type == "text-generation":
                self.prompt_template = "Complete the following text: {text}"
            else:
                self.prompt_template = "{text}"
    
    def predict(self, texts: List[str]) -> List[Any]:
        """Make predictions using the OpenAI API.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of predictions
        """
        results = []
        
        for text in texts:
            # Construct prompt
            if self.config["task_type"] == "text-classification":
                prompt = self.prompt_template.format(
                    text=text,
                    categories=", ".join(self.config.get("categories", []))
                )
            else:
                prompt = self.prompt_template.format(text=text)
            
            # Call API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic output
                **self.config
            )
            
            # Parse response
            content = response["choices"][0]["message"]["content"].strip()
            
            if self.config["task_type"] == "text-classification":
                # Try to extract category index
                try:
                    result = int(content.split()[0])
                except (ValueError, IndexError):
                    # Fallback to category name
                    result = content
            else:
                result = content
                
            results.append(result)
            
        return results 