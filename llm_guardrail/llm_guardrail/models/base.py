from typing import Any, Dict

class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        
    def call(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
        
    def _format_messages(self, prompt: str) -> list[Dict[str, Any]]:
        return [{"role": "user", "content": prompt}] 