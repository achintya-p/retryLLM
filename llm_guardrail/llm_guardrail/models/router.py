from .openai_model import OpenAIModel
from .google_model import GoogleModel
from .groq_model import GroqModel

MODEL_MAP = {
    # OpenAI models
    "gpt-3.5-turbo": OpenAIModel,
    "gpt-4": OpenAIModel,
    "gpt-4-turbo-preview": OpenAIModel,
    
    # Google models
    "gemini-pro": GoogleModel,
    "models/gemini-2.5-pro": GoogleModel,
    
    # Groq models
    "mixtral-8x7b-32768": GroqModel,
    "llama2-70b-4096": GroqModel
}

def get_model(model_name: str):
    """
    Get an instance of the appropriate model class based on the model name.
    
    Args:
        model_name (str): Name of the model to use
        
    Returns:
        BaseModel: An instance of the appropriate model class
        
    Raises:
        ValueError: If the model name is not supported
    """
    for key in MODEL_MAP:
        if model_name.lower() == key.lower():
            return MODEL_MAP[key](model_name)
            
    raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_MAP.keys())}")
