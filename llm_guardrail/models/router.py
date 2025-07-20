from typing import List, Tuple, Dict, Any, Optional
import random
import os
import google.generativeai as genai
import groq

class SmartRouter:
    """Smart router for selecting and calling LLM models."""
    def __init__(self):
        # Load environment variables
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize clients
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
        if self.groq_api_key:
            self.groq_client = groq.Groq(api_key=self.groq_api_key)
        
        # Available models and their weights
        self.models = {
            "gemini-pro": 0.6,
            "llama3-70b-8192": 0.4
        }

    def select_models(self, prompt: str) -> Tuple[str, List[str]]:
        """Select primary and fallback models based on the prompt."""
        available_models = []
        
        # Only include models we have API keys for
        if self.google_api_key:
            available_models.append(("gemini-pro", self.models["gemini-pro"]))
        if self.groq_api_key:
            available_models.append(("llama3-70b-8192", self.models["llama3-70b-8192"]))
        
        if not available_models:
            raise ValueError("No API keys configured. Please set GROQ_API_KEY and/or GOOGLE_API_KEY environment variables.")
        
        models, weights = zip(*available_models)
        primary = random.choices(models, weights=weights)[0]
        fallbacks = [m for m in models if m != primary]
        
        return primary, fallbacks
    
    def call_model(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Call the specified model with the given prompt."""
        try:
            if model == "gemini-pro":
                if not self.google_api_key:
                    raise ValueError("GOOGLE_API_KEY not configured")
                
                # Configure and generate response
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                
                if response.text:
                    return {
                        "result": response.text,
                        "model_used": "gemini-pro",
                        "tokens": {"prompt": 0, "completion": 0, "total": 0}  # Gemini doesn't provide token counts
                    }
                else:
                    raise ValueError("Empty response from Gemini")
                    
            elif model == "llama3-70b-8192":
                if not self.groq_api_key:
                    raise ValueError("GROQ_API_KEY not configured")
                    
                completion = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    **kwargs
                )
                
                return {
                    "result": completion.choices[0].message.content,
                    "model_used": model,
                    "tokens": {
                        "prompt": completion.usage.prompt_tokens,
                        "completion": completion.usage.completion_tokens,
                        "total": completion.usage.total_tokens
                    }
                }
            else:
                raise ValueError(f"Unsupported model: {model}")
                
        except Exception as e:
            raise Exception(f"Error calling {model}: {str(e)}")
