import json
from typing import Dict, Any, Optional, List, Tuple
import time
from datetime import datetime, timedelta
from .models.router import SmartRouter
from .validators.json_validator import JSONValidator
from .validators.llm_judge import LLMJudge

class LLMGuardrail:
    """Main class for handling LLM calls with validation and retries."""
    
    def __init__(self):
        self.router = SmartRouter()
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)
        self.json_validator = JSONValidator()
        self.llm_judge = LLMJudge(self.router)
    
    def _get_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate a cache key."""
        return f"{prompt}:{model}:{str(kwargs)}"
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a value from cache if it exists and hasn't expired."""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        if datetime.utcnow() - entry['timestamp'] > self.cache_ttl:
            del self.cache[key]
            return None
            
        return entry['value']
    
    def _set_cache(self, key: str, value: Dict[str, Any]):
        """Set a value in the cache."""
        self.cache[key] = {
            'value': value,
            'timestamp': datetime.utcnow()
        }
    
    def safe_call(
        self,
        prompt: str,
        model: Optional[str] = None,
        validate: Optional[str] = None,
        judge_model: Optional[str] = None,
        max_retries: int = 3,
        fallback_models: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a safe call to an LLM with validation and retries.
        
        Args:
            prompt: The prompt to send
            model: Optional specific model to use (if not provided, will be auto-selected)
            validate: Validation method ('json' or 'llm_judge')
            judge_model: Optional specific model to use for LLM judging
            max_retries: Maximum number of retries per model
            fallback_models: Optional list of specific fallback models
            **kwargs: Additional arguments for the model
            
        Returns:
            Dict[str, Any]: Response with metadata
        """
        # Get model recommendations if not specified
        if not model or not fallback_models:
            primary, fallbacks = self.router.select_models(prompt)
            model = model or primary
            fallback_models = fallback_models or fallbacks
        
        models_to_try = [model] + (fallback_models or [])
        total_retries = 0
        last_error = None
        
        for current_model in models_to_try:
            current_prompt = prompt
            model_retries = 0
            
            while model_retries < max_retries:
                # Check cache
                cache_key = self._get_cache_key(current_prompt, current_model, **kwargs)
                if cached := self._get_from_cache(cache_key):
                    return cached
                
                try:
                    # Call the model
                    response = self.router.call_model(current_model, current_prompt, **kwargs)
                    
                    # Validate if needed
                    validation_result = None
                    if validate == 'json':
                        is_valid, parsed_content, error_msg = self.json_validator.validate(response['result'])
                        if not is_valid:
                            raise ValueError(error_msg)
                        response['result'] = parsed_content
                    elif validate == 'llm_judge':
                        is_valid, judgment, error_msg = self.llm_judge.validate(
                            prompt=prompt,
                            response=response['result'],
                            judge_model=judge_model,
                            **kwargs
                        )
                        if not is_valid:
                            raise ValueError(error_msg)
                        validation_result = judgment
                    
                    # Success! Cache and return
                    result = {
                        **response,
                        "status": "success" if current_model == model else "recovered",
                        "retry_count": total_retries,
                        "reason": "valid" if total_retries == 0 else "valid_after_retry"
                    }
                    
                    # Add validation results if available
                    if validation_result:
                        result["validation"] = validation_result
                    
                    self._set_cache(cache_key, result)
                    return result
                    
                except Exception as e:
                    last_error = str(e)
                    print(f"Error with {current_model}: {last_error}")
                    
                    # Prepare for retry
                    total_retries += 1
                    model_retries += 1
                    
                    if model_retries < max_retries:
                        # Add context for retry
                        current_prompt = f"{prompt}\n\nNote: This is retry #{model_retries}. Previous attempt failed with: {last_error}"
        
        return {
            "result": None,
            "status": "failed",
            "retry_count": total_retries,
            "model_used": None,
            "reason": "all_models_failed",
            "last_error": str(last_error)
        }

# Convenience function
def safe_call(
    prompt: str,
    model: Optional[str] = None,
    validate: Optional[str] = None,
    judge_model: Optional[str] = None,
    max_retries: int = 3,
    fallback_models: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for making a safe LLM call."""
    guardrail = LLMGuardrail()
    return guardrail.safe_call(
        prompt=prompt,
        model=model,
        validate=validate,
        judge_model=judge_model,
        max_retries=max_retries,
        fallback_models=fallback_models,
        **kwargs
    ) 