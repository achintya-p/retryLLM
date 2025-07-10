import os
from typing import Dict, Any, Optional
import openai
from dotenv import load_dotenv

from .validators.json_validator import JSONValidator
from .retry_strategies.reprompt import RepromptStrategy
from .utils.logger import GuardrailLogger

# Load environment variables
load_dotenv()

class LLMGuardrail:
    """Main class for handling LLM calls with validation and retry logic."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM Guardrail.
        
        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, will look for OPENAI_API_KEY in env
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key
        self.logger = GuardrailLogger()
    
    def _call_llm(self, prompt: str, model: str) -> str:
        """
        Make the actual call to the LLM.
        
        Args:
            prompt (str): The prompt to send
            model (str): The model to use
            
        Returns:
            str: The LLM's response
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides responses in the requested format."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    
    def safe_call(self,
                 prompt: str,
                 model: str = "gpt-4",
                 validate: str = "json",
                 max_retries: int = 3) -> Dict[str, Any]:
        """
        Make an LLM call with validation and automatic retry.
        
        Args:
            prompt (str): The prompt to send to the LLM
            model (str): The OpenAI model to use
            validate (str): Validation strategy to use (currently only 'json' supported)
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            Dict[str, Any]: Result dictionary containing:
                - result: The final output (parsed if JSON)
                - status: 'success', 'recovered', or 'failed'
                - retry_count: Number of retries performed
                - reason: Description of the outcome
        """
        if validate != "json":
            raise ValueError("Currently only 'json' validation is supported")
        
        validator = JSONValidator()
        retry_strategy = RepromptStrategy()
        current_prompt = prompt
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Call the LLM
                response = self._call_llm(current_prompt, model)
                
                # Validate the response
                is_valid, parsed_content, error_message = validator.validate(response)
                
                # Log the attempt
                self.logger.log_attempt(
                    prompt=current_prompt,
                    response=response,
                    validation_result=is_valid,
                    error_message=error_message,
                    retry_count=retry_count
                )
                
                if is_valid:
                    # Success! Return the result
                    result = {
                        "result": parsed_content,
                        "status": "success" if retry_count == 0 else "recovered",
                        "retry_count": retry_count,
                        "reason": "Valid JSON response received"
                    }
                    self.logger.log_final_result(result)
                    return result
                
                # If validation failed and we have retries left
                if retry_count < max_retries:
                    current_prompt = retry_strategy.improve_prompt(
                        original_prompt=prompt,
                        validation_error=error_message,
                        retry_count=retry_count + 1
                    )
                    retry_count += 1
                else:
                    # Out of retries
                    result = {
                        "result": None,
                        "status": "failed",
                        "retry_count": retry_count,
                        "reason": f"Max retries ({max_retries}) reached. Last error: {error_message}"
                    }
                    self.logger.log_final_result(result)
                    return result
                    
            except Exception as e:
                # Handle any unexpected errors
                result = {
                    "result": None,
                    "status": "failed",
                    "retry_count": retry_count,
                    "reason": f"Unexpected error: {str(e)}"
                }
                self.logger.log_final_result(result)
                return result

# Convenience function for direct use
def safe_call(prompt: str,
             model: str = "gpt-4",
             validate: str = "json",
             max_retries: int = 3,
             api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to make a safe LLM call without explicitly creating a LLMGuardrail instance.
    
    Args are the same as LLMGuardrail.safe_call()
    """
    guardrail = LLMGuardrail(api_key=api_key)
    return guardrail.safe_call(prompt=prompt, model=model, validate=validate, max_retries=max_retries)
