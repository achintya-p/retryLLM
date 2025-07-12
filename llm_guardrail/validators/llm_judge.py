from typing import Dict, Any, Tuple, Optional
import json

class LLMJudge:
    """Validator that uses an LLM to judge response quality."""
    
    def __init__(self, router):
        """Initialize with a router instance for making LLM calls."""
        self.router = router
    
    def _create_judge_prompt(self, original_prompt: str, response: str) -> str:
        """Create a prompt for the judge model."""
        return f"""You are an expert judge evaluating an AI's response. Analyze the following response to determine if it is:
1. Relevant to the prompt
2. Accurate and factual
3. Well-structured and clear
4. Complete (addresses all aspects of the prompt)

Original Prompt:
{original_prompt}

AI Response:
{response}

Evaluate ONLY these aspects. Return a JSON object with this exact structure:
{{
    "is_valid": true/false,
    "score": float between 0-1,
    "reason": "Brief explanation of the judgment",
    "improvements_needed": ["specific_issue_1", "specific_issue_2"] (empty list if valid)
}}"""

    def validate(
        self,
        prompt: str,
        response: str,
        judge_model: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Validate a response using an LLM judge.
        
        Args:
            prompt: Original prompt that generated the response
            response: The response to validate
            judge_model: Optional specific model to use for judging
            **kwargs: Additional arguments for the judge model
            
        Returns:
            Tuple[bool, Dict[str, Any], str]: (is_valid, parsed_judgment, message)
        """
        # Use a more capable model for judging if not specified
        if not judge_model:
            judge_model = "gemini-2.5-pro"  # Default to a more capable model
        
        try:
            # Create judge prompt
            judge_prompt = self._create_judge_prompt(prompt, response)
            
            # Get judgment
            result = self.router.call_model(
                model=judge_model,
                prompt=judge_prompt,
                temperature=0.3,  # Lower temperature for more consistent judgments
                **kwargs
            )
            
            # Parse judgment
            judgment = result.get("result", {})
            if isinstance(judgment, str):
                try:
                    judgment = json.loads(judgment)
                except json.JSONDecodeError:
                    return False, {}, "Judge returned invalid JSON"
            
            is_valid = judgment.get("is_valid", False)
            score = judgment.get("score", 0.0)
            reason = judgment.get("reason", "No reason provided")
            improvements = judgment.get("improvements_needed", [])
            
            # Create structured result
            parsed_judgment = {
                "is_valid": is_valid,
                "score": score,
                "reason": reason,
                "improvements_needed": improvements
            }
            
            message = f"{'VALID' if is_valid else 'INVALID'}: {reason}"
            if improvements:
                message += f"\nImprovements needed: {', '.join(improvements)}"
            
            return is_valid, parsed_judgment, message
            
        except Exception as e:
            return False, {}, f"Validation failed: {str(e)}" 