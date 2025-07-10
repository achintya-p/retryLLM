class RepromptStrategy:
    """Strategy for improving prompts when validation fails."""
    
    @staticmethod
    def improve_prompt(original_prompt: str, validation_error: str, retry_count: int) -> str:
        """
        Generate an improved prompt based on the validation error.
        
        Args:
            original_prompt (str): The original prompt that failed
            validation_error (str): The error message from validation
            retry_count (int): Number of retries attempted so far
            
        Returns:
            str: An improved prompt
        """
        # Base improvement for JSON validation
        json_improvement = (
            "You MUST respond with valid, parseable JSON. "
            "Ensure your response contains only a JSON object or array, "
            "with no additional text before or after."
        )
        
        # Add more specific guidance based on retry count
        if retry_count == 1:
            return f"{original_prompt}\n\n{json_improvement}"
        elif retry_count == 2:
            return (
                f"{original_prompt}\n\n"
                f"Previous attempt failed with error: {validation_error}\n"
                f"{json_improvement}\n"
                "Example of valid JSON format:\n"
                '{"key": "value", "numbers": [1, 2, 3]}'
            )
        else:
            return (
                f"{original_prompt}\n\n"
                "CRITICAL: Previous attempts failed to produce valid JSON.\n"
                f"Error: {validation_error}\n"
                "REQUIREMENTS:\n"
                "1. Response must be ONLY valid JSON\n"
                "2. No text before or after the JSON\n"
                "3. Use double quotes for strings\n"
                "4. No trailing commas\n"
                "Example: {\"result\": [1, 2, 3]}"
            )
