import json
from typing import Tuple, Any

class JSONValidator:
    """Validates if a string is valid JSON."""
    
    @staticmethod
    def validate(content: str) -> Tuple[bool, Any, str]:
        """
        Validate if the content is valid JSON.
        
        Args:
            content (str): The string to validate as JSON
            
        Returns:
            Tuple[bool, Any, str]: (is_valid, parsed_content, error_message)
        """
        try:
            # Try to parse the content as JSON
            parsed = json.loads(content)
            return True, parsed, "Valid JSON"
        except json.JSONDecodeError as e:
            # Return the specific JSON parsing error
            return False, None, f"Invalid JSON: {str(e)}"
        except Exception as e:
            # Catch any other unexpected errors
            return False, None, f"Unexpected error validating JSON: {str(e)}"
