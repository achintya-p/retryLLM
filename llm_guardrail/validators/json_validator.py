import json
from typing import Tuple, Optional, Any, Dict

class JSONValidator:
    """Validator for JSON responses."""
    
    @staticmethod
    def validate(text: str) -> Tuple[bool, Optional[Any], str]:
        """
        Validate if the text is valid JSON.
        
        Args:
            text: The text to validate
            
        Returns:
            Tuple[bool, Optional[Any], str]: (is_valid, parsed_content, message)
        """
        try:
            parsed = json.loads(text)
            return True, parsed, "Valid JSON"
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    @staticmethod
    def validate_schema(data: Any, schema: Dict) -> Tuple[bool, str]:
        """
        Validate JSON data against a schema.
        
        Args:
            data: The data to validate
            schema: The schema to validate against
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            # Check type
            if "type" in schema:
                if schema["type"] == "object" and not isinstance(data, dict):
                    return False, "Expected object"
                elif schema["type"] == "array" and not isinstance(data, list):
                    return False, "Expected array"
                elif schema["type"] == "string" and not isinstance(data, str):
                    return False, "Expected string"
                elif schema["type"] == "number" and not isinstance(data, (int, float)):
                    return False, "Expected number"
                elif schema["type"] == "boolean" and not isinstance(data, bool):
                    return False, "Expected boolean"
            
            # Check required fields
            if "required" in schema and isinstance(data, dict):
                missing = [field for field in schema["required"] if field not in data]
                if missing:
                    return False, f"Missing required fields: {', '.join(missing)}"
            
            # Check properties
            if "properties" in schema and isinstance(data, dict):
                for key, prop_schema in schema["properties"].items():
                    if key in data:
                        is_valid, msg = JSONValidator.validate_schema(data[key], prop_schema)
                        if not is_valid:
                            return False, f"Invalid field '{key}': {msg}"
            
            # Check array items
            if "items" in schema and isinstance(data, list):
                for i, item in enumerate(data):
                    is_valid, msg = JSONValidator.validate_schema(item, schema["items"])
                    if not is_valid:
                        return False, f"Invalid item at index {i}: {msg}"
            
            # Check enum
            if "enum" in schema and data not in schema["enum"]:
                return False, f"Value must be one of: {', '.join(map(str, schema['enum']))}"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Schema validation error: {str(e)}" 