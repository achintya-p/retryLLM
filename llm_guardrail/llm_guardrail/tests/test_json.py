import pytest
from ..validators.json_validator import JSONValidator
from ..retry_strategies.reprompt import RepromptStrategy

def test_json_validator_valid():
    validator = JSONValidator()
    valid_json = '{"key": "value", "numbers": [1, 2, 3]}'
    is_valid, parsed, message = validator.validate(valid_json)
    assert is_valid
    assert parsed == {"key": "value", "numbers": [1, 2, 3]}
    assert message == "Valid JSON"

def test_json_validator_invalid():
    validator = JSONValidator()
    invalid_json = '{"key": "value", numbers: [1, 2, 3]}'  # Missing quotes around numbers
    is_valid, parsed, message = validator.validate(invalid_json)
    assert not is_valid
    assert parsed is None
    assert "Invalid JSON" in message

def test_reprompt_strategy():
    strategy = RepromptStrategy()
    original_prompt = "List three colors"
    error_message = "Invalid JSON: Expecting property name enclosed in double quotes"
    
    # Test first retry
    improved1 = strategy.improve_prompt(original_prompt, error_message, 1)
    assert "MUST respond with valid" in improved1
    assert original_prompt in improved1
    
    # Test second retry
    improved2 = strategy.improve_prompt(original_prompt, error_message, 2)
    assert "Example of valid JSON format" in improved2
    assert error_message in improved2
    
    # Test third retry
    improved3 = strategy.improve_prompt(original_prompt, error_message, 3)
    assert "CRITICAL" in improved3
    assert "REQUIREMENTS" in improved3
