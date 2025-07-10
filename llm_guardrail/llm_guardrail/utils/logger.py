import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class GuardrailLogger:
    """Logger for LLM Guardrail operations."""
    
    def __init__(self, log_file: str = "logs.jsonl"):
        """
        Initialize the logger.
        
        Args:
            log_file (str): Path to the log file
        """
        self.log_file = Path(log_file)
        
        # Set up standard Python logger
        self.logger = logging.getLogger("llm_guardrail")
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already added
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_attempt(self, 
                   prompt: str,
                   response: str,
                   validation_result: bool,
                   error_message: str,
                   retry_count: int) -> None:
        """
        Log an LLM call attempt to the JSONL file.
        
        Args:
            prompt (str): The prompt sent to the LLM
            response (str): The response received
            validation_result (bool): Whether validation passed
            error_message (str): Any error message from validation
            retry_count (int): Current retry attempt number
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "response": response,
            "validation_passed": validation_result,
            "error_message": error_message,
            "retry_count": retry_count
        }
        
        # Log to console
        self.logger.info(f"Attempt {retry_count}: {'✓' if validation_result else '✗'} - {error_message}")
        
        # Append to JSONL file
        try:
            with self.log_file.open("a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except Exception as e:
            self.logger.error(f"Failed to write to log file: {e}")
    
    def log_final_result(self, result: Dict[str, Any]) -> None:
        """
        Log the final result of the LLM call.
        
        Args:
            result (Dict[str, Any]): The final result dictionary
        """
        self.logger.info(
            f"Final result - Status: {result['status']}, "
            f"Retries: {result['retry_count']}, "
            f"Reason: {result['reason']}"
        )
