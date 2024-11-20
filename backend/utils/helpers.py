import logging
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..config.settings import settings
from typing import List, Dict, Any

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


class Format:
    def parse_output(self, output: str, max_num: int) -> List[str]:
        try:
            # Split into lines and clean up
            lines = output.strip().split("\n")

            # Process each line
            parsed_outputs = []
            for line in lines:
                # Clean up the line and check if it starts with a number
                cleaned = line.strip().lstrip("1234567890.-) ")
                if line.strip() and any(
                    line.strip().startswith(str(i)) for i in range(1, 10)
                ):
                    parsed_outputs.append(cleaned)
                    if len(parsed_outputs) >= max_num:
                        break

            return parsed_outputs

        except Exception as e:
            logger.error(f"Error parsing output: {str(e)}")
            return []

    def validate_outputs(self, outputs: List[str], original_query: str) -> List[str]:
        if not outputs or len(outputs) < 1:
            logger.warning("Generated fewer than 1 outputs. Using original query.")
            return [original_query]

        return outputs


# TODO: Create Generate class with generate_text method
