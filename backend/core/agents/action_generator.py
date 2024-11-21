from ..base_agent import BaseAgent
from ...config.settings import settings
import torch
from typing import List, Dict
from ...utils.helpers import Format


class ActionGenerator(BaseAgent):
    def __init__(self):
        super().__init__(settings)

    def _create_prompt(self, query: str) -> List[Dict[str, str]]:
        chat_template = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that breaks down user queries into actions."
                    "Your goal is to create step-by-step actions that accurately answer the user's query."
                    "The available actions are 'Search' and 'Summarize'."
                    "Each action must contain all details from the query and must be ordered logically to ensure accuracy."
                    "Do not execute the actions.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "# Example 1\n"
                    "Query: What is the world's first AI-generated video game?\n"
                    "Max Action: 2\n"
                    "Action:\n"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "1. Search for information about the world's first AI-generated video game\n"
                    "2. Summarize the findings on the world's first AI-generated video game"
                ),
            },
            {
                "role": "user",
                "content": (
                    "# Example 2\n"
                    "Query: How do you delete all save data of Animal Crossing in Nintendo Switch?\n"
                    "Max Action: 2\n"
                    "Action:\n"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "1. Search for instructions on deleting save data of Animal Crossing in Nintendo Switch\n"
                    "2. Search user guides or official support pages related to Nintendo Switch save data management"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n"
                    f"Max Action: {self.params.get('max_actions')}\n"
                    "Action:\n"
                ),
            },
        ]

        return chat_template

    def _generate_response(self, query: str) -> str:
        """
        Generate a response using a chat template.
        """
        # Create a chat template from the query
        chat_template = self._create_prompt(query)

        # Tokenize input using apply_chat_template
        inputs = self._tokenize_input(chat_template, max_length=self.params.get("max_length"))

        # Generate output using the model
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.params.get("max_length"),
                temperature=self.params.get("temperature"),
                num_beams=self.params.get("num_beams"),
                do_sample=self.params.get("do_sample"),
                top_p=self.params.get("top_p"),
                top_k=self.params.get("top_k"),
                repetition_penalty=self.params.get("repetition_penalty"),
                early_stopping=True,
            )

        # Decode and return only the generated part of the response
        generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]  # Exclude input tokens
        response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        return response

    async def action_generator(self, query: str) -> List[str]:
        try:
            actions = self._generate_response(query)

            # Parse and validate output
            parsed_actions = Format().parse_output(
                actions, self.params.get("max_actions")
            )
            validated_actions = Format().validate_outputs(parsed_actions, query)

            self.logger.info(f"Generated actions: {validated_actions}")

            return validated_actions

        except Exception as e:
            self.logger.error(f"Error in ActionGenerator: {str(e)}")
            return [query]

    async def __call__(self, query: str) -> List[str]:
        return await self.action_generator(query)
