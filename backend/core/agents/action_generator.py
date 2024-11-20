from ..base_agent import BaseAgent
from ...config.settings import settings
import torch
from typing import List
from ...utils.helpers import Format


class ActionGenerator(BaseAgent):
    def __init__(self):
        super().__init__(settings)

    def _create_prompt(self, query: str) -> str:
        # TODO: Refine the action creation logic. Currently, if you set max actions to more than 2, it creates other actions that is not supported.
        return f"""Today's date is {self.date}\nBreak down this User Query into 'Max Action' or less actions required to accurately answer the User Query. Possible actions are only "Search" and "Summarize". No other actions are possible. Each action should be an action plus the information needed to complete the action, and should contain all details from the User Query. The order of the action should be sorted so that step-by-step execution of the action will end up in an accurate answer. However, do not execute the action.\n\n# Example 1 - Max Action: 2\nQuery: What is the world's first AI-generated video game?\nAction:\n1. Search for information about the world's first AI-generated video game\n2. Summarize the findings on the world's first AI-generated video game\n\n# Example 2 - Max Action: 2\nQuery: How do you delete all save data of animal crossing in Nintendo switch?\nAction:\n1. Search for instructions on deleting save data of Animal Crossing in Nintendo Switch\n2. Search user guides or official support pages related to Nintendo Switch save data management\n\n# User Query - Max Action: {self.params.get("max_actions")}\nQuery: {query}\nAction:\n"""

    def _tokenize_input(self, query: str) -> torch.Tensor:
        return self.tokenizer(
            self._create_prompt(query),
            return_tensors="pt",
            truncation=True,
            max_length=self.params.get("max_length"),
            padding=True,
        ).to(self.model.device)

    def _generate_text(self, query: str) -> str:
        inputs = self._tokenize_input(query)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.params.get("max_length"),
                temperature=self.params.get("temperature"),
                num_beams=self.params.get("num_beams"),
                do_sample=self.params.get("do_sample"),
                top_p=self.params.get("top_p"),
                top_k=self.params.get("top_k"),
                repetition_penalty=self.params.get("repetition_penalty"),
                early_stopping=True,
            )

        raw_actions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.debug(f"Raw Actions:\n{raw_actions}")
        return raw_actions

    async def action_generator(self, query: str) -> List[str]:
        try:
            # Generate actions and only keep the response
            raw_actions = self._generate_text(query).split("Action:\n")[
                3:
            ]  # 3 is the index of the first action
            if len(raw_actions) > 1:
                # Combine all the elements in the array into a single string
                actions = " ".join(raw_actions)
            else:
                actions = raw_actions[0]

            # Parse and validate output
            # parsed_actions = self._parse_output(actions)  # Now returns a list
            # validated_actions = self._validate_actions(parsed_actions, query)
            parsed_actions = Format().parse_output(
                actions, self.params.get("max_actions")
            )
            validated_actions = Format().validate_outputs(parsed_actions, query)

            for i, sq in enumerate(validated_actions, 1):
                self.logger.info(f"Action {i}: {sq}")

            return validated_actions

        except Exception as e:
            self.logger.error(f"Error in query decomposition: {str(e)}")
            return [query]

    async def __call__(self, query: str) -> List[str]:
        return await self.action_generator(query)
