from ..base_agent import BaseAgent
from ...config.settings import settings
import torch
from typing import List

# uses a  model to decide if the query should be devided into actions or just used by itself for final response generation
# if query is simple enough, it will be used as is
# if query is complex, it will be broken down into actions
class ActionDecider(BaseAgent):
    def __init__(self):
        super().__init__(settings)

    def _initialize_model(self):
        return None, None

    def _create_prompt(self, text: str) -> str:
        pass

    async def _action_decider(self, query: str) -> List[str]:
        # use a classifier model to decide if the query should be broken down into actions
        inputs = self._tokenize_input(self._create_prompt(query), self.params.get("max_length"))
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
            )

        decision = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.logger.debug(f"Action decision: {decision}")
        return decision

    async def __call__(self, query: str) -> List[str]:
        # return await self.action_decider(query)
        return True
