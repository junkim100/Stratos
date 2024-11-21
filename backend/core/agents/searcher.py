from ..base_agent import BaseAgent
from ...config.settings import settings
import torch
from typing import List, Dict
from ...utils.helpers import Format
from .retriever import Retriever


class Searcher(BaseAgent):
    def __init__(self):
        super().__init__(settings)
        self.retriever = Retriever()
        self.NUM_SOURCES = settings.NUM_SOURCES

    def _create_prompt(self, query: str, action: str) -> List[Dict[str, str]]:
        chat_template = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that generates Search Queries for the following request"
                    "Your goal is to create a Search Query to look for sources that can answer the user's query."
                    "Each Search Query should be focused and precise. No preamble/postamble.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    "# Example 1\n"
                    "User Query: What is the world's first AI-generated video game?\n"
                    "Request: Search for information about the world's first AI-generated video game\n"
                    "Max Query: 5\n"
                    "Search Queries:\n"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "1. First AI-generated video game history and background\n"
                    "2. Pioneering AI applications in video game creation\n"
                    "3. First AI-driven video game release date and developers\n"
                    "4. Timeline of AI in gaming and first AI-created game\n"
                    "5. Significance of AI-generated games in gaming history"
                ),
            },
            {
                "role": "user",
                "content": (
                    "# Example 2\n"
                    "Query: How do you delete all save data of Animal Crossing in Nintendo Switch?\n"
                    "Request: Search for instructions on deleting save data of Animal Crossing in Nintendo Switch\n"
                    "Max Query: 3\n"
                    "Search Queries:\n"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "1. How to delete save data for Animal Crossing: New Horizons on Nintendo Switch\n"
                    "2. Nintendo Switch guide to deleting game save data for Animal Crossing\n"
                    "3. Steps to reset Animal Crossing island and delete saved progress"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n"
                    f"Request: {action}\n"
                    f"Max Query: {self.params.get('max_actions')}\n"
                    "Search Queries:\n"
                ),
            },
        ]

        return chat_template

    def _generate_response(self, query: str, action: str) -> str:
        """
        Generate a response using a chat template.
        """
        # Create a chat template from the query
        chat_template = self._create_prompt(query, action)

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

    def _extract_content(self, search_results: List[Dict]) -> List[str]:
        """
        Extract and format content from search results.
        """
        extracted_content = []
        for result in search_results:
            try:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")

                formatted_content = (
                    f"Title: {title}\nSummary: {snippet}\nSource: {link}\n"
                )
                extracted_content.append(formatted_content)
            except Exception as e:
                self.logger.error(f"Error extracting content: {str(e)}")

        return extracted_content

    async def search(self, query: str, action: str) -> List[str]:
        try:
            search_queries = self._generate_response(query, action)

            # Parse and validate output
            parsed_queries = Format().parse_output(
                search_queries, self.params.get("max_queries")
            )
            validated_queries = Format().validate_outputs(parsed_queries, action)

            self.logger.info(f"Generated search queries for \"{action}\":\n{validated_queries}")

            # Perform Google searches for each query using Retriever
            all_results = []
            for search_query in validated_queries:
                clean_query = search_query.strip('"')
                search_results = await self.retriever.search(clean_query, self.NUM_SOURCES)

                if search_results:
                    extracted_content = self._extract_content(search_results)
                    all_results.extend(extracted_content)
                else:
                    self.logger.warning(f"No results found for: {clean_query}")

            if not all_results:
                error_msg = (
                    f"No search results found for any queries. "
                    f"API Key: {'Exists' if self.retriever.GOOGLE_API_KEY else 'Missing'}, "
                    f"CSE ID: {'Exists' if self.retriever.GOOGLE_CSE_ID else 'Missing'}"
                    f"Please {'check if the API key and CSE ID are valid' if self.retriever.GOOGLE_API_KEY and self.retriever.GOOGLE_CSE_ID else 'set up the Google API key and CSE ID in the .env file'}"
                )
                self.logger.error(error_msg)
                return [error_msg]

            return all_results

        except Exception as e:
            self.logger.error(f"Error in Searcher: {str(e)}")
            return [f"Error performing search: {str(e)}"]

    async def __call__(self, query: str, action: str) -> List[str]:
        return await self.search(query, action)
