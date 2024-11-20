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

    def _create_prompt(self, query: str) -> str:
        # TODO: Improve the prompt
        return f"""Today's date is {self.date}\nGenerate specific search queries for the following request.\nEach query should be focused and precise.\nGenerate between {self.params.get('min_queries')} and {self.params.get('max_queries')} queries. No preamble/postamble.\n\nRequest: {query}\nSearch queries:"""

    def _tokenize_input(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.params.get("max_length"),
            padding=True,
        ).to(self.model.device)

    def _generate_text(self, query: str) -> str:
        inputs = self._tokenize_input(self._create_prompt(query))
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
                early_stopping=True,
            )
        search_queries = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return search_queries

    async def search(self, query: str) -> List[str]:
        try:
            # Generate and validate search queries using the language model
            generated_text = self._generate_text(query)
            parsed_queries = Format().parse_output(
                generated_text, self.params.get("max_queries")
            )
            validated_queries = Format().validate_outputs(parsed_queries, query)

            for i, sq in enumerate(validated_queries, 1):
                self.logger.info(f"Search query {i}: {sq}")

            # Perform Google searches for each query using Retriever
            all_results = []
            for search_query in validated_queries:
                clean_query = search_query.strip('"')
                search_results = await self.retriever.search(
                    clean_query, self.NUM_SOURCES
                )

                # Add debug logging
                self.logger.debug(
                    f"Search results for '{clean_query}': {search_results}"
                )

                if search_results:
                    extracted_content = self._extract_content(search_results)
                    all_results.extend(extracted_content)
                else:
                    self.logger.warning(f"No results found for: {clean_query}")

            if not all_results:
                error_msg = (
                    f"No search results found for any queries. "
                    f"API Key: {'Valid' if self.retriever.GOOGLE_API_KEY else 'Missing'}, "
                    f"CSE ID: {'Valid' if self.retriever.GOOGLE_CSE_ID else 'Missing'}"
                )
                self.logger.warning(error_msg)
                return [error_msg]

            return all_results

        except Exception as e:
            self.logger.error(f"Error in Searcher: {str(e)}")
            return [f"Error performing search: {str(e)}"]

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

    async def __call__(self, query: str) -> List[str]:
        return await self.search(query)
