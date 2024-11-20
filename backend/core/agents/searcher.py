from ..base_agent import BaseAgent
from ...config.settings import settings
import torch
from typing import List, Dict
from ...utils.helpers import Format
import aiohttp
import json


class Searcher(BaseAgent):
    def __init__(self):
        super().__init__(settings)
        self.GOOGLE_API_KEY = settings.GOOGLE_API_KEY
        self.GOOGLE_CSE_ID = settings.GOOGLE_CSE_ID
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

    async def _google_search(self, query: str, num_results: int) -> List[Dict]:
        """
        Perform a Google search using the Custom Search API
        """
        base_url = "https://www.googleapis.com/customsearch/v1"

        # Add error checking for API credentials
        if not self.GOOGLE_API_KEY or not self.GOOGLE_CSE_ID:
            self.logger.error("Missing API credentials")
            return []

        params = {
            "key": self.GOOGLE_API_KEY,
            "cx": self.GOOGLE_CSE_ID,
            "q": query,
            "num": num_results,
            "safe": "off",
        }

        # Log the request details (be careful not to log the full API key)
        self.logger.debug(
            f"Making API request with params: {json.dumps({k: v for k, v in params.items() if k != 'key'}, indent=2)}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    response_text = await response.text()

                    # Log the raw response for debugging
                    self.logger.debug(f"Raw API Response: {response_text}")

                    if response.status == 200:
                        data = json.loads(response_text)
                        if "items" in data:
                            return data["items"]
                        else:
                            self.logger.warning(
                                f"No results in response for query: {query}. "
                                f"Response: {json.dumps(data, indent=2)}"
                            )
                            return []
                    else:
                        self.logger.error(
                            f"Google API error: Status {response.status}, "
                            f"Response: {response_text}"
                        )
                        return []
        except Exception as e:
            self.logger.error(f"Error in Google search: {str(e)}")
            return []

    def _extract_content(self, search_results: List[Dict]) -> List[str]:
        """
        Extract and format content from search results
        """
        extracted_content = []
        for result in search_results:
            try:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")

                formatted_content = (
                    f"Title: {title}\n" f"Summary: {snippet}\n" f"Source: {link}\n"
                )
                self.logger.debug(f"Extracted content: {formatted_content}")
                extracted_content.append(formatted_content)
            except Exception as e:
                self.logger.error(f"Error extracting content: {str(e)}")
                continue

        return extracted_content

    async def search(self, query: str) -> List[str]:
        try:
            # Generate and validate search queries
            generated_text = self._generate_text(query)
            parsed_queries = Format().parse_output(
                generated_text, self.params.get("max_queries")
            )
            validated_queries = Format().validate_outputs(parsed_queries, query)

            for i, sq in enumerate(validated_queries, 1):
                self.logger.info(f"Search query {i}: {sq}")

            # Perform Google searches for each query
            all_results = []
            for search_query in validated_queries:
                # Remove the quotes from the search query
                clean_query = search_query.strip('"')

                search_results = await self._google_search(
                    clean_query, self.NUM_SOURCES
                )

                # Add debug logging
                self.logger.debug(
                    f"Search results for '{clean_query}': {json.dumps(search_results, indent=2)}"
                )

                if search_results:
                    extracted_content = self._extract_content(search_results)
                    all_results.extend(extracted_content)
                else:
                    self.logger.warning(f"No results found for: {clean_query}")

            if not all_results:
                # Return a more specific error message
                error_msg = (
                    f"No search results found for any queries. "
                    f"API Key: {'Valid' if self.GOOGLE_API_KEY else 'Missing'}, "
                    f"CSE ID: {'Valid' if self.GOOGLE_CSE_ID else 'Missing'}"
                )
                self.logger.warning(error_msg)
                return [error_msg]

            return all_results

        except Exception as e:
            self.logger.error(f"Error in Searcher: {str(e)}")
            return [f"Error performing search: {str(e)}"]

    async def __call__(self, query: str) -> List[str]:
        return await self.search(query)
