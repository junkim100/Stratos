from ..base_agent import BaseAgent
from ...config.settings import settings
import torch
from typing import Dict, Any, List, Tuple


class Responder(BaseAgent):
    def __init__(self):
        super().__init__(settings)

    def _create_prompt(self, query: str, results: Dict[str, Any]) -> str:
        """Create a prompt with properly formatted sources and summaries."""
        try:
            # Format each news source and its content
            formatted_sources = []
            for i, content in enumerate(results["content"], 1):
                formatted_sources.append(
                    f"Source {i}:\n"
                    f"Title: {content['title']}\n"
                    f"Content: {content['text']}\n"
                    f"URL: {content['source']}\n"
                )

            sources_text = "\n".join(formatted_sources)

            # TODO: Use prompt template
            return f"""Today's date is {self.date}\nYou are a helpful assistant who's role is to answer the User Query. Based on the provided sources, generate a concise, comprehensive Response to the following User Query. The Response must answer the User Query directly.\nInclude relevant source citations using [X] format where X is the source number, but do not list the sources after the Response. You are not allowed to rearrange the sources. No preamble is allowed, only the response to show to the user.\n\nUser Query: {query}\n\nAvailable Sources:\n{sources_text}\n\nInstructions:\n1. Provide a clear and concise response that directly addresses the query\n2. Include specific details from the sources\n3. Use [X] citations to reference specific information\n4. Maintain a professional tone\n\nResponse:"""

        except Exception as e:
            self.logger.error(f"Error creating prompt: {str(e)}")
            raise

    def _tokenize_input(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.params.get("max_length"),
            padding=True,
        ).to(self.model.device)

    # TODO: Stream the response generation
    def _generate_text(self, prompt: str) -> str:
        inputs = self._tokenize_input(prompt)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=self.params.get("max_length"),
                temperature=self.params.get("temperature"),
                num_beams=self.params.get("num_beams"),
                do_sample=self.params.get("do_sample"),
                top_p=self.params.get("top_p"),
                top_k=self.params.get("top_k"),
                repetition_penalty=self.params.get("repetition_penalty", 1.2),
                early_stopping=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _format_sources(self, results: Dict[str, Any]) -> List[str]:
        """Format sources into a numbered list for references."""
        formatted_sources = []
        for i, content in enumerate(results["content"], 1):
            formatted_sources.append(f"{i}. {content['title']} ({content['source']})")
        return formatted_sources

    def _validate_response(
        self, response: str, sources: List[str]
    ) -> Tuple[str, List[str]]:
        """Validate and clean up the response and sources."""
        if not response:
            raise ValueError("Generated response is empty")

        # Ensure all source citations are valid
        max_source_num = len(sources)
        response_parts = response.split("[")
        cleaned_parts = []

        # Handle the first part (before any citations)
        if response_parts:
            cleaned_parts.append(response_parts[0])

        # Process parts with citations
        for part in response_parts[1:]:  # Skip the first part as it's already handled
            if "]" in part:
                try:
                    citation, text = part.split("]", 1)
                    try:
                        citation_num = int(citation)
                        if 1 <= citation_num <= max_source_num:
                            cleaned_parts.append(f"[{citation}]{text}")
                        else:
                            cleaned_parts.append(text)
                    except ValueError:
                        cleaned_parts.append(text)
                except ValueError:
                    cleaned_parts.append(part)
            else:
                cleaned_parts.append(part)

        cleaned_response = "".join(cleaned_parts)

        # Find all valid citations in the response
        used_sources = set()
        for i in range(1, max_source_num + 1):
            if f"[{i}]" in cleaned_response:
                used_sources.add(i - 1)  # Convert to 0-based index

        # Get the corresponding source entries
        filtered_sources = [sources[i] for i in sorted(used_sources)]

        return cleaned_response.strip(), filtered_sources

    async def generate_response(
        self, query: str, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate the final response with sources."""
        try:
            # Validate input
            if not results.get("content"):
                return {
                    "answer": "No news sources available to provide a response.",
                    "sources": [],
                }

            # Generate the initial response
            prompt = self._create_prompt(query, results)
            response = self._generate_text(prompt)

            # Extract just the response part
            if "\n\nResponse:" in response:
                response = response.split("\n\nResponse:")[1]

            # Format and validate sources
            formatted_sources = self._format_sources(results)
            cleaned_response, used_sources = self._validate_response(
                response, formatted_sources
            )

            return {"answer": cleaned_response, "sources": used_sources}

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while generating the response. Please try again.",
                "sources": [],
            }

    async def __call__(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for generate_response method."""
        return await self.generate_response(query, results)

    # class Responder(BaseAgent):
    #     def __init__(self):
    #         super().__init__(settings)

    #     def _create_prompt(self, query: str, results: Dict[str, Any]) -> str:
    #         """Create a prompt with properly formatted sources and summaries."""
    #         try:
    #             # Debug log the results structure
    #             self.logger.debug(f"Results: {results}")

    #             # Format sources with their summaries
    #             formatted_sources = []
    #             for i, summary in enumerate(results.get("summaries", []), 1):
    #                 formatted_sources.append(
    #                     f"Source {i}:\n"
    #                     f"Content: {summary['text']}\n"
    #                     f"URL: {summary['source']}\n"
    #                 )

    #             sources_text = "\n".join(formatted_sources)
    #             final_summary = results.get("final_summary", "No final summary available.")

    #             return f"""Today's date is {self.date}\nUsing the provided information, answer the following query. Cite sources using [X] where X is the source number.\nIf specific information comes from multiple sources, cite all relevant sources.\nIf the information isn't found in the sources, don't include it.\nMaintain a professional and informative tone.\n\nQuery: {query}\nSources:\n{sources_text}\n\nSource Summary:\n{results['final_summary']}\n\nGenerate a comprehensive answer that:\n1. Directly addresses the query\n2. Uses information from the sources\n3. Cites sources appropriately using [X] format\n4. Maintains a logical flow\n5. Includes specific details and examples when available\n\nAnswer:"""

    #         except Exception as e:
    #             self.logger.error(f"Error creating prompt: {str(e)}")
    #             raise

    #     def _tokenize_input(self, prompt: str) -> torch.Tensor:
    #         return self.tokenizer(
    #             prompt,
    #             return_tensors="pt",
    #             truncation=True,
    #             max_length=self.params.get("max_length"),
    #             padding=True,
    #         ).to(self.model.device)

    #     def _generate_text(self, prompt: str) -> str:
    #         inputs = self._tokenize_input(prompt)
    #         with torch.no_grad():
    #             outputs = self.model.generate(
    #                 inputs.input_ids,
    #                 attention_mask=inputs.attention_mask,
    #                 pad_token_id=self.tokenizer.pad_token_id,
    #                 max_new_tokens=self.params.get("max_length"),
    #                 temperature=self.params.get("temperature"),
    #                 num_beams=self.params.get("num_beams"),
    #                 do_sample=self.params.get("do_sample"),
    #                 top_p=self.params.get("top_p"),
    #                 top_k=self.params.get("top_k"),
    #                 repetition_penalty=self.params.get("repetition_penalty", 1.2),
    #             )
    #         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    #     def _format_sources(self, results: Dict[str, Any]) -> List[str]:
    #         """Format sources into a numbered list for references."""
    #         formatted_sources = []
    #         for i, summary in enumerate(results.get("summaries", []), 1):
    #             # Extract domain from URL for a cleaner title
    #             url = summary["source"]
    #             domain = url.split("/")[2]  # Get domain from URL
    #             formatted_sources.append(f"{i}. {domain} ({url})")
    #         return formatted_sources

    #     def _validate_response(
    #         self, response: str, sources: List[str]
    #     ) -> Tuple[str, List[str]]:
    #         """Validate and clean up the response and sources."""
    #         if not response:
    #             raise ValueError("Generated response is empty")

    #         # Ensure all source citations are valid
    #         max_source_num = len(sources)
    #         response_parts = response.split("[")
    #         cleaned_parts = []

    #         for part in response_parts:
    #             if "]" in part:
    #                 try:
    #                     citation, text = part.split("]", 1)
    #                     citation_num = int(citation)
    #                     if 1 <= citation_num <= max_source_num:
    #                         cleaned_parts.append(f"[{citation}]{text}")
    #                     else:
    #                         cleaned_parts.append(text)
    #                 except (ValueError, IndexError):
    #                     cleaned_parts.append(part)
    #             else:
    #                 cleaned_parts.append(part)

    #         cleaned_response = "".join(cleaned_parts)

    #         # Remove any sources that aren't cited in the response
    #         used_sources = set()
    #         for i in range(1, max_source_num + 1):
    #             if f"[{i}]" in cleaned_response:
    #                 used_sources.add(i - 1)  # Convert to 0-based index

    #         filtered_sources = [s for i, s in enumerate(sources) if i in used_sources]

    #         return cleaned_response.strip(), filtered_sources

    #     async def generate_response(
    #         self, query: str, results: Dict[str, Any]
    #     ) -> Dict[str, Any]:
    #         """Generate the final response with sources."""
    #         try:
    #             # Validate results structure
    #             if not results.get("summaries"):
    #                 return {
    #                     "answer": "No information available to answer the query.",
    #                     "sources": [],
    #                 }

    #             # Generate the initial response
    #             prompt = self._create_prompt(query, results)
    #             response = self._generate_text(prompt)

    #             print(f"\n\nresponse:\n{response}\n\n")

    #             test0 = response.split("\n\nAnswer:")[0]
    #             test1 = response.split("\n\nAnswer:")[1]

    #             print(f"\n\ntest0:\n{test0}\n\n")
    #             print(f"\n\ntest1:\n{test1}\n\n")

    #             # Extract just the response part
    #             if "\n\nAnswer:" in response:
    #                 response = response.split("\n\nAnswer:")[1]

    #             # Format sources
    #             formatted_sources = self._format_sources(results)

    #             # Validate and clean up the response and sources
    #             cleaned_response, used_sources = self._validate_response(
    #                 response, formatted_sources
    #             )

    #             return {"answer": cleaned_response, "sources": used_sources}

    #         except Exception as e:
    #             self.logger.error(f"Error generating response: {str(e)}")
    #             return {
    #                 "answer": "I apologize, but I encountered an error while generating the response. Please try again.",
    #                 "sources": [],
    #             }

    # async def __call__(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
    #     """Wrapper for generate_response method."""
    #     return await self.generate_response(query, results)
