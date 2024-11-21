from ..base_agent import BaseAgent
from ...config.settings import settings
import torch
from typing import Dict, Any, List, Tuple


class Responder(BaseAgent):
    def __init__(self):
        super().__init__(settings)

    def _create_prompt(self, query: str, results: Dict[str, Any]) -> List[Dict[str, str]]:
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

            chat_template = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant who's role is to answer the User Query."
                        "Your goal is to generate a concise, comprehensive Response to the User Query, based on the provided sources."
                        "The Response must answer the User Query directly."
                        "Include relevant source citations using [X] format where X is the source number, but do not list the sources after the Response. You are not allowed to rearrange the sources."
                        "No preamble is allowed, only the response to show to the user.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Instructions:\n1. Provide a clear and concise response that directly addresses the query\n2. Include specific details from the sources\n3. Use [X] citations to reference specific information\n4. Maintain a professional tone\n"
                        f"User Query: {query}\n"
                        f"Sources:\n{sources_text}\n"
                        "Response:\n"
                    ),
                },
            ]

            return chat_template

        except Exception as e:
            self.logger.error(f"Error creating prompt: {str(e)}")
            raise

    # TODO: Stream the response generation
    def _generate_response(self, query: str, results: Dict[str, Any]) -> str:
        """
        Generate a response using a chat template.
        """
        # Create a chat template from the query
        chat_template = self._create_prompt(query, results)

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
            response = self._generate_response(query, results)

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