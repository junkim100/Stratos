from ..base_agent import BaseAgent
from ...config.settings import settings
from typing import List, Dict, Any
from ..agents.searcher import Searcher
from ..agents.summarizer import Summarizer


class ActionExecutor(BaseAgent):
    def __init__(self):
        super().__init__(settings)
        self.searcher = Searcher()
        self.summarizer = Summarizer()

    def determine_action(self, actions: List[str]) -> Dict[str, str]:
        """
        Determine the type of action for each input action string.
        Returns a dictionary mapping action strings to their types.
        """
        decisions = {}

        for action in actions:
            action_lower = action.lower()
            if action_lower.startswith("search"):
                decisions[action] = "search"
            elif action_lower.startswith("summarize"):
                decisions[action] = "summarize"
            elif "search" in action_lower and "summarize" not in action_lower:
                decisions[action] = "search"
            elif "summarize" in action_lower and "search" not in action_lower:
                decisions[action] = "summarize"
            else:
                self.logger.warning(f"Unable to determine action type for: {action}")
                decisions[action] = "unknown"

            self.logger.debug(f"Action determined - {action} -> {decisions[action]}")

        return decisions

    def validate_actions(self, decisions: Dict[str, str]) -> bool:
        """
        Validate if the combination of actions is executable.
        Rules:
        1. At least one search and one summarize action must exist
        2. Search must come before summarize
        """
        # Count action types
        action_counts = {
            "search": sum(
                1 for action_type in decisions.values() if action_type == "search"
            ),
            "summarize": sum(
                1 for action_type in decisions.values() if action_type == "summarize"
            ),
        }

        # Check if both action types exist
        if action_counts["search"] == 0 or action_counts["summarize"] == 0:
            self.logger.error(
                "Actions must include at least one search and one summarize"
            )
            return False

        # Check order of actions
        found_search = False
        for action_type in decisions.values():
            if action_type == "search":
                found_search = True
            elif action_type == "summarize" and not found_search:
                self.logger.error("Summarize action cannot come before search action")
                return False

        return True

    async def execute_action(self, actions: List[str]) -> Dict[str, Any]:
        """
        Execute actions and format results for the Responder.
        Returns a dictionary with properly formatted search results and summaries.
        """
        try:
            # Determine and validate action types
            decisions = self.determine_action(actions)
            if not self.validate_actions(decisions):
                raise ValueError("Invalid action combination")

            # Initialize results structure
            results = {
                "sources": [],  # List of source information
                "content": [],  # List of content from each source
                "summaries": [],  # List of individual summaries
                "final_summary": None,  # Final combined summary
            }

            # Execute all search actions first
            search_actions = [
                (action, type_)
                for action, type_ in decisions.items()
                if type_ == "search"
            ]
            for search_action, _ in search_actions:
                self.logger.info(f"Executing search: {search_action}")
                search_results = await self.searcher(search_action)

                if not search_results:
                    continue

                # Process each search result
                for result in search_results:
                    try:
                        # Parse the result into components
                        lines = result.strip().split("\n")
                        title = lines[0].replace("Title: ", "")
                        summary = lines[1].replace("Summary: ", "")
                        source = lines[2].replace("Source: ", "")

                        # Add to results structure
                        results["sources"].append({"title": title, "url": source})
                        results["content"].append(
                            {"title": title, "text": summary, "source": source}
                        )
                    except Exception as e:
                        self.logger.error(f"Error parsing search result: {str(e)}")
                        continue

            # Check if we have any content to process
            if not results["content"]:
                self.logger.error("No valid search results found")
                return results

            # Generate individual summaries
            for content in results["content"]:
                try:
                    summary = await self.summarizer(
                        content["text"], summary_type="individual"
                    )
                    results["summaries"].append(
                        {"text": summary, "source": content["source"]}
                    )
                except Exception as e:
                    self.logger.error(f"Error generating summary: {str(e)}")
                    continue

            # Generate final summary if we have individual summaries
            if results["summaries"]:
                try:
                    combined_text = "\n".join(
                        [
                            f"{summary['text']} (Source: {summary['source']})"
                            for summary in results["summaries"]
                        ]
                    )
                    results["final_summary"] = await self.summarizer(
                        combined_text, summary_type="final"
                    )
                except Exception as e:
                    self.logger.error(f"Error generating final summary: {str(e)}")

            return results

        except Exception as e:
            self.logger.error(f"Error in action execution: {str(e)}")
            raise

    async def __call__(self, actions: List[str]) -> Dict[str, Any]:
        """
        Wrapper for execute_action method.
        """
        return await self.execute_action(actions)
