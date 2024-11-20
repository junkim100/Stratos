import aiohttp
import os
from typing import List, Dict


class Retriever:
    def __init__(self):
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

        if not self.GOOGLE_API_KEY or not self.GOOGLE_CSE_ID:
            raise ValueError(
                "Google API key or CSE ID not found in environment variables"
            )

    async def _google_search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform a Google search using the Custom Search API
        """
        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.GOOGLE_API_KEY,
            "cx": self.GOOGLE_CSE_ID,
            "q": query,
            "num": num_results,
            "safe": "off",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "items" in data:
                            return data["items"]
                        else:
                            return []
                    else:
                        return []
        except Exception as e:
            return []

    async def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Perform a search and return the results
        """
        return await self._google_search(query, num_results)
