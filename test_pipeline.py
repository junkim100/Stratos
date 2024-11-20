import asyncio
import logging
from backend.config.settings import settings
from backend.core.agents.action_decider import ActionDecider
from backend.core.agents.action_generator import ActionGenerator
from backend.core.agents.action_executor import ActionExecutor
from backend.core.agents.responder import Responder
import time

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


class SearchPipeline:
    def __init__(self):
        self.action_decider = ActionDecider()
        self.action_generator = ActionGenerator()
        self.action_executor = ActionExecutor()
        self.responder = Responder()

    async def process_query(self, query: str):
        logger.info(f"\n================ Pipeline Start ================\n")
        action_decision = await self.action_decider(query)
        if action_decision:
            actions = await self.action_generator(query)
            results = await self.action_executor(actions)
            response = await self.responder(query, results)
            return response
        else:
            raise NotImplementedError("responder can't recieve just a query")
            response = await self.responder(query)
            return response


# Function to run the pipeline from command line input
async def run_pipeline():
    pipeline = SearchPipeline()

    # Get user input from command line
    # query = input("Enter your query: ")
    # query = "Pour over recipe for an Ethiopian light roast coffee"
    # query = "World news today"
    query = "How are the reviews for Wicked"

    # Run the pipeline with the provided query
    start = time.time()
    try:
        response = await pipeline.process_query(query)
    except Exception as e:
        logger.error(f"\n================ Pipeline Error: {str(e)} ================\n")
        return

    # Print the response
    print("\n================ Answer ================\n")
    print(response["answer"])
    print("\n================ Sources ================\n")
    for source in response["sources"]:
        print(source)

    end = time.time()
    print(f"\n\nTime taken: {end - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(run_pipeline())
