import asyncio
import logging
from backend.core.query_decomposer import QueryDecomposer
from backend.core.retriever import Retriever
from backend.core.processor import Processor
from backend.core.response_generator import ResponseGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchPipeline:
    def __init__(self):
        self.query_decomposer = QueryDecomposer()
        self.retriever = Retriever()
        self.processor = Processor()
        self.response_generator = ResponseGenerator()

    async def process_query(self, query: str):
        """
        Process a search query through the entire pipeline.
        """
        try:
            # Step 1: Decompose query into sub-queries
            # logger.info(f"\n////////// Decomposing query //////////\n")
            sub_queries = await self.query_decomposer(query)

            # Step 2: Retrieve and process chunks for each sub-query in parallel
            async def retrieve_chunks(sub_query: str, index: int):
                # logger.info(f"\n////////// Processing sub-query {index+1}: {sub_query} //////////\n")
                return await self.retriever(sub_query)

            # Use asyncio.gather to retrieve the sub-queries in parallel and log the sub-query index
            all_chunks_nested = await asyncio.gather(*[retrieve_chunks(sub_query, index) for index, sub_query in enumerate(sub_queries)])

            # Flatten the list of lists into a single flat list of ProcessedChunk objects
            all_chunks = [chunk for sublist in all_chunks_nested for chunk in sublist]

            # Step 3: Process chunks (ensure it's a flat list)
            # logger.info(f"\n////////// Processing {len(all_chunks)} chunks //////////\n")
            processed_chunks = await self.processor(all_chunks)

            # Step 4: Generate response
            # logger.info("\n////////// Generating final response //////////\n")
            response = await self.response_generator(query, processed_chunks)

            return {
                "answer": response.answer,
                "sources": list(set(chunk.source for chunk in processed_chunks))
            }

        except Exception as e:
            logger.error(f"\n////////// Pipeline error: {str(e)} //////////\n")
            raise

# Function to run the pipeline from command line input
async def run_pipeline():
    pipeline = SearchPipeline()

    # Get user input from command line
    query = input("Enter your query: ")

    # Run the pipeline with the provided query
    result = await pipeline.process_query(query)

    # Print the result
    print("\n======== Result ========\n")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())
