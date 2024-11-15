from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import logging
from pydantic import BaseModel

from ..core.query_decomposer import QueryDecomposer
from ..core.retriever import Retriever
from ..core.processor import Processor
from ..core.response_generator import ResponseGenerator
from ..models.schema import ProcessedChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Stratos API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    answer: str
    sources: List[str]

# Initialize pipeline components
class SearchPipeline:
    def __init__(self):
        self.query_decomposer = QueryDecomposer()
        self.retriever = Retriever()
        self.processor = Processor()
        self.response_generator = ResponseGenerator()

    async def process_query(self, query: str) -> SearchResponse:
        """
        Process a search query through the entire pipeline.
        """
        try:
            # Step 1: Decompose query into sub-queries
            logger.info(f"\n////////// Decomposing query //////////\n")
            sub_queries = await self.query_decomposer(query)

            # Step 2: Retrieve and process chunks for each sub-query
            all_chunks: List[ProcessedChunk] = []
            # for sub_query in sub_queries:
            #     logger.info(f"\n////////// Processing sub-query: {sub_query} //////////\n")
            #     chunks = await self.retriever(sub_query)
            #     all_chunks.extend(chunks)

            # retrieve the sub-queries in parallel and log the sub-query index
            async def retrieve_chunks(sub_query: str, index: int) -> List[ProcessedChunk]:
                logger.info(f"\n////////// Processing sub-query {index}: {sub_query} //////////\n")
                return await self.retriever(sub_query)

            all_chunks = await asyncio.gather(*[retrieve_chunks(sub_query, index) for index, sub_query in enumerate(sub_queries)])

            # Step 3: Process chunks
            logger.info(f"\n////////// Processing {len(all_chunks)} chunks //////////\n")
            processed_chunks = await self.processor(all_chunks)

            # Step 4: Generate response
            logger.info("\n////////// Generating final response //////////\n")
            response = await self.response_generator(query, processed_chunks)

            return SearchResponse(
                answer=response.answer,
                sources=list(set(chunk.source for chunk in processed_chunks))
            )

        except Exception as e:
            logger.error(f"\n////////// Pipeline error: {str(e)} //////////\n")
            raise HTTPException(
                status_code=500,
                detail=f"Search pipeline error: {str(e)}"
            )

# Initialize pipeline
pipeline = SearchPipeline()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Stratos API is running"
    }

@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Process a search query and return the response.
    """
    try:
        logger.info(f"\n////////// Received search request: {request.query} //////////\n")

        # Process the query through the pipeline
        response = await pipeline.process_query(request.query)

        logger.info("\n////////// Search request completed successfully //////////\n")
        return response

    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        # Log unexpected errors and return 500
        logger.error(f"\n////////// Unexpected error in search endpoint: {str(e)} //////////\n")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"\n////////// Global exception handler caught: {str(exc)} //////////\n")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred",
            "type": type(exc).__name__
        }
    )