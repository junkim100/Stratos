from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from ..core.retriever import Retriever
from ..core.processor import Processor
from ..core.generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Search API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = Retriever()
processor = Processor()
generator = Generator()


@app.get("/")
async def root():
    return {"status": "online", "message": "RAG Search API is running"}


@app.post("/api/search")
async def search(request: Request):
    try:
        # Log the incoming request
        body = await request.json()
        logger.info(f"Received request body: {body}")

        # Validate request
        if not isinstance(body, dict) or "query" not in body:
            raise HTTPException(
                status_code=400,
                detail="Invalid request format. Expected {'query': 'your search query'}",
            )

        query = body["query"]
        logger.info(f"Processing search query: {query}")

        # Process the search
        chunks = retriever.retrieve(query)
        processed_chunks = await processor.process(chunks)
        response = await generator.generate(query, processed_chunks)

        return response

    except Exception as e:
        logger.error(f"Error processing search query: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
