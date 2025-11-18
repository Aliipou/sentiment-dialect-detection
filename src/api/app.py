"""
FastAPI application for sentiment analysis and dialect detection
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from .schemas import (
    TextInput,
    BatchTextInput,
    SentimentResponse,
    DialectResponse,
    CombinedResponse,
    BatchResponse,
    HealthResponse,
    ErrorResponse
)
from .service import analysis_service
from ..utils.config import settings
from ..utils.logger import setup_logger
from .. import __version__

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multilingual Sentiment Analysis & Persian Dialect Detection API",
    description="API for analyzing sentiment in multiple languages and detecting Persian dialects",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {process_time:.3f}s"
    )

    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting API server...")
    try:
        await analysis_service.initialize()
        logger.info("API server started successfully")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server...")


# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multilingual Sentiment Analysis & Persian Dialect Detection API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "sentiment": {
                "path": "/sentiment",
                "method": "POST",
                "description": "Analyze sentiment of text"
            },
            "dialect": {
                "path": "/dialect",
                "method": "POST",
                "description": "Detect Persian dialect"
            },
            "analyze": {
                "path": "/analyze",
                "method": "POST",
                "description": "Combined sentiment and dialect analysis"
            },
            "batch": {
                "path": "/batch",
                "method": "POST",
                "description": "Batch analysis of multiple texts"
            }
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_status = analysis_service.get_models_status()

    return {
        "status": "healthy",
        "version": __version__,
        "models_loaded": models_status
    }


@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """
    Analyze sentiment of text

    - **text**: Text to analyze (required)
    - **language**: Language code (default: fa)
    """
    try:
        result = await analysis_service.analyze_sentiment(
            text=input_data.text,
            language=input_data.language
        )
        return result
    except Exception as e:
        logger.error(f"Error in sentiment endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialect", response_model=DialectResponse)
async def analyze_dialect(input_data: TextInput):
    """
    Detect Persian dialect

    - **text**: Persian text to analyze (required)
    - **language**: Must be 'fa' for dialect detection
    """
    if input_data.language != "fa":
        raise HTTPException(
            status_code=400,
            detail="Dialect detection is only supported for Persian (fa) language"
        )

    try:
        result = await analysis_service.analyze_dialect(text=input_data.text)
        return result
    except Exception as e:
        logger.error(f"Error in dialect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=CombinedResponse)
async def analyze_combined(input_data: TextInput):
    """
    Combined sentiment and dialect analysis

    - **text**: Text to analyze (required)
    - **language**: Language code (default: fa)

    Note: Dialect detection is only available for Persian texts
    """
    try:
        result = await analysis_service.analyze_combined(
            text=input_data.text,
            language=input_data.language
        )
        return result
    except Exception as e:
        logger.error(f"Error in combined analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse)
async def analyze_batch(input_data: BatchTextInput):
    """
    Batch analysis of multiple texts

    - **texts**: List of texts to analyze (1-100 texts)
    - **language**: Language code (default: fa)

    Processes multiple texts and returns combined results
    """
    if len(input_data.texts) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {settings.MAX_BATCH_SIZE} texts"
        )

    try:
        result = await analysis_service.analyze_batch(
            texts=input_data.texts,
            language=input_data.language
        )
        return result
    except Exception as e:
        logger.error(f"Error in batch endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )
