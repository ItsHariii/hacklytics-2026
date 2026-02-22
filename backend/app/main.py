"""RespiLens API — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import analyze, recordings, patients, demo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("RespiLens API starting up...")
    logger.info(f"CORS origins: {settings.cors_origin_list}")
    logger.info(f"Max upload size: {settings.MAX_UPLOAD_SIZE_MB}MB")
    yield
    logger.info("RespiLens API shutting down...")


app = FastAPI(
    title="RespiLens API",
    description="AI-powered lung sound analysis backend for the RespiLens Clinical PWA",
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount API Routers ---
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(recordings.router, prefix="/api", tags=["Recordings"])
app.include_router(patients.router, prefix="/api", tags=["Patients"])
app.include_router(demo.router, prefix="/api", tags=["Demo"])


# --- Health Check ---
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "service": "respi-lens-api",
        "version": "0.1.0",
    }


@app.get("/", tags=["System"])
async def root():
    """Root endpoint — redirects to docs."""
    return {
        "message": "RespiLens API",
        "docs": "/docs",
        "health": "/health",
    }
