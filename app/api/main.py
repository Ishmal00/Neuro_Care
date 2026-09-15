"""NeuroCare-AI FastAPI Application Entrypoint.

Provides clinical REST API endpoints for neurological diagnostics,
system health telemetry, and clinical RAG retrieval.
"""

from datetime import datetime
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Application Metadata
APP_TITLE = "NeuroCare-AI API"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = (
    "Production-grade Clinical AI Backend for Neurological Care, "
    "Diagnostic Intelligence, and Medical RAG Knowledge Retrieval."
)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to trusted origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemHealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    app_name: str = Field(..., example="NeuroCare-AI")
    version: str = Field(..., example="0.1.0")
    timestamp: str = Field(..., example="2026-09-14T00:00:00Z")
    environment: str = Field(..., example="development")
    services: Dict[str, str] = Field(default_factory=dict)


class RootResponse(BaseModel):
    message: str = Field(..., example="Hello NeuroCare")
    version: str = Field(..., example="0.1.0")
    status: str = Field(..., example="online")
    docs_url: str = Field(..., example="/docs")


@app.get("/", response_model=RootResponse, tags=["General"])
async def root() -> Dict[str, Any]:
    """Root entrypoint for the NeuroCare-AI API.

    Returns greeting and status.
    """
    return {
        "message": "Hello NeuroCare",
        "version": APP_VERSION,
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health", response_model=SystemHealthResponse, tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Production health check endpoint verifying core service connectivity."""
    env = os.getenv("APP_ENV", "development")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    return {
        "status": "healthy",
        "app_name": "NeuroCare-AI",
        "version": APP_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": env,
        "services": {
            "api": "online",
            "qdrant_target": qdrant_host,
            "mlflow_target": mlflow_uri,
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("app.api.main:app", host=host, port=port, reload=True)
