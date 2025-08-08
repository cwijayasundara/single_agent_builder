"""
FastAPI main application for configurable agents system.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from .utils.config import config
from .utils.exceptions import ConfigurableAgentException
from .middleware import LoggingMiddleware, ErrorHandlingMiddleware
from .routes import health_router, agents_router
# TODO: Import other routers when implemented
# from .routes import evaluation_router, templates_router, analytics_router, files_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("🚀 Starting Configurable Agents API...")
    
    # Startup logic
    app.state.config = config
    
    # Initialize database connections if needed
    try:
        # Database initialization would go here
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    
    yield
    
    # Shutdown logic
    logger.info("🛑 Shutting down Configurable Agents API...")

# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Creates a FastAPI application with comprehensive configuration including:
    - CORS middleware for cross-origin requests
    - Rate limiting for API protection
    - Custom error handling and logging
    - Security headers and trusted host validation
    - OpenAPI documentation with security schemes
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    
    app = FastAPI(
        title="Configurable Agents API",
        description="""# Configurable Agents API
        
        A comprehensive RESTful API for managing and deploying configurable AI agents. 
        This API supports multiple LLM providers (OpenAI, Anthropic, Google, Groq) 
        and provides full lifecycle management for AI agents.
        
        ## Features
        
        - **Agent Management**: Create, read, update, and delete AI agents
        - **Multi-Provider Support**: OpenAI, Anthropic, Google, Groq
        - **Real-time Execution**: Synchronous and WebSocket streaming
        - **Memory Systems**: Semantic, episodic, and procedural memory
        - **Tool Integration**: Built-in and custom tool support
        - **Evaluation Framework**: Automated agent performance testing
        - **Configuration Templates**: Reusable agent configurations
        
        ## Authentication
        
        This API supports multiple authentication methods:
        - Bearer Token (JWT)
        - API Key (X-API-Key header)
        
        ## Rate Limits
        
        Default rate limits are applied to all endpoints to ensure fair usage.
        
        ## Cloud Deployment
        
        This API is designed for cloud deployment and includes:
        - Health checks for container orchestration
        - Structured logging for observability
        - Error handling with proper HTTP status codes
        - CORS configuration for web frontends
        """,
        version="1.0.0",
        contact={
            "name": "Configurable Agents API",
            "email": "support@example.com",
        },
        license_info={
            "name": "MIT",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://your-api-domain.com",
                "description": "Production server"
            }
        ]
    )
    
    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for security
    # Add trusted host middleware for security (configured for cloud deployment)
    allowed_hosts = ["localhost", "127.0.0.1", "*.localhost"]
    
    # Add cloud-specific hosts if environment variables are set
    cloud_host = os.getenv("ALLOWED_HOSTS")
    if cloud_host:
        allowed_hosts.extend([host.strip() for host in cloud_host.split(",")])
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )
    
    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Include routers
    app.include_router(health_router, prefix="/api")
    app.include_router(agents_router, prefix="/api")
    # TODO: Include other routers when implemented
    # app.include_router(evaluation_router, prefix="/api")
    # app.include_router(templates_router, prefix="/api")
    # app.include_router(analytics_router, prefix="/api")
    # app.include_router(files_router, prefix="/api")
    
    # Note: Health check is now handled by health_router
    
    # Root endpoint
    @app.get("/", tags=["root"], summary="API Information", description="Get basic API information and available endpoints")
    async def root():
        """Root endpoint with API information.
        
        Returns basic information about the API including:
        - API name and version
        - Documentation URLs
        - Health check endpoint
        - Available features
        
        Returns:
            dict: API information object
        """
        return {
            "message": "Configurable Agents API",
            "version": "1.0.0",
            "description": "RESTful API for managing configurable AI agents",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "health_url": "/api/health",
            "features": [
                "Multi-provider LLM support",
                "Agent lifecycle management", 
                "Real-time execution",
                "Memory systems",
                "Tool integration",
                "Evaluation framework"
            ],
            "supported_providers": ["openai", "anthropic", "google", "groq"]
        }
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="Configurable Agents API",
            version="1.0.0",
            description="Comprehensive RESTful API for managing and deploying configurable AI agents with multi-provider support",
            routes=app.routes,
        )
        
        # Add additional OpenAPI metadata
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }
        
        # Add tags metadata for better organization
        openapi_schema["tags"] = [
            {
                "name": "root",
                "description": "Root API endpoints and information"
            },
            {
                "name": "health", 
                "description": "Health check and system status endpoints"
            },
            {
                "name": "agents",
                "description": "Agent management operations (CRUD, execution, status)"
            },
            {
                "name": "evaluation",
                "description": "Agent evaluation and performance testing"
            },
            {
                "name": "templates",
                "description": "Configuration template management"
            },
            {
                "name": "analytics",
                "description": "Usage analytics and metrics"
            },
            {
                "name": "files",
                "description": "File upload and management"
            }
        ]
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT Bearer token authentication"
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key authentication via X-API-Key header"
            }
        }
        
        # Add example responses
        openapi_schema["components"]["examples"] = {
            "AgentCreateExample": {
                "summary": "Basic research agent",
                "description": "Example configuration for a research agent with web search capabilities",
                "value": {
                    "name": "Research Assistant",
                    "description": "AI agent specialized in research and information gathering",
                    "version": "1.0.0",
                    "llm": {
                        "provider": "openai",
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    "prompts": {
                        "system_prompt": "You are a helpful research assistant that provides accurate and well-sourced information.",
                        "variables": {}
                    },
                    "tools": ["web_search", "calculator"],
                    "memory": [
                        {
                            "type": "semantic",
                            "enabled": True
                        }
                    ],
                    "evaluation": {
                        "enabled": True,
                        "evaluators": [
                            {
                                "name": "correctness",
                                "type": "llm_as_judge",
                                "parameters": {}
                            }
                        ],
                        "auto_evaluate": False
                    },
                    "react": {
                        "max_iterations": 10,
                        "recursion_limit": 25
                    },
                    "debug_mode": False,
                    "tags": ["research", "assistant"]
                }
            },
            "AgentRunExample": {
                "summary": "Simple query execution",
                "description": "Example of running an agent with a simple query",
                "value": {
                    "query": "What are the latest developments in artificial intelligence for 2024?",
                    "context": {},
                    "stream": False,
                    "timeout": 300,
                    "include_evaluation": False
                }
            }
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Global exception handler
    @app.exception_handler(ConfigurableAgentException)
    async def configurable_agent_exception_handler(request: Request, exc: ConfigurableAgentException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc) if os.getenv("DEBUG") else None
            }
        )
    
    return app

# Create app instance
app = create_app()

# Development server runner
def run_dev_server():
    """Run development server."""
    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    run_dev_server()