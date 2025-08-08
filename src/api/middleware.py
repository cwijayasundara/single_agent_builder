"""
Custom middleware for the FastAPI application.
"""
import time
import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            logger.info(
                f"Response {request_id}: {response.status_code} "
                f"in {process_time:.4f}s"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error {request_id}: {str(e)} in {process_time:.4f}s",
                exc_info=True
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors consistently."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            # Get request ID if available
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            # Import here to avoid circular import
            from .utils.exceptions import ConfigurableAgentException
            
            # Handle custom exceptions
            if isinstance(e, ConfigurableAgentException):
                logger.error(f"Agent error {request_id}: {e.message}", exc_info=True)
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "error": e.error_code,
                        "message": e.message,
                        "details": e.details,
                        "request_id": request_id,
                        "timestamp": time.time()
                    }
                )
            
            # Handle HTTP exceptions (from FastAPI/Starlette)
            from fastapi import HTTPException
            if isinstance(e, HTTPException):
                # Check if detail is a dict (our custom error format)
                if isinstance(e.detail, dict):
                    error_detail = e.detail.copy()
                    error_detail["request_id"] = request_id
                    error_detail["timestamp"] = time.time()
                    
                    # Log based on status code severity
                    if e.status_code >= 500:
                        logger.error(f"Server error {request_id}: {error_detail}", exc_info=True)
                    elif e.status_code >= 400:
                        logger.warning(f"Client error {request_id}: {error_detail}")
                    
                    return JSONResponse(
                        status_code=e.status_code,
                        content=error_detail
                    )
                else:
                    # Simple string detail
                    logger.warning(f"HTTP error {request_id}: {e.detail}")
                    return JSONResponse(
                        status_code=e.status_code,
                        content={
                            "error": "HTTP_ERROR",
                            "message": str(e.detail),
                            "request_id": request_id,
                            "timestamp": time.time()
                        }
                    )
            
            # Handle unexpected errors
            logger.error(f"Unhandled error {request_id}: {str(e)}", exc_info=True)
            
            # Return consistent error response for unexpected errors
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": time.time(),
                    "type": type(e).__name__
                }
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add HSTS in production
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response