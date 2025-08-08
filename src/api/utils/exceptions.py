"""
Custom exceptions for the API.
"""
from typing import Optional, Dict, Any


class ConfigurableAgentException(Exception):
    """Base exception for configurable agent API."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500, 
        error_code: str = "AGENT_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class AgentNotFoundException(ConfigurableAgentException):
    """Agent not found exception."""
    
    def __init__(self, agent_id: str):
        super().__init__(
            message=f"Agent with ID '{agent_id}' not found",
            status_code=404,
            error_code="AGENT_NOT_FOUND",
            details={"agent_id": agent_id}
        )


class AgentConfigurationError(ConfigurableAgentException):
    """Agent configuration error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Agent configuration error: {message}",
            status_code=400,
            error_code="AGENT_CONFIG_ERROR",
            details=details
        )


class AgentExecutionError(ConfigurableAgentException):
    """Agent execution error."""
    
    def __init__(self, message: str, agent_id: Optional[str] = None):
        super().__init__(
            message=f"Agent execution failed: {message}",
            status_code=500,
            error_code="AGENT_EXECUTION_ERROR",
            details={"agent_id": agent_id} if agent_id else {}
        )


class TeamNotFoundException(ConfigurableAgentException):
    """Team not found exception."""
    
    def __init__(self, team_id: str):
        super().__init__(
            message=f"Team with ID '{team_id}' not found",
            status_code=404,
            error_code="TEAM_NOT_FOUND",
            details={"team_id": team_id}
        )


class TeamConfigurationError(ConfigurableAgentException):
    """Team configuration error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Team configuration error: {message}",
            status_code=400,
            error_code="TEAM_CONFIG_ERROR",
            details=details
        )


class EvaluationError(ConfigurableAgentException):
    """Evaluation error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Evaluation error: {message}",
            status_code=500,
            error_code="EVALUATION_ERROR",
            details=details
        )


class AuthenticationError(ConfigurableAgentException):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(ConfigurableAgentException):
    """Authorization error."""
    
    def __init__(self, message: str = "Access forbidden"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR"
        )


class ValidationError(ConfigurableAgentException):
    """Request validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=f"Validation error: {message}",
            status_code=422,
            error_code="VALIDATION_ERROR",
            details={"field": field} if field else {}
        )


class RateLimitError(ConfigurableAgentException):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_ERROR"
        )


class FileUploadError(ConfigurableAgentException):
    """File upload error."""
    
    def __init__(self, message: str, filename: Optional[str] = None):
        super().__init__(
            message=f"File upload error: {message}",
            status_code=400,
            error_code="FILE_UPLOAD_ERROR",
            details={"filename": filename} if filename else {}
        )