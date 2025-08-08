"""
Centralized logging configuration for the configurable agents system.
"""
import os
import sys
import logging
import logging.config
from typing import Optional, Dict, Any
from pathlib import Path
from enum import Enum

from .formatters import JSONFormatter, StructuredFormatter
from .correlation import correlation_filter


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggerConfig:
    """Logger configuration class."""
    
    def __init__(self):
        self.level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.format_type = os.getenv("LOG_FORMAT", "structured")  # structured or json
        self.enable_correlation = os.getenv("LOG_CORRELATION", "true").lower() == "true"
        self.log_file = os.getenv("LOG_FILE", None)
        self.log_dir = os.getenv("LOG_DIR", "logs")
        self.max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
        self.backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        self.enable_console = os.getenv("LOG_CONSOLE", "true").lower() == "true"
        
    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return self.level == "DEBUG"
    
    def is_json_format(self) -> bool:
        """Check if JSON format is enabled."""
        return self.format_type.lower() == "json"


# Global config instance
logger_config = LoggerConfig()


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_correlation: bool = True
) -> None:
    """
    Set up centralized logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('structured' or 'json')
        log_file: Optional log file path
        enable_correlation: Enable correlation ID tracking
    """
    # Update config if parameters provided
    if level:
        logger_config.level = level.upper()
    if format_type:
        logger_config.format_type = format_type
    if log_file:
        logger_config.log_file = log_file
    logger_config.enable_correlation = enable_correlation
    
    # Create log directory if needed
    if logger_config.log_file:
        log_path = Path(logger_config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Choose formatter
    if logger_config.is_json_format():
        formatter = JSONFormatter()
    else:
        formatter = StructuredFormatter()
    
    # Configure handlers
    handlers = []
    
    # Console handler
    if logger_config.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        if enable_correlation:
            console_handler.addFilter(correlation_filter)
        handlers.append(console_handler)
    
    # File handler
    if logger_config.log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            logger_config.log_file,
            maxBytes=logger_config.max_bytes,
            backupCount=logger_config.backup_count
        )
        file_handler.setFormatter(formatter)
        if enable_correlation:
            file_handler.addFilter(correlation_filter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, logger_config.level),
        handlers=handlers,
        force=True
    )
    
    # Set specific logger levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def get_logger(name: str, component: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        component: Optional component name for better categorization
        
    Returns:
        Configured logger instance
    """
    if component:
        logger_name = f"{name}.{component}"
    else:
        logger_name = name
    
    logger = logging.getLogger(logger_name)
    
    # Add custom methods for common patterns
    def log_agent_start(agent_name: str, agent_type: str = "single"):
        """Log agent initialization start."""
        logger.info(
            "Starting agent initialization",
            extra={
                "agent_name": agent_name,
                "agent_type": agent_type,
                "event": "agent_init_start"
            }
        )
    
    def log_agent_ready(agent_name: str, agent_type: str = "single", config: Dict[str, Any] = None):
        """Log agent ready state."""
        logger.info(
            "Agent ready for execution",
            extra={
                "agent_name": agent_name,
                "agent_type": agent_type,
                "event": "agent_ready",
                "config_summary": config or {}
            }
        )
    
    def log_agent_execution(query: str, agent_name: str, execution_id: str):
        """Log agent execution start."""
        logger.info(
            "Starting agent execution",
            extra={
                "agent_name": agent_name,
                "execution_id": execution_id,
                "query_length": len(query),
                "event": "execution_start"
            }
        )
    
    def log_agent_response(response: str, agent_name: str, execution_id: str, execution_time: float):
        """Log agent execution completion."""
        logger.info(
            "Agent execution completed",
            extra={
                "agent_name": agent_name,
                "execution_id": execution_id,
                "response_length": len(response),
                "execution_time": execution_time,
                "event": "execution_complete"
            }
        )
    
    def log_llm_call(provider: str, model: str, tokens_used: Optional[int] = None):
        """Log LLM API call."""
        logger.debug(
            "LLM API call made",
            extra={
                "provider": provider,
                "model": model,
                "tokens_used": tokens_used,
                "event": "llm_call"
            }
        )
    
    def log_tool_execution(tool_name: str, execution_time: float, success: bool = True):
        """Log tool execution."""
        level = logging.INFO if success else logging.ERROR
        logger.log(
            level,
            f"Tool execution {'completed' if success else 'failed'}",
            extra={
                "tool_name": tool_name,
                "execution_time": execution_time,
                "success": success,
                "event": "tool_execution"
            }
        )
    
    def log_memory_operation(operation: str, memory_type: str, success: bool = True):
        """Log memory operations."""
        level = logging.DEBUG if success else logging.ERROR
        logger.log(
            level,
            f"Memory {operation} {'completed' if success else 'failed'}",
            extra={
                "operation": operation,
                "memory_type": memory_type,
                "success": success,
                "event": "memory_operation"
            }
        )
    
    def log_evaluation_result(evaluator: str, score: float, agent_name: str):
        """Log evaluation results."""
        logger.info(
            "Evaluation completed",
            extra={
                "evaluator": evaluator,
                "score": score,
                "agent_name": agent_name,
                "event": "evaluation_result"
            }
        )
    
    # Add custom methods to logger
    logger.log_agent_start = log_agent_start
    logger.log_agent_ready = log_agent_ready
    logger.log_agent_execution = log_agent_execution
    logger.log_agent_response = log_agent_response
    logger.log_llm_call = log_llm_call
    logger.log_tool_execution = log_tool_execution
    logger.log_memory_operation = log_memory_operation
    logger.log_evaluation_result = log_evaluation_result
    
    return logger


# Initialize logging on module import
if not logging.getLogger().handlers:
    setup_logging()