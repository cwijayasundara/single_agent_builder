"""
Correlation ID management for request/response tracing.
"""
import uuid
import logging
from contextvars import ContextVar
from typing import Optional

# Context variable for correlation ID
correlation_id_context: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationContext:
    """Context manager for correlation ID tracking."""
    
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or self.generate_correlation_id()
        self.token = None
    
    def __enter__(self):
        self.token = correlation_id_context.set(self.correlation_id)
        return self.correlation_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            correlation_id_context.reset(self.token)
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())[:8]
    
    @staticmethod
    def get_current() -> Optional[str]:
        """Get the current correlation ID."""
        return correlation_id_context.get()
    
    @staticmethod
    def set_current(correlation_id: str) -> None:
        """Set the current correlation ID."""
        correlation_id_context.set(correlation_id)


def correlation_filter(record: logging.LogRecord) -> bool:
    """Add correlation ID to log records."""
    record.correlation_id = CorrelationContext.get_current() or 'N/A'
    return True


class CorrelationLogger:
    """Logger wrapper that automatically includes correlation IDs."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def _log_with_correlation(self, level: int, msg: str, *args, **kwargs):
        """Log with correlation ID."""
        extra = kwargs.get('extra', {})
        extra['correlation_id'] = CorrelationContext.get_current() or 'N/A'
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_correlation(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_correlation(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_correlation(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log_with_correlation(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_correlation(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        kwargs['exc_info'] = True
        self.error(msg, *args, **kwargs)