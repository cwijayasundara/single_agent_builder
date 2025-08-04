"""
Custom logging formatters for structured and JSON logging.
"""
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any


class StructuredFormatter(logging.Formatter):
    """Structured text formatter for better readability."""
    
    def __init__(self):
        super().__init__()
        self.format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)-30s | "
            "%(correlation_id)-8s | %(message)s"
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        # Add correlation ID if not present
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = getattr(record, 'correlation_id', 'N/A')
        
        # Format the basic message
        formatted = super().format(record)
        
        # Add extra fields if present
        extras = []
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'asctime', 'correlation_id'
            }:
                if value is not None:
                    extras.append(f"{key}={value}")
        
        if extras:
            formatted += f" | {' | '.join(extras)}"
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON formatter for machine-readable logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "correlation_id": getattr(record, 'correlation_id', None),
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'asctime', 'correlation_id'
            }:
                if value is not None:
                    log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add stack info if present
        if record.stack_info:
            log_data['stack_info'] = record.stack_info
        
        return json.dumps(log_data, default=str, ensure_ascii=False)