"""
Health check API endpoints.
"""
import logging
import time
import psutil
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from ..models.responses import HealthResponse, StatusEnum

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

# Track application start time
_start_time = time.time()


def get_system_info() -> Dict[str, Any]:
    """Get basic system information."""
    try:
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "python_version": psutil.__version__,
        }
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return {"error": "system_info_unavailable"}


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    try:
        uptime = time.time() - _start_time
        system_info = get_system_info()
        
        return HealthResponse(
            status=StatusEnum.SUCCESS,
            message="API is healthy",
            version="1.0.0",
            uptime=uptime,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/ready")
async def readiness_check():
    """Readiness check for container orchestration."""
    try:
        # Check if core components are available
        from ...core.configurable_agent import ConfigurableAgent
        from ...evaluation.evaluation_manager import EvaluationManager
        
        # Basic import test
        checks = {
            "configurable_agent": True,
            "evaluation_manager": True,
            "system_resources": get_system_info().get("error") is None
        }
        
        all_ready = all(checks.values())
        status_code = 200 if all_ready else 503
        
        return {
            "status": "ready" if all_ready else "not_ready",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "not_ready",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/live")
async def liveness_check():
    """Liveness check for container orchestration."""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - _start_time
    }