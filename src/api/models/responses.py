"""
Response models for the API.
"""
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class StatusEnum(str, Enum):
    """Status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    CONFIGURING = "configuring"


# Base Response Models

class BaseResponse(BaseModel):
    """Base response model."""
    status: StatusEnum
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: StatusEnum = StatusEnum.ERROR
    error_code: str
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseResponse):
    """Health check response."""
    status: StatusEnum = StatusEnum.SUCCESS
    version: str
    uptime: float
    system_info: Dict[str, Any] = Field(default_factory=dict)


# Agent Response Models

class AgentInfo(BaseModel):
    """Agent information."""
    id: str
    name: str
    description: Optional[str] = None
    version: str
    status: AgentStatus
    created_at: datetime
    updated_at: datetime
    tags: List[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Agent configuration details."""
    llm: Dict[str, Any]
    prompts: Dict[str, Any]
    tools: List[Union[str, Dict[str, Any]]]
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    react: Dict[str, Any] = Field(default_factory=dict)
    debug_mode: bool = False


class AgentResponse(BaseResponse):
    """Single agent response."""
    status: StatusEnum = StatusEnum.SUCCESS
    agent: AgentInfo
    config: Optional[AgentConfig] = None


class AgentListResponse(BaseResponse):
    """Agent list response."""
    status: StatusEnum = StatusEnum.SUCCESS
    agents: List[AgentInfo]
    total: int
    page: int = 1
    page_size: int = 20


class AgentRunResult(BaseModel):
    """Agent run result."""
    response: str
    execution_time: float
    token_usage: Dict[str, int] = Field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    memory_updates: List[Dict[str, Any]] = Field(default_factory=list)
    debug_info: Optional[Dict[str, Any]] = None


class AgentEvaluationResult(BaseModel):
    """Agent evaluation result."""
    evaluator: str
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseResponse):
    """Agent run response."""
    status: StatusEnum = StatusEnum.SUCCESS
    run_id: str
    agent_id: str
    query: str
    result: Optional[AgentRunResult] = None
    evaluation: List[AgentEvaluationResult] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None


# Evaluation Response Models

class EvaluationMetric(BaseModel):
    """Evaluation metric."""
    name: str
    value: float
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Individual evaluation result."""
    input: str
    output: str
    expected_output: Optional[str] = None
    metrics: List[EvaluationMetric]
    evaluator_results: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    """Evaluation summary statistics."""
    total_samples: int
    passed_samples: int
    failed_samples: int
    average_score: float
    metric_averages: Dict[str, float] = Field(default_factory=dict)
    execution_time: float


class EvaluationResponse(BaseResponse):
    """Evaluation response."""
    status: StatusEnum = StatusEnum.SUCCESS
    evaluation_id: str
    run_name: Optional[str] = None
    agent_id: str
    dataset_name: str
    results: List[EvaluationResult]
    summary: EvaluationSummary
    started_at: datetime
    completed_at: Optional[datetime] = None


class BatchEvaluationResponse(BaseResponse):
    """Batch evaluation response."""
    status: StatusEnum = StatusEnum.SUCCESS
    batch_id: str
    run_name: Optional[str] = None
    evaluations: List[EvaluationResponse]
    summary: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    completed_at: Optional[datetime] = None


# Template Response Models

class TemplateInfo(BaseModel):
    """Template information."""
    id: str
    name: str
    description: Optional[str] = None
    category: str
    template_type: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = Field(default_factory=list)
    usage_count: int = 0


class TemplateResponse(BaseResponse):
    """Single template response."""
    status: StatusEnum = StatusEnum.SUCCESS
    template: TemplateInfo
    config: Optional[Dict[str, Any]] = None


class TemplateListResponse(BaseResponse):
    """Template list response."""
    status: StatusEnum = StatusEnum.SUCCESS
    templates: List[TemplateInfo]
    total: int
    page: int = 1
    page_size: int = 20
    categories: List[str] = Field(default_factory=list)


# Analytics Response Models

class AgentMetrics(BaseModel):
    """Agent performance metrics."""
    agent_id: str
    agent_name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    average_execution_time: float
    average_token_usage: Dict[str, float] = Field(default_factory=dict)
    evaluation_scores: Dict[str, float] = Field(default_factory=dict)
    last_run: Optional[datetime] = None


class SystemMetrics(BaseModel):
    """System-wide metrics."""
    total_agents: int
    active_agents: int
    total_runs_today: int
    successful_runs_today: int
    average_response_time: float
    system_load: float
    memory_usage: float
    cpu_usage: float


class AnalyticsResponse(BaseResponse):
    """Analytics response."""
    status: StatusEnum = StatusEnum.SUCCESS
    system_metrics: SystemMetrics
    agent_metrics: List[AgentMetrics] = Field(default_factory=list)
    time_range: Dict[str, datetime] = Field(default_factory=dict)


# File Response Models

class FileInfo(BaseModel):
    """File information."""
    id: str
    filename: str
    original_filename: str
    content_type: str
    size: int
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    uploaded_at: datetime


class FileUploadResponse(BaseResponse):
    """File upload response."""
    status: StatusEnum = StatusEnum.SUCCESS
    file: FileInfo
    download_url: str


class FileListResponse(BaseResponse):
    """File list response."""
    status: StatusEnum = StatusEnum.SUCCESS
    files: List[FileInfo]
    total: int
    page: int = 1
    page_size: int = 20


# WebSocket Response Models

class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    run_id: Optional[str] = None


class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    chunk_id: int
    content: str
    is_final: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamingResponse(BaseResponse):
    """Streaming response wrapper."""
    status: StatusEnum = StatusEnum.SUCCESS
    run_id: str
    chunks: List[StreamingChunk] = Field(default_factory=list)
    total_chunks: Optional[int] = None
    is_complete: bool = False