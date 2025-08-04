"""
Request models for the API.
"""
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"


class MemoryType(str, Enum):
    """Supported memory types."""
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class EvaluatorType(str, Enum):
    """Supported evaluator types."""
    LLM_AS_JUDGE = "llm_as_judge"
    HEURISTIC = "heuristic"


# Single Agent Request Models

class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: LLMProvider
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PromptConfig(BaseModel):
    """Prompt configuration."""
    system_prompt: str
    variables: Dict[str, Any] = Field(default_factory=dict)


class ToolConfig(BaseModel):
    """Tool configuration."""
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """Memory configuration."""
    type: MemoryType
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EvaluatorConfig(BaseModel):
    """Evaluator configuration."""
    name: str
    type: EvaluatorType
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    enabled: bool = False
    evaluators: List[EvaluatorConfig] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=lambda: ["correctness", "helpfulness", "response_time"])
    auto_evaluate: bool = False


class ReactConfig(BaseModel):
    """ReAct configuration."""
    max_iterations: int = Field(default=10, ge=1, le=50)
    recursion_limit: int = Field(default=25, ge=1, le=100)


class AgentCreateRequest(BaseModel):
    """Request to create a new agent."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    version: str = Field(default="1.0.0")
    
    llm: LLMConfig
    prompts: PromptConfig
    tools: List[Union[str, ToolConfig]] = Field(default_factory=list)
    memory: List[MemoryConfig] = Field(default_factory=list)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    react: ReactConfig = Field(default_factory=ReactConfig)
    
    debug_mode: bool = False
    tags: List[str] = Field(default_factory=list)


class AgentUpdateRequest(BaseModel):
    """Request to update an existing agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    version: Optional[str] = None
    
    llm: Optional[LLMConfig] = None
    prompts: Optional[PromptConfig] = None
    tools: Optional[List[Union[str, ToolConfig]]] = None
    memory: Optional[List[MemoryConfig]] = None
    evaluation: Optional[EvaluationConfig] = None
    react: Optional[ReactConfig] = None
    
    debug_mode: Optional[bool] = None
    tags: Optional[List[str]] = None


class AgentRunRequest(BaseModel):
    """Request to run an agent."""
    query: str = Field(..., min_length=1, max_length=10000)
    context: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    timeout: Optional[int] = Field(default=300, gt=0, le=3600)
    include_evaluation: bool = False
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query string."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


# Evaluation Request Models

class EvaluationDatasetConfig(BaseModel):
    """Evaluation dataset configuration."""
    name: str
    description: Optional[str] = None
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    """Request to run evaluation."""
    agent_id: str
    dataset: EvaluationDatasetConfig
    evaluators: List[EvaluatorConfig]
    run_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchEvaluationRequest(BaseModel):
    """Request to run batch evaluation."""
    agent_ids: List[str]
    dataset: EvaluationDatasetConfig
    evaluators: List[EvaluatorConfig]
    run_name: Optional[str] = None
    parallel: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Configuration Management Request Models

class TemplateCreateRequest(BaseModel):
    """Request to create a configuration template."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    category: str = Field(..., min_length=1, max_length=50)
    template_type: str = Field(..., pattern="^(agent)$")
    config: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)


class TemplateUpdateRequest(BaseModel):
    """Request to update a configuration template."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, min_length=1, max_length=50)
    config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


# File Upload Request Models

class FileUploadRequest(BaseModel):
    """Request for file upload metadata."""
    filename: str
    content_type: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ConfigUploadRequest(BaseModel):
    """Request to upload configuration file."""
    filename: str
    config_type: str = Field(..., pattern="^(agent|template)$")
    validate_only: bool = False
    overwrite: bool = False


# Note: Validators are handled directly in field validation in Pydantic v2