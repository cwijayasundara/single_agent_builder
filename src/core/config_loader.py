"""
Configuration loader and validator for configurable agents.
"""
import yaml
import os
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4000
    api_key_env: str
    base_url: Optional[str] = None
    # Optional fields for Google Vertex AI
    project: Optional[str] = None
    location: Optional[str] = None


class PromptTemplate(BaseModel):
    template: str
    variables: Union[List[str], Dict[str, str]] = []
    
    @model_validator(mode='before')
    @classmethod
    def normalize_variables(cls, data):
        if isinstance(data, dict) and 'variables' in data:
            variables = data['variables']
            # If variables is a dict, convert to list of keys
            if isinstance(variables, dict):
                data['variables'] = list(variables.keys())
        return data


class PromptsConfig(BaseModel):
    system_prompt: PromptTemplate
    user_prompt: PromptTemplate
    tool_prompt: Optional[PromptTemplate] = None


class CustomTool(BaseModel):
    name: str
    module_path: str
    class_name: str
    description: str
    parameters: Dict[str, Any] = {}


class ToolsConfig(BaseModel):
    built_in: List[str] = []
    custom: List[CustomTool] = []


class MemoryStorageConfig(BaseModel):
    backend: str = "memory"
    connection_string: Optional[str] = None


class MemoryTypesConfig(BaseModel):
    semantic: bool = True
    episodic: bool = True
    procedural: bool = True


class MemorySettingsConfig(BaseModel):
    max_memory_size: int = 10000
    retention_days: int = 30
    background_processing: bool = True


class MemoryConfig(BaseModel):
    enabled: bool = False
    provider: str = "langmem"
    types: MemoryTypesConfig = MemoryTypesConfig()
    storage: MemoryStorageConfig = MemoryStorageConfig()
    settings: MemorySettingsConfig = MemorySettingsConfig()


# ReAct pattern uses a simpler configuration - no complex graph needed
class ReactConfig(BaseModel):
    max_iterations: int = 10
    recursion_limit: int = 50


class PromptOptimizationConfig(BaseModel):
    enabled: bool = False
    feedback_collection: bool = False
    ab_testing: bool = False
    optimization_frequency: str = "weekly"


class PerformanceTrackingConfig(BaseModel):
    enabled: bool = False
    metrics: List[str] = ["response_time", "accuracy", "user_satisfaction"]


class OptimizationConfig(BaseModel):
    enabled: bool = False
    prompt_optimization: PromptOptimizationConfig = PromptOptimizationConfig()
    performance_tracking: PerformanceTrackingConfig = PerformanceTrackingConfig()


class RuntimeConfig(BaseModel):
    max_iterations: int = 50
    timeout_seconds: int = 300
    retry_attempts: int = 3


# Logging Configuration Classes
class LoggingFileConfig(BaseModel):
    enabled: bool = False
    path: str = "logs/agent.log"
    level: str = "DEBUG"
    format: str = "json"  # structured or json
    max_size_mb: int = 10
    backup_count: int = 5


class LoggingConsoleConfig(BaseModel):
    enabled: bool = True
    level: str = "INFO"
    format: str = "structured"  # structured or json


class LoggingComponentConfig(BaseModel):
    agent: str = "INFO"
    evaluation: str = "INFO"
    memory: str = "INFO"
    llm: str = "WARNING"
    tools: str = "INFO"
    hierarchical: str = "INFO"


class LoggingCorrelationConfig(BaseModel):
    enabled: bool = True
    include_in_response: bool = False


class LoggingPerformanceConfig(BaseModel):
    log_execution_time: bool = True
    log_token_usage: bool = True
    log_memory_operations: bool = True
    log_tool_executions: bool = True


class LoggingPrivacyConfig(BaseModel):
    mask_api_keys: bool = True
    mask_user_input: bool = False
    mask_responses: bool = False
    excluded_fields: List[str] = []


class LoggingCustomConfig(BaseModel):
    structured_extra_fields: List[str] = ["agent_name", "execution_id", "correlation_id", "timestamp"]
    json_ensure_ascii: bool = False
    json_indent: Optional[int] = None


class LoggingConfig(BaseModel):
    enabled: bool = True
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str = "structured"  # structured or json
    console: LoggingConsoleConfig = LoggingConsoleConfig()
    file: LoggingFileConfig = LoggingFileConfig()
    components: LoggingComponentConfig = LoggingComponentConfig()
    correlation: LoggingCorrelationConfig = LoggingCorrelationConfig()
    performance: LoggingPerformanceConfig = LoggingPerformanceConfig()
    privacy: LoggingPrivacyConfig = LoggingPrivacyConfig()
    custom: LoggingCustomConfig = LoggingCustomConfig()
    debug_mode: bool = False


class EvaluatorConfig(BaseModel):
    name: str
    type: str = "llm_as_judge"  # llm_as_judge, custom, heuristic
    prompt: Optional[str] = None
    model: Optional[str] = None
    parameters: Dict[str, Any] = {}
    enabled: bool = True


class DatasetConfig(BaseModel):
    name: str
    description: Optional[str] = None
    examples: List[Dict[str, Any]] = []
    auto_generate: bool = False
    size: int = 10


class LangSmithConfig(BaseModel):
    enabled: bool = False
    api_key_env: str = "LANGSMITH_API_KEY"
    project_name: Optional[str] = None
    endpoint: str = "https://api.smith.langchain.com"
    tracing: bool = True


class EvaluationConfig(BaseModel):
    enabled: bool = False
    langsmith: LangSmithConfig = LangSmithConfig()
    evaluators: List[EvaluatorConfig] = []
    datasets: List[DatasetConfig] = []
    metrics: List[str] = ["correctness", "helpfulness", "response_time"]
    auto_evaluate: bool = False
    evaluation_frequency: str = "manual"  # manual, per_run, daily, weekly
    batch_size: int = 10
    max_concurrency: int = 2


class AgentInfo(BaseModel):
    name: str
    description: str
    version: str = "1.0.0"


class AgentConfiguration(BaseModel):
    agent: AgentInfo
    llm: LLMConfig
    prompts: PromptsConfig
    tools: ToolsConfig = ToolsConfig()
    memory: MemoryConfig = MemoryConfig()
    react: ReactConfig = ReactConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    logging: LoggingConfig = LoggingConfig()

    @field_validator('llm')
    @classmethod
    def validate_api_key_exists(cls, v):
        if v.api_key_env and not os.getenv(v.api_key_env):
            raise ValueError(f"Environment variable {v.api_key_env} not found")
        return v


class ConfigLoader:
    """Loads and validates agent configurations from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Optional[AgentConfiguration] = None
    
    def load_config(self, config_file: str) -> AgentConfiguration:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Validate and parse configuration
            self._config = AgentConfiguration(**config_data)
            return self._config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Configuration validation error: {e}")
    
    def get_config(self) -> Optional[AgentConfiguration]:
        """Get the loaded configuration."""
        return self._config
    
    def validate_config(self, config_data: Dict[str, Any]) -> bool:
        """Validate configuration data without loading."""
        try:
            AgentConfiguration(**config_data)
            return True
        except Exception:
            return False
    
    def get_prompt_template(self, prompt_type: str, **variables) -> str:
        """Get a formatted prompt template with variables substituted."""
        if not self._config:
            raise ValueError("Configuration not loaded")
        
        prompt_config = getattr(self._config.prompts, prompt_type, None)
        if not prompt_config:
            raise ValueError(f"Prompt type '{prompt_type}' not found")
        
        template = prompt_config.template
        
        # Substitute variables
        for var_name, var_value in variables.items():
            placeholder = f"{{{var_name}}}"
            template = template.replace(placeholder, str(var_value))
        
        return template
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        if not self._config:
            raise ValueError("Configuration not loaded")
        return self._config.llm
    
    def get_tools_config(self) -> ToolsConfig:
        """Get tools configuration."""
        if not self._config:
            raise ValueError("Configuration not loaded")
        return self._config.tools
    
    def get_memory_config(self) -> MemoryConfig:
        """Get memory configuration."""
        if not self._config:
            raise ValueError("Configuration not loaded")
        return self._config.memory
    
    def get_react_config(self) -> ReactConfig:
        """Get ReAct configuration."""
        if not self._config:
            raise ValueError("Configuration not loaded")
        return self._config.react
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation configuration."""
        if not self._config:
            raise ValueError("Configuration not loaded")
        return self._config.evaluation
    
    def validate_evaluation_config(self) -> Dict[str, Any]:
        """Validate evaluation configuration and check dependencies."""
        if not self._config or not self._config.evaluation.enabled:
            return {"valid": True, "warnings": [], "errors": []}
        
        warnings = []
        errors = []
        
        eval_config = self._config.evaluation
        
        # Check for LangSmith dependency if LLM-as-judge evaluators are configured
        llm_evaluators = [e for e in eval_config.evaluators if e.type == "llm_as_judge"]
        if llm_evaluators:
            try:
                import langsmith
                import openevals
            except ImportError as e:
                warnings.append(
                    f"LLM-as-judge evaluators configured but dependencies missing: {str(e)}. "
                    "Install with: pip install langsmith openevals. Will fall back to heuristic evaluation."
                )
        
        # Check LangSmith configuration if enabled
        if eval_config.langsmith.enabled:
            import os
            api_key = os.getenv(eval_config.langsmith.api_key_env)
            if not api_key:
                warnings.append(
                    f"LangSmith enabled but {eval_config.langsmith.api_key_env} environment variable not set. "
                    f"Evaluation will work but without LangSmith integration."
                )
        
        # Validate evaluator configurations
        for evaluator in eval_config.evaluators:
            if evaluator.type == "llm_as_judge" and not evaluator.prompt:
                warnings.append(f"LLM-as-judge evaluator '{evaluator.name}' has no prompt configured")
            
            if evaluator.type == "heuristic" and evaluator.name not in ["correctness", "helpfulness", "response_time", "tool_usage"]:
                warnings.append(f"Unknown heuristic evaluator '{evaluator.name}', may not work properly")
        
        # Check if any evaluators are configured when evaluation is enabled
        if not eval_config.evaluators:
            warnings.append("Evaluation enabled but no evaluators configured. Will use default built-in evaluators.")
        
        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "total_evaluators": len(eval_config.evaluators),
            "llm_evaluators": len(llm_evaluators),
            "heuristic_evaluators": len([e for e in eval_config.evaluators if e.type == "heuristic"])
        }