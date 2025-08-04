"""
Evaluation Manager for coordinating agent evaluations with LangSmith.
"""
import os
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dotenv import load_dotenv

try:
    from langsmith import Client
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import CORRECTNESS_PROMPT, RAG_HELPFULNESS_PROMPT
    LANGSMITH_AVAILABLE = True
except ImportError as e:
    LANGSMITH_AVAILABLE = False
    Client = None

try:
    from ..core.config_loader import EvaluationConfig, EvaluatorConfig, DatasetConfig
except ImportError:
    # Fallback for when relative imports fail
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.config_loader import EvaluationConfig, EvaluatorConfig, DatasetConfig

from .metrics import EvaluationMetrics
from .evaluators.built_in import (
    CorrectnessEvaluator,
    HelpfulnessEvaluator,
    ResponseTimeEvaluator,
    ToolUsageEvaluator
)
from ..custom_logging import get_logger, CorrelationContext

load_dotenv()

# Get logger for this module
logger = get_logger(__name__, "evaluation")


class EvaluationManager:
    """Manages agent evaluations using LangSmith and custom evaluators."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = None
        self.metrics = EvaluationMetrics()
        self.evaluators = {}
        self.datasets = {}
        
        logger.info(
            "Initializing evaluation manager",
            extra={
                "enabled": config.enabled,
                "langsmith_enabled": config.langsmith.enabled if config.langsmith else False,
                "evaluators_count": len(config.evaluators),
                "auto_evaluate": config.auto_evaluate
            }
        )
        
        # Try to setup LangSmith if enabled, but don't fail if it's not configured
        if config.enabled and config.langsmith.enabled:
            try:
                self._setup_langsmith()
            except Exception as e:
                logger.warning(
                    "LangSmith setup failed, continuing without LangSmith integration",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "langsmith_endpoint": config.langsmith.endpoint,
                        "api_key_env": config.langsmith.api_key_env
                    },
                    exc_info=True
                )
                # Continue without LangSmith - we can still use built-in evaluators
        
        self._setup_evaluators()
    
    def _setup_langsmith(self):
        """Initialize LangSmith client."""
        logger.debug("Setting up LangSmith client")
        
        if not LANGSMITH_AVAILABLE:
            logger.error("LangSmith dependencies not available")
            raise ImportError(
                "LangSmith dependencies not available. "
                "Install with: pip install langsmith openevals"
            )
        
        api_key = os.getenv(self.config.langsmith.api_key_env)
        if not api_key:
            logger.error(
                "LangSmith API key not found",
                extra={
                    "api_key_env": self.config.langsmith.api_key_env,
                    "endpoint": self.config.langsmith.endpoint
                }
            )
            raise ValueError(
                f"LangSmith API key not found in environment variable: "
                f"{self.config.langsmith.api_key_env}. "
                f"Please set the {self.config.langsmith.api_key_env} environment variable "
                f"or disable LangSmith integration in the evaluation configuration."
            )
        
        try:
            self.client = Client(
                api_key=api_key,
                api_url=self.config.langsmith.endpoint
            )
            
            # Set project name if specified
            if self.config.langsmith.project_name:
                os.environ["LANGCHAIN_PROJECT"] = self.config.langsmith.project_name
                logger.debug(
                    "Set LangSmith project name",
                    extra={"project_name": self.config.langsmith.project_name}
                )
                
            # Enable tracing if configured
            if self.config.langsmith.tracing:
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = api_key
                logger.debug("Enabled LangSmith tracing")
            
            logger.info(
                "LangSmith client initialized successfully",
                extra={
                    "endpoint": self.config.langsmith.endpoint,
                    "project_name": self.config.langsmith.project_name,
                    "tracing_enabled": self.config.langsmith.tracing
                }
            )
                
        except Exception as e:
            logger.error(
                "Failed to initialize LangSmith client",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "endpoint": self.config.langsmith.endpoint
                },
                exc_info=True
            )
            raise RuntimeError(f"Failed to initialize LangSmith client: {e}")
    
    def _setup_evaluators(self):
        """Initialize configured evaluators."""
        # Built-in evaluators
        self.evaluators.update({
            "correctness": CorrectnessEvaluator(),
            "helpfulness": HelpfulnessEvaluator(),
            "response_time": ResponseTimeEvaluator(),
            "tool_usage": ToolUsageEvaluator()
        })
        
        # Custom evaluators from config
        for evaluator_config in self.config.evaluators:
            if evaluator_config.enabled:
                evaluator = self._create_evaluator(evaluator_config)
                if evaluator is not None:  # Only add if evaluator was created successfully
                    self.evaluators[evaluator_config.name] = evaluator
    
    def _create_evaluator(self, config: EvaluatorConfig) -> Callable:
        """Create an evaluator from configuration."""
        if config.type == "llm_as_judge":
            if not LANGSMITH_AVAILABLE:
                logger.warning(
                    "LangSmith not available for LLM-as-judge evaluator, using built-in evaluators instead",
                    extra={
                        "evaluator_name": config.name,
                        "evaluator_type": config.type,
                        "fallback": "built-in evaluator"
                    }
                )
                return None  # Return None to skip this evaluator
            
            try:
                prompt = config.prompt or CORRECTNESS_PROMPT
                model = config.model or "openai:gpt-4.1-mini"
                
                return create_llm_as_judge(
                    prompt=prompt,
                    model=model,
                    feedback_key=config.name,
                    **config.parameters
                )
            except Exception as e:
                logger.error(
                    "Failed to create LLM-as-judge evaluator",
                    extra={
                        "evaluator_name": config.name,
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
                return None
        elif config.type == "heuristic":
            # Use built-in heuristic evaluators
            if config.name == "correctness":
                return CorrectnessEvaluator()
            elif config.name == "helpfulness":
                return HelpfulnessEvaluator()
            elif config.name == "response_time":
                target_time = config.parameters.get("target_time", 5.0)
                max_time = config.parameters.get("max_time", 20.0)
                return ResponseTimeEvaluator(target_time=target_time, max_time=max_time)
            elif config.name == "tool_usage":
                return ToolUsageEvaluator()
            else:
                logger.warning(
                    "Unknown heuristic evaluator, attempting to use built-in evaluator",
                    extra={
                        "evaluator_name": config.name,
                        "evaluator_type": config.type,
                        "available_built_ins": ["correctness", "helpfulness", "response_time", "tool_usage"]
                    }
                )
                return self.evaluators.get(config.name)
        elif config.type == "custom":
            # For custom evaluators, they should be registered separately
            return self.evaluators.get(config.name)
        else:
            raise ValueError(f"Unsupported evaluator type: {config.type}. Supported types: llm_as_judge, heuristic, custom")
    
    def create_dataset(self, dataset_config: DatasetConfig) -> str:
        """Create a dataset in LangSmith."""
        if not self.client:
            raise RuntimeError("LangSmith client not initialized")
        
        try:
            dataset = self.client.create_dataset(
                dataset_name=dataset_config.name,
                description=dataset_config.description or f"Dataset for {dataset_config.name}",
            )
            
            # Add examples if provided
            if dataset_config.examples:
                examples = []
                for i, example in enumerate(dataset_config.examples):
                    examples.append({
                        "inputs": example.get("inputs", {}),
                        "outputs": example.get("outputs", {}),
                        "metadata": example.get("metadata", {"example_id": i})
                    })
                
                self.client.create_examples(
                    dataset_id=dataset.id,
                    examples=examples
                )
            
            self.datasets[dataset_config.name] = dataset.id
            return dataset.id
            
        except Exception as e:
            raise RuntimeError(f"Failed to create dataset '{dataset_config.name}': {e}")
    
    def evaluate_single(
        self, 
        input_data: Dict[str, Any], 
        output_data: Dict[str, Any], 
        reference_output: Optional[Dict[str, Any]] = None,
        evaluator_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single input-output pair."""
        start_time = time.time()
        
        # Determine which evaluators to use
        if evaluator_names is None:
            evaluator_names = list(self.evaluators.keys())
        
        results = {}
        
        for evaluator_name in evaluator_names:
            if evaluator_name not in self.evaluators:
                continue
                
            evaluator = self.evaluators[evaluator_name]
            
            try:
                if evaluator_name == "response_time":
                    # Special handling for response time
                    result = evaluator.evaluate(
                        inputs=input_data,
                        outputs=output_data,
                        reference_outputs=reference_output,
                        start_time=start_time,
                        end_time=time.time()
                    )
                else:
                    result = evaluator.evaluate(
                        inputs=input_data,
                        outputs=output_data,
                        reference_outputs=reference_output
                    )
                
                results[evaluator_name] = result
                
            except Exception as e:
                results[evaluator_name] = {
                    "error": str(e),
                    "score": 0.0,
                    "reasoning": f"Evaluation failed: {str(e)}"
                }
        
        # Update metrics
        self.metrics.add_evaluation_result(results)
        
        return results
    
    def evaluate_dataset(
        self, 
        target_function: Callable,
        dataset_name: str,
        evaluator_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a target function against a dataset."""
        if not self.client:
            raise RuntimeError("LangSmith client not initialized")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Determine which evaluators to use
        if evaluator_names is None:
            evaluator_names = [name for name in self.evaluators.keys() 
                             if self.evaluators[name] is not None]
        
        # Create evaluator functions for LangSmith
        langsmith_evaluators = []
        for evaluator_name in evaluator_names:
            if evaluator_name in self.evaluators:
                evaluator = self.evaluators[evaluator_name]
                
                def make_evaluator(eval_name, eval_func):
                    def langsmith_evaluator(inputs, outputs, reference_outputs=None):
                        return eval_func.evaluate(inputs, outputs, reference_outputs)
                    return langsmith_evaluator
                
                langsmith_evaluators.append(make_evaluator(evaluator_name, evaluator))
        
        try:
            # Run evaluation using LangSmith
            experiment_results = self.client.evaluate(
                target_function,
                data=dataset_name,
                evaluators=langsmith_evaluators,
                max_concurrency=self.config.max_concurrency,
                experiment_prefix=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            return {
                "experiment_id": str(experiment_results.experiment_id),
                "experiment_name": experiment_results.experiment_name,
                "dataset_name": dataset_name,
                "evaluator_names": evaluator_names,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate dataset '{dataset_name}': {e}")
    
    def register_evaluator(self, name: str, evaluator: Callable):
        """Register a custom evaluator."""
        self.evaluators[name] = evaluator
    
    def get_evaluation_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get results from a completed evaluation experiment."""
        if not self.client:
            raise RuntimeError("LangSmith client not initialized")
        
        try:
            # Get experiment results
            runs = list(self.client.list_runs(
                project_name=experiment_id,
                execution_order=1
            ))
            
            results = []
            for run in runs:
                if run.outputs and run.feedback_stats:
                    results.append({
                        "run_id": str(run.id),
                        "inputs": run.inputs,
                        "outputs": run.outputs,
                        "feedback": run.feedback_stats,
                        "start_time": run.start_time,
                        "end_time": run.end_time
                    })
            
            return {
                "experiment_id": experiment_id,
                "total_runs": len(results),
                "results": results,
                "retrieved_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to get evaluation results: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation metrics."""
        return self.metrics.get_summary()
    
    def export_results(self, format_type: str = "json") -> str:
        """Export evaluation results."""
        if format_type == "json":
            import json
            return json.dumps(self.get_metrics_summary(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def clear_metrics(self):
        """Clear collected metrics."""
        self.metrics.clear()