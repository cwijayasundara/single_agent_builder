"""
Base evaluator class for implementing custom evaluators.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def evaluate(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reference_outputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the inputs and outputs.
        
        Args:
            inputs: The input data that was provided to the agent
            outputs: The output data produced by the agent
            reference_outputs: Optional reference/expected outputs for comparison
            **kwargs: Additional parameters specific to the evaluator
            
        Returns:
            Dictionary containing:
            - score: Numerical score (typically 0.0 to 1.0)
            - reasoning: Text explanation of the evaluation
            - metadata: Additional evaluation metadata
        """
        pass
    
    def preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess inputs before evaluation. Override if needed."""
        return inputs
    
    def preprocess_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess outputs before evaluation. Override if needed.""" 
        return outputs
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess evaluation result. Override if needed."""
        return result
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        return f"BaseEvaluator(name='{self.name}', description='{self.description}')"