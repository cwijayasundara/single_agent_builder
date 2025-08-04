"""
Dynamic prompt optimization system for configurable agents.
"""
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics


class OptimizationMetric(Enum):
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    USER_SATISFACTION = "user_satisfaction"
    TASK_SUCCESS = "task_success"
    TOKEN_EFFICIENCY = "token_efficiency"


@dataclass
class PromptVariant:
    """Represents a prompt variant for A/B testing."""
    id: str
    template: str
    variables: List[str]
    created_at: datetime
    performance_scores: Dict[OptimizationMetric, List[float]]
    usage_count: int = 0
    
    def get_average_score(self, metric: OptimizationMetric) -> float:
        """Get average score for a specific metric."""
        scores = self.performance_scores.get(metric, [])
        return statistics.mean(scores) if scores else 0.0
    
    def add_score(self, metric: OptimizationMetric, score: float):
        """Add a performance score."""
        if metric not in self.performance_scores:
            self.performance_scores[metric] = []
        self.performance_scores[metric].append(score)
        self.usage_count += 1


class PromptOptimizer:
    """Manages dynamic prompt optimization and A/B testing."""
    
    def __init__(self, optimization_config):
        self.config = optimization_config
        self.prompt_variants: Dict[str, List[PromptVariant]] = {}
        self.active_experiments: Dict[str, Dict] = {}
        self.feedback_history: List[Dict] = []
        self.optimization_history: List[Dict] = []
        
    def register_prompt_type(self, prompt_type: str, base_template: str, variables: List[str]):
        """Register a new prompt type for optimization."""
        if prompt_type not in self.prompt_variants:
            self.prompt_variants[prompt_type] = []
        
        # Create base variant
        base_variant = PromptVariant(
            id=f"{prompt_type}_base",
            template=base_template,
            variables=variables,
            created_at=datetime.now(),
            performance_scores={}
        )
        
        self.prompt_variants[prompt_type].append(base_variant)
    
    def generate_variants(self, prompt_type: str, num_variants: int = 3) -> List[PromptVariant]:
        """Generate prompt variants for A/B testing."""
        if prompt_type not in self.prompt_variants:
            raise ValueError(f"Prompt type {prompt_type} not registered")
        
        base_variant = self.prompt_variants[prompt_type][0]
        variants = []
        
        # Generate variants using different optimization strategies
        for i in range(num_variants):
            variant_template = self._create_variant(base_variant.template, i)
            variant = PromptVariant(
                id=f"{prompt_type}_variant_{i+1}",
                template=variant_template,
                variables=base_variant.variables,
                created_at=datetime.now(),
                performance_scores={}
            )
            variants.append(variant)
        
        self.prompt_variants[prompt_type].extend(variants)
        return variants
    
    def _create_variant(self, base_template: str, variant_index: int) -> str:
        """Create a variant of the base template."""
        # Different optimization strategies
        strategies = [
            self._add_clarity_instructions,
            self._add_step_by_step_guidance,
            self._add_context_emphasis,
            self._add_role_specification,
            self._add_output_format_specification
        ]
        
        strategy = strategies[variant_index % len(strategies)]
        return strategy(base_template)
    
    def _add_clarity_instructions(self, template: str) -> str:
        """Add clarity instructions to prompt."""
        addition = "\\nPlease be clear and specific in your response."
        return template + addition
    
    def _add_step_by_step_guidance(self, template: str) -> str:
        """Add step-by-step guidance."""
        addition = "\\nThink through this step by step before providing your response."
        return template + addition
    
    def _add_context_emphasis(self, template: str) -> str:
        """Add context emphasis."""
        addition = "\\nConsider the full context when formulating your response."
        return template + addition
    
    def _add_role_specification(self, template: str) -> str:
        """Add role specification."""
        addition = "\\nAs an expert assistant, provide a comprehensive response."
        return template + addition
    
    def _add_output_format_specification(self, template: str) -> str:
        """Add output format specification."""
        addition = "\\nStructure your response clearly with main points highlighted."
        return template + addition
    
    def select_prompt_variant(self, prompt_type: str) -> PromptVariant:
        """Select the best performing prompt variant or use A/B testing."""
        if prompt_type not in self.prompt_variants:
            raise ValueError(f"Prompt type {prompt_type} not registered")
        
        variants = self.prompt_variants[prompt_type]
        
        if not self.config.ab_testing:
            # Use best performing variant
            return self._get_best_variant(variants)
        else:
            # Use A/B testing strategy
            return self._ab_test_selection(variants)
    
    def _get_best_variant(self, variants: List[PromptVariant]) -> PromptVariant:
        """Get the best performing variant based on metrics."""
        if not variants:
            raise ValueError("No variants available")
        
        # Score variants based on performance
        scored_variants = []
        for variant in variants:
            total_score = 0
            metric_count = 0
            
            for metric in OptimizationMetric:
                avg_score = variant.get_average_score(metric)
                if avg_score > 0:
                    total_score += avg_score
                    metric_count += 1
            
            final_score = total_score / metric_count if metric_count > 0 else 0
            scored_variants.append((variant, final_score))
        
        # Return best scoring variant
        scored_variants.sort(key=lambda x: x[1], reverse=True)
        return scored_variants[0][0]
    
    def _ab_test_selection(self, variants: List[PromptVariant]) -> PromptVariant:
        """Select variant using A/B testing strategy (epsilon-greedy)."""
        import random
        
        epsilon = 0.1  # 10% exploration
        
        if random.random() < epsilon:
            # Explore: random selection
            return random.choice(variants)
        else:
            # Exploit: best variant
            return self._get_best_variant(variants)
    
    def record_feedback(self, prompt_type: str, variant_id: str, 
                       metrics: Dict[OptimizationMetric, float]):
        """Record feedback for a prompt variant."""
        # Find the variant and update its scores
        for variant in self.prompt_variants.get(prompt_type, []):
            if variant.id == variant_id:
                for metric, score in metrics.items():
                    variant.add_score(metric, score)
                break
        
        # Store feedback history
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt_type": prompt_type,
            "variant_id": variant_id,
            "metrics": {metric.value: score for metric, score in metrics.items()}
        }
        self.feedback_history.append(feedback_entry)
    
    def optimize_prompts(self) -> Dict[str, str]:
        """Perform prompt optimization based on collected feedback."""
        optimization_results = {}
        
        for prompt_type, variants in self.prompt_variants.items():
            if len(variants) > 1:  # Only optimize if we have multiple variants
                best_variant = self._get_best_variant(variants)
                
                # Check if optimization is needed
                if self._should_optimize(variants):
                    optimized_template = self._evolve_prompt(best_variant)
                    
                    # Create new optimized variant
                    new_variant = PromptVariant(
                        id=f"{prompt_type}_optimized_{len(variants)}",
                        template=optimized_template,
                        variables=best_variant.variables,
                        created_at=datetime.now(),
                        performance_scores={}
                    )
                    
                    variants.append(new_variant)
                    optimization_results[prompt_type] = optimized_template
        
        # Record optimization event
        self.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "optimized_prompts": list(optimization_results.keys()),
            "total_variants": sum(len(variants) for variants in self.prompt_variants.values())
        })
        
        return optimization_results
    
    def _should_optimize(self, variants: List[PromptVariant]) -> bool:
        """Determine if optimization is needed."""
        # Check if we have enough data
        total_usage = sum(variant.usage_count for variant in variants)
        if total_usage < 50:  # Need at least 50 uses
            return False
        
        # Check performance variance
        best_variant = self._get_best_variant(variants)
        worst_performance = min(
            variant.get_average_score(OptimizationMetric.TASK_SUCCESS) 
            for variant in variants
        )
        best_performance = best_variant.get_average_score(OptimizationMetric.TASK_SUCCESS)
        
        # Optimize if there's significant room for improvement
        return (best_performance - worst_performance) > 0.1
    
    def _evolve_prompt(self, best_variant: PromptVariant) -> str:
        """Evolve a prompt based on performance data."""
        base_template = best_variant.template
        
        # Apply improvements based on metrics
        improvements = []
        
        # If response time is slow, add efficiency instruction
        if best_variant.get_average_score(OptimizationMetric.RESPONSE_TIME) < 0.7:
            improvements.append("Be concise and efficient in your response.")
        
        # If accuracy is low, add precision instruction
        if best_variant.get_average_score(OptimizationMetric.ACCURACY) < 0.8:
            improvements.append("Double-check your response for accuracy.")
        
        # If user satisfaction is low, add engagement instruction
        if best_variant.get_average_score(OptimizationMetric.USER_SATISFACTION) < 0.7:
            improvements.append("Ensure your response is helpful and user-friendly.")
        
        # Combine improvements
        if improvements:
            improvement_text = " ".join(improvements)
            evolved_template = f"{base_template}\\n\\nAdditional instructions: {improvement_text}"
        else:
            evolved_template = base_template
        
        return evolved_template
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            "total_prompt_types": len(self.prompt_variants),
            "total_variants": sum(len(variants) for variants in self.prompt_variants.values()),
            "total_feedback_entries": len(self.feedback_history),
            "optimization_runs": len(self.optimization_history),
            "prompt_performance": {}
        }
        
        # Add performance stats for each prompt type
        for prompt_type, variants in self.prompt_variants.items():
            best_variant = self._get_best_variant(variants) if variants else None
            if best_variant:
                stats["prompt_performance"][prompt_type] = {
                    "best_variant_id": best_variant.id,
                    "usage_count": best_variant.usage_count,
                    "average_scores": {
                        metric.value: best_variant.get_average_score(metric)
                        for metric in OptimizationMetric
                    }
                }
        
        return stats
    
    def export_optimization_data(self) -> str:
        """Export optimization data for analysis."""
        export_data = {
            "prompt_variants": {
                prompt_type: [
                    {
                        "id": variant.id,
                        "template": variant.template,
                        "variables": variant.variables,
                        "created_at": variant.created_at.isoformat(),
                        "usage_count": variant.usage_count,
                        "performance_scores": {
                            metric.value: scores 
                            for metric, scores in variant.performance_scores.items()
                        }
                    }
                    for variant in variants
                ]
                for prompt_type, variants in self.prompt_variants.items()
            },
            "feedback_history": self.feedback_history,
            "optimization_history": self.optimization_history,
            "export_timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(export_data, indent=2)
    
    def clear_optimization_data(self):
        """Clear all optimization data."""
        self.prompt_variants.clear()
        self.active_experiments.clear()
        self.feedback_history.clear()
        self.optimization_history.clear()