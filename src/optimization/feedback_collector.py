"""
Feedback collection system for prompt optimization.
"""
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from langchain_core.messages import BaseMessage

from .prompt_optimizer import OptimizationMetric, PromptOptimizer


class FeedbackCollector:
    """Collects feedback for prompt optimization."""
    
    def __init__(self, prompt_optimizer: PromptOptimizer):
        self.prompt_optimizer = prompt_optimizer
        self.interaction_start_times: Dict[str, float] = {}
        self.pending_feedback: Dict[str, Dict] = {}
    
    def start_interaction(self, interaction_id: str, prompt_type: str, variant_id: str):
        """Mark the start of an interaction for timing."""
        self.interaction_start_times[interaction_id] = time.time()
        self.pending_feedback[interaction_id] = {
            "prompt_type": prompt_type,
            "variant_id": variant_id,
            "start_time": time.time()
        }
    
    def end_interaction(self, interaction_id: str, response: BaseMessage, 
                       success: bool = True, user_feedback: Optional[Dict] = None):
        """End an interaction and collect automatic feedback."""
        if interaction_id not in self.pending_feedback:
            return
        
        interaction_data = self.pending_feedback[interaction_id]
        
        # Calculate response time
        response_time = time.time() - interaction_data["start_time"]
        response_time_score = self._score_response_time(response_time)
        
        # Calculate token efficiency
        token_efficiency_score = self._score_token_efficiency(response)
        
        # Task success score
        task_success_score = 1.0 if success else 0.0
        
        # Collect metrics
        metrics = {
            OptimizationMetric.RESPONSE_TIME: response_time_score,
            OptimizationMetric.TOKEN_EFFICIENCY: token_efficiency_score,
            OptimizationMetric.TASK_SUCCESS: task_success_score
        }
        
        # Add user feedback if provided
        if user_feedback:
            if "satisfaction" in user_feedback:
                metrics[OptimizationMetric.USER_SATISFACTION] = user_feedback["satisfaction"]
            if "accuracy" in user_feedback:
                metrics[OptimizationMetric.ACCURACY] = user_feedback["accuracy"]
        
        # Record feedback
        self.prompt_optimizer.record_feedback(
            interaction_data["prompt_type"],
            interaction_data["variant_id"],
            metrics
        )
        
        # Clean up
        del self.pending_feedback[interaction_id]
        if interaction_id in self.interaction_start_times:
            del self.interaction_start_times[interaction_id]
    
    def _score_response_time(self, response_time: float) -> float:
        """Score response time (0.0 to 1.0, higher is better)."""
        # Excellent: < 2s, Good: < 5s, Fair: < 10s, Poor: >= 10s
        if response_time < 2.0:
            return 1.0
        elif response_time < 5.0:
            return 0.8
        elif response_time < 10.0:
            return 0.6
        else:
            return 0.3
    
    def _score_token_efficiency(self, response: BaseMessage) -> float:
        """Score token efficiency based on response length and quality."""
        response_length = len(response.content)
        
        # Optimal range: 100-500 characters
        if 100 <= response_length <= 500:
            return 1.0
        elif 50 <= response_length < 100 or 500 < response_length <= 1000:
            return 0.8
        elif response_length < 50:
            return 0.5  # Too short
        else:
            return 0.6  # Too long
    
    def collect_user_feedback(self, interaction_id: str, satisfaction: float, 
                             accuracy: Optional[float] = None, 
                             additional_feedback: Optional[Dict] = None):
        """Collect explicit user feedback."""
        if interaction_id in self.pending_feedback:
            user_feedback = {"satisfaction": satisfaction}
            if accuracy is not None:
                user_feedback["accuracy"] = accuracy
            if additional_feedback:
                user_feedback.update(additional_feedback)
            
            # Update pending feedback
            self.pending_feedback[interaction_id]["user_feedback"] = user_feedback
    
    def collect_implicit_feedback(self, interaction_id: str, 
                                 user_continued: bool, 
                                 follow_up_questions: int = 0,
                                 task_completed: bool = True):
        """Collect implicit feedback from user behavior."""
        if interaction_id not in self.pending_feedback:
            return
        
        # Score based on user behavior
        satisfaction_score = 0.5  # Base score
        
        if user_continued:
            satisfaction_score += 0.2
        
        if follow_up_questions == 0:
            satisfaction_score += 0.2  # No clarification needed
        elif follow_up_questions <= 2:
            satisfaction_score += 0.1
        
        if task_completed:
            satisfaction_score += 0.2
        
        satisfaction_score = min(1.0, satisfaction_score)
        
        implicit_feedback = {
            "satisfaction": satisfaction_score,
            "accuracy": 1.0 if task_completed else 0.5
        }
        
        self.pending_feedback[interaction_id]["implicit_feedback"] = implicit_feedback
    
    def auto_evaluate_response(self, response: BaseMessage, expected_criteria: Dict[str, Any]) -> Dict[str, float]:
        """Automatically evaluate response quality."""
        scores = {}
        
        # Check if response addresses key criteria
        response_lower = response.content.lower()
        
        # Completeness check
        if "required_elements" in expected_criteria:
            required_elements = expected_criteria["required_elements"]
            found_elements = sum(1 for element in required_elements if element.lower() in response_lower)
            completeness_score = found_elements / len(required_elements) if required_elements else 1.0
            scores["completeness"] = completeness_score
        
        # Relevance check
        if "keywords" in expected_criteria:
            keywords = expected_criteria["keywords"]
            found_keywords = sum(1 for keyword in keywords if keyword.lower() in response_lower)
            relevance_score = min(1.0, found_keywords / len(keywords) * 2) if keywords else 1.0
            scores["relevance"] = relevance_score
        
        # Length appropriateness
        target_length = expected_criteria.get("target_length", 300)
        actual_length = len(response.content)
        length_ratio = min(actual_length, target_length) / max(actual_length, target_length)
        scores["length_appropriateness"] = length_ratio
        
        return scores
    
    def setup_continuous_feedback(self, feedback_callback: Callable[[str, Dict], None]):
        """Setup continuous feedback collection."""
        self.feedback_callback = feedback_callback
    
    def get_pending_feedback_count(self) -> int:
        """Get number of pending feedback items."""
        return len(self.pending_feedback)
    
    def force_complete_interaction(self, interaction_id: str, default_success: bool = True):
        """Force complete an interaction with default values."""
        if interaction_id in self.pending_feedback:
            from langchain_core.messages import AIMessage
            default_response = AIMessage(content="Interaction completed")
            self.end_interaction(interaction_id, default_response, default_success)
    
    def cleanup_stale_interactions(self, max_age_seconds: int = 3600):
        """Clean up interactions older than max_age_seconds."""
        current_time = time.time()
        stale_interactions = []
        
        for interaction_id, data in self.pending_feedback.items():
            if current_time - data["start_time"] > max_age_seconds:
                stale_interactions.append(interaction_id)
        
        for interaction_id in stale_interactions:
            self.force_complete_interaction(interaction_id, default_success=False)
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback collection statistics."""
        return {
            "pending_interactions": len(self.pending_feedback),
            "tracked_start_times": len(self.interaction_start_times),
            "average_pending_age": self._calculate_average_pending_age()
        }
    
    def _calculate_average_pending_age(self) -> float:
        """Calculate average age of pending interactions."""
        if not self.pending_feedback:
            return 0.0
        
        current_time = time.time()
        total_age = sum(
            current_time - data["start_time"] 
            for data in self.pending_feedback.values()
        )
        
        return total_age / len(self.pending_feedback)