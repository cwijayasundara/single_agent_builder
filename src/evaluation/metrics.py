"""
Evaluation metrics collection and analysis.
"""
import time
import statistics
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict


class EvaluationMetrics:
    """Collects and analyzes evaluation metrics."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.time()
        self.evaluation_count = 0
        self.evaluator_stats = defaultdict(list)
        self.time_series_data = []
    
    def add_evaluation_result(self, result: Dict[str, Any]):
        """Add a single evaluation result."""
        timestamp = datetime.now()
        self.evaluation_count += 1
        
        # Store the full result with metadata
        full_result = {
            "timestamp": timestamp.isoformat(),
            "evaluation_id": self.evaluation_count,
            "results": result
        }
        
        self.results.append(full_result)
        
        # Update per-evaluator statistics
        for evaluator_name, eval_result in result.items():
            if isinstance(eval_result, dict) and "score" in eval_result:
                self.evaluator_stats[evaluator_name].append({
                    "score": eval_result["score"],
                    "timestamp": timestamp,
                    "reasoning": eval_result.get("reasoning", ""),
                    "metadata": eval_result.get("metadata", {})
                })
        
        # Add to time series
        self.time_series_data.append({
            "timestamp": timestamp,
            "evaluation_count": self.evaluation_count,
            "average_score": self._calculate_current_average_score(result)
        })
    
    def _calculate_current_average_score(self, result: Dict[str, Any]) -> float:
        """Calculate average score for the current result."""
        scores = []
        for eval_result in result.values():
            if isinstance(eval_result, dict) and "score" in eval_result:
                scores.append(eval_result["score"])
        
        return statistics.mean(scores) if scores else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of evaluation metrics."""
        if not self.results:
            return {
                "total_evaluations": 0,
                "metrics": {},
                "time_period": {"start": None, "end": None, "duration_hours": 0}
            }
        
        current_time = datetime.now()
        duration = (current_time - datetime.fromisoformat(self.results[0]["timestamp"]))
        
        summary = {
            "total_evaluations": self.evaluation_count,
            "time_period": {
                "start": self.results[0]["timestamp"],
                "end": self.results[-1]["timestamp"],
                "duration_hours": duration.total_seconds() / 3600
            },
            "evaluator_metrics": {},
            "overall_metrics": {},
            "trends": self._calculate_trends()
        }
        
        # Calculate per-evaluator metrics
        for evaluator_name, eval_data in self.evaluator_stats.items():
            scores = [item["score"] for item in eval_data]
            
            if scores:
                summary["evaluator_metrics"][evaluator_name] = {
                    "count": len(scores),
                    "average_score": statistics.mean(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "median_score": statistics.median(scores),
                    "score_distribution": self._calculate_score_distribution(scores)
                }
        
        # Calculate overall metrics
        all_scores = []
        for eval_data in self.evaluator_stats.values():
            all_scores.extend([item["score"] for item in eval_data])
        
        if all_scores:
            summary["overall_metrics"] = {
                "average_score": statistics.mean(all_scores),
                "min_score": min(all_scores),
                "max_score": max(all_scores),
                "std_deviation": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                "median_score": statistics.median(all_scores),
                "total_scores": len(all_scores)
            }
        
        return summary
    
    def _calculate_score_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate score distribution in bins."""
        distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for score in scores:
            if score < 0.2:
                distribution["0.0-0.2"] += 1
            elif score < 0.4:
                distribution["0.2-0.4"] += 1
            elif score < 0.6:
                distribution["0.4-0.6"] += 1
            elif score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1
        
        return distribution
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if len(self.time_series_data) < 2:
            return {"trend": "insufficient_data", "slope": 0, "direction": "stable"}
        
        # Simple linear trend calculation
        x_values = list(range(len(self.time_series_data)))
        y_values = [item["average_score"] for item in self.time_series_data]
        
        # Calculate slope using least squares
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x_squared = sum(x * x for x in x_values)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
        
        # Determine trend direction
        if slope > 0.01:
            direction = "improving"
        elif slope < -0.01:
            direction = "declining"
        else:
            direction = "stable"
        
        return {
            "slope": slope,
            "direction": direction,
            "trend": "linear_regression",
            "data_points": len(self.time_series_data)
        }
    
    def get_evaluator_performance(self, evaluator_name: str) -> Dict[str, Any]:
        """Get detailed performance data for a specific evaluator."""
        if evaluator_name not in self.evaluator_stats:
            return {"error": f"Evaluator '{evaluator_name}' not found"}
        
        eval_data = self.evaluator_stats[evaluator_name]
        scores = [item["score"] for item in eval_data]
        
        performance = {
            "evaluator_name": evaluator_name,
            "total_evaluations": len(eval_data),
            "score_statistics": {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "mode": statistics.mode(scores) if len(scores) > 1 else scores[0],
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores)
            },
            "recent_performance": self._get_recent_performance(eval_data),
            "score_history": scores[-10:],  # Last 10 scores
            "distribution": self._calculate_score_distribution(scores)
        }
        
        return performance
    
    def _get_recent_performance(self, eval_data: List[Dict], days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for recent period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_data = [
            item for item in eval_data 
            if datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) > cutoff_date
        ]
        
        if not recent_data:
            return {"message": f"No data in the last {days} days"}
        
        recent_scores = [item["score"] for item in recent_data]
        
        return {
            "period_days": days,
            "evaluations_count": len(recent_data),
            "average_score": statistics.mean(recent_scores),
            "score_trend": "improving" if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else "stable"
        }
    
    def export_detailed_results(self) -> List[Dict[str, Any]]:
        """Export all detailed evaluation results."""
        return self.results.copy()
    
    def get_time_series_data(self, evaluator_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get time series data for visualization."""
        if evaluator_name and evaluator_name in self.evaluator_stats:
            return [
                {
                    "timestamp": item["timestamp"],
                    "score": item["score"]
                }
                for item in self.evaluator_stats[evaluator_name]
            ]
        else:
            return self.time_series_data.copy()
    
    def clear(self):
        """Clear all collected metrics."""
        self.results.clear()
        self.evaluator_stats.clear()
        self.time_series_data.clear()
        self.evaluation_count = 0
        self.start_time = time.time()
    
    def filter_results(
        self, 
        evaluator_name: Optional[str] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Filter results based on criteria."""
        filtered_results = []
        
        for result in self.results:
            # Time filtering
            result_time = datetime.fromisoformat(result["timestamp"])
            if start_date and result_time < start_date:
                continue
            if end_date and result_time > end_date:
                continue
            
            # Evaluator filtering
            if evaluator_name:
                if evaluator_name not in result["results"]:
                    continue
                
                eval_result = result["results"][evaluator_name]
                if not isinstance(eval_result, dict) or "score" not in eval_result:
                    continue
                
                score = eval_result["score"]
                
                # Score filtering
                if min_score is not None and score < min_score:
                    continue
                if max_score is not None and score > max_score:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results