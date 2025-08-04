"""
Evaluation workflows for automated and batch evaluation processes.
"""
import asyncio
import time
import schedule
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from .evaluation_manager import EvaluationManager
from ..core.configurable_agent import ConfigurableAgent


class EvaluationWorkflow:
    """Manages evaluation workflows and automation."""
    
    def __init__(self, agent: ConfigurableAgent, evaluation_manager: Optional[EvaluationManager] = None):
        self.agent = agent
        self.evaluation_manager = evaluation_manager or agent.evaluation_manager
        self.scheduled_jobs = []
        self.workflow_history = []
        
        if not self.evaluation_manager:
            raise ValueError(
                "Evaluation manager is required for workflows. "
                "Please enable evaluation in the agent configuration and ensure evaluators are configured."
            )
    
    def run_batch_evaluation(
        self,
        test_cases: List[Dict[str, Any]],
        evaluator_names: Optional[List[str]] = None,
        max_workers: int = 5,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run batch evaluation on multiple test cases."""
        
        start_time = datetime.now()
        results = []
        errors = []
        
        def evaluate_single_case(test_case):
            try:
                input_text = test_case.get("input") or test_case.get("query", "")
                expected_output = test_case.get("expected_output")
                
                # Run agent
                agent_result = self.agent.run(input_text)
                
                # Prepare evaluation data
                input_data = {"query": input_text, "input": input_text}
                output_data = agent_result.copy()
                
                # Run evaluation
                eval_result = self.evaluation_manager.evaluate_single(
                    input_data=input_data,
                    output_data=output_data,
                    reference_output=expected_output,
                    evaluator_names=evaluator_names
                )
                
                return {
                    "test_case": test_case,
                    "agent_result": agent_result,
                    "evaluation": eval_result,
                    "status": "success"
                }
                
            except Exception as e:
                return {
                    "test_case": test_case,
                    "error": str(e),
                    "status": "error"
                }
        
        # Run evaluations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {
                executor.submit(evaluate_single_case, case): case 
                for case in test_cases
            }
            
            for future in as_completed(future_to_case):
                result = future.result()
                if result["status"] == "success":
                    results.append(result)
                else:
                    errors.append(result)
        
        end_time = datetime.now()
        
        # Compile summary
        workflow_result = {
            "workflow_id": f"batch_{int(time.time())}",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "total_cases": len(test_cases),
            "successful_cases": len(results),
            "failed_cases": len(errors),
            "success_rate": len(results) / len(test_cases) if test_cases else 0,
            "results": results,
            "errors": errors,
            "summary_metrics": self._calculate_batch_metrics(results)
        }
        
        if save_results:
            self.workflow_history.append(workflow_result)
        
        return workflow_result
    
    def _calculate_batch_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary metrics from batch results."""
        if not results:
            return {}
        
        # Collect all evaluation scores
        evaluator_scores = {}
        
        for result in results:
            evaluation = result.get("evaluation", {})
            for evaluator_name, eval_data in evaluation.items():
                if isinstance(eval_data, dict) and "score" in eval_data:
                    if evaluator_name not in evaluator_scores:
                        evaluator_scores[evaluator_name] = []
                    evaluator_scores[evaluator_name].append(eval_data["score"])
        
        # Calculate metrics for each evaluator
        summary = {}
        for evaluator_name, scores in evaluator_scores.items():
            if scores:
                summary[evaluator_name] = {
                    "count": len(scores),
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "scores_above_threshold": {
                        "0.5": len([s for s in scores if s >= 0.5]),
                        "0.7": len([s for s in scores if s >= 0.7]),
                        "0.9": len([s for s in scores if s >= 0.9])
                    }
                }
        
        return summary
    
    def run_a_b_test(
        self,
        test_cases: List[Dict[str, Any]],
        variant_configs: Dict[str, str],  # name -> config_file_path
        evaluator_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run A/B test comparing different agent configurations."""
        
        start_time = datetime.now()
        variant_results = {}
        
        for variant_name, config_path in variant_configs.items():
            print(f"Testing variant: {variant_name}")
            
            try:
                # Create agent with variant configuration
                variant_agent = ConfigurableAgent(config_path)
                variant_workflow = EvaluationWorkflow(variant_agent)
                
                # Run batch evaluation for this variant
                variant_result = variant_workflow.run_batch_evaluation(
                    test_cases=test_cases,
                    evaluator_names=evaluator_names,
                    save_results=False
                )
                
                variant_results[variant_name] = variant_result
                
            except Exception as e:
                variant_results[variant_name] = {
                    "error": str(e),
                    "status": "error"
                }
        
        end_time = datetime.now()
        
        # Compare variants
        comparison = self._compare_variants(variant_results)
        
        ab_test_result = {
            "test_id": f"ab_test_{int(time.time())}",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "variants": list(variant_configs.keys()),
            "test_cases_count": len(test_cases),
            "variant_results": variant_results,
            "comparison": comparison,
            "winner": comparison.get("recommended_variant"),
            "confidence": comparison.get("confidence_level", "low")
        }
        
        self.workflow_history.append(ab_test_result)
        
        return ab_test_result
    
    def _compare_variants(self, variant_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare A/B test variant results."""
        comparison = {
            "evaluator_comparisons": {},
            "overall_scores": {},
            "recommended_variant": None,
            "confidence_level": "low"
        }
        
        # Extract successful variants
        successful_variants = {
            name: result for name, result in variant_results.items()
            if "error" not in result and result.get("successful_cases", 0) > 0
        }
        
        if len(successful_variants) < 2:
            comparison["note"] = "Insufficient successful variants for comparison"
            return comparison
        
        # Compare each evaluator across variants
        for variant_name, result in successful_variants.items():
            summary_metrics = result.get("summary_metrics", {})
            comparison["overall_scores"][variant_name] = {}
            
            for evaluator_name, metrics in summary_metrics.items():
                if evaluator_name not in comparison["evaluator_comparisons"]:
                    comparison["evaluator_comparisons"][evaluator_name] = {}
                
                comparison["evaluator_comparisons"][evaluator_name][variant_name] = {
                    "average_score": metrics.get("average", 0),
                    "count": metrics.get("count", 0),
                    "max_score": metrics.get("max", 0)
                }
                
                comparison["overall_scores"][variant_name][evaluator_name] = metrics.get("average", 0)
        
        # Determine winner (simple approach: highest average across all evaluators)
        variant_averages = {}
        for variant_name, evaluator_scores in comparison["overall_scores"].items():
            if evaluator_scores:
                variant_averages[variant_name] = sum(evaluator_scores.values()) / len(evaluator_scores)
        
        if variant_averages:
            winner = max(variant_averages.items(), key=lambda x: x[1])
            comparison["recommended_variant"] = winner[0]
            comparison["winner_average_score"] = winner[1]
            
            # Simple confidence calculation
            scores = list(variant_averages.values())
            if len(scores) > 1:
                score_range = max(scores) - min(scores)
                if score_range > 0.2:
                    comparison["confidence_level"] = "high"
                elif score_range > 0.1:
                    comparison["confidence_level"] = "medium"
        
        return comparison
    
    def schedule_evaluation(
        self,
        dataset_name: str,
        frequency: str = "daily",
        time_str: str = "09:00",
        evaluator_names: Optional[List[str]] = None
    ):
        """Schedule periodic evaluation runs."""
        
        def evaluation_job():
            try:
                print(f"Running scheduled evaluation for dataset: {dataset_name}")
                result = self.agent.run_dataset_evaluation(
                    dataset_name=dataset_name,
                    evaluator_names=evaluator_names
                )
                
                # Store result
                scheduled_result = {
                    "type": "scheduled_evaluation",
                    "dataset": dataset_name,
                    "timestamp": datetime.now().isoformat(),
                    "result": result
                }
                self.workflow_history.append(scheduled_result)
                
                print(f"Scheduled evaluation completed: {result.get('experiment_id', 'N/A')}")
                
            except Exception as e:
                error_result = {
                    "type": "scheduled_evaluation_error",
                    "dataset": dataset_name,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                self.workflow_history.append(error_result)
                print(f"Scheduled evaluation failed: {e}")
        
        # Schedule the job
        if frequency == "daily":
            job = schedule.every().day.at(time_str).do(evaluation_job)
        elif frequency == "weekly":
            job = schedule.every().week.at(time_str).do(evaluation_job)
        elif frequency == "hourly":
            job = schedule.every().hour.do(evaluation_job)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        self.scheduled_jobs.append({
            "job": job,
            "dataset": dataset_name,
            "frequency": frequency,
            "time": time_str,
            "created_at": datetime.now().isoformat()
        })
        
        print(f"Scheduled {frequency} evaluation for {dataset_name} at {time_str}")
    
    def run_continuous_evaluation(
        self,
        test_cases: List[Dict[str, Any]],
        duration_hours: int = 24,
        interval_minutes: int = 60,
        evaluator_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run continuous evaluation over a time period."""
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        continuous_results = []
        
        print(f"Starting continuous evaluation for {duration_hours} hours")
        
        while datetime.now() < end_time:
            try:
                # Run batch evaluation
                batch_result = self.run_batch_evaluation(
                    test_cases=test_cases,
                    evaluator_names=evaluator_names,
                    save_results=False
                )
                
                # Add to continuous results
                continuous_results.append({
                    "batch_id": len(continuous_results) + 1,
                    "timestamp": datetime.now().isoformat(),
                    "result": batch_result
                })
                
                print(f"Completed batch {len(continuous_results)}, success rate: {batch_result.get('success_rate', 0):.2%}")
                
                # Wait for next interval
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("Continuous evaluation interrupted by user")
                break
            except Exception as e:
                print(f"Error in continuous evaluation: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        # Compile continuous evaluation summary
        continuous_summary = {
            "continuous_eval_id": f"continuous_{int(time.time())}",
            "start_time": start_time.isoformat(),
            "actual_end_time": datetime.now().isoformat(),
            "planned_duration_hours": duration_hours,
            "interval_minutes": interval_minutes,
            "total_batches": len(continuous_results),
            "results": continuous_results,
            "trend_analysis": self._analyze_continuous_trends(continuous_results)
        }
        
        self.workflow_history.append(continuous_summary)
        
        return continuous_summary
    
    def _analyze_continuous_trends(self, continuous_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in continuous evaluation results."""
        if len(continuous_results) < 2:
            return {"note": "Insufficient data for trend analysis"}
        
        # Extract success rates over time
        success_rates = [
            result["result"].get("success_rate", 0) 
            for result in continuous_results
        ]
        
        # Calculate trend
        if len(success_rates) >= 3:
            # Simple linear trend
            x_values = list(range(len(success_rates)))
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(success_rates)
            sum_xy = sum(x * y for x, y in zip(x_values, success_rates))
            sum_x_squared = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
            
            trend_direction = "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable"
        else:
            trend_direction = "insufficient_data"
            slope = 0
        
        return {
            "success_rate_trend": trend_direction,
            "trend_slope": slope,
            "initial_success_rate": success_rates[0] if success_rates else 0,
            "final_success_rate": success_rates[-1] if success_rates else 0,
            "average_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
            "success_rate_variance": self._calculate_variance(success_rates)
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def get_workflow_history(self, workflow_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        if workflow_type:
            return [
                wf for wf in self.workflow_history 
                if wf.get("type") == workflow_type or workflow_type in str(wf.get("workflow_id", ""))
            ]
        return self.workflow_history.copy()
    
    def run_scheduled_jobs(self):
        """Run all pending scheduled jobs (call this periodically)."""
        schedule.run_pending()
    
    def clear_scheduled_jobs(self):
        """Clear all scheduled jobs."""
        schedule.clear()
        self.scheduled_jobs.clear()
    
    def export_workflow_results(self, format_type: str = "json") -> str:
        """Export workflow results."""
        if format_type == "json":
            import json
            return json.dumps(self.workflow_history, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")